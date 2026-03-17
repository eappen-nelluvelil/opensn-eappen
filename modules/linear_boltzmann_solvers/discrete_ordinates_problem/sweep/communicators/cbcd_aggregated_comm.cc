// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_aggregated_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <set>

namespace opensn
{

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

CBCD_AggregatedCommunicator::CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                                                         const MPICommunicatorSet& comm_set,
                                                         size_t max_message_bytes,
                                                         const std::vector<AngleSetCapacity>& capacities)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    mpi_tag_(static_cast<int>(num_angle_sets_)),
    max_message_bytes_(max_message_bytes),
    incoming_mailboxes_(num_angle_sets_),
    send_batch_by_angle_set_(num_angle_sets_),
    angle_set_done_(num_angle_sets_)
{
  // Collect the union of source/destination MPI ranks across all angle sets.
  std::set<int> sources;
  std::set<int> destinations;
  size_t total_outgoing_faces = 0;

  for (size_t i = 0; i < angle_sets.size(); ++i)
  {
    const auto* as = angle_sets[i];
    const auto& spds = as->GetSPDS();
    for (int dep : spds.GetLocationDependencies())
      sources.insert(dep);
    for (int succ : spds.GetLocationSuccessors())
      destinations.insert(succ);

    total_outgoing_faces += capacities[i].outgoing_faces;
    size_t num_incoming = capacities[i].incoming_faces;
    if (num_incoming > 0)
      incoming_mailboxes_[as->GetID()].Preallocate(num_incoming + 1);
  }

  source_ranks_.assign(sources.begin(), sources.end());

  // Create one outgoing ring buffer per destination MPI rank.
  int queue_idx = 0;
  for (int dest : destinations)
  {
    DestinationQueue dq;
    dq.dest_rank = dest;
    dq.queue = std::make_unique<LockFreeRingBuffer<OutgoingFaceData>>();
    if (total_outgoing_faces > 0)
      dq.queue->Preallocate(total_outgoing_faces + 1);
    outgoing_queues_.push_back(std::move(dq));
    dest_to_queue_index_[dest] = queue_idx++;
  }

  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);

  if (max_message_bytes > 0)
    recv_buffer_.Data().reserve(max_message_bytes);
}

CBCD_AggregatedCommunicator::~CBCD_AggregatedCommunicator()
{
  if (comm_thread_.joinable())
    Stop();
}

// ---------------------------------------------------------------------------
// Worker thread interface
// ---------------------------------------------------------------------------

void
CBCD_AggregatedCommunicator::SignalAngleSetComplete(size_t angle_set_id)
{
  assert(angle_set_id < num_angle_sets_);
  angle_set_done_[angle_set_id].store(true, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void
CBCD_AggregatedCommunicator::Start()
{
  stop_requested_.store(false, std::memory_order_relaxed);
  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);
  in_flight_sends_.clear();
  comm_thread_ = std::thread(&CBCD_AggregatedCommunicator::CommThreadLoop, this);
}

void
CBCD_AggregatedCommunicator::Stop()
{
  stop_requested_.store(true, std::memory_order_release);
  if (comm_thread_.joinable())
    comm_thread_.join();
}

// ---------------------------------------------------------------------------
// Communication thread
// ---------------------------------------------------------------------------

void
CBCD_AggregatedCommunicator::CommThreadLoop()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::CommThreadLoop");

  while (true)
  {
    bool work_done = SerializeAndSend();
    work_done |= ProbeAndReceive();
    work_done |= PollInFlightSends();

    if (stop_requested_.load(std::memory_order_acquire) and AllAngleSetsComplete())
    {
      // Final flush: drain any remaining outgoing entries.
      SerializeAndSend();

      // Wait for all in-flight sends to complete.
      while (not in_flight_sends_.empty())
      {
        PollInFlightSends();
        if (not in_flight_sends_.empty())
          std::this_thread::yield();
      }
      break;
    }

    if (not work_done)
      std::this_thread::yield();
  }
}

bool
CBCD_AggregatedCommunicator::SerializeAndSend()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::SerializeAndSend");

  bool sent_any = false;

  for (auto& dq : outgoing_queues_)
  {
    dq.queue->GetReadySlots(slot_cache_);
    if (slot_cache_.empty())
      continue;

    size_t current_payload_bytes = sizeof(size_t); // num_active_angle_sets header
    size_t active_angle_sets = 0;
    size_t slots_processed = 0;

    // Serialize accumulated entries into a single MPI message and initiate async send.
    auto SendBatch = [&]()
    {
      InFlightSend ifs;
      ifs.data.Data().resize(current_payload_bytes);
      size_t offset = 0;

      auto WriteBytes = [&](const void* ptr, size_t size)
      {
        std::memcpy(ifs.data.Data().data() + offset, ptr, size);
        offset += size;
      };

      WriteBytes(&active_angle_sets, sizeof(size_t));

      for (size_t as_id = 0; as_id < num_angle_sets_; ++as_id)
      {
        auto& ptrs = send_batch_by_angle_set_[as_id];
        if (ptrs.empty())
          continue;

        WriteBytes(&as_id, sizeof(size_t));
        size_t num_entries = ptrs.size();
        WriteBytes(&num_entries, sizeof(size_t));

        for (const auto* e : ptrs)
        {
          WriteBytes(&e->cell_global_id, sizeof(uint64_t));
          WriteBytes(&e->face_id, sizeof(unsigned int));
          const size_t data_size = e->psi_data.size();
          WriteBytes(&data_size, sizeof(size_t));
          WriteBytes(e->psi_data.data(), data_size * sizeof(double));
        }
        ptrs.clear();
      }

      const auto& comm = comm_set_.LocICommunicator(dq.dest_rank);
      auto mapped_rank = comm_set_.MapIonJ(dq.dest_rank, dq.dest_rank);
      ifs.request = comm.isend(mapped_rank, mpi_tag_, ifs.data.Data());
      in_flight_sends_.push_back(std::move(ifs));
    };

    for (size_t s = 0; s < slot_cache_.size(); ++s)
    {
      auto* slot = slot_cache_[s];
      const auto& entry = slot->payload;

      size_t entry_bytes = sizeof(uint64_t) + sizeof(unsigned int) + sizeof(size_t) +
                           entry.psi_data.size() * sizeof(double);

      // Split into multiple messages if the batch exceeds the maximum message size.
      if (max_message_bytes_ > 0 and
          current_payload_bytes + entry_bytes > max_message_bytes_ and
          active_angle_sets > 0)
      {
        SendBatch();
        dq.queue->FreeSlots(slots_processed);

        current_payload_bytes = sizeof(size_t);
        active_angle_sets = 0;
        slots_processed = 0;
      }

      auto& ptrs = send_batch_by_angle_set_[entry.angle_set_id];
      if (ptrs.empty())
      {
        active_angle_sets++;
        current_payload_bytes += sizeof(size_t) + sizeof(size_t); // as_id + num_entries
      }
      ptrs.push_back(&entry);
      current_payload_bytes += entry_bytes;
      slots_processed++;
    }

    if (active_angle_sets > 0)
    {
      SendBatch();
      dq.queue->FreeSlots(slots_processed);
    }

    sent_any = true;
  }
  return sent_any;
}

bool
CBCD_AggregatedCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::ProbeAndReceive");

  bool received_any = false;
  const int my_rank = opensn::mpi_comm.rank();

  for (int source_loc : source_ranks_)
  {
    const auto& comm = comm_set_.LocICommunicator(my_rank);
    auto mapped_source = comm_set_.MapIonJ(source_loc, my_rank);

    mpi::Status status;
    while (comm.iprobe(mapped_source, mpi_tag_, status))
    {
      received_any = true;
      int num_bytes = status.count<std::byte>();
      recv_buffer_.Data().resize(num_bytes);

      comm.recv(mapped_source, status.tag(), recv_buffer_.Data().data(), num_bytes);

      // Deserialize the aggregated wire format (see class-level doc for layout).
      recv_buffer_.Seek(0);
      auto num_active_angle_sets = recv_buffer_.Read<size_t>();

      for (size_t as_batch = 0; as_batch < num_active_angle_sets; ++as_batch)
      {
        auto as_id = recv_buffer_.Read<size_t>();
        auto num_entries = recv_buffer_.Read<size_t>();
        assert(as_id < num_angle_sets_);

        auto& slot = incoming_mailboxes_[as_id].ReserveSlot();
        auto& batch = slot.payload;
        batch.resize(num_entries);

        for (size_t e = 0; e < num_entries; ++e)
        {
          batch[e].cell_global_id = recv_buffer_.Read<uint64_t>();
          batch[e].face_id = recv_buffer_.Read<unsigned int>();

          auto data_size = recv_buffer_.Read<size_t>();
          batch[e].psi_data.resize(data_size);

          std::memcpy(batch[e].psi_data.data(),
                      &recv_buffer_.Data()[recv_buffer_.Offset()],
                      data_size * sizeof(double));

          recv_buffer_.Seek(recv_buffer_.Offset() + data_size * sizeof(double));
        }

        incoming_mailboxes_[as_id].PublishSlot(slot);
      }
    }
  }
  return received_any;
}

bool
CBCD_AggregatedCommunicator::PollInFlightSends()
{
  bool completed_any = false;
  // O(1) swap-and-pop removal of completed sends.
  for (size_t i = 0; i < in_flight_sends_.size();)
  {
    if (mpi::test(in_flight_sends_[i].request))
    {
      completed_any = true;
      in_flight_sends_[i] = std::move(in_flight_sends_.back());
      in_flight_sends_.pop_back();
    }
    else
    {
      ++i;
    }
  }
  return completed_any;
}

bool
CBCD_AggregatedCommunicator::AllAngleSetsComplete() const
{
  for (size_t i = 0; i < num_angle_sets_; ++i)
    if (not angle_set_done_[i].load(std::memory_order_acquire))
      return false;

  for (const auto& dq : outgoing_queues_)
    if (not dq.queue->Empty())
      return false;

  return true;
}

} // namespace opensn

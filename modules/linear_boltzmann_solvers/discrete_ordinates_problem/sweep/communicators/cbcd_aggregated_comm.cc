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
                                                         size_t max_single_message_size_in_bytes,
                                                         const std::vector<AngleSetCapacity>& capacities)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    incoming_mailboxes_(num_angle_sets_),
    aggregated_tag_(static_cast<int>(num_angle_sets_)),
    max_single_message_size_in_bytes_(max_single_message_size_in_bytes),
    angle_set_done_(num_angle_sets_)
{
  std::set<int> temp_dependencies;
  std::set<int> temp_successors;

  size_t total_outgoing_faces = 0;

  for (size_t i = 0; i < angle_sets.size(); ++i)
  {
    const auto* as = angle_sets[i];
    const auto& spds = as->GetSPDS();
    for (int dep : spds.GetLocationDependencies())
      temp_dependencies.insert(dep);
    for (int succ : spds.GetLocationSuccessors())
      temp_successors.insert(succ);

    total_outgoing_faces += capacities[i].outgoing_faces;
    size_t num_incoming = capacities[i].incoming_faces;
    if (num_incoming > 0)
      incoming_mailboxes_[as->GetID()].Preallocate(num_incoming + 1);
  }

  location_dependencies_.assign(temp_dependencies.begin(), temp_dependencies.end());

  int queue_idx = 0;
  for (int succ : temp_successors)
  {
    NeighborQueue nq;
    nq.dest_location = succ;
    nq.queue = std::make_unique<LockFreeRingBuffer<OutgoingEntry>>();
    if (total_outgoing_faces > 0)
      nq.queue->Preallocate(total_outgoing_faces + 1);
    outgoing_queues_.push_back(std::move(nq));
    dest_to_queue_index_[succ] = queue_idx++;
  }

  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);

  // Reserve receive buffer
  if (max_single_message_size_in_bytes > 0)
    persistent_recv_buffer_.Data().reserve(max_single_message_size_in_bytes);
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
CBCD_AggregatedCommunicator::EnqueueOutgoing(int dest_location,
                                             size_t angle_set_id,
                                             uint64_t cell_global_id,
                                             unsigned int face_id,
                                             const double* psi_data,
                                             size_t data_size)
{
  auto it = dest_to_queue_index_.find(dest_location);
  assert(it != dest_to_queue_index_.end());
  auto& queue = outgoing_queues_[it->second].queue;

  // Wait-free slot reservation via atomic fetch_add
  auto& slot = queue->ReserveSlot();

  slot.payload.angle_set_id = angle_set_id;
  slot.payload.cell_global_id = cell_global_id;
  slot.payload.face_id = face_id;

  // Reuse pre-allocated vector capacity from previous iterations
  slot.payload.psi_data.resize(data_size);
  std::memcpy(slot.payload.psi_data.data(), psi_data, data_size * sizeof(double));

  queue->PublishSlot(slot);
}

std::vector<std::vector<IncomingEntry>>
CBCD_AggregatedCommunicator::DequeueIncoming(size_t angle_set_id)
{
  assert(angle_set_id < num_angle_sets_);
  return incoming_mailboxes_[angle_set_id].Drain();
}

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
  pending_sends_.clear();
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

  std::vector<std::vector<const OutgoingEntry*>> by_angle_set(num_angle_sets_);

  while (true)
  {
    FlushOutgoing(by_angle_set);
    ProbeAndReceive();
    PollPendingSends();

    if (stop_requested_.load(std::memory_order_acquire) and AllWorkComplete())
    {
      FlushOutgoing(by_angle_set);

      while (not pending_sends_.empty())
      {
        PollPendingSends();
        if (not pending_sends_.empty())
          std::this_thread::yield();
      }
      break;
    }

    std::this_thread::yield();
  }
}

bool
CBCD_AggregatedCommunicator::FlushOutgoing(
  std::vector<std::vector<const OutgoingEntry*>>& by_angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::FlushOutgoing");

  bool flushed_any = false;

  for (auto& nq : outgoing_queues_)
  {
    auto ready_slots = nq.queue->GetReadySlots();
    if (ready_slots.empty())
      continue;

    size_t current_payload_bytes = sizeof(size_t); // num_anglesets_in_batch header
    size_t active_angle_sets = 0;
    size_t slots_processed = 0;

    auto DispatchBatch = [&]()
    {
      PendingSend ps;
      ps.data.Data().resize(current_payload_bytes);
      size_t offset = 0;

      auto WriteBytes = [&](const void* ptr, size_t size)
      {
        std::memcpy(ps.data.Data().data() + offset, ptr, size);
        offset += size;
      };

      WriteBytes(&active_angle_sets, sizeof(size_t));

      for (size_t as_id = 0; as_id < num_angle_sets_; ++as_id)
      {
        auto& ptrs = by_angle_set[as_id];
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

      const auto& comm = comm_set_.LocICommunicator(nq.dest_location);
      auto dest_rank = comm_set_.MapIonJ(nq.dest_location, nq.dest_location);
      ps.request = comm.isend(dest_rank, aggregated_tag_, ps.data.Data());
      pending_sends_.push_back(std::move(ps));
    };

    for (size_t s = 0; s < ready_slots.size(); ++s)
    {
      auto* slot = ready_slots[s];
      const auto& entry = slot->payload;

      size_t entry_bytes = sizeof(uint64_t) + sizeof(unsigned int) + sizeof(size_t) +
                           entry.psi_data.size() * sizeof(double);

      // Split messages if they exceed the maximum size
      if (max_single_message_size_in_bytes_ > 0 and
          current_payload_bytes + entry_bytes > max_single_message_size_in_bytes_ and
          active_angle_sets > 0)
      {
        DispatchBatch();
        nq.queue->FreeSlots(slots_processed);

        current_payload_bytes = sizeof(size_t);
        active_angle_sets = 0;
        slots_processed = 0;
      }

      auto& ptrs = by_angle_set[entry.angle_set_id];
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
      DispatchBatch();
      nq.queue->FreeSlots(slots_processed);
    }

    flushed_any = true;
  }
  return flushed_any;
}

void
CBCD_AggregatedCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::ProbeAndReceive");

  const int my_rank = opensn::mpi_comm.rank();

  for (int locJ : location_dependencies_)
  {
    const auto& comm = comm_set_.LocICommunicator(my_rank);
    auto source_rank = comm_set_.MapIonJ(locJ, my_rank);

    mpi::Status status;
    while (comm.iprobe(source_rank, aggregated_tag_, status))
    {
      int num_bytes = status.count<std::byte>();
      persistent_recv_buffer_.Data().resize(num_bytes);

      comm.recv(source_rank, status.tag(),
                persistent_recv_buffer_.Data().data(), num_bytes);

      persistent_recv_buffer_.Seek(0);

      auto num_as_in_batch = persistent_recv_buffer_.Read<size_t>();

      for (size_t as_batch = 0; as_batch < num_as_in_batch; ++as_batch)
      {
        auto as_id = persistent_recv_buffer_.Read<size_t>();
        auto num_entries = persistent_recv_buffer_.Read<size_t>();
        assert(as_id < num_angle_sets_);

        // Use ring buffer slot for incoming batch
        auto& slot = incoming_mailboxes_[as_id].ReserveSlot();
        auto& batch = slot.payload;
        batch.resize(num_entries);

        for (size_t e = 0; e < num_entries; ++e)
        {
          batch[e].cell_global_id = persistent_recv_buffer_.Read<uint64_t>();
          batch[e].face_id = persistent_recv_buffer_.Read<unsigned int>();

          auto data_size = persistent_recv_buffer_.Read<size_t>();
          batch[e].psi_data.resize(data_size);

          std::memcpy(batch[e].psi_data.data(),
                      &persistent_recv_buffer_.Data()[persistent_recv_buffer_.Offset()],
                      data_size * sizeof(double));

          persistent_recv_buffer_.Seek(persistent_recv_buffer_.Offset() +
                                       data_size * sizeof(double));
        }

        incoming_mailboxes_[as_id].PublishSlot(slot);
      }
    }
  }
}

void
CBCD_AggregatedCommunicator::PollPendingSends()
{
  // O(1) swap-and-pop removal logic
  for (size_t i = 0; i < pending_sends_.size();)
  {
    if (mpi::test(pending_sends_[i].request))
    {
      pending_sends_[i] = std::move(pending_sends_.back());
      pending_sends_.pop_back();
    }
    else
    {
      ++i;
    }
  }
}

bool
CBCD_AggregatedCommunicator::AllWorkComplete() const
{
  for (size_t i = 0; i < num_angle_sets_; ++i)
    if (not angle_set_done_[i].load(std::memory_order_acquire))
      return false;

  for (const auto& nq : outgoing_queues_)
    if (not nq.queue->Empty())
      return false;

  return true;
}

} // namespace opensn

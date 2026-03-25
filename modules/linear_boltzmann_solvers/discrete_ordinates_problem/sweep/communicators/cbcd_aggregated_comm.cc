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
                                                         size_t max_message_bytes)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    mpi_tag_(static_cast<int>(num_angle_sets_)),
    incoming_mailboxes_(num_angle_sets_),
    angle_set_done_(num_angle_sets_)
{
  std::set<int> sources;
  std::set<int> destinations;

  for (size_t i = 0; i < angle_sets.size(); ++i)
  {
    const auto* as = angle_sets[i];
    const auto& spds = as->GetSPDS();
    for (int dep : spds.GetLocationDependencies())
      sources.insert(dep);
    for (int succ : spds.GetLocationSuccessors())
      destinations.insert(succ);
  }

  source_ranks_.assign(sources.begin(), sources.end());

  // Create one Treiber stack per destination MPI rank (no pre-allocation needed).
  int queue_idx = 0;
  for (int dest : destinations)
  {
    NeighborQueue nq;
    nq.dest_location = dest;
    nq.queue = std::make_unique<LockFreeTreiberStack<ByteArray>>();
    outgoing_queues_.push_back(std::move(nq));
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

int
CBCD_AggregatedCommunicator::GetQueueIndex(int dest_location) const
{
  auto it = dest_to_queue_index_.find(dest_location);
  if (it == dest_to_queue_index_.end())
    return -1;
  return it->second;
}

void
CBCD_AggregatedCommunicator::EnqueuePrepackedByIndex(int queue_index, ByteArray&& data)
{
  assert(queue_index >= 0 and queue_index < static_cast<int>(outgoing_queues_.size()));
  outgoing_queues_[queue_index].queue->Push(std::move(data));
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
    bool work_done = FlushOutgoing();
    work_done |= ProbeAndReceive();
    work_done |= PollInFlightSends();

    if (stop_requested_.load(std::memory_order_acquire) and AllWorkComplete())
    {
      FlushOutgoing();

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
CBCD_AggregatedCommunicator::FlushOutgoing()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::FlushOutgoing");

  bool any_sent = false;

  for (auto& nq : outgoing_queues_)
  {
    // Drain pre-packed wire-format sections from the Treiber stack.
    // Each section was packed by CopyOutgoingPsiBackToHost on a worker thread:
    //   [angle_set_id : size_t][num_entries : size_t][entries...]
    auto sections = nq.queue->Drain();
    if (sections.empty())
      continue;
    any_sent = true;

    // Compute total message size: header + all sections concatenated.
    size_t total_bytes = sizeof(size_t); // num_sections header
    for (const auto& section : sections)
      total_bytes += section.Size();

    // Assemble the message: [num_sections][section_0][section_1]...
    InFlightSend ifs;
    ifs.data.Data().resize(total_bytes);
    size_t offset = 0;

    size_t num_sections = sections.size();
    std::memcpy(ifs.data.Data().data() + offset, &num_sections, sizeof(size_t));
    offset += sizeof(size_t);

    for (auto& section : sections)
    {
      std::memcpy(ifs.data.Data().data() + offset, section.Data().data(), section.Size());
      offset += section.Size();
    }

    // Dispatch the MPI Isend.
    const auto& comm = comm_set_.LocICommunicator(nq.dest_location);
    auto dest_rank = comm_set_.MapIonJ(nq.dest_location, nq.dest_location);
    ifs.request = comm.isend(dest_rank, mpi_tag_, ifs.data.Data());
    in_flight_sends_.push_back(std::move(ifs));
  }

  return any_sent;
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

      recv_buffer_.Seek(0);
      auto num_sections = recv_buffer_.Read<size_t>();

      for (size_t s = 0; s < num_sections; ++s)
      {
        auto as_id = recv_buffer_.Read<size_t>();
        auto num_entries = recv_buffer_.Read<size_t>();
        assert(as_id < num_angle_sets_);

        std::vector<IncomingFaceData> batch(num_entries);

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

        incoming_mailboxes_[as_id].Push(std::move(batch));
      }
    }
  }
  return received_any;
}

bool
CBCD_AggregatedCommunicator::PollInFlightSends()
{
  bool completed_any = false;
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

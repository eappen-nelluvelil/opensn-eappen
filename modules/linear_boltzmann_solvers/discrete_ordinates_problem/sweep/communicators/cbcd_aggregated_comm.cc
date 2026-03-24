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
                                                         size_t begin_as,
                                                         int aggregated_tag)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    begin_as_(begin_as),
    incoming_mailboxes_(num_angle_sets_),
    aggregated_tag_(aggregated_tag >= 0 ? aggregated_tag
                                        : static_cast<int>(begin_as_ + num_angle_sets_)),
    angle_set_done_(num_angle_sets_)
{
  std::set<int> temp_dependencies;
  std::set<int> temp_successors;

  for (const auto* as : angle_sets)
  {
    const auto& spds = as->GetSPDS();
    for (int dep : spds.GetLocationDependencies())
      temp_dependencies.insert(dep);
    for (int succ : spds.GetLocationSuccessors())
      temp_successors.insert(succ);
  }

  location_dependencies_.assign(temp_dependencies.begin(), temp_dependencies.end());

  int queue_idx = 0;
  for (int succ : temp_successors)
  {
    NeighborQueue nq;
    nq.dest_location = succ;
    nq.queue = std::make_unique<LockFreeTreiberStack<std::vector<OutgoingEntry>>>();
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

int CBCD_AggregatedCommunicator::GetQueueIndex(int dest_location) const
{
  auto it = dest_to_queue_index_.find(dest_location);
  if (it == dest_to_queue_index_.end())
    return -1;
  return it->second;
}

void
CBCD_AggregatedCommunicator::EnqueueOutgoingBatchByIndex(int queue_index,
                                                         std::vector<OutgoingEntry>&& batch)
{
  assert(queue_index >= 0 and queue_index < static_cast<int>(outgoing_queues_.size()));
  outgoing_queues_[queue_index].queue->Push(std::move(batch));
}

std::vector<std::vector<IncomingEntry>>
CBCD_AggregatedCommunicator::DequeueIncoming(size_t angle_set_id)
{
  assert(angle_set_id >= begin_as_ and angle_set_id < begin_as_ + num_angle_sets_);
  return incoming_mailboxes_[angle_set_id - begin_as_].Drain();
}

void
CBCD_AggregatedCommunicator::SignalAngleSetComplete(size_t angle_set_id)
{
  assert(angle_set_id >= begin_as_ and angle_set_id < begin_as_ + num_angle_sets_);
  angle_set_done_[angle_set_id - begin_as_].store(true, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void
CBCD_AggregatedCommunicator::Start()
{
  // Skip launching a comm thread if this communicator has no angle sets
  // (can happen when num_workers > num_angle_sets).
  if (num_angle_sets_ == 0)
    return;

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

  // Maybe pre-reserve pointer vectors to avoid dynamic growth during FlushOutgoing?
  // I think this can be done exactly instead of using a heuristic

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
CBCD_AggregatedCommunicator::FlushOutgoing(std::vector<std::vector<const OutgoingEntry*>>& by_angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::FlushOutgoing");

  bool any_sent = false;

  for (auto& nq : outgoing_queues_)
  {
    // Drain returns std::vector<std::vector<OutgoingEntry>>
    // (each element is one batch push)
    auto entry_batches = nq.queue->Drain();
    if (entry_batches.empty())
      continue;
    any_sent = true;

    // Flatten the batches-of-batches in to angle_set grouping and compute total
    // bytes needed
    size_t total_bytes = sizeof(size_t); // num_anglesets_in_batch
    size_t active_anglesets = 0;
    for (const auto& batch : entry_batches)
    {
      for (const auto& entry : batch)
      {
        assert(entry.angle_set_id >= begin_as_ and
               entry.angle_set_id < begin_as_ + num_angle_sets_);
        auto& ptrs = by_angle_set[entry.angle_set_id - begin_as_];
        if (ptrs.empty())
        {
          active_anglesets++;
          total_bytes += sizeof(size_t) + sizeof(size_t); // angle_set_id + num_entries
        }
        ptrs.push_back(&entry);

        total_bytes += sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(size_t)
                      + entry.psi_data.size() * sizeof(double);
      }
    }

    // Allocate and manually pack the contiguous byte buffer
    PendingSend ps;
    ps.data.Data().resize(total_bytes);
    size_t offset = 0;

    auto WriteBytes = [&](const void* ptr, size_t size)
    {
      std::memcpy(ps.data.Data().data() + offset, ptr, size);
      offset += size;
    };

    WriteBytes(&active_anglesets, sizeof(size_t));

    for (size_t local_id = 0; local_id < num_angle_sets_; ++local_id)
    {
      auto& ptrs = by_angle_set[local_id];
      if (ptrs.empty())
        continue;
      // Write global angle set ID to the wire format
      size_t global_id = local_id + begin_as_;
      WriteBytes(&global_id, sizeof(size_t));
      size_t num_entries = ptrs.size();
      WriteBytes(&num_entries, sizeof(size_t));
      for (const auto* e : ptrs)
      {
        WriteBytes(&e->cell_global_id, sizeof(std::uint64_t));
        WriteBytes(&e->face_id, sizeof(unsigned int));
        size_t data_size = e->psi_data.size();
        WriteBytes(&data_size, sizeof(size_t));
        WriteBytes(e->psi_data.data(), data_size * sizeof(double));
      }
      // Reset for the next destination partition
      ptrs.clear();
    }

    // Dispatch the MPI Isend request
    const auto& comm = comm_set_.LocICommunicator(nq.dest_location);
    auto dest_rank = comm_set_.MapIonJ(nq.dest_location, nq.dest_location);
    ps.request = comm.isend(dest_rank, aggregated_tag_, ps.data.Data());
    pending_sends_.push_back(std::move(ps));
  }

  return any_sent;
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

      // Use pointer+size overload to go directly to MPI_Recv.
      // The vector overload would do a redundant blocking MPI_Probe.
      comm.recv(source_rank, status.tag(),
                persistent_recv_buffer_.Data().data(), num_bytes);

      persistent_recv_buffer_.Seek(0);

      auto num_as_in_batch = persistent_recv_buffer_.Read<size_t>();

      for (size_t as_batch = 0; as_batch < num_as_in_batch; ++as_batch)
      {
        auto global_as_id = persistent_recv_buffer_.Read<size_t>();
        auto num_entries = persistent_recv_buffer_.Read<size_t>();
        assert(global_as_id >= begin_as_ and global_as_id < begin_as_ + num_angle_sets_);
        auto local_as_id = global_as_id - begin_as_;

        std::vector<IncomingEntry> batch;
        batch.reserve(num_entries);

        for (size_t e = 0; e < num_entries; ++e)
        {
          IncomingEntry entry;
          entry.cell_global_id = persistent_recv_buffer_.Read<uint64_t>();
          entry.face_id = persistent_recv_buffer_.Read<unsigned int>();

          auto data_size = persistent_recv_buffer_.Read<size_t>();
          entry.psi_data.resize(data_size);

          std::memcpy(entry.psi_data.data(),
                      &persistent_recv_buffer_.Data()[persistent_recv_buffer_.Offset()],
                      data_size * sizeof(double));

          persistent_recv_buffer_.Seek(persistent_recv_buffer_.Offset() + data_size * sizeof(double));

          batch.push_back(std::move(entry));
        }

        incoming_mailboxes_[local_as_id].Push(std::move(batch));
      }
    }
  }
}

void
CBCD_AggregatedCommunicator::PollPendingSends()
{
  // O(1) swap-and-pop removal logic
  for (size_t i = 0; i < pending_sends_.size(); )
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
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

namespace opensn
{

CBCD_AggregatedCommunicator::CBCD_AggregatedCommunicator(
  const std::vector<AngleSet*>& angle_sets,
  const MPICommunicatorSet& comm_set)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    incoming_mailboxes_(num_angle_sets_),
    aggregated_tag_(static_cast<int>(num_angle_sets_)),
    angle_set_done_(num_angle_sets_)
{
  // Collect the union of all location dependencies and successors across all angle sets
  for (const auto* as : angle_sets)
  {
    const auto& spds = as->GetSPDS();
    for (int dep : spds.GetLocationDependencies())
      all_location_dependencies_.insert(dep);
    for (int succ : spds.GetLocationSuccessors())
      all_location_successors_.insert(succ);
  }

  // Pre-create per-destination queues for all successors
  for (int succ : all_location_successors_)
    outgoing_queues_[succ]; // default-construct

  // Initialize completion flags
  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);
}

CBCD_AggregatedCommunicator::~CBCD_AggregatedCommunicator()
{
  if (comm_thread_.joinable())
    Stop();
}

void
CBCD_AggregatedCommunicator::EnqueueOutgoing(int dest_location,
                                              size_t angle_set_id,
                                              uint64_t cell_global_id,
                                              unsigned int face_id,
                                              std::vector<double>&& psi_data)
{
  auto it = outgoing_queues_.find(dest_location);
  assert(it != outgoing_queues_.end());

  auto& queue = it->second;
  std::lock_guard<std::mutex> lock(queue.mutex);
  queue.entries.push_back({angle_set_id, cell_global_id, face_id, std::move(psi_data)});
}

std::vector<IncomingEntry>
CBCD_AggregatedCommunicator::DequeueIncoming(size_t angle_set_id)
{
  assert(angle_set_id < num_angle_sets_);
  auto& mailbox = incoming_mailboxes_[angle_set_id];

  std::vector<IncomingEntry> result;
  {
    std::lock_guard<std::mutex> lock(mailbox.mutex);
    result.swap(mailbox.entries);
  }
  return result;
}

void
CBCD_AggregatedCommunicator::SignalAngleSetComplete(size_t angle_set_id)
{
  assert(angle_set_id < num_angle_sets_);
  angle_set_done_[angle_set_id].store(true, std::memory_order_release);
}

void
CBCD_AggregatedCommunicator::Start()
{
  stop_requested_.store(false, std::memory_order_relaxed);
  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);
  pending_sends_.clear();
  last_flush_time_ = std::chrono::steady_clock::now();
  comm_thread_ = std::thread(&CBCD_AggregatedCommunicator::CommThreadLoop, this);
}

void
CBCD_AggregatedCommunicator::Stop()
{
  stop_requested_.store(true, std::memory_order_release);
  if (comm_thread_.joinable())
    comm_thread_.join();
}

void
CBCD_AggregatedCommunicator::CommThreadLoop()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::CommThreadLoop");

  while (true)
  {
    FlushOutgoing();
    ProbeAndReceive();
    PollPendingSends();

    if (stop_requested_.load(std::memory_order_acquire) and AllWorkComplete())
    {
      // Final flush to ensure no data is left in queues
      FlushOutgoing();

      // Wait for all pending sends to complete
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

void
CBCD_AggregatedCommunicator::FlushOutgoing()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::FlushOutgoing");

  auto now = std::chrono::steady_clock::now();

  for (auto& [dest, queue] : outgoing_queues_)
  {
    std::vector<OutgoingEntry> entries_to_send;

    {
      std::lock_guard<std::mutex> lock(queue.mutex);
      if (queue.entries.empty())
        continue;

      bool should_flush =
        queue.entries.size() >= kFlushSizeThreshold or
        (now - last_flush_time_) >= kFlushTimeInterval or
        stop_requested_.load(std::memory_order_relaxed);

      if (not should_flush)
        continue;

      entries_to_send.swap(queue.entries);
    }

    // Group entries by angle set id for serialization
    std::map<size_t, std::vector<const OutgoingEntry*>> by_angle_set;
    for (const auto& entry : entries_to_send)
      by_angle_set[entry.angle_set_id].push_back(&entry);

    // Serialize using ByteArray wire format:
    // [num_angle_sets_in_batch : size_t]
    // repeated {
    //   [angle_set_id : size_t]
    //   [num_entries : size_t]
    //   repeated {
    //     [cell_global_id : uint64_t]
    //     [face_id : unsigned int]
    //     [data_size : size_t]
    //     [psi_data : double x data_size]
    //   }
    // }
    ByteArray buffer;
    size_t num_as_in_batch = by_angle_set.size();
    buffer.Write(num_as_in_batch);

    for (const auto& [as_id, entry_ptrs] : by_angle_set)
    {
      buffer.Write(as_id);
      size_t num_entries = entry_ptrs.size();
      buffer.Write(num_entries);

      for (const auto* entry : entry_ptrs)
      {
        buffer.Write(entry->cell_global_id);
        buffer.Write(entry->face_id);
        size_t data_size = entry->psi_data.size();
        buffer.Write(data_size);
        for (double val : entry->psi_data)
          buffer.Write(val);
      }
    }

    // Send via MPI
    const auto& comm = comm_set_.LocICommunicator(dest);
    auto dest_rank = comm_set_.MapIonJ(dest, dest);

    PendingSend ps;
    ps.data = std::move(buffer);
    ps.request = comm.isend(dest_rank, aggregated_tag_, ps.data.Data());
    pending_sends_.push_back(std::move(ps));
  }

  last_flush_time_ = now;
}

void
CBCD_AggregatedCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::ProbeAndReceive");

  const int my_rank = opensn::mpi_comm.rank();

  for (int locJ : all_location_dependencies_)
  {
    const auto& comm = comm_set_.LocICommunicator(my_rank);
    auto source_rank = comm_set_.MapIonJ(locJ, my_rank);

    mpi::Status status;
    while (comm.iprobe(source_rank, aggregated_tag_, status))
    {
      int num_bytes = status.count<std::byte>();
      std::vector<std::byte> recv_buffer(num_bytes);
      comm.recv(source_rank, status.tag(), recv_buffer.data(), num_bytes);

      ByteArray data_array(std::move(recv_buffer));

      // Deserialize
      auto num_as_in_batch = data_array.Read<size_t>();

      for (size_t as_batch = 0; as_batch < num_as_in_batch; ++as_batch)
      {
        auto as_id = data_array.Read<size_t>();
        auto num_entries = data_array.Read<size_t>();

        assert(as_id < num_angle_sets_);
        auto& mailbox = incoming_mailboxes_[as_id];
        std::lock_guard<std::mutex> lock(mailbox.mutex);

        for (size_t e = 0; e < num_entries; ++e)
        {
          IncomingEntry entry;
          entry.cell_global_id = data_array.Read<uint64_t>();
          entry.face_id = data_array.Read<unsigned int>();
          auto data_size = data_array.Read<size_t>();
          entry.psi_data.resize(data_size);
          for (size_t k = 0; k < data_size; ++k)
            entry.psi_data[k] = data_array.Read<double>();

          mailbox.entries.push_back(std::move(entry));
        }
      }
    } // while iprobe
  }
}

void
CBCD_AggregatedCommunicator::PollPendingSends()
{
  pending_sends_.erase(
    std::remove_if(pending_sends_.begin(),
                   pending_sends_.end(),
                   [](PendingSend& ps) { return mpi::test(ps.request); }),
    pending_sends_.end());
}

bool
CBCD_AggregatedCommunicator::AllWorkComplete() const
{
  // Check all angle sets are done
  for (size_t i = 0; i < num_angle_sets_; ++i)
    if (not angle_set_done_[i].load(std::memory_order_acquire))
      return false;

  // Check all outgoing queues are empty
  for (const auto& [dest, queue] : outgoing_queues_)
  {
    // We can't lock here as we're const, but after all angle sets signal done,
    // no more enqueues happen. The atomic acquire fence above ensures visibility.
    // A quick non-locked check is safe at this point.
    if (not queue.entries.empty())
      return false;
  }

  return true;
}

} // namespace opensn

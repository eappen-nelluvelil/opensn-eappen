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
  const int my_rank = opensn::mpi_comm.rank();

  for (size_t i = 0; i < angle_sets.size(); ++i)
  {
    const auto* as = angle_sets[i];
    const auto& spds = as->GetSPDS();
    for (int dep : spds.GetLocationDependencies())
      sources.insert(dep);
    for (int succ : spds.GetLocationSuccessors())
      destinations.insert(succ);
  }

  source_queues_.reserve(sources.size());
  for (int source : sources)
    source_queues_.push_back({comm_set_.MapIonJ(source, my_rank)});

  // Create one Treiber stack per destination MPI rank.
  outgoing_queues_.reserve(destinations.size());
  dest_to_queue_index_.reserve(destinations.size());
  int queue_idx = 0;
  for (int dest : destinations)
  {
    NeighborQueue nq;
    nq.dest_location = dest;
    nq.dest_rank = comm_set_.MapIonJ(dest, dest);
    nq.queue = std::make_unique<LockFreeTreiberStack<ByteArray>>();
    outgoing_queues_.push_back(std::move(nq));
    dest_to_queue_index_[dest] = queue_idx++;
  }

  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);

  if (max_message_bytes > 0)
    recv_buffer_.Data().reserve(max_message_bytes);

  // Pre-allocate Treiber stack nodes to eliminate heap allocations during sweeps.
  // Outgoing: at most one section per angle set may be queued before the comm thread drains.
  for (auto& nq : outgoing_queues_)
    nq.queue->Preallocate(num_angle_sets_);

  // Incoming: at most one section per source rank may arrive before the worker thread drains.
  const size_t num_sources = source_queues_.size();
  for (auto& mailbox : incoming_mailboxes_)
    mailbox.Preallocate(num_sources);

  // Recycler: workers may return up to one ByteArray per mailbox drain between comm thread
  // recycler drains.  Pre-allocate conservatively for all angle sets × sources.
  incoming_recycler_.Preallocate(num_angle_sets_ * num_sources);
  incoming_reuse_cache_.reserve(num_angle_sets_ * num_sources);
  in_flight_sends_.reserve(outgoing_queues_.size());
  send_buffer_pool_.reserve(outgoing_queues_.size());
}

CBCD_AggregatedCommunicator::~CBCD_AggregatedCommunicator()
{
  if (comm_thread_.joinable())
    Stop();
}

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
    // Skip empty queues to avoid touching the send-buffer pool.
    if (nq.queue->Empty())
      continue;

    ByteArray send_buf = AcquireSendBuffer();
    size_t num_sections = 0;
    send_buf.Data().resize(sizeof(size_t));

    bool has_data = nq.queue->DrainAndProcess(
      [&](ByteArray&& section)
      {
        send_buf.Append(section);
        ++num_sections;
      });

    if (not has_data)
    {
      ReleaseSendBuffer(std::move(send_buf));
      continue;
    }
    any_sent = true;

    std::memcpy(send_buf.Data().data(), &num_sections, sizeof(size_t));

    InFlightSend ifs;
    ifs.data = std::move(send_buf);
    const auto& comm = comm_set_.LocICommunicator(nq.dest_location);
    ifs.request = comm.isend(nq.dest_rank, mpi_tag_, ifs.data.Data());
    in_flight_sends_.push_back(std::move(ifs));
  }

  return any_sent;
}

bool
CBCD_AggregatedCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::ProbeAndReceive");

  incoming_recycler_.DrainAndProcess(
    [this](ByteArray&& buf) { incoming_reuse_cache_.push_back(std::move(buf)); });

  bool received_any = false;
  const int my_rank = opensn::mpi_comm.rank();
  const auto& recv_comm = comm_set_.LocICommunicator(my_rank);

  for (const auto& source_queue : source_queues_)
  {
    mpi::Status status;
    while (recv_comm.iprobe(source_queue.mapped_rank, mpi_tag_, status))
    {
      received_any = true;
      int num_bytes = status.count<std::byte>();
      recv_buffer_.Data().resize(num_bytes);

      recv_comm.recv(source_queue.mapped_rank, status.tag(), recv_buffer_.Data().data(), num_bytes);

      const auto* raw = recv_buffer_.Data().data();
      size_t offset = 0;

      size_t num_sections;
      std::memcpy(&num_sections, raw + offset, sizeof(size_t));
      offset += sizeof(size_t);

      for (size_t s = 0; s < num_sections; ++s)
      {
        size_t as_id;
        std::memcpy(&as_id, raw + offset, sizeof(size_t));
        offset += sizeof(size_t);
        assert(as_id < num_angle_sets_);

        const size_t section_start = offset;
        size_t num_entries;
        std::memcpy(&num_entries, raw + offset, sizeof(size_t));
        offset += sizeof(size_t);

        for (size_t e = 0; e < num_entries; ++e)
        {
          offset += sizeof(uint64_t) + sizeof(unsigned int);
          size_t data_size;
          std::memcpy(&data_size, raw + offset, sizeof(size_t));
          offset += sizeof(size_t);
          offset += data_size * sizeof(double);
        }

        const size_t section_size = offset - section_start;
        ByteArray section;
        if (not incoming_reuse_cache_.empty())
        {
          section = std::move(incoming_reuse_cache_.back());
          incoming_reuse_cache_.pop_back();
        }
        section.Data().resize(section_size);
        std::memcpy(section.Data().data(), raw + section_start, section_size);

        incoming_mailboxes_[as_id].Push(std::move(section));
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
      ReleaseSendBuffer(std::move(in_flight_sends_[i].data));
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

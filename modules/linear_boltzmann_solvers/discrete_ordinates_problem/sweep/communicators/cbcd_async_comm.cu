// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <set>

namespace opensn
{

CBCD_AsynchronousCommunicator::CBCD_AsynchronousCommunicator(
  const std::vector<AngleSet*>& angle_sets,
  const MPICommunicatorSet& comm_set,
  size_t max_message_bytes)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    max_message_bytes_(max_message_bytes),
    mpi_tag_(static_cast<int>(num_angle_sets_)),
    incoming_mailboxes_(num_angle_sets_),
    angle_set_done_(num_angle_sets_)
{
  std::set<int> sources;
  std::set<int> destinations;
  my_rank_ = opensn::mpi_comm.rank();

  for (size_t i = 0; i < angle_sets.size(); ++i)
  {
    const auto* as = angle_sets[i];
    const auto& spds = as->GetSPDS();
    for (int dep : spds.GetLocationDependencies())
      sources.insert(dep);
    for (int succ : spds.GetLocationSuccessors())
      destinations.insert(succ);
  }

  const size_t num_sources = sources.size();
  recv_communicator_ = &comm_set_.LocICommunicator(my_rank_);
  num_outgoing_stacks_per_queue_ = std::max<size_t>(1, num_angle_sets_);

  outgoing_queues_.reserve(destinations.size());
  dest_to_queue_index_.reserve(destinations.size());
  outgoing_stacks_.resize(destinations.size() * num_outgoing_stacks_per_queue_);
  int queue_idx = 0;
  for (int dest : destinations)
  {
    NeighborQueue queue;
    queue.communicator = &comm_set_.LocICommunicator(dest);
    queue.dest_rank = comm_set_.MapIonJ(dest, dest);
    queue.shard_offset = static_cast<size_t>(queue_idx) * num_outgoing_stacks_per_queue_;
    outgoing_queues_.push_back(std::move(queue));
    dest_to_queue_index_[dest] = queue_idx++;
  }

  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);

  for (auto& stack : outgoing_stacks_)
    stack.Preallocate(1);

  for (auto& mailbox : incoming_mailboxes_)
    mailbox.Preallocate(num_sources);

  recv_buffer_recycler_.Preallocate(num_sources);
  recv_buffer_reuse_cache_.reserve(num_sources);
  in_flight_sends_.reserve(outgoing_queues_.size());
  send_buffer_pool_.reserve(outgoing_queues_.size());
}

CBCD_AsynchronousCommunicator::~CBCD_AsynchronousCommunicator()
{
  if (comm_thread_.joinable())
    Stop();
}

int
CBCD_AsynchronousCommunicator::GetQueueIndex(int dest_location) const
{
  const auto it = dest_to_queue_index_.find(dest_location);
  if (it == dest_to_queue_index_.end())
    return -1;
  return it->second;
}

void
CBCD_AsynchronousCommunicator::EnqueuePrepackedByIndex(int queue_index,
                                                       size_t producer_id,
                                                       ByteArray&& data)
{
  assert(queue_index >= 0 and queue_index < static_cast<int>(outgoing_queues_.size()));
  const auto& queue = outgoing_queues_[queue_index];
  const size_t shard_index = std::min(producer_id, num_outgoing_stacks_per_queue_ - 1);
  outgoing_stacks_[queue.shard_offset + shard_index].Push(std::move(data));
}

void
CBCD_AsynchronousCommunicator::SignalAngleSetComplete(size_t angle_set_id)
{
  assert(angle_set_id < num_angle_sets_);
  angle_set_done_[angle_set_id].store(true, std::memory_order_release);
}

void
CBCD_AsynchronousCommunicator::Start()
{
  stop_requested_.store(false, std::memory_order_relaxed);
  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);
  in_flight_sends_.clear();
  comm_thread_ = std::thread(&CBCD_AsynchronousCommunicator::CommThreadLoop, this);
}

void
CBCD_AsynchronousCommunicator::Stop()
{
  stop_requested_.store(true, std::memory_order_release);
  if (comm_thread_.joinable())
    comm_thread_.join();
}

void
CBCD_AsynchronousCommunicator::CommThreadLoop()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::CommThreadLoop");

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

std::shared_ptr<CBCD_AsynchronousCommunicator::IncomingSection::IncomingBuffer>
CBCD_AsynchronousCommunicator::AcquireReceiveBuffer(size_t num_bytes)
{
  auto recv_buffer = std::make_shared<IncomingSection::IncomingBuffer>();
  recv_buffer->recycler = &recv_buffer_recycler_;
  if (not recv_buffer_reuse_cache_.empty())
  {
    recv_buffer->data = std::move(recv_buffer_reuse_cache_.back());
    recv_buffer_reuse_cache_.pop_back();
  }
  else if (max_message_bytes_ > 0)
    recv_buffer->data.Data().reserve(max_message_bytes_);
  recv_buffer->data.Data().resize(num_bytes);
  return recv_buffer;
}

bool
CBCD_AsynchronousCommunicator::FlushOutgoing()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::FlushOutgoing");

  bool any_sent = false;
  for (auto& queue : outgoing_queues_)
  {
    ByteArray send_buffer = AcquireSendBuffer();
    auto& send_data = send_buffer.Data();
    send_data.resize(Wire::AGGREGATE_HEADER_BYTES);
    size_t num_sections = 0;

    for (size_t shard = 0; shard < num_outgoing_stacks_per_queue_; ++shard)
      outgoing_stacks_[queue.shard_offset + shard].DrainAndProcess(
        [&](ByteArray&& section)
        {
          const auto& sec = section.Data();
          const auto old_size = send_data.size();
          send_data.resize(old_size + sec.size());
          std::memcpy(send_data.data() + old_size, sec.data(), sec.size());
          ++num_sections;
        });

    if (num_sections == 0)
    {
      ReleaseSendBuffer(std::move(send_buffer));
      continue;
    }
    any_sent = true;

    Wire::StoreSize(send_data.data(), num_sections);

    InFlightSend send;
    send.data = std::move(send_buffer);
    send.request = queue.communicator->isend(queue.dest_rank, mpi_tag_, send.data.Data());
    in_flight_sends_.push_back(std::move(send));
  }

  return any_sent;
}

bool
CBCD_AsynchronousCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::ProbeAndReceive");

  recv_buffer_recycler_.DrainAndProcess(
    [this](ByteArray&& buf) { recv_buffer_reuse_cache_.push_back(std::move(buf)); });

  bool received_any = false;

  mpi::Status status;
  while (recv_communicator_->iprobe(ANY_SOURCE, mpi_tag_, status))
  {
    received_any = true;
    const int source_rank = status.source();
    const int num_bytes = status.count<std::byte>();

    auto recv_buffer = AcquireReceiveBuffer(num_bytes);
    recv_communicator_->recv(source_rank, status.tag(), recv_buffer->data.Data().data(), num_bytes);

    const auto* ptr = recv_buffer->data.Data().data();
    const size_t num_sections = Wire::LoadSize(ptr);
    for (size_t s = 0; s < num_sections; ++s)
    {
      const size_t angle_set_id = Wire::LoadSize(ptr);
      const size_t num_entries = Wire::LoadSize(ptr);
      assert(angle_set_id < num_angle_sets_);

      const auto* section_data = ptr;
      for (size_t e = 0; e < num_entries; ++e)
      {
        const auto entry_header = Wire::LoadEntryHeader(ptr);
        ptr += entry_header.data_size * sizeof(double);
      }
      incoming_mailboxes_[angle_set_id].Push(
        IncomingSection{recv_buffer, section_data, num_entries});
    }
  }
  return received_any;
}

bool
CBCD_AsynchronousCommunicator::PollInFlightSends()
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
      ++i;
  }
  return completed_any;
}

bool
CBCD_AsynchronousCommunicator::AllWorkComplete() const
{
  for (size_t i = 0; i < num_angle_sets_; ++i)
    if (not angle_set_done_[i].load(std::memory_order_acquire))
      return false;

  for (const auto& queue : outgoing_queues_)
    for (size_t shard = 0; shard < num_outgoing_stacks_per_queue_; ++shard)
      if (not outgoing_stacks_[queue.shard_offset + shard].Empty())
        return false;

  return true;
}

} // namespace opensn

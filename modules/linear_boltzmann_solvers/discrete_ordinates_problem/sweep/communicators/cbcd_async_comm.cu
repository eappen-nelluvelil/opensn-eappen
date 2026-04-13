// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/mpi/mpi_utils.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <thread>
#include <unordered_map>

namespace opensn
{

struct CBCD_AsynchronousCommunicator::Impl
{
  struct ReceiveBufferPoolState
  {
    LockFreeTreiberStack<std::shared_ptr<IncomingSection::IncomingBuffer>> recycler;
    std::atomic<bool> shutting_down{false};

    void Preallocate(std::size_t count) { recycler.Preallocate(count); }

    void Push(std::shared_ptr<IncomingSection::IncomingBuffer>&& buffer)
    {
      recycler.Push(std::move(buffer));
    }

    template <class F>
    bool DrainAndProcess(F&& callback)
    {
      return recycler.DrainAndProcess(std::forward<F>(callback));
    }

    bool DrainAndDiscard() { return recycler.DrainAndDiscard(); }
  };

  struct NeighborQueue
  {
    const mpi::Communicator* comm = nullptr;
    int dest_rank = 0;
    std::size_t shard_offset = 0;
  };

  struct InFlightSend
  {
    ByteArray data;
  };

  explicit Impl(const MPICommunicatorSet& in_comm_set,
                const std::size_t in_num_angle_sets,
                const std::size_t in_max_message_bytes)
    : comm_set(in_comm_set),
      num_angle_sets(in_num_angle_sets),
      max_message_bytes(in_max_message_bytes),
      mpi_tag(static_cast<int>(in_num_angle_sets)),
      angle_sets_done(in_num_angle_sets)
  {
  }

  const MPICommunicatorSet& comm_set;
  std::size_t num_angle_sets;
  std::size_t max_message_bytes;
  int mpi_tag;
  int my_rank = 0;
  const mpi::Communicator* recv_comm = nullptr;
  std::vector<NeighborQueue> outgoing_queues;
  std::size_t num_outgoing_stacks_per_queue = 1;
  std::vector<LockFreeSPSCQueue<ByteArray>> outgoing_stacks;
  std::size_t num_active_queue_words = 0;
  std::unique_ptr<std::atomic<std::uint64_t>[]> active_outgoing_queue_bits;
  std::unordered_map<int, int> dest_to_queue_index;
  std::shared_ptr<ReceiveBufferPoolState> recv_buffer_pool_state;
  std::vector<LockFreeSPSCQueue<IncomingSection>> incoming_mailboxes;
  std::vector<std::shared_ptr<IncomingSection::IncomingBuffer>> recv_buffer_reuse_cache;
  std::vector<mpi::Request> in_flight_send_requests;
  std::vector<InFlightSend> in_flight_sends;
  std::vector<int> completed_send_indices;
  std::vector<std::uint8_t> completed_send_mask;
  std::vector<ByteArray> send_buffer_pool;
  std::atomic<bool> stop_requested{false};
  std::vector<std::atomic<bool>> angle_sets_done;
  std::thread comm_thread;
};

CBCD_AsynchronousCommunicator::CBCD_AsynchronousCommunicator(
  const std::vector<AngleSet*>& angle_sets,
  const MPICommunicatorSet& comm_set,
  const std::vector<int>& outgoing_dest_localities,
  const std::vector<std::size_t>& outgoing_shard_capacities,
  const std::vector<std::size_t>& incoming_mailbox_capacities,
  const size_t max_message_bytes)
  : impl_(std::make_unique<Impl>(comm_set, angle_sets.size(), max_message_bytes))
{
  assert(incoming_mailbox_capacities.size() == impl_->num_angle_sets);
  impl_->my_rank = opensn::mpi_comm.rank();
  std::set<int> sources;
  for (std::size_t i = 0; i < impl_->num_angle_sets; ++i)
  {
    const auto* angle_set = angle_sets[i];
    const auto& spds = angle_set->GetSPDS();
    for (const auto& dep : spds.GetLocationDependencies())
      sources.insert(dep);
  }
  const std::size_t num_sources = sources.size();
  impl_->recv_comm = &impl_->comm_set.LocICommunicator(impl_->my_rank);
  impl_->num_outgoing_stacks_per_queue = std::max<std::size_t>(1, impl_->num_angle_sets);
  impl_->recv_buffer_pool_state = std::make_shared<Impl::ReceiveBufferPoolState>();

  impl_->incoming_mailboxes.reserve(impl_->num_angle_sets);
  for (const auto capacity : incoming_mailbox_capacities)
    impl_->incoming_mailboxes.emplace_back(capacity);

  impl_->outgoing_queues.reserve(outgoing_dest_localities.size());
  impl_->dest_to_queue_index.reserve(outgoing_dest_localities.size());
  impl_->outgoing_stacks.reserve(outgoing_dest_localities.size() *
                                 impl_->num_outgoing_stacks_per_queue);
  impl_->num_active_queue_words = (outgoing_dest_localities.size() + 63) / 64;
  impl_->active_outgoing_queue_bits =
    std::make_unique<std::atomic<std::uint64_t>[]>(impl_->num_active_queue_words);
  assert(outgoing_shard_capacities.size() ==
         outgoing_dest_localities.size() * impl_->num_outgoing_stacks_per_queue);

  int queue_idx = 0;
  for (const auto& dest : outgoing_dest_localities)
  {
    Impl::NeighborQueue queue;
    queue.comm = &impl_->comm_set.LocICommunicator(dest);
    queue.dest_rank = impl_->comm_set.MapIonJ(dest, dest);
    queue.shard_offset = static_cast<std::size_t>(queue_idx) * impl_->num_outgoing_stacks_per_queue;
    impl_->outgoing_queues.push_back(queue);
    impl_->dest_to_queue_index[dest] = queue_idx++;
  }

  for (const auto capacity : outgoing_shard_capacities)
    impl_->outgoing_stacks.emplace_back(capacity);

  for (std::size_t i = 0; i < impl_->num_angle_sets; ++i)
    impl_->angle_sets_done[i].store(false, std::memory_order_relaxed);
  for (std::size_t i = 0; i < impl_->num_active_queue_words; ++i)
    impl_->active_outgoing_queue_bits[i].store(0, std::memory_order_relaxed);

  impl_->recv_buffer_pool_state->Preallocate(num_sources);
  impl_->recv_buffer_reuse_cache.reserve(num_sources);
  impl_->in_flight_send_requests.reserve(impl_->outgoing_queues.size());
  impl_->in_flight_sends.reserve(impl_->outgoing_queues.size());
  impl_->completed_send_indices.reserve(impl_->outgoing_queues.size());
  impl_->send_buffer_pool.reserve(impl_->outgoing_queues.size());
}

CBCD_AsynchronousCommunicator::~CBCD_AsynchronousCommunicator()
{
  if (impl_->comm_thread.joinable())
    Stop();
  impl_->recv_buffer_pool_state->shutting_down.store(true, std::memory_order_release);
  impl_->recv_buffer_reuse_cache.clear();
  impl_->recv_buffer_pool_state->DrainAndDiscard();
}

void
CBCD_AsynchronousCommunicator::ResolveQueueIndices(std::span<const int> dest_localities,
                                                   std::span<int> queue_indices) const
{
  assert(dest_localities.size() == queue_indices.size());
  for (std::size_t i = 0; i < dest_localities.size(); ++i)
  {
    const auto it = impl_->dest_to_queue_index.find(dest_localities[i]);
    queue_indices[i] = (it == impl_->dest_to_queue_index.end()) ? -1 : it->second;
  }
}

void
CBCD_AsynchronousCommunicator::EnqueuePrepackedByIndex(const int queue_index,
                                                       const std::size_t producer_id,
                                                       ByteArray&& data)
{
  assert(queue_index >= 0 and queue_index < static_cast<int>(impl_->outgoing_queues.size()));
  const auto& queue = impl_->outgoing_queues[queue_index];
  const auto shard_index = std::min(producer_id, impl_->num_outgoing_stacks_per_queue - 1);
  assert(impl_->outgoing_stacks[queue.shard_offset + shard_index].Push(std::move(data)));
  const auto word_index = static_cast<std::size_t>(queue_index) / 64;
  const auto bit_index = static_cast<std::size_t>(queue_index) % 64;
  impl_->active_outgoing_queue_bits[word_index].fetch_or(std::uint64_t{1} << bit_index,
                                                         std::memory_order_release);
}

bool
CBCD_AsynchronousCommunicator::DrainIncomingImpl(const std::size_t angle_set_id,
                                                 const void* callback_context,
                                                 IncomingSectionCallback callback)
{
  assert(angle_set_id < impl_->num_angle_sets);
  return impl_->incoming_mailboxes[angle_set_id].DrainAndProcess(
    [this, callback_context, callback](IncomingSection&& section)
    {
      callback(callback_context, std::move(section));
      if (section.buffer and section.buffer.use_count() == 1)
      {
        section.buffer->data.Data().clear();
        impl_->recv_buffer_pool_state->Push(std::move(section.buffer));
      }
    });
}

void
CBCD_AsynchronousCommunicator::SignalAngleSetComplete(const size_t angle_set_id)
{
  assert(angle_set_id < impl_->num_angle_sets);
  impl_->angle_sets_done[angle_set_id].store(true, std::memory_order_release);
}

void
CBCD_AsynchronousCommunicator::Start()
{
  impl_->stop_requested.store(false, std::memory_order_relaxed);
  for (std::size_t i = 0; i < impl_->num_angle_sets; ++i)
    impl_->angle_sets_done[i].store(false, std::memory_order_relaxed);
  for (std::size_t i = 0; i < impl_->num_active_queue_words; ++i)
    impl_->active_outgoing_queue_bits[i].store(0, std::memory_order_relaxed);
  impl_->in_flight_send_requests.clear();
  impl_->in_flight_sends.clear();
  impl_->comm_thread = std::thread(&CBCD_AsynchronousCommunicator::CommThreadLoop, this);
}

void
CBCD_AsynchronousCommunicator::Stop()
{
  impl_->stop_requested.store(true, std::memory_order_release);
  if (impl_->comm_thread.joinable())
    impl_->comm_thread.join();
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
    if (impl_->stop_requested.load(std::memory_order_acquire) and AllWorkComplete())
    {
      work_done |= FlushOutgoing();
      while (not impl_->in_flight_sends.empty())
      {
        work_done |= PollInFlightSends();
        if (not impl_->in_flight_sends.empty())
          std::this_thread::yield();
      }
      break;
    }
    if (not work_done)
      std::this_thread::yield();
  }
}

std::shared_ptr<CBCD_AsynchronousCommunicator::IncomingSection::IncomingBuffer>
CBCD_AsynchronousCommunicator::AcquireReceiveBuffer(const std::size_t num_bytes)
{
  if (impl_->recv_buffer_reuse_cache.empty())
    impl_->recv_buffer_pool_state->DrainAndProcess(
      [this](const std::shared_ptr<IncomingSection::IncomingBuffer>& buffer)
      { impl_->recv_buffer_reuse_cache.push_back(buffer); });

  std::shared_ptr<IncomingSection::IncomingBuffer> recv_buffer;
  if (not impl_->recv_buffer_reuse_cache.empty())
  {
    recv_buffer = std::move(impl_->recv_buffer_reuse_cache.back());
    impl_->recv_buffer_reuse_cache.pop_back();
  }
  else
  {
    struct Recycler
    {
      static std::shared_ptr<IncomingSection::IncomingBuffer> Make()
      {
        return std::make_shared<IncomingSection::IncomingBuffer>();
      }
    };

    recv_buffer = Recycler::Make();
  }

  if (impl_->max_message_bytes > 0 and
      recv_buffer->data.Data().capacity() < impl_->max_message_bytes)
    recv_buffer->data.Data().reserve(impl_->max_message_bytes);
  recv_buffer->data.Data().resize(num_bytes);
  return recv_buffer;
}

ByteArray
CBCD_AsynchronousCommunicator::AcquireSendBuffer()
{
  if (not impl_->send_buffer_pool.empty())
  {
    ByteArray buf = std::move(impl_->send_buffer_pool.back());
    impl_->send_buffer_pool.pop_back();
    return buf;
  }

  ByteArray buf;
  if (impl_->max_message_bytes > 0)
    buf.Data().reserve(impl_->max_message_bytes);
  return buf;
}

void
CBCD_AsynchronousCommunicator::ReleaseSendBuffer(ByteArray&& buf)
{
  buf.Data().clear();
  impl_->send_buffer_pool.push_back(std::move(buf));
}

bool
CBCD_AsynchronousCommunicator::FlushOutgoing()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::FlushOutgoing");

  bool any_sent = false;
  for (std::size_t word_index = 0; word_index < impl_->num_active_queue_words; ++word_index)
  {
    auto pending =
      impl_->active_outgoing_queue_bits[word_index].exchange(0, std::memory_order_acq_rel);
    while (pending != 0)
    {
      const auto bit_index = static_cast<std::size_t>(std::countr_zero(pending));
      pending &= (pending - 1);
      const auto queue_index = word_index * 64 + bit_index;
      if (queue_index < impl_->outgoing_queues.size())
        any_sent |= FlushActiveOutgoingQueue(queue_index);
    }
  }
  return any_sent;
}

bool
CBCD_AsynchronousCommunicator::FlushActiveOutgoingQueue(const std::size_t queue_index)
{
  const auto& queue = impl_->outgoing_queues[queue_index];
  bool any_sent = false;

  while (true)
  {
    ByteArray first_section;
    bool have_first_section = false;
    ByteArray send_buffer;
    auto* send_data = static_cast<std::vector<std::byte>*>(nullptr);
    std::size_t num_sections = 0;

    for (std::size_t shard = 0; shard < impl_->num_outgoing_stacks_per_queue; ++shard)
      impl_->outgoing_stacks[queue.shard_offset + shard].DrainAndProcess(
        [this, &first_section, &have_first_section, &send_buffer, &send_data, &num_sections](
          ByteArray&& section)
        {
          const auto& section_data = section.Data();
          if (not have_first_section)
          {
            first_section = std::move(section);
            have_first_section = true;
            num_sections = 1;
            return;
          }

          if (num_sections == 1)
          {
            send_buffer = std::move(first_section);
            send_data = &send_buffer.Data();
            if (impl_->max_message_bytes > 0 && send_data->capacity() < impl_->max_message_bytes)
              send_data->reserve(impl_->max_message_bytes);
          }

          const auto payload_bytes = section_data.size() - Wire::AGGREGATE_HEADER_BYTES;
          const auto old_size = send_data->size();
          send_data->resize(old_size + payload_bytes);
          std::memcpy(send_data->data() + old_size,
                      section_data.data() + Wire::AGGREGATE_HEADER_BYTES,
                      payload_bytes);
          ++num_sections;
        });

    if (num_sections == 0)
      break;

    any_sent = true;
    if (num_sections == 1)
    {
      Wire::StoreSize(first_section.Data().data(), 1);
      Impl::InFlightSend send;
      send.data = std::move(first_section);
      impl_->in_flight_send_requests.push_back(
        queue.comm->isend(queue.dest_rank, impl_->mpi_tag, send.data.Data()));
      impl_->in_flight_sends.push_back(std::move(send));
      continue;
    }

    Wire::StoreSize(send_data->data(), num_sections);
    Impl::InFlightSend send;
    send.data = std::move(send_buffer);
    impl_->in_flight_send_requests.push_back(
      queue.comm->isend(queue.dest_rank, impl_->mpi_tag, send.data.Data()));
    impl_->in_flight_sends.push_back(std::move(send));
  }

  return any_sent;
}

bool
CBCD_AsynchronousCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::ProbeAndReceive");

  bool received_any = false;
  mpi::Status status;
  while (impl_->recv_comm->iprobe(ANY_SOURCE, impl_->mpi_tag, status))
  {
    received_any = true;
    const int source_rank = status.source();
    const int num_bytes = status.count<std::byte>();
    auto recv_buffer = AcquireReceiveBuffer(static_cast<std::size_t>(num_bytes));
    impl_->recv_comm->recv(source_rank, status.tag(), recv_buffer->data.Data().data(), num_bytes);

    const auto* ptr = recv_buffer->data.Data().data();
    const auto num_sections = Wire::LoadSize(ptr);
    if (num_sections == 0)
      continue;
    for (std::size_t s = 0; s < num_sections; ++s)
    {
      const auto angle_set_id = Wire::LoadSize(ptr);
      const auto num_entries = Wire::LoadSize(ptr);
      assert(angle_set_id < impl_->num_angle_sets);

      const auto* section_data = ptr;
      for (std::size_t e = 0; e < num_entries; ++e)
      {
        const auto& entry_header = Wire::LoadEntryHeader(ptr);
        ptr += entry_header.data_size * sizeof(double);
      }
      assert(impl_->incoming_mailboxes[angle_set_id].Push(
        IncomingSection{recv_buffer, section_data, num_entries}));
    }
  }
  return received_any;
}

bool
CBCD_AsynchronousCommunicator::PollInFlightSends()
{
  if (impl_->in_flight_sends.empty())
    return false;

  const int outcount =
    mpi_utils::TestSome(impl_->in_flight_send_requests, impl_->completed_send_indices);
  if (outcount <= 0)
    return false;

  const auto num_in_flight = impl_->in_flight_sends.size();
  if (impl_->completed_send_mask.size() < num_in_flight)
    impl_->completed_send_mask.resize(num_in_flight);
  std::fill_n(impl_->completed_send_mask.begin(), num_in_flight, std::uint8_t{0});
  for (const int completed_index : impl_->completed_send_indices)
    impl_->completed_send_mask[static_cast<std::size_t>(completed_index)] = std::uint8_t{1};

  std::size_t write_index = 0;
  for (std::size_t read_index = 0; read_index < num_in_flight; ++read_index)
  {
    if (impl_->completed_send_mask[read_index] != 0)
    {
      ReleaseSendBuffer(std::move(impl_->in_flight_sends[read_index].data));
      continue;
    }

    if (write_index != read_index)
    {
      impl_->in_flight_sends[write_index] = std::move(impl_->in_flight_sends[read_index]);
      impl_->in_flight_send_requests[write_index] =
        std::move(impl_->in_flight_send_requests[read_index]);
    }
    ++write_index;
  }
  impl_->in_flight_sends.resize(write_index);
  impl_->in_flight_send_requests.resize(write_index);
  impl_->completed_send_indices.clear();
  return true;
}

bool
CBCD_AsynchronousCommunicator::AllWorkComplete() const
{
  for (std::size_t i = 0; i < impl_->num_angle_sets; ++i)
    if (not impl_->angle_sets_done[i].load(std::memory_order_acquire))
      return false;

  for (const auto& queue : impl_->outgoing_queues)
    for (std::size_t shard = 0; shard < impl_->num_outgoing_stacks_per_queue; ++shard)
      if (not impl_->outgoing_stacks[queue.shard_offset + shard].Empty())
        return false;

  return true;
}

} // namespace opensn

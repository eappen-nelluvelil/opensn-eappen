// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep_parallel_for.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>

namespace opensn
{

namespace detail
{

int
CheckedMpiTag(const std::size_t num_angle_sets)
{
  if (num_angle_sets > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("CBCD communicator: angle-set count exceeds the MPI tag range.");
  return static_cast<int>(num_angle_sets);
}

std::uint32_t
CheckedSourceSlot(const std::size_t source_slot)
{
  if (source_slot > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("CBCD communicator: source-slot index exceeds 32-bit storage.");
  return static_cast<std::uint32_t>(source_slot);
}

std::size_t
CheckedAdd(const std::size_t lhs, const std::size_t rhs, const char* const description)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::overflow_error(description);
  return lhs + rhs;
}

std::size_t
CheckedMultiply(const std::size_t lhs, const std::size_t rhs, const char* const description)
{
  if (lhs != 0 and rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::overflow_error(description);
  return lhs * rhs;
}

constexpr std::size_t message_header_bytes = sizeof(std::size_t);
constexpr std::size_t section_header_bytes = sizeof(std::uint8_t) + 2 * sizeof(std::size_t);
constexpr std::size_t entry_header_bytes = sizeof(std::uint32_t) + sizeof(std::size_t);

// Bounded byte reader for communicator payload deserialization.
struct BufferReader
{
  const std::byte* ptr = nullptr;
  std::size_t remaining_bytes = 0;

  void Require(const std::size_t num_bytes) const
  {
    if (remaining_bytes < num_bytes)
      throw std::runtime_error("CBCD communicator: truncated wire payload.");
  }

  std::size_t LoadSize()
  {
    Require(sizeof(std::size_t));
    std::size_t value{};
    std::memcpy(&value, ptr, sizeof(std::size_t));
    ptr += sizeof(std::size_t);
    remaining_bytes -= sizeof(std::size_t);
    return value;
  }

  std::uint32_t LoadFaceIndex()
  {
    Require(sizeof(std::uint32_t));
    std::uint32_t value{};
    std::memcpy(&value, ptr, sizeof(std::uint32_t));
    ptr += sizeof(std::uint32_t);
    remaining_bytes -= sizeof(std::uint32_t);
    return value;
  }

  CBCDMessageKind LoadKind()
  {
    Require(sizeof(std::uint8_t));
    std::uint8_t value{};
    std::memcpy(&value, ptr, sizeof(std::uint8_t));
    ptr += sizeof(std::uint8_t);
    remaining_bytes -= sizeof(std::uint8_t);
    if (value > static_cast<std::uint8_t>(CBCDMessageKind::DELAYED_COMPLETION))
      throw std::runtime_error("CBCD communicator: invalid wire-section kind.");
    return static_cast<CBCDMessageKind>(value);
  }

  void SkipBytes(const std::size_t num_bytes)
  {
    Require(num_bytes);
    ptr += num_bytes;
    remaining_bytes -= num_bytes;
  }

  const std::byte* Data() const noexcept { return ptr; }
};

} // namespace detail

CBCD_AsynchronousCommunicator::CBCD_AsynchronousCommunicator(
  const std::vector<AngleSet*>& angle_sets,
  const MPICommunicatorSet& comm_set,
  const std::vector<std::vector<int>>& incoming_source_partitions,
  const std::vector<std::vector<int>>& delayed_incoming_source_partitions,
  const std::size_t max_message_bytes,
  const std::vector<AngleSetCapacity>& capacities)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    capacities_(capacities),
    mpi_tag_(detail::CheckedMpiTag(angle_sets.size())),
    max_message_bytes_(max_message_bytes),
    angle_set_done_(angle_sets.size()),
    delayed_sources_remaining_by_angle_set_(angle_sets.size())
{
  if (angle_sets.empty() or incoming_source_partitions.size() != angle_sets.size() or
      delayed_incoming_source_partitions.size() != angle_sets.size() or
      capacities.size() != angle_sets.size())
    throw std::invalid_argument("CBCD communicator: inconsistent angle-set metadata.");

  std::set<int> sources;
  std::set<int> destinations;

  delayed_source_partitions_by_angle_set_.resize(angle_sets.size());
  delayed_destination_partitions_by_angle_set_.resize(angle_sets.size());

  for (std::size_t i = 0; i < angle_sets.size(); ++i)
  {
    const auto* angle_set = angle_sets[i];
    const auto& spds = angle_set->GetSPDS();
    for (const int dep : spds.GetLocationDependencies())
      sources.insert(dep);
    for (const int succ : spds.GetLocationSuccessors())
      destinations.insert(succ);

    // Cycle-aware: delayed nonlocal traffic also flows along the dependency-removed edges,
    // so every delayed-source rank must appear in our incoming `sources` set and every
    // delayed-destination rank must appear in `destinations`.  The per-angle-set delayed
    // source partition list comes from the caller (matches `CBCD_FLUDSCommonData`'s
    // delayed-incoming source enumeration so slot indices agree between this map and the
    // delayed face-table lookup).  Delayed destinations come from the SPDS — they only
    // affect outgoing-completion enqueueing, not slot indexing.
    const auto& delayed_succs = spds.GetDelayedLocationSuccessors();
    for (const int dep : delayed_incoming_source_partitions[i])
      sources.insert(dep);
    for (const int succ : delayed_succs)
      destinations.insert(succ);
    delayed_source_partitions_by_angle_set_[i] = delayed_incoming_source_partitions[i];
    delayed_destination_partitions_by_angle_set_[i].assign(delayed_succs.begin(),
                                                           delayed_succs.end());

    if (capacities[i].incoming_faces > 0)
    {
      // Each mailbox slot stores one incoming batch for a single angle set. Entry and value
      // buffers are reserved once from the angle-set-local capacity summary and then reused.
      auto mailbox = std::make_unique<LockFreeSPSCSlotQueue<IncomingFaceBatch>>();
      if (capacities[i].incoming_faces == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("CBCD communicator: incoming mailbox capacity overflow.");
      mailbox->Preallocate(capacities[i].incoming_faces + 1);
      mailbox->InitializeSlots(
        [&](IncomingFaceBatch& batch)
        {
          batch.entries.reserve(capacities[i].max_incoming_batch_entries);
          batch.psi_data.reserve(capacities[i].max_incoming_batch_values);
          batch.entries.clear();
          batch.psi_data.clear();
          batch.source_slot = 0;
        });
      incoming_mailboxes_.push_back(std::move(mailbox));
    }
    else
    {
      incoming_mailboxes_.push_back(std::make_unique<LockFreeSPSCSlotQueue<IncomingFaceBatch>>());
    }
  }

  my_rank_ = opensn::mpi_comm.rank();
  source_partitions_.assign(sources.begin(), sources.end());
  source_partition_by_rank_.reserve(source_partitions_.size());
  for (const int source_partition : source_partitions_)
  {
    const auto source_rank = comm_set_.MapIonJ(source_partition, my_rank_);
    const auto [it, inserted] = source_partition_by_rank_.emplace(source_rank, source_partition);
    if (not inserted and it->second != source_partition)
      throw std::logic_error("CBCD communicator: source partitions alias one communicator rank.");
  }

  source_partition_to_slot_by_angle_set_.resize(angle_sets.size());
  delayed_source_partition_to_slot_by_angle_set_.resize(angle_sets.size());
  for (std::size_t angle_set_id = 0; angle_set_id < angle_sets.size(); ++angle_set_id)
  {
    auto& source_to_slot = source_partition_to_slot_by_angle_set_[angle_set_id];
    const auto& source_partitions = incoming_source_partitions[angle_set_id];
    source_to_slot.reserve(source_partitions.size());
    for (std::size_t source_slot = 0; source_slot < source_partitions.size(); ++source_slot)
      if (not source_to_slot
                .emplace(source_partitions[source_slot], detail::CheckedSourceSlot(source_slot))
                .second)
        throw std::logic_error("CBCD communicator: duplicate normal source partition.");

    auto& delayed_source_to_slot = delayed_source_partition_to_slot_by_angle_set_[angle_set_id];
    const auto& delayed_source_partitions = delayed_incoming_source_partitions[angle_set_id];
    delayed_source_to_slot.reserve(delayed_source_partitions.size());
    for (std::size_t source_slot = 0; source_slot < delayed_source_partitions.size(); ++source_slot)
      if (not delayed_source_to_slot
                .emplace(delayed_source_partitions[source_slot],
                         detail::CheckedSourceSlot(source_slot))
                .second)
        throw std::logic_error("CBCD communicator: duplicate delayed source partition.");
  }

  destination_ranks_.assign(destinations.begin(), destinations.end());
  dest_to_queue_index_.reserve(destination_ranks_.size());
  for (std::size_t queue_index = 0; queue_index < destination_ranks_.size(); ++queue_index)
    dest_to_queue_index_.emplace(destination_ranks_[queue_index], queue_index);
  for (auto& done : angle_set_done_)
    done.store(false, std::memory_order_relaxed);

  // Only the communication thread accesses the per-source flags. Workers observe the
  // aggregate remaining-source count.
  delayed_completion_received_by_angle_set_.resize(num_angle_sets_);
  for (std::size_t i = 0; i < num_angle_sets_; ++i)
    delayed_completion_received_by_angle_set_[i].resize(
      delayed_source_partitions_by_angle_set_[i].size());

  if (max_message_bytes_ > 0)
    recv_buffer_.Data().reserve(max_message_bytes_);
}

CBCD_AsynchronousCommunicator::~CBCD_AsynchronousCommunicator()
{
  if (comm_thread_.joinable())
    Stop();
}

void
CBCD_AsynchronousCommunicator::SignalAngleSetComplete(const std::size_t angle_set_id,
                                                      const std::size_t producer_id)
{
  assert(angle_set_id < num_angle_sets_);
  assert(producer_id < num_producers_);

  // Enqueue one delayed-completion marker for each delayed destination of this angle set so
  // the receiver knows when its lagged incoming bank for the angle set can be rotated.
  // The completion markers travel through the producer's own outgoing shards and are
  // therefore FIFO-ordered after every delayed face payload produced by this angle set.
  for (const int dest_rank : delayed_destination_partitions_by_angle_set_[angle_set_id])
    EnqueueDelayedCompletion(dest_rank, producer_id, angle_set_id);

  angle_set_done_[angle_set_id].store(true, std::memory_order_release);
}

void
CBCD_AsynchronousCommunicator::EnqueueDelayedCompletion(const int dest_rank,
                                                        const std::size_t producer_id,
                                                        const std::size_t angle_set_id)
{
  const auto it = dest_to_queue_index_.find(dest_rank);
  assert(it != dest_to_queue_index_.end());
  assert(producer_id < num_producers_);
  auto& shard = *outgoing_queues_[it->second].producer_shards[producer_id];
  auto& slot = shard.queue.ReserveSlot();
  slot.payload.kind = CBCDMessageKind::DELAYED_COMPLETION;
  slot.payload.angle_set_id = angle_set_id;
  slot.payload.remote_face_index = 0;
  slot.payload.psi_data = nullptr;
  slot.payload.data_size = 0;
  shard.queue.PublishSlot();
  if (not shard.scheduled.exchange(true, std::memory_order_acq_rel))
  {
    auto& doorbell = *producer_doorbells_[producer_id];
    auto& doorbell_slot = doorbell.ReserveSlot();
    doorbell_slot.payload = it->second;
    doorbell.PublishSlot();
  }
}

bool
CBCD_AsynchronousCommunicator::AreDelayedReceivesComplete(
  const std::size_t angle_set_id) const noexcept
{
  assert(angle_set_id < num_angle_sets_);
  return delayed_sources_remaining_by_angle_set_[angle_set_id].load(std::memory_order_acquire) == 0;
}

void
CBCD_AsynchronousCommunicator::ConfigureProducerShards(const std::size_t num_producers)
{
  assert(num_producers > 0);
  if ((num_producers_ == num_producers) and (producer_doorbells_.size() == num_producers) and
      (outgoing_queues_.size() == destination_ranks_.size()))
    return;

  num_producers_ = num_producers;

  std::vector<std::unordered_map<int, std::size_t>> producer_face_counts(num_producers);

  for (std::size_t producer_id = 0; producer_id < num_producers; ++producer_id)
  {
    const auto [begin, end] = GetStaticPartition(num_angle_sets_, num_producers, producer_id);
    auto& face_counts = producer_face_counts[producer_id];
    for (std::size_t angle_set_id = begin; angle_set_id < end; ++angle_set_id)
    {
      const auto& capacity = capacities_[angle_set_id];
      for (const auto& destination_capacity : capacity.outgoing_faces_by_destination)
      {
        auto& count = face_counts[destination_capacity.dest_rank];
        if (destination_capacity.face_count > std::numeric_limits<std::size_t>::max() - count)
          throw std::overflow_error("CBCD communicator: producer queue capacity overflow.");
        count += destination_capacity.face_count;
      }
    }
  }

  producer_doorbells_.clear();
  producer_doorbells_.reserve(num_producers);
  for (std::size_t producer_id = 0; producer_id < num_producers; ++producer_id)
  {
    auto doorbell = std::make_unique<DoorbellQueue>();
    if (producer_face_counts[producer_id].size() == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("CBCD communicator: producer doorbell capacity overflow.");
    doorbell->Preallocate(producer_face_counts[producer_id].size() + 1);
    producer_doorbells_.push_back(std::move(doorbell));
  }

  outgoing_queues_.clear();
  outgoing_queues_.resize(destination_ranks_.size());
  destination_active_local_.assign(destination_ranks_.size(), 0);
  active_destinations_.clear();

  for (std::size_t queue_index = 0; queue_index < destination_ranks_.size(); ++queue_index)
  {
    auto& destination_queue = outgoing_queues_[queue_index];
    destination_queue.dest_rank = destination_ranks_[queue_index];
    destination_queue.producer_shards.resize(num_producers);
    destination_queue.active_producers.clear();
    destination_queue.producer_active_local.assign(num_producers, 0);
    destination_queue.rr_cursor = 0;

    for (std::size_t producer_id = 0; producer_id < num_producers; ++producer_id)
    {
      auto shard = std::make_unique<OutgoingShard>();
      const auto count_it = producer_face_counts[producer_id].find(destination_queue.dest_rank);
      if (count_it != producer_face_counts[producer_id].end())
      {
        if (count_it->second == std::numeric_limits<std::size_t>::max())
          throw std::overflow_error("CBCD communicator: outgoing queue capacity overflow.");
        shard->queue.Preallocate(count_it->second + 1);
        shard->queue.InitializeSlots(
          [](OutgoingFaceData& payload)
          {
            payload.kind = CBCDMessageKind::NORMAL_FACE_PSI;
            payload.angle_set_id = 0;
            payload.remote_face_index = 0;
            payload.psi_data = nullptr;
            payload.data_size = 0;
          });
      }
      destination_queue.producer_shards[producer_id] = std::move(shard);
    }
  }
}

void
CBCD_AsynchronousCommunicator::Start(const std::size_t num_producers)
{
  ConfigureProducerShards(num_producers);

  stop_requested_.store(false, std::memory_order_relaxed);
  for (auto& done : angle_set_done_)
    done.store(false, std::memory_order_relaxed);
  for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
  {
    auto& received = delayed_completion_received_by_angle_set_[angle_set_id];
    std::fill(received.begin(), received.end(), std::uint8_t{0});
    delayed_sources_remaining_by_angle_set_[angle_set_id].store(
      detail::CheckedSourceSlot(received.size()), std::memory_order_relaxed);
  }
  send_requests_.clear();
  in_flight_send_buffers_.clear();
  completed_send_indices_.clear();
  active_destinations_.clear();
  std::fill(destination_active_local_.begin(), destination_active_local_.end(), 0);
  for (auto& destination_queue : outgoing_queues_)
  {
    destination_queue.active_producers.clear();
    std::fill(destination_queue.producer_active_local.begin(),
              destination_queue.producer_active_local.end(),
              0);
    destination_queue.rr_cursor = 0;
    assert(destination_queue.open_send_buffer.Data().empty());
    destination_queue.num_open_sections = 0;
    destination_queue.has_open_face_section = false;
    for (auto& shard : destination_queue.producer_shards)
    {
      shard->scheduled.store(false, std::memory_order_relaxed);
      assert(shard->queue.Empty());
    }
  }
  for (const auto& doorbell : producer_doorbells_)
    assert(doorbell->Empty());
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

  // The communication thread handles all MPI communication for CBCD.
  // Each iteration advances all three communication phases: outgoing batching,
  // incoming pushes to angleset mailboxes, and retirement of completed nonblocking sends.
  while (true)
  {
    bool work_done = SerializeAndSend();
    work_done |= ProbeAndReceive();
    work_done |= PollInFlightSends();

    if (stop_requested_.load(std::memory_order_acquire) and AllAngleSetsComplete())
    {
      while (not send_requests_.empty())
      {
        ProbeAndReceive();
        PollInFlightSends();
        if (not send_requests_.empty())
          std::this_thread::yield();
      }
      break;
    }

    if (not work_done)
      std::this_thread::yield();
  }
}

bool
CBCD_AsynchronousCommunicator::DrainProducerDoorbells()
{
  bool activated_any = false;

  for (std::size_t producer_id = 0; producer_id < producer_doorbells_.size(); ++producer_id)
  {
    activated_any |= producer_doorbells_[producer_id]->ProcessReady(
                       [this, producer_id](const std::size_t destination_queue_index)
                       {
                         assert(destination_queue_index < outgoing_queues_.size());
                         auto& destination_queue = outgoing_queues_[destination_queue_index];

                         if (not destination_queue.producer_active_local[producer_id])
                         {
                           destination_queue.producer_active_local[producer_id] = 1;
                           destination_queue.active_producers.push_back(producer_id);
                         }

                         if (not destination_active_local_[destination_queue_index])
                         {
                           destination_active_local_[destination_queue_index] = 1;
                           active_destinations_.push_back(destination_queue_index);
                         }
                       }) > 0;
  }

  return activated_any;
}

void
CBCD_AsynchronousCommunicator::PostSend(DestinationQueue& destination_queue)
{
  auto& bytes = destination_queue.open_send_buffer.Data();
  if (bytes.empty())
    return;
  if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("CBCD communicator: MPI payload exceeds the count range.");

  std::memcpy(bytes.data(), &destination_queue.num_open_sections, detail::message_header_bytes);
  ByteArray send_buffer = std::move(destination_queue.open_send_buffer);
  destination_queue.open_send_buffer.Clear();
  const auto& comm = comm_set_.LocICommunicator(destination_queue.dest_rank);
  const auto mapped_rank =
    comm_set_.MapIonJ(destination_queue.dest_rank, destination_queue.dest_rank);
  send_requests_.push_back(comm.isend(mapped_rank, mpi_tag_, send_buffer.Data()));
  in_flight_send_buffers_.push_back(std::move(send_buffer));

  destination_queue.num_open_sections = 0;
  destination_queue.has_open_face_section = false;
}

void
CBCD_AsynchronousCommunicator::AppendOutgoing(DestinationQueue& destination_queue,
                                              const OutgoingFaceData& entry)
{
  const auto kind_idx = static_cast<std::size_t>(entry.kind);
  if (kind_idx > static_cast<std::size_t>(CBCDMessageKind::DELAYED_COMPLETION) or
      entry.angle_set_id >= num_angle_sets_)
    throw std::logic_error("CBCD communicator: invalid outgoing queue record.");
  const bool is_completion = entry.kind == CBCDMessageKind::DELAYED_COMPLETION;
  if ((is_completion and entry.data_size != 0) or
      ((not is_completion) and entry.data_size > 0 and entry.psi_data == nullptr))
    throw std::logic_error("CBCD communicator: invalid outgoing payload view.");

  const auto payload_bytes =
    is_completion ? std::size_t{0}
                  : detail::CheckedMultiply(entry.data_size,
                                            sizeof(double),
                                            "CBCD communicator: outgoing payload-size overflow.");
  const auto entry_bytes =
    is_completion ? std::size_t{0}
                  : detail::CheckedAdd(detail::entry_header_bytes,
                                       payload_bytes,
                                       "CBCD communicator: outgoing entry-size overflow.");
  bool extends_last_section = not is_completion and destination_queue.has_open_face_section and
                              destination_queue.last_section_kind == entry.kind and
                              destination_queue.last_section_angle_set_id == entry.angle_set_id;
  const auto additional_bytes =
    detail::CheckedAdd(extends_last_section ? 0 : detail::section_header_bytes,
                       entry_bytes,
                       "CBCD communicator: outgoing message-size overflow.");

  const auto current_size = destination_queue.open_send_buffer.Data().size();
  const bool exceeds_message_limit =
    max_message_bytes_ > 0 and not destination_queue.open_send_buffer.Data().empty() and
    (current_size >= max_message_bytes_ or additional_bytes > max_message_bytes_ - current_size);
  if (exceeds_message_limit)
  {
    PostSend(destination_queue);
    extends_last_section = false;
  }

  auto& open_bytes = destination_queue.open_send_buffer.Data();
  if (open_bytes.empty())
  {
    if (not available_send_buffers_.empty())
    {
      destination_queue.open_send_buffer = std::move(available_send_buffers_.back());
      available_send_buffers_.pop_back();
    }
    destination_queue.open_send_buffer.Clear();
    auto& fresh_bytes = destination_queue.open_send_buffer.Data();
    if (max_message_bytes_ > 0 and fresh_bytes.capacity() < max_message_bytes_)
      fresh_bytes.reserve(max_message_bytes_);
    fresh_bytes.resize(detail::message_header_bytes);
  }

  auto append_bytes = [&destination_queue](const void* const source, const std::size_t size)
  {
    auto& buffer = destination_queue.open_send_buffer.Data();
    const auto offset = buffer.size();
    buffer.resize(offset + size);
    std::memcpy(buffer.data() + offset, source, size);
  };

  if (not extends_last_section or
      destination_queue.open_send_buffer.Data().size() == detail::message_header_bytes)
  {
    const auto kind = static_cast<std::uint8_t>(entry.kind);
    const std::size_t num_entries = 0;
    append_bytes(&kind, sizeof(kind));
    append_bytes(&entry.angle_set_id, sizeof(entry.angle_set_id));
    destination_queue.last_entry_count_offset = destination_queue.open_send_buffer.Data().size();
    append_bytes(&num_entries, sizeof(num_entries));
    ++destination_queue.num_open_sections;
    destination_queue.last_section_kind = entry.kind;
    destination_queue.last_section_angle_set_id = entry.angle_set_id;
    destination_queue.has_open_face_section = not is_completion;
  }

  if (not is_completion)
  {
    std::size_t num_entries = 0;
    auto& open_data = destination_queue.open_send_buffer.Data();
    std::memcpy(&num_entries,
                open_data.data() + destination_queue.last_entry_count_offset,
                sizeof(num_entries));
    ++num_entries;
    std::memcpy(open_data.data() + destination_queue.last_entry_count_offset,
                &num_entries,
                sizeof(num_entries));
    append_bytes(&entry.remote_face_index, sizeof(entry.remote_face_index));
    append_bytes(&entry.data_size, sizeof(entry.data_size));
    append_bytes(entry.psi_data, payload_bytes);
  }

  if (max_message_bytes_ > 0 and
      destination_queue.open_send_buffer.Data().size() >= max_message_bytes_)
    PostSend(destination_queue);
}

bool
CBCD_AsynchronousCommunicator::FlushActiveDestination(const std::size_t destination_queue_index)
{
  if (destination_queue_index >= outgoing_queues_.size())
    throw std::out_of_range("CBCD communicator: invalid destination queue index.");
  auto& destination_queue = outgoing_queues_[destination_queue_index];
  bool work_done = false;

  const auto producer_count = destination_queue.active_producers.size();
  for (std::size_t visited = 0;
       visited < producer_count and not destination_queue.active_producers.empty();
       ++visited)
  {
    if (destination_queue.rr_cursor >= destination_queue.active_producers.size())
      destination_queue.rr_cursor = 0;

    const auto active_index = destination_queue.rr_cursor;
    const auto producer_id = destination_queue.active_producers[active_index];
    auto& shard = *destination_queue.producer_shards[producer_id];
    shard.queue.PeekReadySlots(slot_cache_);

    if (slot_cache_.empty())
    {
      shard.scheduled.store(false, std::memory_order_release);
      if (not shard.queue.Empty())
      {
        shard.scheduled.store(true, std::memory_order_release);
        if (not destination_queue.active_producers.empty())
          destination_queue.rr_cursor =
            (active_index + 1) % destination_queue.active_producers.size();
      }
      else
      {
        destination_queue.producer_active_local[producer_id] = 0;
        std::swap(destination_queue.active_producers[active_index],
                  destination_queue.active_producers.back());
        destination_queue.active_producers.pop_back();
        if (destination_queue.active_producers.empty())
          destination_queue.rr_cursor = 0;
        else if (active_index >= destination_queue.active_producers.size())
          destination_queue.rr_cursor = 0;
        else
          destination_queue.rr_cursor = active_index;
      }
      continue;
    }

    for (const auto* slot : slot_cache_)
      AppendOutgoing(destination_queue, slot->payload);

    shard.queue.ReleaseReadySlots(slot_cache_.size());
    work_done = true;
    destination_queue.rr_cursor = (active_index + 1) % destination_queue.active_producers.size();
  }

  if (not destination_queue.open_send_buffer.Data().empty())
  {
    PostSend(destination_queue);
    work_done = true;
  }

  return work_done;
}

bool
CBCD_AsynchronousCommunicator::FlushActiveDestinations()
{
  bool work_done = false;

  for (std::size_t i = 0; i < active_destinations_.size();)
  {
    const auto destination_queue_index = active_destinations_[i];
    work_done |= FlushActiveDestination(destination_queue_index);

    const auto& destination_queue = outgoing_queues_[destination_queue_index];
    if (destination_queue.active_producers.empty() and
        destination_queue.open_send_buffer.Data().empty())
    {
      destination_active_local_[destination_queue_index] = 0;
      std::swap(active_destinations_[i], active_destinations_.back());
      active_destinations_.pop_back();
    }
    else
      ++i;
  }

  return work_done;
}

bool
CBCD_AsynchronousCommunicator::SerializeAndSend()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::SerializeAndSend");

  const auto activated_any = DrainProducerDoorbells();
  const auto sent_any = FlushActiveDestinations();
  return activated_any or sent_any;
}

bool
CBCD_AsynchronousCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::ProbeAndReceive");

  bool received_any = false;
  const auto& recv_comm = comm_set_.LocICommunicator(my_rank_);

  while (true)
  {
    auto message = recv_comm.improbe(mpi::ANY_SOURCE, mpi_tag_);
    if (not message)
      break;

    received_any = true;
    const auto source_it = source_partition_by_rank_.find(message.source());
    if (source_it == source_partition_by_rank_.end())
      throw std::logic_error("CBCD communicator: message arrived from an unknown source rank.");
    const int source_partition = source_it->second;
    message.recv(recv_buffer_.Data());

    detail::BufferReader reader{reinterpret_cast<const std::byte*>(recv_buffer_.Data().data()),
                                recv_buffer_.Data().size()};

    // Walk each (kind, angle_set) section to determine its source slot, entry count, and
    // total number of doubles, which allows exactly one mailbox payload allocation per
    // face-psi section.  DELAYED_COMPLETION sections carry zero entries and only
    // increment the receiver's per-(angle_set, source_slot) completion counter.
    const auto num_sections = reader.LoadSize();
    for (std::size_t section_index = 0; section_index < num_sections; ++section_index)
    {
      const auto kind = reader.LoadKind();
      const auto angle_set_id = reader.LoadSize();
      const auto num_entries = reader.LoadSize();
      if (angle_set_id >= num_angle_sets_)
        throw std::runtime_error("CBCD communicator: invalid wire angle-set ID.");

      // Normal and delayed traffic use disjoint per-(angle_set) source-slot spaces; the
      // kind tag selects which map to consult.  `DELAYED_COMPLETION` markers use the
      // delayed map because the counter table is keyed by the delayed-source slot index.
      const auto& slot_map = (kind == CBCDMessageKind::NORMAL_FACE_PSI)
                               ? source_partition_to_slot_by_angle_set_[angle_set_id]
                               : delayed_source_partition_to_slot_by_angle_set_[angle_set_id];
      const auto slot_it = slot_map.find(source_partition);
      if (slot_it == slot_map.end())
        throw std::runtime_error("CBCD communicator: source has no angle-set slot mapping.");
      const auto source_slot = slot_it->second;

      if (kind == CBCDMessageKind::DELAYED_COMPLETION)
      {
        if (num_entries != 0 or
            source_slot >= delayed_completion_received_by_angle_set_[angle_set_id].size())
          throw std::runtime_error("CBCD communicator: invalid delayed-completion section.");
        auto& received = delayed_completion_received_by_angle_set_[angle_set_id][source_slot];
        if (received == 0)
        {
          received = 1;
          const auto previous = delayed_sources_remaining_by_angle_set_[angle_set_id].fetch_sub(
            1, std::memory_order_release);
          if (previous == 0)
            throw std::logic_error("CBCD communicator: delayed-completion counter underflow.");
        }
        continue;
      }

      auto& slot = incoming_mailboxes_[angle_set_id]->ReserveSlot();
      auto& batch = slot.payload;
      batch.kind = kind;
      batch.source_slot = source_slot;
      batch.entries.resize(num_entries);
      batch.psi_data.clear();
      std::size_t value_offset = 0;
      for (std::size_t entry_index = 0; entry_index < num_entries; ++entry_index)
      {
        auto& entry = batch.entries[entry_index];
        entry.source_face_index = reader.LoadFaceIndex();
        entry.payload_offset = value_offset;
        entry.payload_size = reader.LoadSize();
        if (entry.payload_size > std::numeric_limits<std::size_t>::max() / sizeof(double) or
            value_offset > std::numeric_limits<std::size_t>::max() - entry.payload_size)
          throw std::runtime_error("CBCD communicator: wire face-payload size overflow.");
        const auto payload_bytes = entry.payload_size * sizeof(double);
        reader.Require(payload_bytes);
        batch.psi_data.resize(value_offset + entry.payload_size);
        std::memcpy(batch.psi_data.data() + value_offset, reader.Data(), payload_bytes);
        reader.SkipBytes(payload_bytes);
        value_offset += entry.payload_size;
      }

      incoming_mailboxes_[angle_set_id]->PublishSlot();
    }

    if (reader.remaining_bytes != 0)
      throw std::runtime_error("CBCD communicator: trailing bytes in wire payload.");
  }

  return received_any;
}

bool
CBCD_AsynchronousCommunicator::PollInFlightSends()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::PollInFlightSends");

  if (send_requests_.empty())
    return false;

  completed_send_indices_.clear();
  mpi::test_some(send_requests_, completed_send_indices_);
  std::ranges::sort(completed_send_indices_, std::greater<>{});
  for (const int completed_index : completed_send_indices_)
  {
    const auto i = static_cast<std::size_t>(completed_index);
    available_send_buffers_.push_back(std::move(in_flight_send_buffers_[i]));
    if (i + 1 != send_requests_.size())
    {
      send_requests_[i] = send_requests_.back();
      in_flight_send_buffers_[i] = std::move(in_flight_send_buffers_.back());
    }
    send_requests_.pop_back();
    in_flight_send_buffers_.pop_back();
  }
  return not completed_send_indices_.empty();
}

bool
CBCD_AsynchronousCommunicator::AllAngleSetsComplete() const
{
  for (const auto& done : angle_set_done_)
    if (not done.load(std::memory_order_acquire))
      return false;

  if (not active_destinations_.empty())
    return false;

  for (const auto& doorbell : producer_doorbells_)
    if (not doorbell->Empty())
      return false;

  for (const auto& destination_queue : outgoing_queues_)
  {
    if (not destination_queue.open_send_buffer.Data().empty())
      return false;
    for (const auto& shard : destination_queue.producer_shards)
      if (not shard->queue.Empty())
        return false;
  }

  // Cycle-aware: the communicator must also wait for every delayed-completion marker so
  // that the lagged incoming-nonlocal banks have been fully populated for the next sweep.
  for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
    if (not AreDelayedReceivesComplete(angle_set_id))
      return false;

  return true;
}

} // namespace opensn

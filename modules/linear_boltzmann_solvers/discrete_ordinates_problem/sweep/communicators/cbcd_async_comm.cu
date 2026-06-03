// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <set>
#include <cstddef>

namespace opensn
{

namespace detail
{

// Bounded byte reader for communicator payload deserialization.
struct BufferReader
{
  const std::byte* ptr = nullptr;
  std::size_t remaining_bytes = 0;

  std::size_t LoadSize()
  {
    assert(remaining_bytes >= sizeof(std::size_t));
    std::size_t value{};
    std::memcpy(&value, ptr, sizeof(std::size_t));
    ptr += sizeof(std::size_t);
    remaining_bytes -= sizeof(std::size_t);
    return value;
  }

  std::uint32_t LoadFaceIndex()
  {
    assert(remaining_bytes >= sizeof(std::uint32_t));
    std::uint32_t value{};
    std::memcpy(&value, ptr, sizeof(std::uint32_t));
    ptr += sizeof(std::uint32_t);
    remaining_bytes -= sizeof(std::uint32_t);
    return value;
  }

  CBCDMessageKind LoadKind()
  {
    assert(remaining_bytes >= sizeof(std::uint8_t));
    std::uint8_t value{};
    std::memcpy(&value, ptr, sizeof(std::uint8_t));
    ptr += sizeof(std::uint8_t);
    remaining_bytes -= sizeof(std::uint8_t);
    return static_cast<CBCDMessageKind>(value);
  }

  void SkipBytes(const std::size_t num_bytes)
  {
    assert(remaining_bytes >= num_bytes);
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
    mpi_tag_(static_cast<int>(angle_sets.size())),
    max_message_bytes_(max_message_bytes),
    angle_set_done_(angle_sets.size())
{
  assert(incoming_source_partitions.size() == angle_sets.size());
  assert(delayed_incoming_source_partitions.size() == angle_sets.size());
  assert(capacities.size() == angle_sets.size());

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
  source_ranks_.reserve(source_partitions_.size());
  for (const int source_partition : source_partitions_)
    source_ranks_.push_back(comm_set_.MapIonJ(source_partition, my_rank_));

  source_partition_to_slot_by_angle_set_.resize(angle_sets.size());
  delayed_source_partition_to_slot_by_angle_set_.resize(angle_sets.size());
  for (std::size_t angle_set_id = 0; angle_set_id < angle_sets.size(); ++angle_set_id)
  {
    auto& source_to_slot = source_partition_to_slot_by_angle_set_[angle_set_id];
    const auto& source_partitions = incoming_source_partitions[angle_set_id];
    source_to_slot.reserve(source_partitions.size());
    for (std::size_t source_slot = 0; source_slot < source_partitions.size(); ++source_slot)
      source_to_slot.emplace(source_partitions[source_slot],
                             static_cast<std::uint32_t>(source_slot));

    auto& delayed_source_to_slot = delayed_source_partition_to_slot_by_angle_set_[angle_set_id];
    const auto& delayed_source_partitions = delayed_incoming_source_partitions[angle_set_id];
    delayed_source_to_slot.reserve(delayed_source_partitions.size());
    for (std::size_t source_slot = 0; source_slot < delayed_source_partitions.size(); ++source_slot)
      delayed_source_to_slot.emplace(delayed_source_partitions[source_slot],
                                     static_cast<std::uint32_t>(source_slot));
  }

  destination_ranks_.assign(destinations.begin(), destinations.end());
  dest_to_queue_index_.reserve(destination_ranks_.size());
  for (std::size_t queue_index = 0; queue_index < destination_ranks_.size(); ++queue_index)
    dest_to_queue_index_.emplace(destination_ranks_[queue_index], queue_index);

  for (auto& per_kind : send_batch_by_kind_and_angle_set_)
    per_kind.resize(num_angle_sets_);
  for (auto& done : angle_set_done_)
    done.store(false, std::memory_order_relaxed);

  // One atomic per `(angle_set, delayed_source)` pair so the receiver can count completion
  // markers without taking a lock.  Each atomic is initialised to zero; `Start` resets them
  // before every sweep.
  delayed_completion_received_by_angle_set_.resize(num_angle_sets_);
  for (std::size_t i = 0; i < num_angle_sets_; ++i)
    delayed_completion_received_by_angle_set_[i] =
      std::vector<std::atomic<std::uint32_t>>(delayed_source_partitions_by_angle_set_[i].size());

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
  slot.payload.psi_data.clear();
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
  const auto& counters = delayed_completion_received_by_angle_set_[angle_set_id];
  for (const auto& counter : counters)
    if (counter.load(std::memory_order_acquire) == 0)
      return false;
  return true;
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
  std::vector<std::size_t> producer_max_outgoing_face_values(num_producers, 0);

  const auto chunk_size = (num_angle_sets_ + num_producers - 1) / num_producers;
  for (std::size_t producer_id = 0; producer_id < num_producers; ++producer_id)
  {
    const auto begin = producer_id * chunk_size;
    const auto end = std::min(begin + chunk_size, num_angle_sets_);
    auto& face_counts = producer_face_counts[producer_id];
    for (std::size_t angle_set_id = begin; angle_set_id < end; ++angle_set_id)
    {
      const auto& capacity = capacities_[angle_set_id];
      producer_max_outgoing_face_values[producer_id] =
        std::max(producer_max_outgoing_face_values[producer_id], capacity.max_outgoing_face_values);
      for (const auto& destination_capacity : capacity.outgoing_faces_by_destination)
        face_counts[destination_capacity.dest_rank] += destination_capacity.face_count;
    }
  }

  producer_doorbells_.clear();
  producer_doorbells_.reserve(num_producers);
  for (std::size_t producer_id = 0; producer_id < num_producers; ++producer_id)
  {
    auto doorbell = std::make_unique<DoorbellQueue>();
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
        shard->queue.Preallocate(count_it->second + 1);
        shard->queue.InitializeSlots(
          [reserve = producer_max_outgoing_face_values[producer_id]](OutgoingFaceData& payload)
          {
            payload.kind = CBCDMessageKind::NORMAL_FACE_PSI;
            payload.angle_set_id = 0;
            payload.remote_face_index = 0;
            payload.psi_data.clear();
            payload.psi_data.reserve(reserve);
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
  for (auto& per_angle_set : delayed_completion_received_by_angle_set_)
    for (auto& counter : per_angle_set)
      counter.store(0, std::memory_order_relaxed);
  in_flight_sends_.clear();
  active_destinations_.clear();
  std::fill(destination_active_local_.begin(), destination_active_local_.end(), 0);
  for (auto& destination_queue : outgoing_queues_)
  {
    destination_queue.active_producers.clear();
    std::fill(destination_queue.producer_active_local.begin(),
              destination_queue.producer_active_local.end(),
              0);
    destination_queue.rr_cursor = 0;
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
      SerializeAndSend();
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

bool
CBCD_AsynchronousCommunicator::FlushActiveDestination(const std::size_t destination_queue_index)
{
  auto& destination_queue = outgoing_queues_[destination_queue_index];
  if (destination_queue.active_producers.empty())
    return false;

  bool work_done = false;
  std::size_t current_payload_bytes = sizeof(std::size_t);
  std::size_t active_sections = 0;

  // Per-section overhead: kind byte + angle-set id + entry count.
  constexpr std::size_t section_header_bytes = sizeof(std::uint8_t) + 2 * sizeof(std::size_t);

  const auto send_batch = [&]()
  {
    assert(active_sections > 0);

    // Wire format:
    // [num_sections]
    //   repeated:
    //   [kind: u8][angle_set_id: size_t][num_entries: size_t]
    //     repeated entries (none if kind == DELAYED_COMPLETION):
    //     [remote_face_index: u32][payload_size: size_t][payload doubles...]
    InFlightSend in_flight;
    in_flight.data.Data().resize(current_payload_bytes);
    std::size_t offset = 0;

    const auto write_bytes = [&](const void* ptr, const std::size_t size)
    {
      std::memcpy(in_flight.data.Data().data() + offset, ptr, size);
      offset += size;
    };

    write_bytes(&active_sections, sizeof(std::size_t));
    for (std::size_t kind_idx = 0; kind_idx < send_batch_by_kind_and_angle_set_.size(); ++kind_idx)
    {
      const auto kind = static_cast<CBCDMessageKind>(kind_idx);
      const auto kind_byte = static_cast<std::uint8_t>(kind_idx);
      for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
      {
        auto& entries = send_batch_by_kind_and_angle_set_[kind_idx][angle_set_id];
        if (entries.empty())
          continue;

        write_bytes(&kind_byte, sizeof(std::uint8_t));
        write_bytes(&angle_set_id, sizeof(std::size_t));
        // Completion markers carry the kind tag with zero entries; face-psi sections
        // pack one (remote_face_index, payload_size, doubles) triple per entry.
        const auto num_entries =
          kind == CBCDMessageKind::DELAYED_COMPLETION ? std::size_t{0} : entries.size();
        write_bytes(&num_entries, sizeof(std::size_t));
        if (kind != CBCDMessageKind::DELAYED_COMPLETION)
          for (const auto* entry : entries)
          {
            write_bytes(&entry->remote_face_index, sizeof(std::uint32_t));
            const auto data_size = entry->psi_data.size();
            write_bytes(&data_size, sizeof(std::size_t));
            write_bytes(entry->psi_data.data(), data_size * sizeof(double));
          }
        entries.clear();
      }
    }

    const auto& comm = comm_set_.LocICommunicator(destination_queue.dest_rank);
    const auto mapped_rank =
      comm_set_.MapIonJ(destination_queue.dest_rank, destination_queue.dest_rank);
    in_flight.request = comm.isend(mapped_rank, mpi_tag_, in_flight.data.Data());
    in_flight_sends_.push_back(std::move(in_flight));
    current_payload_bytes = sizeof(std::size_t);
    active_sections = 0;
  };

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
    {
      const auto& entry = slot->payload;
      const bool is_completion = entry.kind == CBCDMessageKind::DELAYED_COMPLETION;
      const auto entry_bytes = is_completion ? std::size_t{0}
                                             : sizeof(std::uint32_t) + sizeof(std::size_t) +
                                                 entry.psi_data.size() * sizeof(double);

      // Attempt to adhere to the message-size limit. Once the next entry would exceed the
      // limit, flush the current batch and continue packing the remaining queue entries.
      if (max_message_bytes_ > 0 and current_payload_bytes + entry_bytes > max_message_bytes_ and
          active_sections > 0)
        send_batch();

      const auto kind_idx = static_cast<std::size_t>(entry.kind);
      auto& entries = send_batch_by_kind_and_angle_set_[kind_idx][entry.angle_set_id];
      if (entries.empty())
      {
        ++active_sections;
        current_payload_bytes += section_header_bytes;
      }
      entries.push_back(&entry);
      current_payload_bytes += entry_bytes;
    }

    shard.queue.ReleaseReadySlots(slot_cache_.size());
    work_done = true;
    destination_queue.rr_cursor = (active_index + 1) % destination_queue.active_producers.size();
  }

  if (active_sections > 0)
    send_batch();

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

    if (outgoing_queues_[destination_queue_index].active_producers.empty())
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

  for (std::size_t source_index = 0; source_index < source_ranks_.size(); ++source_index)
  {
    const int source_partition = source_partitions_[source_index];
    const int source_rank = source_ranks_[source_index];
    mpi::Status status;

    while (recv_comm.iprobe(source_rank, mpi_tag_, status))
    {
      received_any = true;
      const auto num_bytes = status.count<std::byte>();
      recv_buffer_.Data().resize(static_cast<std::size_t>(num_bytes));
      recv_comm.recv(source_rank, status.tag(), recv_buffer_.Data().data(), num_bytes);

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
        assert(angle_set_id < num_angle_sets_);

        // Normal and delayed traffic use disjoint per-(angle_set) source-slot spaces; the
        // kind tag selects which map to consult.  `DELAYED_COMPLETION` markers use the
        // delayed map because the counter table is keyed by the delayed-source slot index.
        const auto& slot_map =
          (kind == CBCDMessageKind::NORMAL_FACE_PSI)
            ? source_partition_to_slot_by_angle_set_[angle_set_id]
            : delayed_source_partition_to_slot_by_angle_set_[angle_set_id];
        const auto slot_it = slot_map.find(source_partition);
        assert(slot_it != slot_map.end());
        const auto source_slot = slot_it->second;

        if (kind == CBCDMessageKind::DELAYED_COMPLETION)
        {
          assert(num_entries == 0);
          // Mark this `(angle_set_id, source_slot)` as complete.  The atomic value carries
          // the number of completion markers received; for the current protocol only one
          // marker per `(angle_set, source)` per sweep is emitted, so any non-zero value
          // counts as complete.
          if (source_slot < delayed_completion_received_by_angle_set_[angle_set_id].size())
            delayed_completion_received_by_angle_set_[angle_set_id][source_slot].fetch_add(
              1, std::memory_order_release);
          continue;
        }

        const auto* const section_ptr = reader.Data();
        std::size_t total_values = 0;
        for (std::size_t entry_index = 0; entry_index < num_entries; ++entry_index)
        {
          reader.LoadFaceIndex();
          const auto data_size = reader.LoadSize();
          reader.SkipBytes(data_size * sizeof(double));
          total_values += data_size;
        }
        const auto section_num_bytes = static_cast<std::size_t>(reader.Data() - section_ptr);

        auto& slot = incoming_mailboxes_[angle_set_id]->ReserveSlot();
        auto& batch = slot.payload;
        batch.kind = kind;
        batch.source_slot = source_slot;
        batch.entries.resize(num_entries);
        batch.psi_data.resize(total_values);
        detail::BufferReader section_reader{section_ptr, section_num_bytes};
        std::size_t value_offset = 0;
        // Walk the compact mailbox payload with per-face offsets into one contiguous
        // `psi_data` block.
        for (std::size_t entry_index = 0; entry_index < num_entries; ++entry_index)
        {
          auto& entry = batch.entries[entry_index];
          entry.source_face_index = section_reader.LoadFaceIndex();
          entry.payload_offset = value_offset;
          entry.payload_size = section_reader.LoadSize();
          std::memcpy(batch.psi_data.data() + value_offset,
                      section_reader.Data(),
                      entry.payload_size * sizeof(double));
          section_reader.SkipBytes(entry.payload_size * sizeof(double));
          value_offset += entry.payload_size;
        }

        incoming_mailboxes_[angle_set_id]->PublishSlot();
      }
    }
  }

  return received_any;
}

bool
CBCD_AsynchronousCommunicator::PollInFlightSends()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::PollInFlightSends");

  // Compact the in-flight vector in place by swapping completed requests with the back.
  bool completed_any = false;
  for (std::size_t i = 0; i < in_flight_sends_.size();)
  {
    if (mpi::test(in_flight_sends_[i].request))
    {
      completed_any = true;
      in_flight_sends_[i] = std::move(in_flight_sends_.back());
      in_flight_sends_.pop_back();
    }
    else
      ++i;
  }
  return completed_any;
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
    for (const auto& shard : destination_queue.producer_shards)
      if (not shard->queue.Empty())
        return false;

  // Cycle-aware: the communicator must also wait for every delayed-completion marker so
  // that the lagged incoming-nonlocal banks have been fully populated for the next sweep.
  for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
    if (not AreDelayedReceivesComplete(angle_set_id))
      return false;

  return true;
}

} // namespace opensn

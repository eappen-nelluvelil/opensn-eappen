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
    if (value > static_cast<std::uint8_t>(CBCDMessageKind::DELAYED_FACE_PSI))
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
    mpi_tag_(static_cast<int>(angle_sets.size())),
    max_message_bytes_(max_message_bytes),
    angle_set_done_(angle_sets.size()),
    delayed_faces_expected_by_angle_set_(angle_sets.size()),
    delayed_faces_remaining_by_angle_set_(angle_sets.size())
{
  assert(incoming_source_partitions.size() == angle_sets.size());
  assert(delayed_incoming_source_partitions.size() == angle_sets.size());
  assert(capacities.size() == angle_sets.size());

  std::set<int> sources;
  std::set<int> destinations;
  std::size_t total_outgoing_faces = 0;
  std::size_t max_outgoing_face_values = 0;

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
    // delayed face-table lookup).
    const auto& delayed_succs = spds.GetDelayedLocationSuccessors();
    for (const int dep : delayed_incoming_source_partitions[i])
      sources.insert(dep);
    for (const int succ : delayed_succs)
      destinations.insert(succ);
    delayed_faces_expected_by_angle_set_[i] = capacities[i].delayed_incoming_faces;

    total_outgoing_faces += capacities[i].outgoing_faces;
    max_outgoing_face_values =
      std::max(max_outgoing_face_values, capacities[i].max_outgoing_face_values);
    if (capacities[i].incoming_faces > 0)
    {
      // Each mailbox slot stores one incoming batch for a single angle set. Entry and value
      // buffers are reserved once from the angle-set-local capacity summary and then reused.
      auto mailbox = std::make_unique<LockFreeRingBuffer<IncomingFaceBatch>>();
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
      incoming_mailboxes_.push_back(std::make_unique<LockFreeRingBuffer<IncomingFaceBatch>>());
    }
  }

  my_rank_ = opensn::mpi_comm.rank();
  source_partition_by_rank_.reserve(sources.size());
  for (const int source_partition : sources)
  {
    const int source_rank = comm_set_.MapIonJ(source_partition, my_rank_);
    const bool inserted = source_partition_by_rank_.emplace(source_rank, source_partition).second;
    if (not inserted)
      throw std::logic_error("CBCD communicator: duplicate mapped source rank.");
  }

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

  outgoing_queues_.reserve(destinations.size());
  dest_to_queue_index_.reserve(destinations.size());
  int queue_index = 0;
  for (const int dest_rank : destinations)
  {
    // Each destination rank receives one bounded MPSC queue. The slots are preallocated once
    // and their payload vectors retain capacity across all subsequent publications.
    auto queue = std::make_unique<DestinationQueue>();
    queue->dest_rank = dest_rank;
    queue->queue = std::make_unique<LockFreeRingBuffer<OutgoingFaceData>>();
    if (total_outgoing_faces > 0)
      queue->queue->Preallocate(total_outgoing_faces + 1);
    queue->queue->InitializeSlots([max_outgoing_face_values](OutgoingFaceData& payload)
                                  { payload.psi_data.reserve(max_outgoing_face_values); });
    outgoing_queues_.push_back(std::move(queue));
    dest_to_queue_index_[dest_rank] = queue_index++;
  }

  for (auto& per_kind : send_batch_by_kind_and_angle_set_)
    per_kind.resize(num_angle_sets_);
  for (auto& done : angle_set_done_)
    done.store(false, std::memory_order_relaxed);

  if (max_message_bytes_ > 0)
    recv_buffer_.Data().reserve(max_message_bytes_);
}

CBCD_AsynchronousCommunicator::~CBCD_AsynchronousCommunicator()
{
  if (comm_thread_.joinable())
    Stop();
}

void
CBCD_AsynchronousCommunicator::SignalAngleSetComplete(const std::size_t angle_set_id)
{
  assert(angle_set_id < num_angle_sets_);

  angle_set_done_[angle_set_id].store(true, std::memory_order_release);
}

bool
CBCD_AsynchronousCommunicator::AreDelayedReceivesComplete(
  const std::size_t angle_set_id) const noexcept
{
  assert(angle_set_id < num_angle_sets_);
  return delayed_faces_remaining_by_angle_set_[angle_set_id].load(std::memory_order_acquire) == 0;
}

void
CBCD_AsynchronousCommunicator::Start()
{
  stop_requested_.store(false, std::memory_order_relaxed);
  for (auto& done : angle_set_done_)
    done.store(false, std::memory_order_relaxed);
  for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
    delayed_faces_remaining_by_angle_set_[angle_set_id].store(
      delayed_faces_expected_by_angle_set_[angle_set_id], std::memory_order_relaxed);
  send_requests_.clear();
  in_flight_send_buffers_.clear();
  completed_send_indices_.clear();
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
      while (not send_requests_.empty())
      {
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
CBCD_AsynchronousCommunicator::SerializeAndSend()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::SerializeAndSend");

  bool sent_any = false;

  // Per-section overhead: kind byte + angle-set id + entry count.
  constexpr std::size_t section_header_bytes = sizeof(std::uint8_t) + 2 * sizeof(std::size_t);

  for (auto& destination_queue : outgoing_queues_)
  {
    // Gather the currently published outgoing face payloads for this destination. The queue
    // is drained in FIFO order, but the serialized message is batched by `(kind, angle_set)`
    // so the receiver can publish one mailbox payload per `(kind, angle_set)` section.
    destination_queue->queue->GetReadySlots(slot_cache_);
    if (slot_cache_.empty())
      continue;

    std::size_t current_payload_bytes = sizeof(std::size_t);
    std::size_t active_sections = 0;
    std::size_t slots_processed = 0;

    const auto send_batch = [&]()
    {
      // Wire format:
      // [num_sections]
      //   repeated:
      //   [kind: u8][angle_set_id: size_t][num_entries: size_t]
      //     repeated entries:
      //     [remote_face_index: u32][payload_size: size_t][payload doubles...]
      ByteArray send_buffer;
      if (current_payload_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("CBCD communicator: MPI payload exceeds the count range.");
      send_buffer.Data().resize(current_payload_bytes);
      std::size_t offset = 0;

      const auto write_bytes = [&](const void* ptr, const std::size_t size)
      {
        std::memcpy(send_buffer.Data().data() + offset, ptr, size);
        offset += size;
      };

      write_bytes(&active_sections, sizeof(std::size_t));
      for (const auto& section : active_send_sections_)
      {
        const auto kind_byte = static_cast<std::uint8_t>(section.kind_index);
        const auto angle_set_id = section.angle_set_id;
        auto& entries = send_batch_by_kind_and_angle_set_[section.kind_index][angle_set_id];
        assert(not entries.empty());

        write_bytes(&kind_byte, sizeof(std::uint8_t));
        write_bytes(&angle_set_id, sizeof(std::size_t));
        const auto num_entries = entries.size();
        write_bytes(&num_entries, sizeof(std::size_t));
        for (const auto* entry : entries)
        {
          write_bytes(&entry->remote_face_index, sizeof(std::uint32_t));
          const auto data_size = entry->psi_data.size();
          write_bytes(&data_size, sizeof(std::size_t));
          write_bytes(entry->psi_data.data(), data_size * sizeof(double));
        }
        entries.clear();
      }
      active_send_sections_.clear();

      const auto& comm = comm_set_.LocICommunicator(destination_queue->dest_rank);
      const auto mapped_rank =
        comm_set_.MapIonJ(destination_queue->dest_rank, destination_queue->dest_rank);
      send_requests_.push_back(comm.isend(mapped_rank, mpi_tag_, send_buffer.Data()));
      in_flight_send_buffers_.push_back(std::move(send_buffer));
    };

    for (std::size_t slot_index = 0; slot_index < slot_cache_.size(); ++slot_index)
    {
      const auto* slot = slot_cache_[slot_index];
      const auto& entry = slot->payload;
      const auto payload_bytes = detail::CheckedMultiply(
        entry.psi_data.size(), sizeof(double), "CBCD communicator: outgoing face size overflow.");
      const auto entry_bytes =
        detail::CheckedAdd(sizeof(std::uint32_t) + sizeof(std::size_t),
                           payload_bytes,
                           "CBCD communicator: outgoing record size overflow.");

      const auto kind_idx = static_cast<std::size_t>(entry.kind);
      if (kind_idx >= send_batch_by_kind_and_angle_set_.size() or
          entry.angle_set_id >= num_angle_sets_)
        throw std::logic_error("CBCD communicator: invalid outgoing record metadata.");
      auto& entries = send_batch_by_kind_and_angle_set_[kind_idx][entry.angle_set_id];
      bool opens_section = entries.empty();
      auto added_bytes = detail::CheckedAdd(entry_bytes,
                                            opens_section ? section_header_bytes : 0,
                                            "CBCD communicator: outgoing section size overflow.");
      auto projected_bytes = detail::CheckedAdd(
        current_payload_bytes, added_bytes, "CBCD communicator: outgoing message size overflow.");

      // Close the current message before adding an entry that would exceed the cap. Continue
      // through the same ready snapshot so a backlog is scanned only once. A single
      // indivisible face record may exceed the limit and is sent by itself.
      if (max_message_bytes_ > 0 and projected_bytes > max_message_bytes_ and active_sections > 0)
      {
        send_batch();
        current_payload_bytes = sizeof(std::size_t);
        active_sections = 0;
        opens_section = true;
        added_bytes = detail::CheckedAdd(
          entry_bytes, section_header_bytes, "CBCD communicator: outgoing section size overflow.");
        projected_bytes = detail::CheckedAdd(
          current_payload_bytes, added_bytes, "CBCD communicator: outgoing message size overflow.");
      }

      if (opens_section)
      {
        ++active_sections;
        active_send_sections_.push_back({kind_idx, entry.angle_set_id});
      }
      entries.push_back(&entry);
      current_payload_bytes = projected_bytes;
      ++slots_processed;
    }

    if (active_sections > 0)
      send_batch();
    if (slots_processed > 0)
      destination_queue->queue->FreeSlots(slots_processed);

    sent_any |= slots_processed > 0;
  }

  return sent_any;
}

bool
CBCD_AsynchronousCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::ProbeAndReceive");

  const auto& recv_comm = comm_set_.LocICommunicator(my_rank_);
  bool received_any = false;
  while (auto message = recv_comm.improbe(mpi::ANY_SOURCE, mpi_tag_))
  {
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
    // face-psi section.
    const auto num_sections = reader.LoadSize();
    for (std::size_t section_index = 0; section_index < num_sections; ++section_index)
    {
      const auto kind = reader.LoadKind();
      const auto angle_set_id = reader.LoadSize();
      const auto num_entries = reader.LoadSize();
      if (angle_set_id >= num_angle_sets_)
        throw std::runtime_error("CBCD communicator: invalid wire angle-set ID.");

      // Normal and delayed traffic use disjoint per-(angle_set) source-slot spaces; the
      // kind tag selects which map to consult.
      const auto& slot_map = (kind == CBCDMessageKind::NORMAL_FACE_PSI)
                               ? source_partition_to_slot_by_angle_set_[angle_set_id]
                               : delayed_source_partition_to_slot_by_angle_set_[angle_set_id];
      const auto slot_it = slot_map.find(source_partition);
      if (slot_it == slot_map.end())
        throw std::runtime_error("CBCD communicator: source has no angle-set slot mapping.");
      const auto source_slot = slot_it->second;
      if (num_entries == 0)
        throw std::runtime_error("CBCD communicator: empty face-data section.");

      const auto* const section_ptr = reader.Data();
      std::size_t total_values = 0;
      for (std::size_t entry_index = 0; entry_index < num_entries; ++entry_index)
      {
        reader.LoadFaceIndex();
        const auto data_size = reader.LoadSize();
        const auto data_bytes = detail::CheckedMultiply(
          data_size, sizeof(double), "CBCD communicator: wire face size overflow.");
        reader.SkipBytes(data_bytes);
        total_values = detail::CheckedAdd(
          total_values, data_size, "CBCD communicator: wire section size overflow.");
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
        const auto payload_bytes = detail::CheckedMultiply(
          entry.payload_size, sizeof(double), "CBCD communicator: wire face size overflow.");
        section_reader.Require(payload_bytes);
        std::memcpy(batch.psi_data.data() + value_offset, section_reader.Data(), payload_bytes);
        section_reader.SkipBytes(payload_bytes);
        value_offset = detail::CheckedAdd(
          value_offset, entry.payload_size, "CBCD communicator: wire section size overflow.");
      }
      if (section_reader.remaining_bytes != 0 or value_offset != total_values)
        throw std::runtime_error("CBCD communicator: inconsistent face-data section.");

      incoming_mailboxes_[angle_set_id]->PublishSlot(slot);
      if (kind == CBCDMessageKind::DELAYED_FACE_PSI)
      {
        const auto previous = delayed_faces_remaining_by_angle_set_[angle_set_id].fetch_sub(
          num_entries, std::memory_order_release);
        if (num_entries > previous)
          throw std::logic_error("CBCD communicator: delayed face counter underflow.");
      }
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

  for (const auto& destination_queue : outgoing_queues_)
    if (not destination_queue->queue->Empty())
      return false;

  // The lagged incoming-nonlocal banks must be fully populated for the next sweep.
  for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
    if (not AreDelayedReceivesComplete(angle_set_id))
      return false;

  return true;
}

} // namespace opensn

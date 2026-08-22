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
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace opensn
{

namespace detail
{

/// A finite per-peer window permits overlap without exhausting MPI request/resources.
constexpr std::size_t max_in_flight_sends_per_destination = 4;
/// Bound one receive-service pass so outgoing and completion progress cannot starve.
constexpr std::size_t max_receives_per_progress_round = 64;
/// Avoid repeatedly materializing an entire full-sweep backlog while credits are bounded.
constexpr std::size_t max_records_per_producer_scan = 256;

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
    delayed_faces_remaining_by_angle_set_(angle_sets.size()),
    delayed_faces_expected_by_source_and_angle_set_(angle_sets.size()),
    delayed_faces_remaining_by_source_and_angle_set_(angle_sets.size()),
    normal_face_seen_by_source_and_angle_set_(angle_sets.size()),
    delayed_face_seen_by_source_and_angle_set_(angle_sets.size())
{
  assert(incoming_source_partitions.size() == angle_sets.size());
  assert(delayed_incoming_source_partitions.size() == angle_sets.size());
  assert(capacities.size() == angle_sets.size());

  std::set<int> sources;
  std::set<int> destinations;
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
    if (capacities[i].incoming_faces_by_source.size() != incoming_source_partitions[i].size())
      throw std::logic_error("CBCD communicator: normal source-count table has wrong extent.");
    if (capacities[i].delayed_incoming_faces_by_source.size() !=
        delayed_incoming_source_partitions[i].size())
      throw std::logic_error("CBCD communicator: delayed source-count table has wrong extent.");
    const auto normal_incoming_faces =
      std::accumulate(capacities[i].incoming_faces_by_source.begin(),
                      capacities[i].incoming_faces_by_source.end(),
                      std::size_t{0});
    if (std::accumulate(capacities[i].delayed_incoming_faces_by_source.begin(),
                        capacities[i].delayed_incoming_faces_by_source.end(),
                        std::size_t{0}) != capacities[i].delayed_incoming_faces)
      throw std::logic_error("CBCD communicator: delayed source counts do not match total.");
    if (capacities[i].delayed_incoming_faces > capacities[i].incoming_faces or
        normal_incoming_faces !=
          capacities[i].incoming_faces - capacities[i].delayed_incoming_faces)
      throw std::logic_error("CBCD communicator: incoming source counts do not match total.");
    delayed_faces_expected_by_source_and_angle_set_[i] =
      capacities[i].delayed_incoming_faces_by_source;
    delayed_faces_remaining_by_source_and_angle_set_[i].resize(
      capacities[i].delayed_incoming_faces_by_source.size());
    auto& normal_seen_by_source = normal_face_seen_by_source_and_angle_set_[i];
    normal_seen_by_source.reserve(capacities[i].incoming_faces_by_source.size());
    for (const auto face_count : capacities[i].incoming_faces_by_source)
      normal_seen_by_source.emplace_back(face_count, std::uint8_t{0});
    auto& seen_by_source = delayed_face_seen_by_source_and_angle_set_[i];
    seen_by_source.reserve(capacities[i].delayed_incoming_faces_by_source.size());
    for (const auto face_count : capacities[i].delayed_incoming_faces_by_source)
      seen_by_source.emplace_back(face_count, std::uint8_t{0});

    if (capacities[i].incoming_faces > 0)
    {
      // One slot per possible face record guarantees that the communication thread can
      // continue routing other angle sets even if this owner cannot initialize yet. Batch
      // storage itself is released on consumption so packet allocations do not accumulate
      // in every logical slot over repeated sweeps.
      auto mailbox = std::make_unique<LockFreeRingBuffer<IncomingFaceBatch>>();
      mailbox->Preallocate(capacities[i].incoming_faces + 1);
      mailbox->InitializeSlots(
        [&](IncomingFaceBatch& batch)
        {
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

  destination_states_.reserve(destinations.size());
  dest_to_state_index_.reserve(destinations.size());
  for (const int dest_rank : destinations)
  {
    const auto destination_index = destination_states_.size();
    destination_states_.push_back({dest_rank, 0});
    dest_to_state_index_[dest_rank] = destination_index;
  }
  send_builders_.resize(destination_states_.size());
  active_builder_destinations_.reserve(destination_states_.size());
  builder_is_active_.resize(destination_states_.size(), 0);

  log.Log0Verbose1() << "CBCD communicator: packet_target_bytes=" << max_message_bytes_
                     << ", sends_per_destination=" << detail::max_in_flight_sends_per_destination
                     << ", receive_progress_budget=" << detail::max_receives_per_progress_round
                     << ", producer_queues=" << num_angle_sets_
                     << ", destinations=" << destination_states_.size() << ".";

  // Static angle-set scheduling gives each queue exactly one producer and the communication
  // thread is its only consumer. Capacity is exact for that angle set rather than multiplied
  // by every destination rank as in the former per-destination MPSC layout.
  outgoing_queues_.reserve(num_angle_sets_);
  for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
  {
    auto producer_queue = std::make_unique<ProducerQueue>();
    producer_queue->queue = std::make_unique<CommittedSPSCQueue<OutgoingFaceData>>();
    if (capacities[angle_set_id].outgoing_faces > 0)
      producer_queue->queue->Preallocate(capacities[angle_set_id].outgoing_faces + 1);
    producer_queue->queue->InitializeSlots(
      [reserve_values = capacities[angle_set_id].max_outgoing_face_values](
        OutgoingFaceData& payload) { payload.psi_data.reserve(reserve_values); });
    outgoing_queues_.push_back(std::move(producer_queue));
  }

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

void
CBCD_AsynchronousCommunicator::CommitOutgoingBatch(const std::size_t angle_set_id)
{
  assert(angle_set_id < outgoing_queues_.size());
  outgoing_queues_[angle_set_id]->queue->Commit();
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
  {
    delayed_faces_remaining_by_angle_set_[angle_set_id].store(
      delayed_faces_expected_by_angle_set_[angle_set_id], std::memory_order_relaxed);
    delayed_faces_remaining_by_source_and_angle_set_[angle_set_id] =
      delayed_faces_expected_by_source_and_angle_set_[angle_set_id];
    for (auto& seen_by_source : normal_face_seen_by_source_and_angle_set_[angle_set_id])
      std::fill(seen_by_source.begin(), seen_by_source.end(), std::uint8_t{0});
    for (auto& seen_by_source : delayed_face_seen_by_source_and_angle_set_[angle_set_id])
      std::fill(seen_by_source.begin(), seen_by_source.end(), std::uint8_t{0});
  }
  in_flight_sends_.clear();
  for (auto& destination : destination_states_)
    destination.in_flight_sends = 0;
  next_outgoing_queue_ = 0;
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

  try
  {
    // Retire sends first so credits are available, service a bounded incoming burst next so
    // dependency-unlocking traffic has priority, then admit committed producer batches. No
    // phase is allowed to monopolize the dedicated progress thread.
    while (true)
    {
      bool work_done = PollInFlightSends();
      work_done |= ProbeAndReceive();
      work_done |= SerializeAndSend();

      if (stop_requested_.load(std::memory_order_acquire) and AllAngleSetsComplete())
      {
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
  catch (const std::exception& error)
  {
    log.LogAllError() << "CBCD communication thread failed: " << error.what();
    opensn::mpi_comm.abort(EXIT_FAILURE);
    std::terminate();
  }
  catch (...)
  {
    log.LogAllError() << "CBCD communication thread failed with an unknown exception.";
    opensn::mpi_comm.abort(EXIT_FAILURE);
    std::terminate();
  }
}

void
CBCD_AsynchronousCommunicator::PostSendBuilder(const std::size_t destination_index,
                                               const std::size_t angle_set_id,
                                               SendBuilder& builder)
{
  assert(destination_index < destination_states_.size());
  assert(not builder.Empty());

  auto& destination = destination_states_[destination_index];
  assert(destination.in_flight_sends < detail::max_in_flight_sends_per_destination);
  if (builder.payload_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("CBCD communicator: MPI payload exceeds the count range.");

  InFlightSend in_flight;
  in_flight.destination_index = destination_index;
  in_flight.data.Data().resize(builder.payload_bytes);
  std::size_t offset = 0;
  const auto write_bytes = [&](const void* ptr, const std::size_t size)
  {
    std::memcpy(in_flight.data.Data().data() + offset, ptr, size);
    offset += size;
  };

  std::size_t num_sections = 0;
  for (const auto& entries : builder.entries_by_kind)
    num_sections += not entries.empty();
  write_bytes(&num_sections, sizeof(std::size_t));

  for (std::size_t kind_index = 0; kind_index < builder.entries_by_kind.size(); ++kind_index)
  {
    const auto& entries = builder.entries_by_kind[kind_index];
    if (entries.empty())
      continue;

    const auto kind_byte = static_cast<std::uint8_t>(kind_index);
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
  }
  assert(offset == in_flight.data.Data().size());

  const auto& comm = comm_set_.LocICommunicator(destination.dest_rank);
  const auto mapped_rank = comm_set_.MapIonJ(destination.dest_rank, destination.dest_rank);
  in_flight.request = comm.isend(mapped_rank, mpi_tag_, in_flight.data.Data());
  ++destination.in_flight_sends;
  in_flight_sends_.push_back(std::move(in_flight));
  builder.Clear();
}

bool
CBCD_AsynchronousCommunicator::SerializeAndSend()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::SerializeAndSend");

  constexpr std::size_t section_header_bytes = sizeof(std::uint8_t) + 2 * sizeof(std::size_t);
  if (outgoing_queues_.empty())
    return false;

  bool sent_any = false;
  const auto num_queues = outgoing_queues_.size();
  // Normal records unlock the current sweep DAG; delayed records only seed the next sweep.
  // Service all currently available normal prefixes before allowing delayed traffic to
  // consume a destination credit. Each producer queue is FIFO and publishes its delayed
  // tail only after its local normal DAG has completed.
  constexpr std::array traffic_priority{CBCDMessageKind::NORMAL_FACE_PSI,
                                        CBCDMessageKind::DELAYED_FACE_PSI};
  for (const auto traffic_kind : traffic_priority)
  {
    for (std::size_t scan = 0; scan < num_queues; ++scan)
    {
      const auto angle_set_id = (next_outgoing_queue_ + scan) % num_queues;
      auto& queue = *outgoing_queues_[angle_set_id]->queue;
      queue.GetReadySlots(slot_cache_, detail::max_records_per_producer_scan);
      if (slot_cache_.empty() or slot_cache_.front()->payload.kind != traffic_kind)
        continue;

      active_builder_destinations_.clear();
      std::size_t slots_processed = 0;
      for (const auto* slot : slot_cache_)
      {
        const auto& entry = slot->payload;
        if (entry.kind != traffic_kind)
          break;
        const auto destination_it = dest_to_state_index_.find(entry.dest_rank);
        if (destination_it == dest_to_state_index_.end())
          throw std::logic_error("CBCD communicator: outgoing record has an unknown destination.");
        const auto destination_index = destination_it->second;
        auto& destination = destination_states_[destination_index];
        auto& builder = send_builders_[destination_index];

        if (builder.Empty() and
            destination.in_flight_sends >= detail::max_in_flight_sends_per_destination)
          break;
        if (not builder_is_active_[destination_index])
        {
          builder_is_active_[destination_index] = 1;
          active_builder_destinations_.push_back(destination_index);
        }

        const auto payload_bytes = detail::CheckedMultiply(
          entry.psi_data.size(), sizeof(double), "CBCD communicator: outgoing face size overflow.");
        const auto entry_bytes =
          detail::CheckedAdd(sizeof(std::uint32_t) + sizeof(std::size_t),
                             payload_bytes,
                             "CBCD communicator: outgoing record size overflow.");

        const auto kind_idx = static_cast<std::size_t>(entry.kind);
        if (kind_idx >= builder.entries_by_kind.size() or entry.angle_set_id != angle_set_id)
          throw std::logic_error("CBCD communicator: invalid outgoing record metadata.");
        auto& entries = builder.entries_by_kind[kind_idx];
        const bool opens_section = entries.empty();
        auto added_bytes = detail::CheckedAdd(entry_bytes,
                                              opens_section ? section_header_bytes : 0,
                                              "CBCD communicator: outgoing section size overflow.");
        auto projected_bytes = detail::CheckedAdd(
          builder.payload_bytes, added_bytes, "CBCD communicator: outgoing message size overflow.");

        // One builder consumes one destination credit. Post it and open another only
        // when a second credit is available. One indivisible face may exceed the target.
        const bool exceeds_byte_target =
          max_message_bytes_ > 0 and projected_bytes > max_message_bytes_;
        if (exceeds_byte_target and not builder.Empty())
        {
          PostSendBuilder(destination_index, angle_set_id, builder);
          sent_any = true;
          if (destination.in_flight_sends >= detail::max_in_flight_sends_per_destination)
            break;
          added_bytes = detail::CheckedAdd(entry_bytes,
                                           section_header_bytes,
                                           "CBCD communicator: outgoing section size overflow.");
          projected_bytes =
            detail::CheckedAdd(builder.payload_bytes,
                               added_bytes,
                               "CBCD communicator: outgoing message size overflow.");
        }

        entries.push_back(&entry);
        builder.payload_bytes = projected_bytes;
        ++slots_processed;
      }

      for (const auto destination_index : active_builder_destinations_)
      {
        auto& builder = send_builders_[destination_index];
        if (not builder.Empty())
        {
          PostSendBuilder(destination_index, angle_set_id, builder);
          sent_any = true;
        }
        builder_is_active_[destination_index] = 0;
      }
      if (slots_processed > 0)
        queue.FreeSlots(slots_processed);
    }
  }

  next_outgoing_queue_ = (next_outgoing_queue_ + 1) % num_queues;
  return sent_any;
}

bool
CBCD_AsynchronousCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::ProbeAndReceive");

  const auto& recv_comm = comm_set_.LocICommunicator(my_rank_);
  bool received_any = false;
  std::size_t messages_received = 0;
  while (messages_received < detail::max_receives_per_progress_round)
  {
    auto message = recv_comm.improbe(mpi::ANY_SOURCE, mpi_tag_);
    if (not message)
      break;
    received_any = true;
    ++messages_received;

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
      if (kind == CBCDMessageKind::DELAYED_FACE_PSI)
      {
        const auto& remaining_by_source =
          delayed_faces_remaining_by_source_and_angle_set_[angle_set_id];
        if (source_slot >= remaining_by_source.size() or
            num_entries > remaining_by_source[source_slot])
          throw std::logic_error("CBCD communicator: delayed source-face counter underflow.");
      }

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
        auto& seen = (kind == CBCDMessageKind::NORMAL_FACE_PSI)
                       ? normal_face_seen_by_source_and_angle_set_[angle_set_id][source_slot]
                       : delayed_face_seen_by_source_and_angle_set_[angle_set_id][source_slot];
        if (entry.source_face_index >= seen.size())
          throw std::runtime_error(
            kind == CBCDMessageKind::NORMAL_FACE_PSI
              ? "CBCD communicator: normal face index exceeds its source table."
              : "CBCD communicator: delayed face index exceeds its source table.");
        if (seen[entry.source_face_index] != 0)
          throw std::logic_error(kind == CBCDMessageKind::NORMAL_FACE_PSI
                                   ? "CBCD communicator: duplicate normal face record."
                                   : "CBCD communicator: duplicate delayed face record.");
        seen[entry.source_face_index] = 1;
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
        // The communication thread is the sole decrementer. Validate before mutating so a
        // malformed/duplicate section cannot wrap the unsigned liveness counter.
        auto& remaining = delayed_faces_remaining_by_angle_set_[angle_set_id];
        const auto previous = remaining.load(std::memory_order_relaxed);
        if (num_entries > previous)
          throw std::logic_error("CBCD communicator: delayed face counter underflow.");
        delayed_faces_remaining_by_source_and_angle_set_[angle_set_id][source_slot] -= num_entries;
        remaining.store(previous - num_entries, std::memory_order_release);
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

  bool completed_any = false;
  for (std::size_t i = 0; i < in_flight_sends_.size();)
  {
    if (mpi::test(in_flight_sends_[i].request))
    {
      completed_any = true;
      auto& destination = destination_states_[in_flight_sends_[i].destination_index];
      assert(destination.in_flight_sends > 0);
      --destination.in_flight_sends;
      if (i + 1 != in_flight_sends_.size())
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

  for (const auto& producer_queue : outgoing_queues_)
    if (not producer_queue->queue->Empty())
      return false;

  // The lagged incoming-nonlocal banks must be fully populated for the next sweep.
  for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
    if (not AreDelayedReceivesComplete(angle_set_id))
      return false;

  return true;
}

} // namespace opensn

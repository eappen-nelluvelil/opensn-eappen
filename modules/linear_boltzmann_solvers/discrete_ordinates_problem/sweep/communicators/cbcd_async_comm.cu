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

void
CBCD_AsynchronousCommunicator::SendBuilder::Clear()
{
  data.Clear();
  data.Data().resize(sizeof(std::size_t));
  const std::size_t zero = 0;
  std::memcpy(data.Data().data(), &zero, sizeof(zero));
  num_sections = 0;
  current_angle_set_id = static_cast<std::size_t>(-1);
  current_section_entry_count_offset = static_cast<std::size_t>(-1);
  num_records = 0;
}

CBCD_AsynchronousCommunicator::CBCD_AsynchronousCommunicator(
  const std::vector<AngleSet*>& angle_sets,
  const MPICommunicatorSet& comm_set,
  const std::vector<std::vector<int>>& incoming_source_partitions,
  const std::vector<std::vector<int>>& delayed_incoming_source_partitions,
  const std::vector<AngleSetCapacity>& capacities,
  const std::vector<DestinationCapacity>& destination_capacities)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    mpi_tag_(static_cast<int>(angle_sets.size())),
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
      auto mailbox = std::make_unique<CommittedSPSCQueue<IncomingFaceBatch>>();
      mailbox->Preallocate(capacities[i].incoming_faces);
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
      incoming_mailboxes_.push_back(std::make_unique<CommittedSPSCQueue<IncomingFaceBatch>>());
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
  std::unordered_map<int, const DestinationCapacity*> capacity_by_destination;
  capacity_by_destination.reserve(destination_capacities.size());
  for (const auto& capacity : destination_capacities)
    if (not capacity_by_destination.emplace(capacity.dest_rank, &capacity).second)
      throw std::logic_error("CBCD communicator: duplicate destination capacity.");
  if (capacity_by_destination.size() != destinations.size())
    throw std::logic_error("CBCD communicator: destination capacity table has wrong extent.");
  for (const int dest_rank : destinations)
  {
    const auto destination_index = destination_states_.size();
    const auto mapped_rank = comm_set_.MapIonJ(dest_rank, dest_rank);
    const auto capacity_it = capacity_by_destination.find(dest_rank);
    if (capacity_it == capacity_by_destination.end())
      throw std::logic_error("CBCD communicator: destination is missing its static capacity.");
    DestinationState state;
    state.dest_rank = dest_rank;
    state.mapped_rank = mapped_rank;
    state.record_bounds = capacity_it->second->records;
    state.builder_byte_bounds = capacity_it->second->builder_bytes;
    destination_states_.push_back(state);
    dest_to_state_index_[dest_rank] = destination_index;
  }
  send_builders_.resize(destination_states_.size());
  pending_send_packets_.resize(destination_states_.size());
  for (auto& builders : send_builders_)
    for (auto& builder : builders)
      builder.Clear();
  log.Log0Verbose1() << "CBCD communicator: producer_queues=" << num_angle_sets_
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
      producer_queue->queue->Preallocate(capacities[angle_set_id].outgoing_faces);
    producer_queue->queue->InitializeSlots(
      [reserve_values = capacities[angle_set_id].max_outgoing_face_values](
        OutgoingFaceData& payload) { payload.psi_data.reserve(reserve_values); });
    outgoing_queues_.push_back(std::move(producer_queue));
  }

  for (auto& done : angle_set_done_)
    done.store(false, std::memory_order_relaxed);
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
  if (comm_thread_.joinable())
    throw std::logic_error("CBCD communicator: cannot start an active communication thread.");
  if (in_flight_sends_.size() != in_flight_send_requests_.size() or not in_flight_sends_.empty())
    throw std::logic_error("CBCD communicator: stale in-flight send state at sweep start.");
  for (const auto& builders : send_builders_)
    for (const auto& builder : builders)
      if (not builder.Empty())
        throw std::logic_error("CBCD communicator: stale send builder at sweep start.");
  for (const auto& packets_by_kind : pending_send_packets_)
    for (const auto& packets : packets_by_kind)
      if (not packets.empty())
        throw std::logic_error("CBCD communicator: stale pending packet at sweep start.");
  for (const auto& destination : destination_states_)
    for (const bool active : destination.send_in_flight)
      if (active)
        throw std::logic_error("CBCD communicator: active destination channel at sweep start.");
  for (const auto& producer : outgoing_queues_)
    if (not producer->queue->Empty())
      throw std::logic_error("CBCD communicator: stale producer state at sweep start.");

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
  completed_send_indices_.clear();
  metrics_ = {};
  for (auto& builders : send_builders_)
    for (auto& builder : builders)
      builder.Clear();
  for (auto& destination : destination_states_)
  {
    destination.sent_records = {};
  }
  comm_thread_ = std::thread(&CBCD_AsynchronousCommunicator::CommThreadLoop, this);
}

void
CBCD_AsynchronousCommunicator::Stop()
{
  if (not comm_thread_.joinable())
    return;
  stop_requested_.store(true, std::memory_order_release);
  comm_thread_.join();

  const auto sum_lanes = [](const auto& values) { return values[0] + values[1]; };
  log.Log0Verbose1() << "CBCD communication summary: sent_messages="
                     << sum_lanes(metrics_.sent_messages)
                     << ", sent_normal_messages=" << metrics_.sent_messages[0]
                     << ", sent_delayed_messages=" << metrics_.sent_messages[1]
                     << ", sent_sections=" << sum_lanes(metrics_.sent_sections)
                     << ", sent_records=" << sum_lanes(metrics_.sent_records)
                     << ", sent_bytes=" << sum_lanes(metrics_.sent_bytes)
                     << ", received_messages=" << sum_lanes(metrics_.received_messages)
                     << ", received_normal_messages=" << metrics_.received_messages[0]
                     << ", received_delayed_messages=" << metrics_.received_messages[1]
                     << ", received_sections=" << sum_lanes(metrics_.received_sections)
                     << ", received_records=" << sum_lanes(metrics_.received_records)
                     << ", received_bytes=" << sum_lanes(metrics_.received_bytes)
                     << ", producer_queue_visits=" << metrics_.producer_queue_visits
                     << ", idle_progress_turns=" << metrics_.idle_progress_turns
                     << ", peak_outstanding_sends=" << metrics_.peak_outstanding_sends << ".";
}

void
CBCD_AsynchronousCommunicator::CommThreadLoop()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::CommThreadLoop");

  try
  {
    // Retire completed sends first, service one incoming message next, then inspect every
    // producer. These are complete semantic progress rounds rather than empirical work quotas.
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
      {
        ++metrics_.idle_progress_turns;
        std::this_thread::yield();
      }
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
CBCD_AsynchronousCommunicator::AppendToSendBuilder(SendBuilder& builder,
                                                   const OutgoingFaceData& entry)
{
  auto& raw = builder.data.Data();
  const auto append_bytes = [&raw](const void* const ptr, const std::size_t size)
  {
    const auto old_size = raw.size();
    raw.resize(detail::CheckedAdd(old_size, size, "CBCD communicator: message size overflow."));
    std::memcpy(raw.data() + old_size, ptr, size);
  };

  if (builder.current_angle_set_id != entry.angle_set_id)
  {
    const auto kind_byte = static_cast<std::uint8_t>(entry.kind);
    append_bytes(&kind_byte, sizeof(kind_byte));
    append_bytes(&entry.angle_set_id, sizeof(entry.angle_set_id));
    builder.current_section_entry_count_offset = raw.size();
    const std::size_t zero = 0;
    append_bytes(&zero, sizeof(zero));
    builder.current_angle_set_id = entry.angle_set_id;
    ++builder.num_sections;
    std::memcpy(raw.data(), &builder.num_sections, sizeof(builder.num_sections));
  }

  append_bytes(&entry.remote_face_index, sizeof(entry.remote_face_index));
  const auto data_size = entry.psi_data.size();
  append_bytes(&data_size, sizeof(data_size));
  const auto payload_bytes = detail::CheckedMultiply(
    data_size, sizeof(double), "CBCD communicator: outgoing face size overflow.");
  append_bytes(entry.psi_data.data(), payload_bytes);

  std::size_t section_entries = 0;
  std::memcpy(&section_entries,
              raw.data() + builder.current_section_entry_count_offset,
              sizeof(section_entries));
  ++section_entries;
  std::memcpy(raw.data() + builder.current_section_entry_count_offset,
              &section_entries,
              sizeof(section_entries));
  ++builder.num_records;
}

void
CBCD_AsynchronousCommunicator::QueueSendBuilder(const std::size_t destination_index,
                                                const CBCDMessageKind kind,
                                                SendBuilder& builder)
{
  assert(destination_index < destination_states_.size());
  assert(not builder.Empty());
  if (builder.data.Size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("CBCD communicator: MPI payload exceeds the count range.");

  const auto kind_index = static_cast<std::size_t>(kind);
  pending_send_packets_[destination_index][kind_index].push_back(std::move(builder));
  if (not reusable_send_buffers_.empty())
  {
    builder.data = std::move(reusable_send_buffers_.back());
    reusable_send_buffers_.pop_back();
  }
  builder.Clear();
}

bool
CBCD_AsynchronousCommunicator::PostNextSend(const std::size_t destination_index,
                                            const CBCDMessageKind kind)
{
  assert(destination_index < destination_states_.size());
  auto& destination = destination_states_[destination_index];
  const auto kind_index = static_cast<std::size_t>(kind);
  if (destination.send_in_flight[kind_index])
    return false;

  auto& pending_packets = pending_send_packets_[destination_index][kind_index];
  if (pending_packets.empty())
    return false;

  auto packet = std::move(pending_packets.front());
  pending_packets.pop_front();
  assert(not packet.Empty());

  InFlightSend in_flight;
  in_flight.destination_index = destination_index;
  in_flight.kind = kind;
  in_flight.data = std::move(packet.data);
  destination.send_in_flight[kind_index] = true;

  ++metrics_.sent_messages[kind_index];
  metrics_.sent_sections[kind_index] += packet.num_sections;
  metrics_.sent_records[kind_index] += packet.num_records;
  metrics_.sent_bytes[kind_index] += in_flight.data.Size();
  destination.sent_records[kind_index] =
    detail::CheckedAdd(destination.sent_records[kind_index],
                       packet.num_records,
                       "CBCD communicator: destination record count overflow.");
  if (destination.sent_records[kind_index] > destination.record_bounds[kind_index])
    throw std::logic_error("CBCD communicator: sent records exceed the static topology.");

  const auto& comm = comm_set_.LocICommunicator(destination.dest_rank);
  auto request = comm.isend(destination.mapped_rank, mpi_tag_, in_flight.data.Data());
  in_flight_sends_.push_back(std::move(in_flight));
  in_flight_send_requests_.push_back(std::move(request));
  metrics_.peak_outstanding_sends =
    std::max(metrics_.peak_outstanding_sends, in_flight_sends_.size());
  assert(in_flight_sends_.size() == in_flight_send_requests_.size());
  assert(in_flight_sends_.size() <= NUM_CBCD_MESSAGE_KINDS * destination_states_.size());

  return true;
}

bool
CBCD_AsynchronousCommunicator::FlushSendBuilders()
{
  bool sent_any = false;
  for (std::size_t destination_index = 0; destination_index < send_builders_.size();
       ++destination_index)
  {
    constexpr std::array traffic_priority{CBCDMessageKind::NORMAL_FACE_PSI,
                                          CBCDMessageKind::DELAYED_FACE_PSI};
    for (const auto kind : traffic_priority)
    {
      const auto kind_index = static_cast<std::size_t>(kind);
      auto& destination = destination_states_[destination_index];

      if (destination.send_in_flight[kind_index])
        continue;

      auto& builder = send_builders_[destination_index][kind_index];
      auto& pending_packets = pending_send_packets_[destination_index][kind_index];
      if (pending_packets.empty() and not builder.Empty())
        QueueSendBuilder(destination_index, kind, builder);

      sent_any |= PostNextSend(destination_index, kind);
    }
  }
  return sent_any;
}

bool
CBCD_AsynchronousCommunicator::SerializeAndSend()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::SerializeAndSend");

  if (outgoing_queues_.empty())
    return false;

  bool made_progress = false;
  constexpr std::size_t section_header_bytes = sizeof(std::uint8_t) + 2 * sizeof(std::size_t);
  constexpr auto mpi_max_message_bytes = static_cast<std::size_t>(std::numeric_limits<int>::max());
  // Inspect each SPSC shard exactly once per progress round. Commit() publishes the producer's
  // complete reserved prefix with release ordering and GetReadySlots() acquires it; no separate
  // notification state is necessary for correctness or liveness.
  for (std::size_t angle_set_id = 0; angle_set_id < outgoing_queues_.size(); ++angle_set_id)
  {
    auto& producer = *outgoing_queues_[angle_set_id];
    ++metrics_.producer_queue_visits;
    auto& queue = *producer.queue;
    queue.GetReadySlots(slot_cache_);
    std::size_t slots_processed = 0;
    for (const auto* slot : slot_cache_)
    {
      const auto& entry = slot->payload;
      const auto destination_it = dest_to_state_index_.find(entry.dest_rank);
      if (destination_it == dest_to_state_index_.end())
        throw std::logic_error("CBCD communicator: outgoing record has an unknown destination.");
      const auto destination_index = destination_it->second;
      const auto kind_index = static_cast<std::size_t>(entry.kind);
      if (kind_index >= send_builders_[destination_index].size() or
          entry.angle_set_id != angle_set_id)
        throw std::logic_error("CBCD communicator: invalid outgoing record metadata.");
      auto& builder = send_builders_[destination_index][kind_index];

      const auto payload_bytes = detail::CheckedMultiply(
        entry.psi_data.size(), sizeof(double), "CBCD communicator: outgoing face size overflow.");
      const auto entry_bytes =
        detail::CheckedAdd(sizeof(std::uint32_t) + sizeof(std::size_t),
                           payload_bytes,
                           "CBCD communicator: outgoing record size overflow.");

      const bool opens_section = builder.current_angle_set_id != angle_set_id;
      auto added_bytes = detail::CheckedAdd(entry_bytes,
                                            opens_section ? section_header_bytes : 0,
                                            "CBCD communicator: outgoing section size overflow.");
      auto projected_bytes = detail::CheckedAdd(
        builder.data.Size(), added_bytes, "CBCD communicator: outgoing message size overflow.");

      // CBCD does not use the AAH/AAHD maximum-message option. Splitting at the MPI
      // integer-count representation is a correctness requirement, not a tuning policy.
      if (projected_bytes > mpi_max_message_bytes and not builder.Empty())
      {
        made_progress = true;
        QueueSendBuilder(destination_index, entry.kind, builder);
        added_bytes = detail::CheckedAdd(
          entry_bytes, section_header_bytes, "CBCD communicator: outgoing section size overflow.");
        projected_bytes = detail::CheckedAdd(
          builder.data.Size(), added_bytes, "CBCD communicator: outgoing message size overflow.");
      }
      if (projected_bytes > mpi_max_message_bytes)
        throw std::overflow_error(
          "CBCD communicator: one face record exceeds the MPI count range.");
      if (projected_bytes > destination_states_[destination_index].builder_byte_bounds[kind_index])
        throw std::logic_error("CBCD communicator: packet exceeds its static topology bound.");

      AppendToSendBuilder(builder, entry);
      assert(builder.data.Size() == projected_bytes);
      ++slots_processed;
      made_progress = true;

      if (builder.data.Size() == mpi_max_message_bytes)
        QueueSendBuilder(destination_index, entry.kind, builder);
    }

    if (slots_processed > 0)
      queue.FreeSlots(slots_processed);
  }

  // One complete producer scan is the semantic flush boundary. Active channels keep accumulating
  // into their persistent builders; idle channels start exactly one transaction.
  made_progress |= FlushSendBuilders();
  return made_progress;
}

bool
CBCD_AsynchronousCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::ProbeAndReceive");

  const auto& recv_comm = comm_set_.LocICommunicator(my_rank_);
  auto message = recv_comm.improbe(mpi::ANY_SOURCE, mpi_tag_);
  if (not message)
    return false;

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
  if (num_sections == 0)
    throw std::runtime_error("CBCD communicator: empty wire message.");
  auto packet_kind = CBCDMessageKind::NORMAL_FACE_PSI;
  for (std::size_t section_index = 0; section_index < num_sections; ++section_index)
  {
    const auto kind = reader.LoadKind();
    const auto kind_index = static_cast<std::size_t>(kind);
    if (section_index == 0)
      packet_kind = kind;
    else if (kind != packet_kind)
      throw std::runtime_error("CBCD communicator: mixed traffic kinds in one wire message.");
    const auto angle_set_id = reader.LoadSize();
    const auto num_entries = reader.LoadSize();
    ++metrics_.received_sections[kind_index];
    metrics_.received_records[kind_index] += num_entries;
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

    incoming_mailboxes_[angle_set_id]->Commit();
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

  // A packet contains sections from one traffic lane because normal and delayed builders
  // are disjoint. Attribute its wire bytes once after validating every section.
  const auto packet_kind_index = static_cast<std::size_t>(packet_kind);
  ++metrics_.received_messages[packet_kind_index];
  metrics_.received_bytes[packet_kind_index] += recv_buffer_.Data().size();

  return true;
}

bool
CBCD_AsynchronousCommunicator::PollInFlightSends()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::PollInFlightSends");

  if (in_flight_send_requests_.empty())
    return false;

  completed_send_indices_.clear();
  mpi::test_some(in_flight_send_requests_, completed_send_indices_);
  std::ranges::sort(completed_send_indices_, std::greater<>{});
  for (const int completed_index : completed_send_indices_)
  {
    const auto i = static_cast<std::size_t>(completed_index);
    auto& in_flight = in_flight_sends_[i];
    assert(in_flight.destination_index < destination_states_.size());
    auto& destination = destination_states_[in_flight.destination_index];
    const auto kind_index = static_cast<std::size_t>(in_flight.kind);
    assert(kind_index < destination.send_in_flight.size());
    assert(destination.send_in_flight[kind_index]);
    destination.send_in_flight[kind_index] = false;
    in_flight.data.Clear();
    reusable_send_buffers_.push_back(std::move(in_flight.data));
    if (i + 1 != in_flight_sends_.size())
    {
      in_flight_sends_[i] = std::move(in_flight_sends_.back());
      in_flight_send_requests_[i] = in_flight_send_requests_.back();
    }
    in_flight_sends_.pop_back();
    in_flight_send_requests_.pop_back();
  }
  assert(in_flight_sends_.size() == in_flight_send_requests_.size());
  return not completed_send_indices_.empty();
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

  for (const auto& builders : send_builders_)
    for (const auto& builder : builders)
      if (not builder.Empty())
        return false;

  for (const auto& packets_by_kind : pending_send_packets_)
    for (const auto& packets : packets_by_kind)
      if (not packets.empty())
        return false;

  for (const auto& destination : destination_states_)
  {
    if (destination.sent_records != destination.record_bounds)
      throw std::logic_error(
        "CBCD communicator: completed sweep does not match the static outgoing topology.");
    if (destination.send_in_flight != std::array<bool, NUM_CBCD_MESSAGE_KINDS>{})
      return false;
  }

  // The lagged incoming-nonlocal banks must be fully populated for the next sweep.
  for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
    if (not AreDelayedReceivesComplete(angle_set_id))
      return false;

  return true;
}

} // namespace opensn

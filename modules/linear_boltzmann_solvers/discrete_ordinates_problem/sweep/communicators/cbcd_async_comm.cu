// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "framework/utils/error.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <set>

namespace opensn
{

namespace detail
{

constexpr std::size_t MPI_BYTE_COUNT_LIMIT =
  static_cast<std::size_t>(std::numeric_limits<int>::max());

struct BufferReader
{
  const std::byte* ptr = nullptr;

  std::size_t LoadSize()
  {
    std::size_t value{};
    std::memcpy(&value, ptr, sizeof(std::size_t));
    ptr += sizeof(std::size_t);
    return value;
  }

  std::uint32_t LoadFaceIndex()
  {
    std::uint32_t value{};
    std::memcpy(&value, ptr, sizeof(std::uint32_t));
    ptr += sizeof(std::uint32_t);
    return value;
  }

  void SkipBytes(const std::size_t num_bytes)
  {
    ptr += num_bytes;
  }

  const std::byte* Data() const noexcept { return ptr; }
};

} // namespace detail

CBCD_AsynchronousCommunicator::CBCD_AsynchronousCommunicator(
  const std::vector<AngleSet*>& angle_sets,
  const MPICommunicatorSet& comm_set,
  const std::vector<std::vector<int>>& incoming_source_partitions,
  const std::size_t max_message_bytes,
  const std::vector<AngleSetCapacity>& capacities)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    capacities_(capacities),
    mpi_tag_(static_cast<int>(angle_sets.size())),
    angle_set_done_(angle_sets.size())
{
  std::set<int> sources;
  std::set<int> destinations;
  for (std::size_t i = 0; i < angle_sets.size(); ++i)
  {
    const auto& spds = angle_sets[i]->GetSPDS();
    sources.insert(spds.GetLocationDependencies().begin(), spds.GetLocationDependencies().end());
    destinations.insert(spds.GetLocationSuccessors().begin(), spds.GetLocationSuccessors().end());

    if (capacities[i].incoming_faces > 0)
    {
      auto mailbox = std::make_unique<LockFreeSPSCSlotQueue<IncomingFaceBatch>>();
      mailbox->Preallocate(capacities[i].incoming_faces);
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
      incoming_mailboxes_.push_back(std::make_unique<LockFreeSPSCSlotQueue<IncomingFaceBatch>>());
  }

  my_rank_ = opensn::mpi_comm.rank();
  source_partitions_.assign(sources.begin(), sources.end());
  source_ranks_.reserve(source_partitions_.size());
  for (const int source_partition : source_partitions_)
    source_ranks_.push_back(comm_set_.MapIonJ(source_partition, my_rank_));

  source_partition_to_slot_by_angle_set_.resize(angle_sets.size());
  for (std::size_t angle_set_id = 0; angle_set_id < angle_sets.size(); ++angle_set_id)
  {
    auto& source_to_slot = source_partition_to_slot_by_angle_set_[angle_set_id];
    const auto& source_partitions = incoming_source_partitions[angle_set_id];
    source_to_slot.reserve(source_partitions.size());
    for (std::size_t source_slot = 0; source_slot < source_partitions.size(); ++source_slot)
      source_to_slot.emplace(source_partitions[source_slot],
                             static_cast<std::uint32_t>(source_slot));
  }

  destination_ranks_.assign(destinations.begin(), destinations.end());
  dest_to_queue_index_.reserve(destination_ranks_.size());
  for (std::size_t queue_index = 0; queue_index < destination_ranks_.size(); ++queue_index)
    dest_to_queue_index_.emplace(destination_ranks_[queue_index], queue_index);

  send_batch_by_angle_set_.resize(num_angle_sets_);
  for (auto& done : angle_set_done_)
    done.store(false, std::memory_order_relaxed);

  message_limit_ = max_message_bytes == 0
                     ? detail::MPI_BYTE_COUNT_LIMIT
                     : std::min(max_message_bytes, detail::MPI_BYTE_COUNT_LIMIT);
  if (max_message_bytes > 0)
    recv_buffer_.Data().reserve(message_limit_);
}

CBCD_AsynchronousCommunicator::~CBCD_AsynchronousCommunicator()
{
  if (comm_thread_.joinable())
    Stop();
}

void
CBCD_AsynchronousCommunicator::SignalAngleSetComplete(const std::size_t angle_set_id)
{
  angle_set_done_[angle_set_id].store(true, std::memory_order_release);
}

void
CBCD_AsynchronousCommunicator::ConfigureProducerQueues(const std::size_t num_producers)
{
  if (num_producers_ == num_producers and outgoing_queues_.size() == destination_ranks_.size())
    return;

  num_producers_ = num_producers;
  std::vector<std::unordered_map<int, std::size_t>> producer_face_counts(num_producers);
  std::vector<std::size_t> producer_max_outgoing_face_values(num_producers, 0);

  // Match the scheduler's static angle-set partitioning.
  const auto chunk_size = (num_angle_sets_ + num_producers - 1) / num_producers;
  for (std::size_t producer_id = 0; producer_id < num_producers; ++producer_id)
  {
    const auto begin = std::min(producer_id * chunk_size, num_angle_sets_);
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

  outgoing_queues_.clear();
  outgoing_queues_.resize(destination_ranks_.size());

  std::size_t realized_queues = 0;
  for (std::size_t queue_index = 0; queue_index < destination_ranks_.size(); ++queue_index)
  {
    auto& destination_queue = outgoing_queues_[queue_index];
    destination_queue.dest_rank = destination_ranks_[queue_index];
    destination_queue.producer_queues.resize(num_producers);
    destination_queue.realized_producers.clear();

    for (std::size_t producer_id = 0; producer_id < num_producers; ++producer_id)
    {
      auto queue = std::make_unique<OutgoingQueue>();
      const auto count_it = producer_face_counts[producer_id].find(destination_queue.dest_rank);
      if (count_it != producer_face_counts[producer_id].end())
      {
        ++realized_queues;
        destination_queue.realized_producers.push_back(producer_id);
        queue->Preallocate(count_it->second);
        queue->InitializeSlots(
          [reserve = producer_max_outgoing_face_values[producer_id]](OutgoingFaceData& payload)
          {
            payload.angle_set_id = 0;
            payload.remote_face_index = 0;
            payload.psi_data.clear();
            payload.psi_data.reserve(reserve);
          });
      }
      destination_queue.producer_queues[producer_id] = std::move(queue);
    }
  }

  log.Log0Verbose1() << "CBCD communicator: producer_queues=" << realized_queues
                     << ", destinations=" << destination_ranks_.size() << ".";
}

void
CBCD_AsynchronousCommunicator::Start(const std::size_t num_producers)
{
  ConfigureProducerQueues(num_producers);

  stop_requested_.store(false, std::memory_order_relaxed);
  for (auto& done : angle_set_done_)
    done.store(false, std::memory_order_relaxed);

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
CBCD_AsynchronousCommunicator::FlushDestination(const std::size_t destination_queue_index)
{
  auto& destination_queue = outgoing_queues_[destination_queue_index];
  if (destination_queue.realized_producers.empty())
    return false;

  bool work_done = false;
  std::size_t current_payload_bytes = sizeof(std::size_t);
  constexpr std::size_t section_header_bytes = 2 * sizeof(std::size_t);

  const auto send_batch = [&]()
  {
    // [section count], followed by [angle-set id, record count] sections.
    InFlightSend in_flight;
    in_flight.data.Data().resize(current_payload_bytes);
    std::size_t offset = 0;
    const auto write_bytes = [&](const void* ptr, const std::size_t size)
    {
      std::memcpy(in_flight.data.Data().data() + offset, ptr, size);
      offset += size;
    };

    const auto num_sections = active_section_ids_.size();
    write_bytes(&num_sections, sizeof(std::size_t));
    for (const auto angle_set_id : active_section_ids_)
    {
      auto& entries = send_batch_by_angle_set_[angle_set_id];
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
    const auto& comm = comm_set_.LocICommunicator(destination_queue.dest_rank);
    const auto mapped_rank =
      comm_set_.MapIonJ(destination_queue.dest_rank, destination_queue.dest_rank);
    in_flight.request = comm.isend(mapped_rank, mpi_tag_, in_flight.data.Data());
    in_flight_sends_.push_back(std::move(in_flight));
    current_payload_bytes = sizeof(std::size_t);
    active_section_ids_.clear();

    for (const auto& [queue, count] : deferred_slot_releases_)
      queue->ReleaseReadySlots(count);
    deferred_slot_releases_.clear();
  };

  // Poll every realizable SPSC queue without separate wake-up state.
  for (const auto producer_id : destination_queue.realized_producers)
  {
    auto& queue = *destination_queue.producer_queues[producer_id];
    queue.PeekReadySlots(slot_cache_);
    if (slot_cache_.empty())
      continue;

    for (const auto* slot : slot_cache_)
    {
      const auto& entry = slot->payload;
      constexpr std::size_t record_header_bytes = sizeof(std::uint32_t) + sizeof(std::size_t);
      OpenSnLogicalErrorIf(
        entry.psi_data.size() > (detail::MPI_BYTE_COUNT_LIMIT - sizeof(std::size_t) -
                                 section_header_bytes - record_header_bytes) /
                                  sizeof(double),
        "One CBCD face record exceeds the MPI int byte-count limit and cannot be serialized.");
      const auto entry_bytes = record_header_bytes + entry.psi_data.size() * sizeof(double);
      auto& entries = send_batch_by_angle_set_[entry.angle_set_id];
      const auto appended_bytes = entry_bytes + (entries.empty() ? section_header_bytes : 0);

      if (current_payload_bytes + appended_bytes > message_limit_ and
          not active_section_ids_.empty())
        send_batch();

      if (entries.empty())
      {
        active_section_ids_.push_back(entry.angle_set_id);
        current_payload_bytes += section_header_bytes;
      }
      entries.push_back(&entry);
      current_payload_bytes += entry_bytes;
    }

    // The section vectors still point into these slots. Return them only after serialization.
    deferred_slot_releases_.emplace_back(&queue, slot_cache_.size());
    work_done = true;
  }

  if (not active_section_ids_.empty())
    send_batch();
  return work_done;
}

bool
CBCD_AsynchronousCommunicator::SerializeAndSend()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::SerializeAndSend");

  bool work_done = false;
  for (std::size_t queue_index = 0; queue_index < outgoing_queues_.size(); ++queue_index)
    work_done |= FlushDestination(queue_index);
  return work_done;
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

      detail::BufferReader reader{reinterpret_cast<const std::byte*>(recv_buffer_.Data().data())};
      const auto num_sections = reader.LoadSize();
      for (std::size_t section_index = 0; section_index < num_sections; ++section_index)
      {
        const auto angle_set_id = reader.LoadSize();
        const auto num_entries = reader.LoadSize();

        const auto& slot_map = source_partition_to_slot_by_angle_set_[angle_set_id];
        const auto slot_it = slot_map.find(source_partition);
        const auto source_slot = slot_it->second;

        const auto* const section_ptr = reader.Data();
        std::size_t total_values = 0;
        for (std::size_t entry_index = 0; entry_index < num_entries; ++entry_index)
        {
          reader.LoadFaceIndex();
          const auto data_size = reader.LoadSize();
          reader.SkipBytes(data_size * sizeof(double));
          total_values += data_size;
        }
        auto& slot = incoming_mailboxes_[angle_set_id]->ReserveSlot();
        auto& batch = slot.payload;
        batch.source_slot = source_slot;
        batch.entries.resize(num_entries);
        batch.psi_data.resize(total_values);
        detail::BufferReader section_reader{section_ptr};
        std::size_t value_offset = 0;
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

  for (const auto& destination_queue : outgoing_queues_)
    for (const auto& queue : destination_queue.producer_queues)
      if (not queue->Empty())
        return false;

  return true;
}

} // namespace opensn

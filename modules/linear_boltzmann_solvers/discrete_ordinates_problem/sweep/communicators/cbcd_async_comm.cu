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
#include <limits>
#include <set>

namespace opensn
{

CBCD_AsynchronousCommunicator::CBCD_AsynchronousCommunicator(
  const std::vector<AngleSet*>& angle_sets,
  const MPICommunicatorSet& comm_set,
  const std::vector<std::vector<int>>& incoming_source_partitions,
  const std::vector<std::vector<int>>& delayed_source_partitions,
  const std::size_t max_message_bytes,
  const std::vector<AngleSetCapacity>& capacities)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    mpi_tag_(static_cast<int>(angle_sets.size())),
    delayed_mpi_tag_(static_cast<int>(2 * angle_sets.size())),
    max_message_bytes_(max_message_bytes),
    angle_set_done_(angle_sets.size())
{
  assert(incoming_source_partitions.size() == angle_sets.size());
  assert(delayed_source_partitions.size() == angle_sets.size());
  assert(capacities.size() == angle_sets.size());

  std::set<int> sources;
  std::set<int> delayed_sources;
  std::set<int> destinations;
  std::set<int> delayed_destinations;
  std::size_t total_outgoing_faces = 0;
  std::size_t max_outgoing_face_values = 0;
  std::size_t total_delayed_outgoing_faces = 0;
  std::size_t max_delayed_outgoing_face_values = 0;

  for (std::size_t i = 0; i < angle_sets.size(); ++i)
  {
    const auto* angle_set = angle_sets[i];
    const auto& spds = angle_set->GetSPDS();
    for (const int dep : spds.GetLocationDependencies())
      sources.insert(dep);
    for (const int dep : spds.GetDelayedLocationDependencies())
      delayed_sources.insert(dep);
    for (const int succ : spds.GetLocationSuccessors())
      destinations.insert(succ);
    for (const int succ : spds.GetDelayedLocationSuccessors())
      delayed_destinations.insert(succ);

    total_outgoing_faces += capacities[i].outgoing_faces;
    max_outgoing_face_values =
      std::max(max_outgoing_face_values, capacities[i].max_outgoing_face_values);
    total_delayed_outgoing_faces += capacities[i].outgoing_faces;
    max_delayed_outgoing_face_values =
      std::max(max_delayed_outgoing_face_values, capacities[i].max_outgoing_face_values);
    if (capacities[i].incoming_faces > 0)
    {
      auto mailbox = std::make_unique<LockFreeRingBuffer<std::vector<IncomingFaceData>>>();
      mailbox->Preallocate(capacities[i].incoming_faces + 1);
      mailbox->InitializeSlots(
        [&](std::vector<IncomingFaceData>& batch)
        {
          batch.reserve(capacities[i].max_incoming_batch_entries);
          batch.resize(capacities[i].max_incoming_batch_entries);
          for (auto& entry : batch)
            entry.psi_data.reserve(capacities[i].max_incoming_face_values);
          batch.clear();
        });
      incoming_mailboxes_.push_back(std::move(mailbox));
    }
    else
    {
      incoming_mailboxes_.push_back(
        std::make_unique<LockFreeRingBuffer<std::vector<IncomingFaceData>>>());
    }

    if (capacities[i].delayed_incoming_faces > 0)
    {
      auto mailbox = std::make_unique<LockFreeRingBuffer<std::vector<IncomingFaceData>>>();
      mailbox->Preallocate(capacities[i].delayed_incoming_faces + 1);
      mailbox->InitializeSlots(
        [&](std::vector<IncomingFaceData>& batch)
        {
          batch.reserve(capacities[i].max_delayed_incoming_batch_entries);
          batch.resize(capacities[i].max_delayed_incoming_batch_entries);
          for (auto& entry : batch)
            entry.psi_data.reserve(capacities[i].max_delayed_incoming_face_values);
          batch.clear();
        });
      delayed_incoming_mailboxes_.push_back(std::move(mailbox));
    }
    else
    {
      delayed_incoming_mailboxes_.push_back(
        std::make_unique<LockFreeRingBuffer<std::vector<IncomingFaceData>>>());
    }
  }

  my_rank_ = opensn::mpi_comm.rank();
  source_partitions_.assign(sources.begin(), sources.end());
  source_ranks_.reserve(source_partitions_.size());
  for (const int source_partition : source_partitions_)
    source_ranks_.push_back(comm_set_.MapIonJ(source_partition, my_rank_));
  delayed_source_partitions_.assign(delayed_sources.begin(), delayed_sources.end());
  delayed_source_ranks_.reserve(delayed_source_partitions_.size());
  for (const int source_partition : delayed_source_partitions_)
    delayed_source_ranks_.push_back(comm_set_.MapIonJ(source_partition, my_rank_));

  source_partition_to_slot_by_angle_set_.resize(angle_sets.size());
  delayed_source_partition_to_slot_by_angle_set_.resize(angle_sets.size());
  delayed_destinations_by_angle_set_.resize(angle_sets.size());
  for (std::size_t angle_set_id = 0; angle_set_id < angle_sets.size(); ++angle_set_id)
  {
    const auto* angle_set = angle_sets[angle_set_id];
    auto& source_to_slot = source_partition_to_slot_by_angle_set_[angle_set_id];
    const auto& source_partitions = incoming_source_partitions[angle_set_id];
    source_to_slot.reserve(source_partitions.size());
    for (std::size_t source_slot = 0; source_slot < source_partitions.size(); ++source_slot)
      source_to_slot.emplace(source_partitions[source_slot],
                             static_cast<std::uint32_t>(source_slot));

    auto& delayed_source_to_slot = delayed_source_partition_to_slot_by_angle_set_[angle_set_id];
    const auto& delayed_partitions = delayed_source_partitions[angle_set_id];
    delayed_source_to_slot.reserve(delayed_partitions.size());
    for (std::size_t source_slot = 0; source_slot < delayed_partitions.size(); ++source_slot)
      delayed_source_to_slot.emplace(delayed_partitions[source_slot],
                                     static_cast<std::uint32_t>(source_slot));

    delayed_destinations_by_angle_set_[angle_set_id] =
      angle_set->GetSPDS().GetDelayedLocationSuccessors();
  }

  outgoing_queues_.reserve(destinations.size());
  dest_to_queue_index_.reserve(destinations.size());
  int queue_index = 0;
  for (const int dest_rank : destinations)
  {
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

  delayed_outgoing_queues_.reserve(delayed_destinations.size());
  int delayed_queue_index = 0;
  for (const int dest_rank : delayed_destinations)
  {
    auto queue = std::make_unique<DestinationQueue>();
    queue->dest_rank = dest_rank;
    queue->queue = std::make_unique<LockFreeRingBuffer<OutgoingFaceData>>();
    if (total_delayed_outgoing_faces > 0)
      queue->queue->Preallocate(total_delayed_outgoing_faces + 1);
    queue->queue->InitializeSlots([max_delayed_outgoing_face_values](OutgoingFaceData& payload)
                                  { payload.psi_data.reserve(max_delayed_outgoing_face_values); });
    delayed_outgoing_queues_.push_back(std::move(queue));
    delayed_dest_to_queue_index_[dest_rank] = delayed_queue_index++;
  }

  send_batch_by_angle_set_.resize(num_angle_sets_);
  delayed_send_batch_by_angle_set_.resize(num_angle_sets_);
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
CBCD_AsynchronousCommunicator::Start()
{
  stop_requested_.store(false, std::memory_order_relaxed);
  for (auto& done : angle_set_done_)
    done.store(false, std::memory_order_relaxed);
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
CBCD_AsynchronousCommunicator::SerializeAndSend()
{
  CALI_CXX_MARK_SCOPE("CBCD_AsynchronousCommunicator::SerializeAndSend");

  bool sent_any = false;

  const auto serialize_queue_set =
    [this, &sent_any](std::vector<std::unique_ptr<DestinationQueue>>& queues,
                      std::vector<std::vector<const OutgoingFaceData*>>& send_batches,
                      const int mpi_tag)
  {
    for (auto& destination_queue : queues)
    {
      destination_queue->queue->GetReadySlots(slot_cache_);
      if (slot_cache_.empty())
        continue;

      std::size_t current_payload_bytes = sizeof(std::size_t);
      std::size_t active_angle_sets = 0;
      std::size_t slots_processed = 0;

      const auto send_batch = [&]()
      {
        InFlightSend in_flight;
        in_flight.data.Data().resize(current_payload_bytes);
        std::size_t offset = 0;

        const auto write_bytes = [&](const void* ptr, const std::size_t size)
        {
          std::memcpy(in_flight.data.Data().data() + offset, ptr, size);
          offset += size;
        };

        write_bytes(&active_angle_sets, sizeof(std::size_t));
        for (std::size_t angle_set_id = 0; angle_set_id < num_angle_sets_; ++angle_set_id)
        {
          auto& entries = send_batches[angle_set_id];
          if (entries.empty())
            continue;

          write_bytes(&angle_set_id, sizeof(std::size_t));
          const auto num_entries = entries.size();
          write_bytes(&num_entries, sizeof(std::size_t));
          for (const auto* entry : entries)
          {
            write_bytes(&entry->cell_global_id, sizeof(std::uint64_t));
            write_bytes(&entry->face_id, sizeof(unsigned int));
            const auto data_size = entry->psi_data.size();
            write_bytes(&data_size, sizeof(std::size_t));
            write_bytes(entry->psi_data.data(), data_size * sizeof(double));
          }
          entries.clear();
        }

        const auto& comm = comm_set_.LocICommunicator(destination_queue->dest_rank);
        const auto mapped_rank =
          comm_set_.MapIonJ(destination_queue->dest_rank, destination_queue->dest_rank);
        in_flight.request = comm.isend(mapped_rank, mpi_tag, in_flight.data.Data());
        in_flight_sends_.push_back(std::move(in_flight));
      };

      for (std::size_t slot_index = 0; slot_index < slot_cache_.size(); ++slot_index)
      {
        const auto* slot = slot_cache_[slot_index];
        const auto& entry = slot->payload;
        const auto entry_bytes = sizeof(std::uint64_t) + sizeof(unsigned int) +
                                 sizeof(std::size_t) + entry.psi_data.size() * sizeof(double);

        if (max_message_bytes_ > 0 && current_payload_bytes + entry_bytes > max_message_bytes_ &&
            active_angle_sets > 0)
        {
          send_batch();
          destination_queue->queue->FreeSlots(slots_processed);
          current_payload_bytes = sizeof(std::size_t);
          active_angle_sets = 0;
          slots_processed = 0;
        }

        auto& entries = send_batches[entry.angle_set_id];
        if (entries.empty())
        {
          ++active_angle_sets;
          current_payload_bytes += 2 * sizeof(std::size_t);
        }
        entries.push_back(&entry);
        current_payload_bytes += entry_bytes;
        ++slots_processed;
      }

      if (active_angle_sets > 0)
      {
        send_batch();
        destination_queue->queue->FreeSlots(slots_processed);
      }

      sent_any = true;
    }
  };

  serialize_queue_set(outgoing_queues_, send_batch_by_angle_set_, mpi_tag_);
  serialize_queue_set(delayed_outgoing_queues_, delayed_send_batch_by_angle_set_, delayed_mpi_tag_);

  return sent_any;
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

      recv_buffer_.Seek(0);
      const auto num_active_angle_sets = recv_buffer_.Read<std::size_t>();
      for (std::size_t as_batch = 0; as_batch < num_active_angle_sets; ++as_batch)
      {
        const auto angle_set_id = recv_buffer_.Read<std::size_t>();
        const auto num_entries = recv_buffer_.Read<std::size_t>();
        assert(angle_set_id < num_angle_sets_);

        const auto slot_it =
          source_partition_to_slot_by_angle_set_[angle_set_id].find(source_partition);
        assert(slot_it != source_partition_to_slot_by_angle_set_[angle_set_id].end());
        const auto source_slot = slot_it->second;

        auto& slot = incoming_mailboxes_[angle_set_id]->ReserveSlot();
        auto& batch = slot.payload;
        batch.resize(num_entries);
        for (std::size_t entry_index = 0; entry_index < num_entries; ++entry_index)
        {
          auto& entry = batch[entry_index];
          entry.cell_global_id = recv_buffer_.Read<std::uint64_t>();
          entry.face_id = recv_buffer_.Read<unsigned int>();
          entry.source_slot = source_slot;
          const auto data_size = recv_buffer_.Read<std::size_t>();
          entry.psi_data.resize(data_size);
          std::memcpy(entry.psi_data.data(),
                      recv_buffer_.Data().data() + recv_buffer_.Offset(),
                      data_size * sizeof(double));
          recv_buffer_.Seek(recv_buffer_.Offset() + data_size * sizeof(double));
        }

        incoming_mailboxes_[angle_set_id]->PublishSlot(slot);
      }
    }
  }

  for (std::size_t source_index = 0; source_index < delayed_source_ranks_.size(); ++source_index)
  {
    const int source_partition = delayed_source_partitions_[source_index];
    const int source_rank = delayed_source_ranks_[source_index];
    mpi::Status status;

    while (recv_comm.iprobe(source_rank, delayed_mpi_tag_, status))
    {
      received_any = true;
      const auto num_bytes = status.count<std::byte>();
      recv_buffer_.Data().resize(static_cast<std::size_t>(num_bytes));
      recv_comm.recv(source_rank, status.tag(), recv_buffer_.Data().data(), num_bytes);

      recv_buffer_.Seek(0);
      const auto num_active_angle_sets = recv_buffer_.Read<std::size_t>();
      for (std::size_t as_batch = 0; as_batch < num_active_angle_sets; ++as_batch)
      {
        const auto angle_set_id = recv_buffer_.Read<std::size_t>();
        const auto num_entries = recv_buffer_.Read<std::size_t>();
        assert(angle_set_id < num_angle_sets_);

        const auto slot_it =
          delayed_source_partition_to_slot_by_angle_set_[angle_set_id].find(source_partition);
        assert(slot_it != delayed_source_partition_to_slot_by_angle_set_[angle_set_id].end());
        const auto source_slot = slot_it->second;

        auto& slot = delayed_incoming_mailboxes_[angle_set_id]->ReserveSlot();
        auto& batch = slot.payload;
        batch.resize(num_entries);
        for (std::size_t entry_index = 0; entry_index < num_entries; ++entry_index)
        {
          auto& entry = batch[entry_index];
          entry.cell_global_id = recv_buffer_.Read<std::uint64_t>();
          entry.face_id = recv_buffer_.Read<unsigned int>();
          entry.source_slot = source_slot;
          const auto data_size = recv_buffer_.Read<std::size_t>();
          entry.psi_data.resize(data_size);
          std::memcpy(entry.psi_data.data(),
                      recv_buffer_.Data().data() + recv_buffer_.Offset(),
                      data_size * sizeof(double));
          recv_buffer_.Seek(recv_buffer_.Offset() + data_size * sizeof(double));
        }

        delayed_incoming_mailboxes_[angle_set_id]->PublishSlot(slot);
      }
    }
  }

  return received_any;
}

void
CBCD_AsynchronousCommunicator::EnqueueDelayedCompletionMarkers(const std::size_t angle_set_id)
{
  for (const int dest_rank : delayed_destinations_by_angle_set_[angle_set_id])
  {
    const auto it = delayed_dest_to_queue_index_.find(dest_rank);
    assert(it != delayed_dest_to_queue_index_.end());
    auto& destination_queue = delayed_outgoing_queues_[it->second];
    auto& queue = *destination_queue->queue;
    auto& slot = queue.ReserveSlot();
    slot.payload.angle_set_id = angle_set_id;
    slot.payload.cell_global_id = std::numeric_limits<std::uint64_t>::max();
    slot.payload.face_id = std::numeric_limits<unsigned int>::max();
    slot.payload.psi_data.clear();
    queue.PublishSlot(slot);
  }
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
    if (not destination_queue->queue->Empty())
      return false;

  for (const auto& destination_queue : delayed_outgoing_queues_)
    if (not destination_queue->queue->Empty())
      return false;

  return true;
}

} // namespace opensn

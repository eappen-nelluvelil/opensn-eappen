// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class MPICommunicatorSet;

/// Metadata for one received non-local face payload inside an incoming batch.
struct IncomingFaceBatchEntry
{
  std::uint32_t source_face_index = 0;
  std::size_t payload_offset = 0;
  std::size_t payload_size = 0;
};

/// One received mailbox payload grouped by sending source slot and angle set.
struct IncomingFaceBatch
{
  std::uint32_t source_slot = 0;
  std::vector<IncomingFaceBatchEntry> entries;
  std::vector<double> psi_data;
};

/// One outgoing non-local face payload published by a sweep worker.
struct OutgoingFaceData
{
  std::size_t angle_set_id = 0;
  std::uint32_t remote_face_index = 0;
  std::vector<double> psi_data;
};

/// Exact outgoing-record capacity contributed by one angle set to one destination.
struct OutgoingDestinationCapacity
{
  int dest_rank = -1;
  std::size_t face_count = 0;
};

/// Queue-capacity summary for one angle set.
struct AngleSetCapacity
{
  std::size_t incoming_faces = 0;
  std::size_t max_outgoing_face_values = 0;
  std::size_t max_incoming_batch_entries = 0;
  std::size_t max_incoming_batch_values = 0;
  std::vector<OutgoingDestinationCapacity> outgoing_faces_by_destination;
};

/** Aggregated CBCD communicator with per-worker SPSC queues and one MPI progress thread. */
class CBCD_AsynchronousCommunicator
{
public:
  CBCD_AsynchronousCommunicator(const std::vector<AngleSet*>& angle_sets,
                                const MPICommunicatorSet& comm_set,
                                const std::vector<std::vector<int>>& incoming_source_partitions,
                                std::size_t max_message_bytes,
                                const std::vector<AngleSetCapacity>& capacities);

  ~CBCD_AsynchronousCommunicator();

  template <typename FillCallback>
  void EnqueueOutgoing(int dest_rank,
                       std::size_t producer_id,
                       std::size_t angle_set_id,
                       std::uint32_t remote_face_index,
                       std::size_t data_size,
                       FillCallback&& fill)
  {
    const auto it = dest_to_queue_index_.find(dest_rank);
    auto& queue = *outgoing_queues_[it->second].producer_queues[producer_id];
    auto& slot = queue.ReserveSlot();
    slot.payload.angle_set_id = angle_set_id;
    slot.payload.remote_face_index = remote_face_index;
    slot.payload.psi_data.resize(data_size);
    fill(slot.payload.psi_data.data());
    queue.PublishSlot();
  }

  template <typename Callback>
  bool ProcessIncoming(std::size_t angle_set_id, Callback&& callback)
  {
    return incoming_mailboxes_[angle_set_id]->ProcessReady(std::forward<Callback>(callback)) > 0;
  }

  bool HasIncoming(std::size_t angle_set_id) const
  {
    return not incoming_mailboxes_[angle_set_id]->Empty();
  }

  void SignalAngleSetComplete(std::size_t angle_set_id);
  void Start(std::size_t num_producers);
  void Stop();

private:
  using OutgoingQueue = LockFreeSPSCSlotQueue<OutgoingFaceData>;

  struct DestinationQueue
  {
    int dest_rank = 0;
    std::vector<std::unique_ptr<OutgoingQueue>> producer_queues;
    /// Producers owning at least one outgoing face toward this destination.
    std::vector<std::size_t> realized_producers;
  };

  struct InFlightSend
  {
    mpi::Request request;
    ByteArray data;
  };

  void CommThreadLoop();
  void ConfigureProducerQueues(std::size_t num_producers);
  bool FlushDestination(std::size_t destination_queue_index);
  bool SerializeAndSend();
  bool ProbeAndReceive();
  bool PollInFlightSends();
  bool AllAngleSetsComplete() const;

  const MPICommunicatorSet& comm_set_;
  std::size_t num_angle_sets_;
  std::vector<AngleSetCapacity> capacities_;
  std::size_t num_producers_ = 0;
  int mpi_tag_;
  std::size_t message_limit_ = 0;
  int my_rank_ = 0;
  std::vector<int> source_partitions_;
  std::vector<int> source_ranks_;
  std::vector<std::unordered_map<int, std::uint32_t>> source_partition_to_slot_by_angle_set_;
  std::vector<int> destination_ranks_;
  std::vector<DestinationQueue> outgoing_queues_;
  std::unordered_map<int, std::size_t> dest_to_queue_index_;
  std::vector<std::unique_ptr<LockFreeSPSCSlotQueue<IncomingFaceBatch>>> incoming_mailboxes_;
  std::vector<std::vector<const OutgoingFaceData*>> send_batch_by_angle_set_;
  /// Angle sets holding records for the message currently being packed.
  std::vector<std::size_t> active_section_ids_;
  ByteArray recv_buffer_;
  std::vector<InFlightSend> in_flight_sends_;
  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_done_;
  std::thread comm_thread_;
  std::vector<OutgoingQueue::Slot*> slot_cache_;
  std::vector<std::pair<OutgoingQueue*, std::size_t>> deferred_slot_releases_;
};

} // namespace opensn

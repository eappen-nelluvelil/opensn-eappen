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
  /// Source-slot-local face index carried on the wire.
  std::uint32_t source_face_index = 0;
  /// Offset of this payload within `IncomingFaceBatch::psi_data`.
  std::size_t payload_offset = 0;
  /// Number of doubles in this payload.
  std::size_t payload_size = 0;
};

/// One received mailbox payload grouped by sending source slot and angle set.
struct IncomingFaceBatch
{
  /// Source-locality slot for the sending partition.
  std::uint32_t source_slot = 0;
  /// Per-face metadata for the packed payload block.
  std::vector<IncomingFaceBatchEntry> entries;
  /// Packed received doubles for all faces in the batch.
  std::vector<double> psi_data;
};

/// One outgoing non-local face payload published by a sweep worker.
struct OutgoingFaceData
{
  /// Producing angle-set ID.
  std::size_t angle_set_id = 0;
  /// Receiver-local face index understood by the destination rank.
  std::uint32_t remote_face_index = 0;
  /// Packed outgoing doubles for one non-local face.
  std::vector<double> psi_data;
};

/// Outgoing-face capacity contributed by one angle set to one destination.
struct OutgoingDestinationCapacity
{
  /// Destination rank.
  int dest_rank = -1;
  /// Number of outgoing faces sent by this angle set to `dest_rank`.
  std::size_t face_count = 0;
};

/// Queue-capacity summary for one angle set.
struct AngleSetCapacity
{
  /// Number of outgoing non-local faces produced by this angle set.
  std::size_t outgoing_faces = 0;
  /// Number of incoming non-local faces consumed by this angle set.
  std::size_t incoming_faces = 0;
  /// Maximum number of doubles in one outgoing face payload.
  std::size_t max_outgoing_face_values = 0;
  /// Maximum number of face entries in one received batch.
  std::size_t max_incoming_batch_entries = 0;
  /// Maximum number of doubles in one received batch.
  std::size_t max_incoming_batch_values = 0;
  /// Outgoing-face counts grouped by destination rank.
  std::vector<OutgoingDestinationCapacity> outgoing_faces_by_destination;
};

/**
 * Aggregated CBCD communicator with one dedicated progress thread.
 *
 * Sweep worker threads publish outgoing non-local face payloads into producer-sharded
 * per-destination SPSC queues. Each worker owns one doorbell queue used to notify the
 * communication thread when a previously idle shard becomes active. The communication
 * thread drains doorbells, maintains active-producer lists per destination, batches
 * queued payloads by angle set subject to the configured message-size limit, serializes
 * them into MPI messages, and posts nonblocking sends. The communication thread also
 * probes for incoming messages, deserializes them into compact `IncomingFaceBatch`
 * payloads, and publishes those batches into per-angle-set incoming mailboxes.
 *
 * The aggregated communicator assumes the following communication patterns and sweep worker
 * thread interactions:
 * - sweep worker threads only write outgoing queue slots,
 * - the communication thread handles only the draining of outgoing queues and routing of
 *   incoming batches to angle-set mailboxes,
 * - each angle-set owner thread only drains its own incoming mailbox.
 *
 * Aggregated communicator flow:
 * 1. A sweep worker publishes one completed non-local face payload into its own shard for
 *    the destination rank and rings the worker-local doorbell when that shard becomes active.
 * 2. The communication thread drains doorbells, gathers ready shard slots, groups them by
 *    angle set, and serializes one or more MPI messages subject to the configured byte limit.
 * 3. The destination rank probes for those messages, maps the sending partition to its local
 *    source slot, and reconstructs one compact `IncomingFaceBatch` per angle-set section.
 * 4. The communication thread publishes each reconstructed batch into the mailbox owned by
 *    that angle set.
 * 5. The angle-set owner thread drains its mailbox and copies the received face data into
 *    the corresponding non-local FLUDS storage.
 */
class CBCD_AsynchronousCommunicator
{
public:
  /**
   * Construct the CBCD asynchronous communicator.
   *
   * \param angle_sets Angle sets served by the communicator.
   * \param comm_set MPI communicator set used for point-to-point exchanges.
   * \param incoming_source_partitions Incoming source partitions grouped by angle set.
   * \param max_message_bytes Maximum serialized MPI payload size. A value of zero disables
   * message-size splitting.
   * \param capacities Queue-capacity summary for each angle set.
   */
  CBCD_AsynchronousCommunicator(const std::vector<AngleSet*>& angle_sets,
                                const MPICommunicatorSet& comm_set,
                                const std::vector<std::vector<int>>& incoming_source_partitions,
                                std::size_t max_message_bytes,
                                const std::vector<AngleSetCapacity>& capacities);

  ~CBCD_AsynchronousCommunicator();

  /**
   * Publish one outgoing non-local face payload.
   *
   * \param dest_rank Destination rank.
   * \param producer_id Worker ID that owns the publishing angle set.
   * \param angle_set_id Producing angle-set ID.
   * \param remote_face_index Receiver-local face index.
   * \param data_size Number of doubles in the payload.
   * \param fill Callback that fills the reserved payload buffer.
   */
  template <typename FillCallback>
  void EnqueueOutgoing(int dest_rank,
                       std::size_t producer_id,
                       std::size_t angle_set_id,
                       std::uint32_t remote_face_index,
                       std::size_t data_size,
                       FillCallback&& fill)
  {
    const auto it = dest_to_queue_index_.find(dest_rank);
    assert(it != dest_to_queue_index_.end());
    assert(producer_id < num_producers_);
    auto& shard = *outgoing_queues_[it->second].producer_shards[producer_id];
    auto& slot = shard.queue.ReserveSlot();
    slot.payload.angle_set_id = angle_set_id;
    slot.payload.remote_face_index = remote_face_index;
    slot.payload.psi_data.resize(data_size);
    fill(slot.payload.psi_data.data());
    shard.queue.PublishSlot();
    if (not shard.scheduled.exchange(true, std::memory_order_acq_rel))
    {
      auto& doorbell = *producer_doorbells_[producer_id];
      auto& doorbell_slot = doorbell.ReserveSlot();
      doorbell_slot.payload = it->second;
      doorbell.PublishSlot();
    }
  }

  /**
   * Drain all currently ready incoming batches for one angle set.
   *
   * \param angle_set_id Angle-set ID.
   * \param callback Callback invoked for each incoming batch payload.
   * \return `true` if at least one batch was consumed.
   */
  template <typename Callback>
  bool ProcessIncoming(std::size_t angle_set_id, Callback&& callback)
  {
    assert(angle_set_id < num_angle_sets_);
    return incoming_mailboxes_[angle_set_id]->ProcessReady(std::forward<Callback>(callback)) > 0;
  }

  /// Report whether the specified angle set currently has a published incoming batch.
  bool HasIncoming(std::size_t angle_set_id) const
  {
    assert(angle_set_id < num_angle_sets_);
    return not incoming_mailboxes_[angle_set_id]->Empty();
  }

  /// Mark one angle set as locally complete.
  void SignalAngleSetComplete(std::size_t angle_set_id);
  /// Start the communication thread for the given number of sweep workers.
  void Start(std::size_t num_producers);
  /// Request termination and join the communication thread.
  void Stop();

private:
  /// Outgoing shard for one `(destination, producer)` pair.
  struct OutgoingShard
  {
    /// Single-producer, single-consumer payload queue for this shard.
    LockFreeSPSCSlotQueue<OutgoingFaceData> queue;
    /// Doorbell suppression flag for this shard.
    std::atomic<bool> scheduled{false};
  };

  /// Outgoing queues and comm-thread-local scheduling state for one destination rank.
  struct DestinationQueue
  {
    /// Destination rank.
    int dest_rank = 0;
    /// Outgoing shards, one per producer.
    std::vector<std::unique_ptr<OutgoingShard>> producer_shards;
    /// Producer IDs currently scheduled for draining by the communication thread.
    std::vector<std::size_t> active_producers;
    /// Comm-thread-local membership flags for `active_producers`.
    std::vector<std::uint8_t> producer_active_local;
    /// Round-robin cursor over `active_producers`.
    std::size_t rr_cursor = 0;
  };

  /// Worker-local destination activation queue.
  using DoorbellQueue = LockFreeSPSCSlotQueue<std::size_t>;

  /// One in-flight nonblocking MPI send and its owned serialized bytes.
  struct InFlightSend
  {
    /// Nonblocking MPI request.
    mpi::Request request;
    /// Owned serialized payload storage.
    ByteArray data;
  };

  /// Run the communication-thread progress loop.
  void CommThreadLoop();
  /// Allocate or resize outgoing shards and doorbell queues for the current worker count.
  void ConfigureProducerShards(std::size_t num_producers);
  /// Drain worker-local doorbell queues into comm-thread-local active lists.
  bool DrainProducerDoorbells();
  /// Drain all currently active destination queues.
  bool FlushActiveDestinations();
  /// Drain one active destination queue in round-robin producer order.
  bool FlushActiveDestination(std::size_t destination_queue_index);
  /// Drain outgoing queues, serialize batches, and post MPI sends.
  bool SerializeAndSend();
  /// Probe for incoming MPI messages, deserialize them, and publish mailbox batches.
  bool ProbeAndReceive();
  /// Retire completed nonblocking sends.
  bool PollInFlightSends();
  /// Report whether all angle sets are complete and no local outgoing work remains.
  bool AllAngleSetsComplete() const;

  /// Communicator set used for all CBCD point-to-point exchanges.
  const MPICommunicatorSet& comm_set_;
  /// Number of managed angle sets.
  std::size_t num_angle_sets_;
  /// Capacity summary for each managed angle set.
  std::vector<AngleSetCapacity> capacities_;
  /// Number of worker threads publishing into the communicator.
  std::size_t num_producers_ = 0;
  /// MPI tag shared by all communicator messages in this instance.
  int mpi_tag_;
  /// Maximum serialized MPI payload size.
  std::size_t max_message_bytes_;
  /// Local MPI rank.
  int my_rank_ = 0;
  /// Source partitions that can send to this rank.
  std::vector<int> source_partitions_;
  /// Source ranks mapped into the local communicator for receives.
  std::vector<int> source_ranks_;
  /// Source-partition to source-slot map grouped by angle set.
  std::vector<std::unordered_map<int, std::uint32_t>> source_partition_to_slot_by_angle_set_;
  /// Ordered outgoing destination-rank table.
  std::vector<int> destination_ranks_;
  /// Outgoing destination queues.
  std::vector<DestinationQueue> outgoing_queues_;
  /// Destination-rank to outgoing-queue index map.
  std::unordered_map<int, std::size_t> dest_to_queue_index_;
  /// One worker-local destination doorbell queue per producer.
  std::vector<std::unique_ptr<DoorbellQueue>> producer_doorbells_;
  /// Active destination queue indices owned by the communication thread.
  std::vector<std::size_t> active_destinations_;
  /// Comm-thread-local membership flags for `active_destinations_`.
  std::vector<std::uint8_t> destination_active_local_;
  /// Per-angle-set incoming mailboxes.
  std::vector<std::unique_ptr<LockFreeSPSCSlotQueue<IncomingFaceBatch>>> incoming_mailboxes_;
  /// Per-angle-set transient send batches assembled by the communication thread.
  std::vector<std::vector<const OutgoingFaceData*>> send_batch_by_angle_set_;
  /// Reusable receive buffer for one incoming MPI payload.
  ByteArray recv_buffer_;
  /// Outstanding nonblocking sends owned by the communication thread.
  std::vector<InFlightSend> in_flight_sends_;
  /// Termination flag for the communication thread.
  std::atomic<bool> stop_requested_{false};
  /// Per-angle-set local completion flags.
  std::vector<std::atomic<bool>> angle_set_done_;
  /// Dedicated communication thread.
  std::thread comm_thread_;
  /// Scratch vector used while gathering ready outgoing queue slots.
  std::vector<LockFreeSPSCSlotQueue<OutgoingFaceData>::Slot*> slot_cache_;
};

} // namespace opensn

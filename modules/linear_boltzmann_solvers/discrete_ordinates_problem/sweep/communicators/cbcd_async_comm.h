// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <array>
#include <atomic>
#include <cassert>
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

/**
 * Kind of one wire-format section in a CBCD aggregated message.
 *
 * The aggregated wire format carries a sequence of sections, each tagged with a kind byte.
 * `NORMAL_FACE_PSI` sections feed the normal incoming non-local bank and decrement the
 * receiving angle set's task dependencies.  `DELAYED_FACE_PSI` sections populate the
 * lagged incoming non-local "new" bank without touching dependency counters.
 * `DELAYED_COMPLETION` sections carry no face entries and only signal that the sending
 * angle set has finished publishing its delayed outgoing data.
 */
enum class CBCDMessageKind : std::uint8_t
{
  NORMAL_FACE_PSI = 0,
  DELAYED_FACE_PSI = 1,
  DELAYED_COMPLETION = 2,
};

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
  /// Wire-format section kind that produced this batch.
  CBCDMessageKind kind = CBCDMessageKind::NORMAL_FACE_PSI;
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
  /// Stream kind for this payload.
  CBCDMessageKind kind = CBCDMessageKind::NORMAL_FACE_PSI;
  /// Producing angle-set ID.
  std::size_t angle_set_id = 0;
  /// Receiver-local face index understood by the destination rank.
  std::uint32_t remote_face_index = 0;
  /// Stable mapped-host pointer to one receiver-node-ordered face payload.
  const double* psi_data = nullptr;
  /// Number of doubles in `psi_data`.
  std::size_t data_size = 0;
};

/// Outgoing-face capacity contributed by one angle set to one destination.
struct OutgoingDestinationCapacity
{
  /// Destination rank.
  int dest_rank = -1;
  /// Number of outgoing queue records sent by this angle set to `dest_rank`.
  std::size_t face_count = 0;
};

/// Queue-capacity summary for one angle set.
struct AngleSetCapacity
{
  /// Number of outgoing non-local faces produced by this angle set.
  std::size_t outgoing_faces = 0;
  /// Number of incoming non-local faces consumed by this angle set.
  std::size_t incoming_faces = 0;
  /// Maximum number of face entries in one received batch.
  std::size_t max_incoming_batch_entries = 0;
  /// Maximum number of doubles in one received batch.
  std::size_t max_incoming_batch_values = 0;
  /// Outgoing queue-record counts grouped by destination rank.
  std::vector<OutgoingDestinationCapacity> outgoing_faces_by_destination;
};

/**
 * Aggregated CBCD communicator with one dedicated progress thread.
 *
 * Sweep worker threads publish outgoing non-local face payloads into producer-sharded
 * per-destination SPSC queues. Each worker owns one producer ID and one doorbell queue
 * used to notify the communication thread when a previously idle shard becomes active.
 * The communication thread drains those doorbells, batches payloads by message kind and
 * angle set subject to the configured message-size limit, serializes them into MPI messages,
 * and posts nonblocking sends. The communication thread also probes for incoming messages,
 * deserializes them into compact `IncomingFaceBatch` payloads, and publishes those batches
 * into per-angle-set incoming mailboxes.
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
 *    the destination rank and rings its worker-local doorbell when the shard becomes active.
 * 2. The communication thread drains doorbells, gathers ready shard slots, groups them by
 *    message kind and angle set, and serializes one or more MPI messages subject to the
 *    configured byte limit.
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
   * \param incoming_source_partitions Normal-incoming source partitions grouped by angle set.
   * \param delayed_incoming_source_partitions Delayed-incoming source partitions grouped by
   * angle set.  Empty for acyclic SPDSes.
   * \param max_message_bytes Maximum serialized MPI payload size. A value of zero disables
   * message-size splitting.
   * \param capacities Queue-capacity summary for each angle set.
   */
  CBCD_AsynchronousCommunicator(
    const std::vector<AngleSet*>& angle_sets,
    const MPICommunicatorSet& comm_set,
    const std::vector<std::vector<int>>& incoming_source_partitions,
    const std::vector<std::vector<int>>& delayed_incoming_source_partitions,
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
   * \param psi_data Stable mapped-host face payload. The storage must remain valid and
   * unmodified until the current sweep's communicator has stopped.
   * \param kind Wire-format section kind: `NORMAL_FACE_PSI` (default) for normal traffic
   * unlocking task dependencies on the receiver, `DELAYED_FACE_PSI` for lagged traffic
   * that populates the receiver's delayed-incoming bank.
   */
  void EnqueueOutgoing(int dest_rank,
                       std::size_t producer_id,
                       std::size_t angle_set_id,
                       std::uint32_t remote_face_index,
                       std::size_t data_size,
                       const double* psi_data,
                       CBCDMessageKind kind = CBCDMessageKind::NORMAL_FACE_PSI)
  {
    const auto it = dest_to_queue_index_.find(dest_rank);
    assert(it != dest_to_queue_index_.end());
    assert(producer_id < num_producers_);
    assert(data_size == 0 or psi_data != nullptr);
    auto& shard = *outgoing_queues_[it->second].producer_shards[producer_id];
    auto& slot = shard.queue.ReserveSlot();
    slot.payload.kind = kind;
    slot.payload.angle_set_id = angle_set_id;
    slot.payload.remote_face_index = remote_face_index;
    slot.payload.psi_data = psi_data;
    slot.payload.data_size = data_size;
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
   * Publish a delayed-completion marker for one angle set toward one destination rank.
   *
   * Completion markers carry no face entries; they only signal to the receiver that the
   * sending rank has finished publishing every delayed outgoing payload for the given
   * angle set.  The receiver uses these markers to decide when its lagged incoming bank
   * can be promoted from `new` to `old`.
   */
  void EnqueueDelayedCompletion(int dest_rank, std::size_t producer_id, std::size_t angle_set_id);

  /// Report whether every expected delayed-completion marker for one angle set has arrived.
  bool AreDelayedReceivesComplete(std::size_t angle_set_id) const noexcept;

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
  void SignalAngleSetComplete(std::size_t angle_set_id, std::size_t producer_id);
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

  /// Key identifying one nonempty `(message kind, angle set)` send section.
  struct SendSectionKey
  {
    /// Message-kind array index.
    std::uint8_t kind_index = 0;
    /// Producing angle-set ID.
    std::size_t angle_set_id = 0;
  };

  /// Queue slots retained until every pointer into them has been serialized.
  struct PendingQueueRelease
  {
    /// Queue whose consumer slots remain owned by the communication thread.
    LockFreeSPSCSlotQueue<OutgoingFaceData>* queue = nullptr;
    /// Number of consecutive ready slots to release.
    std::size_t count = 0;
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
  /// Local-communicator source rank to global partition map.
  std::unordered_map<int, int> source_partition_by_rank_;
  /// Source-partition to source-slot map grouped by angle set (normal traffic).
  std::vector<std::unordered_map<int, std::uint32_t>> source_partition_to_slot_by_angle_set_;
  /// Source-partition to source-slot map grouped by angle set (delayed traffic).
  std::vector<std::unordered_map<int, std::uint32_t>>
    delayed_source_partition_to_slot_by_angle_set_;
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
  /// Transient send batches assembled by the communication thread, indexed first by
  /// `CBCDMessageKind` (cast to its underlying integer) and then by angle-set id.
  std::array<std::vector<std::vector<const OutgoingFaceData*>>, 3>
    send_batch_by_kind_and_angle_set_;
  /// Nonempty send sections in first-activation order.
  std::vector<SendSectionKey> active_send_sections_;
  /// Reusable receive buffer for one incoming MPI payload.
  ByteArray recv_buffer_;
  /// Outstanding nonblocking sends owned by the communication thread.
  std::vector<InFlightSend> in_flight_sends_;
  /// Completed aggregate buffers retained for capacity-preserving reuse.
  std::vector<ByteArray> available_send_buffers_;
  /// Termination flag for the communication thread.
  std::atomic<bool> stop_requested_{false};
  /// Per-angle-set local completion flags.
  std::vector<std::atomic<bool>> angle_set_done_;
  /// Number of delayed source slots still incomplete for each angle set.
  std::vector<std::atomic<std::uint32_t>> delayed_sources_remaining_by_angle_set_;
  /// Delayed source partitions expected to send completion markers for each angle set.
  std::vector<std::vector<int>> delayed_source_partitions_by_angle_set_;
  /// Delayed destination partitions to send completion markers to for each angle set.
  std::vector<std::vector<int>> delayed_destination_partitions_by_angle_set_;
  /// Communication-thread-local delayed-completion flags by angle set and source slot.
  std::vector<std::vector<std::uint8_t>> delayed_completion_received_by_angle_set_;
  /// Dedicated communication thread.
  std::thread comm_thread_;
  /// Scratch vector used while gathering ready outgoing queue slots.
  std::vector<LockFreeSPSCSlotQueue<OutgoingFaceData>::Slot*> slot_cache_;
  /// Scratch release list that preserves queue-slot ownership through serialization.
  std::vector<PendingQueueRelease> pending_queue_releases_;
};

} // namespace opensn

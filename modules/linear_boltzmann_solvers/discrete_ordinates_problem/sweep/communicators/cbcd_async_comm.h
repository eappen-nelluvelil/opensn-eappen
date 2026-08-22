// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <unordered_map>
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
 * receiving angle set's task dependencies. `DELAYED_FACE_PSI` sections populate the
 * lagged incoming non-local "new" bank without touching dependency counters.
 */
enum class CBCDMessageKind : std::uint8_t
{
  NORMAL_FACE_PSI = 0,
  DELAYED_FACE_PSI = 1,
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
  /// Destination partition rank.
  int dest_rank = 0;
  /// Stream kind for this payload.
  CBCDMessageKind kind = CBCDMessageKind::NORMAL_FACE_PSI;
  /// Producing angle-set ID.
  std::size_t angle_set_id = 0;
  /// Receiver-local face index understood by the destination rank.
  std::uint32_t remote_face_index = 0;
  /// Packed outgoing doubles for one non-local face.
  std::vector<double> psi_data;
};

/// Queue-capacity summary for one angle set.
struct AngleSetCapacity
{
  /// Number of outgoing non-local faces produced by this angle set.
  std::size_t outgoing_faces = 0;
  /// Number of incoming non-local faces consumed by this angle set.
  std::size_t incoming_faces = 0;
  /// Expected normal face records by normal-source slot.
  std::vector<std::size_t> incoming_faces_by_source;
  /// Number of delayed face records expected during each sweep.
  std::size_t delayed_incoming_faces = 0;
  /// Expected delayed face records by delayed-source slot.
  std::vector<std::size_t> delayed_incoming_faces_by_source;
  /// Maximum number of doubles in one outgoing face payload.
  std::size_t max_outgoing_face_values = 0;
};

/**
 * Aggregated CBCD communicator with one dedicated progress thread.
 *
 * Each angle-set worker publishes completed device batches through its own SPSC queue. The
 * communication thread drains only committed batches, serializes packets subject to the configured
 * message-size limit, and maintains a finite nonblocking-send window for every destination. The
 * communication thread also probes for incoming messages, deserializes them into compact
 * `IncomingFaceBatch` payloads, and publishes those batches into per-angle-set incoming mailboxes.
 *
 * The aggregated communicator assumes the following communication patterns and sweep worker
 * thread interactions:
 * - sweep worker threads only write outgoing queue slots,
 * - the communication thread handles only the draining of outgoing queues and routing of
 *   incoming batches to angle-set mailboxes,
 * - each angle-set owner thread only drains its own incoming mailbox.
 *
 * Aggregated communicator flow:
 * 1. A sweep worker fills all non-local face payloads from one completed device batch and commits
 *    the angle-set queue once.
 * 2. The communication thread gathers committed slots and serializes one or more MPI messages
 *    subject to the configured byte and per-destination in-flight limits.
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
   * \param angle_set_id Producing angle-set ID.
   * \param remote_face_index Receiver-local face index.
   * \param data_size Number of doubles in the payload.
   * \param fill Callback that fills the reserved payload buffer.
   * \param kind Wire-format section kind: `NORMAL_FACE_PSI` (default) for normal traffic
   * unlocking task dependencies on the receiver, `DELAYED_FACE_PSI` for lagged traffic
   * that populates the receiver's delayed-incoming bank.
   */
  template <typename FillCallback>
  void EnqueueOutgoing(int dest_rank,
                       std::size_t angle_set_id,
                       std::uint32_t remote_face_index,
                       std::size_t data_size,
                       FillCallback&& fill,
                       CBCDMessageKind kind = CBCDMessageKind::NORMAL_FACE_PSI)
  {
    assert(angle_set_id < outgoing_queues_.size());
    assert(dest_to_state_index_.contains(dest_rank));
    auto& queue = *outgoing_queues_[angle_set_id]->queue;
    auto& slot = queue.ReserveSlot();
    slot.payload.dest_rank = dest_rank;
    slot.payload.kind = kind;
    slot.payload.angle_set_id = angle_set_id;
    slot.payload.remote_face_index = remote_face_index;
    slot.payload.psi_data.resize(data_size);
    fill(slot.payload.psi_data.data());
  }

  /// Atomically expose every outgoing record produced by the latest completed device batch.
  void CommitOutgoingBatch(std::size_t angle_set_id);

  /// Report whether every expected delayed face record for one angle set has arrived.
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
    return incoming_mailboxes_[angle_set_id]->ProcessReady(
             [&callback](IncomingFaceBatch& batch)
             {
               callback(batch);
               // A mailbox has one logical slot per possible face record to guarantee
               // progress even when an angle set cannot initialize yet. Do not let each
               // of those slots permanently retain a packet-sized allocation across
               // iterations; return consumed storage to the allocator for immediate reuse.
               std::vector<IncomingFaceBatchEntry>().swap(batch.entries);
               std::vector<double>().swap(batch.psi_data);
             }) > 0;
  }

  /// Report whether the specified angle set currently has a published incoming batch.
  bool HasIncoming(std::size_t angle_set_id) const
  {
    assert(angle_set_id < num_angle_sets_);
    return not incoming_mailboxes_[angle_set_id]->Empty();
  }

  /// Mark one angle set as locally complete.
  void SignalAngleSetComplete(std::size_t angle_set_id);
  /// Start the communication thread.
  void Start();
  /// Request termination and join the communication thread.
  void Stop();

private:
  /// Outgoing queue written by exactly one angle-set owner.
  struct ProducerQueue
  {
    std::unique_ptr<CommittedSPSCQueue<OutgoingFaceData>> queue;
  };

  /// Communication state for one destination partition.
  struct DestinationState
  {
    int dest_rank = 0;
    std::size_t in_flight_sends = 0;
  };

  /// One in-flight nonblocking MPI send and its owned serialized bytes.
  struct InFlightSend
  {
    mpi::Request request;
    ByteArray data;
    std::size_t destination_index = 0;
  };

  /// One message under construction for a destination and a single angle set.
  struct SendBuilder
  {
    std::array<std::vector<const OutgoingFaceData*>, 2> entries_by_kind;
    std::size_t payload_bytes = sizeof(std::size_t);

    bool Empty() const noexcept
    {
      return entries_by_kind[0].empty() and entries_by_kind[1].empty();
    }

    void Clear()
    {
      entries_by_kind[0].clear();
      entries_by_kind[1].clear();
      payload_bytes = sizeof(std::size_t);
    }
  };

  /// Run the communication-thread progress loop.
  void CommThreadLoop();
  /// Drain outgoing queues, serialize batches, and post MPI sends.
  bool SerializeAndSend();
  /// Serialize and post one completed message builder.
  void
  PostSendBuilder(std::size_t destination_index, std::size_t angle_set_id, SendBuilder& builder);
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
  /// MPI tag shared by all communicator messages in this instance.
  int mpi_tag_;
  /// Maximum serialized MPI payload size.
  std::size_t max_message_bytes_;
  /// Local MPI rank.
  int my_rank_ = 0;
  /// Local-communicator source rank to global partition map.
  std::unordered_map<int, int> source_partition_by_rank_;
  /// Source-partition to source-slot map grouped by angle set (normal traffic).
  std::vector<std::unordered_map<int, std::uint32_t>> source_partition_to_slot_by_angle_set_;
  /// Source-partition to source-slot map grouped by angle set (delayed traffic).
  std::vector<std::unordered_map<int, std::uint32_t>>
    delayed_source_partition_to_slot_by_angle_set_;
  /// Per-angle-set outgoing queues. Static scheduling gives each queue one producer.
  std::vector<std::unique_ptr<ProducerQueue>> outgoing_queues_;
  /// Communication state for every possible destination partition.
  std::vector<DestinationState> destination_states_;
  /// Destination-rank to communication-state index map.
  std::unordered_map<int, std::size_t> dest_to_state_index_;
  /// Reusable message builders indexed by destination state.
  std::vector<SendBuilder> send_builders_;
  /// Nonempty builder indices in the current producer-queue scan.
  std::vector<std::size_t> active_builder_destinations_;
  /// Membership flags for `active_builder_destinations_`.
  std::vector<std::uint8_t> builder_is_active_;
  /// Per-angle-set incoming mailboxes.
  std::vector<std::unique_ptr<LockFreeRingBuffer<IncomingFaceBatch>>> incoming_mailboxes_;
  /// Reusable receive buffer for one incoming MPI payload.
  ByteArray recv_buffer_;
  /// Outstanding nonblocking sends owned by the communication thread.
  std::vector<InFlightSend> in_flight_sends_;
  /// Termination flag for the communication thread.
  std::atomic<bool> stop_requested_{false};
  /// Per-angle-set local completion flags.
  std::vector<std::atomic<bool>> angle_set_done_;
  /// Number of delayed face records expected for each angle set at sweep start.
  std::vector<std::size_t> delayed_faces_expected_by_angle_set_;
  /// Number of delayed face records still expected for each angle set.
  std::vector<std::atomic<std::size_t>> delayed_faces_remaining_by_angle_set_;
  /// Expected delayed records by `(angle set, delayed source slot)`.
  std::vector<std::vector<std::size_t>> delayed_faces_expected_by_source_and_angle_set_;
  /// Communication-thread-local delayed records remaining by source and angle set.
  std::vector<std::vector<std::size_t>> delayed_faces_remaining_by_source_and_angle_set_;
  /// Per-sweep duplicate detector indexed by `(angle set, normal source slot, face index)`.
  std::vector<std::vector<std::vector<std::uint8_t>>> normal_face_seen_by_source_and_angle_set_;
  /// Per-sweep duplicate detector indexed by `(angle set, delayed source slot, face index)`.
  std::vector<std::vector<std::vector<std::uint8_t>>> delayed_face_seen_by_source_and_angle_set_;
  /// Dedicated communication thread.
  std::thread comm_thread_;
  /// Next producer queue serviced by the communication thread.
  std::size_t next_outgoing_queue_ = 0;
  /// Scratch vector used while gathering one committed outgoing prefix.
  std::vector<CommittedSPSCQueue<OutgoingFaceData>::Slot*> slot_cache_;
};

} // namespace opensn

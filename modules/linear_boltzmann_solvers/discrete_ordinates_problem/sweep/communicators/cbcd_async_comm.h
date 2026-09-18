// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_peer_transport.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_receive_packet.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/spmd_threadpool.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class MPICommunicatorSet;

/// One outgoing nonlocal face published by a sweep worker.
struct OutgoingFaceRecord
{
  /// Owning angle-set identifier.
  std::size_t angle_set_id = 0;
  /// Face index assigned by the destination for this source rank.
  std::uint32_t destination_face_index = 0;
  /// Completed nonlocal face psi in destination-node order, valid until serialization.
  const double* psi_values = nullptr;
  /// Number of values referenced by psi_values.
  std::size_t num_psi_values = 0;
};

/// Exact face counts and serialized record sizes for one angle set and peer.
struct PeerCommunicationBounds
{
  int rank = -1;
  std::size_t num_faces = 0;
  std::size_t num_bytes = 0;
  std::size_t largest_record_bytes = 0;
};

/// Precomputed storage bounds and exact outgoing queue counts for one angle set.
struct AngleSetCommunicationBounds
{
  /// Safe mailbox capacity: at most one received batch per incoming face.
  std::size_t incoming_mailbox_capacity = 0;
  /// Exact queue bounds for each destination reached by this angle set.
  std::vector<PeerCommunicationBounds> outgoing_queue_bounds;
  std::vector<PeerCommunicationBounds> incoming_queue_bounds;
};

/** Aggregated CBCD communicator with per-worker SPSC queues and one MPI progress thread. */
class CBCD_AsynchronousCommunicator
{
public:
  /**
   * Construct preallocated mailboxes and deterministic peer mappings.
   *
   * \param angle_sets Angle sets served by this communicator.
   * \param comm_set Partition communicator mapping.
   * \param incoming_source_partitions Source partitions for each angle set.
   * \param bounds Per-angle-set storage bounds and outgoing queue counts.
   */
  CBCD_AsynchronousCommunicator(const std::vector<AngleSet*>& angle_sets,
                                const MPICommunicatorSet& comm_set,
                                const std::vector<std::vector<int>>& incoming_source_partitions,
                                const std::vector<AngleSetCommunicationBounds>& bounds);

  ~CBCD_AsynchronousCommunicator();

  /** Publish one outgoing face through the calling worker's SPSC queue. */
  void EnqueueOutgoing(int destination_rank,
                       std::size_t worker_id,
                       std::size_t angle_set_id,
                       std::uint32_t destination_face_index,
                       const double* psi_values,
                       std::size_t num_psi_values)
  {
    const auto destination = destination_to_channel_.find(destination_rank);
    assert(destination != destination_to_channel_.end());
    const auto channel = destination->second;
    auto& queue = *destination_channels_[channel].worker_queues[worker_id];
    auto& record = queue.ReserveSlot();
    record.angle_set_id = angle_set_id;
    record.destination_face_index = destination_face_index;
    record.psi_values = psi_values;
    record.num_psi_values = num_psi_values;
    queue.PublishSlot();
  }

  /** Process all received batches currently visible for one angle set. */
  template <typename Callback>
  bool ProcessIncoming(std::size_t angle_set_id, Callback callback)
  {
    return incoming_mailboxes_[angle_set_id]->ProcessReady(
             [&](const IncomingFaceBatch& batch)
             {
               callback(batch);
               batch.packet->readers.fetch_sub(1, std::memory_order_release);
             }) > 0;
  }

  /// Return whether one angle set has at least one received batch.
  bool HasIncoming(std::size_t angle_set_id) const
  {
    return not incoming_mailboxes_[angle_set_id]->Empty();
  }

  /// Mark an angle set locally complete after all of its faces have been published.
  void SignalAngleSetComplete(std::size_t angle_set_id);
  /// Allocate worker-owned queues and start a sweep on the reusable MPI progress thread.
  void Start(std::size_t num_workers);
  /// Drain published work and wait for the MPI progress thread to become idle.
  void Stop();

private:
  using OutgoingQueue = LockFreeSPSCSlotQueue<OutgoingFaceRecord>;

  struct DestinationChannel
  {
    /// Destination MPI rank.
    int destination_rank = 0;
    /// One SPSC queue per scheduler worker; empty queues require no storage.
    std::vector<std::unique_ptr<OutgoingQueue>> worker_queues;
    /// Workers owning at least one outgoing face toward this destination.
    std::vector<std::size_t> active_workers;
    std::size_t next_worker = 0;
  };

  void CommThreadLoop();
  void ConfigureWorkerQueues(std::size_t num_workers);
  bool FlushDestination(std::size_t destination_channel_index);
  bool FlushOutgoing();
  bool ProgressPackets();
  bool DispatchPacket(CBCDReceivePacket& packet, int num_bytes);
  bool AllAngleSetsComplete() const;

  /// Immutable communicator topology and per-angle-set bounds.
  const MPICommunicatorSet& comm_set_;
  std::size_t num_angle_sets_;
  std::vector<AngleSetCommunicationBounds> communication_bounds_;
  /// Worker count and MPI message parameters.
  std::size_t num_workers_ = 0;
  int mpi_tag_;
  int my_rank_ = 0;
  /// Unique receive peers in partition and communicator-rank coordinates.
  std::vector<int> source_partitions_;
  std::vector<std::size_t> source_face_counts_;
  std::vector<std::size_t> remaining_source_faces_;
  /// Per-angle-set map from source partition to compact source index.
  std::vector<std::unordered_map<int, std::uint32_t>> source_partition_to_index_by_angle_set_;
  /// Unique destinations and their compact communication channels.
  std::vector<int> destination_ranks_;
  std::vector<DestinationChannel> destination_channels_;
  std::unordered_map<int, std::size_t> destination_to_channel_;
  /// Progress-thread-to-worker SPSC mailboxes indexed by angle-set ID.
  std::vector<std::unique_ptr<LockFreeSPSCSlotQueue<IncomingFaceBatch>>> incoming_mailboxes_;
  /// Serialization scratch grouped by angle-set section.
  std::vector<std::vector<const OutgoingFaceRecord*>> pending_records_by_angle_set_;
  std::vector<std::size_t> active_angle_set_ids_;
  std::unique_ptr<CBCDPeerTransport> transport_;
  /// Progress-thread lifecycle and per-angle-set completion state.
  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_complete_;
  bool sweep_active_ = false;
  SPMD_ThreadPool comm_pool_;
  /// Queues released after serialization.
  std::vector<std::pair<OutgoingQueue*, std::size_t>> pending_slot_releases_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_receive_packet.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/spmd_threadpool.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class SweepCommunicator;

/// One outgoing nonlocal face published by a sweep worker.
struct OutgoingFaceRecord
{
  std::size_t angle_set_id = 0;
  std::uint32_t destination_face_index = 0;
  /// Completed nonlocal face psi in destination-node order, valid until serialization.
  const double* psi_values = nullptr;
  std::size_t num_psi_values = 0;
};

/// Exact outgoing queue bounds contributed by one angle set to one destination.
struct DestinationQueueBounds
{
  int destination_rank = -1;
  std::size_t num_faces = 0;
};

/// Precomputed storage bounds and exact outgoing queue counts for one angle set.
struct AngleSetCommunicationBounds
{
  /// Safe mailbox capacity: at most one received batch per incoming face.
  std::size_t incoming_mailbox_capacity = 0;
  /// Exact queue bounds for each destination reached by this angle set.
  std::vector<DestinationQueueBounds> outgoing_queue_bounds;
};

/** Aggregated CBCD communicator with per-worker SPSC queues and one MPI progress thread. */
class CBCD_AsynchronousCommunicator
{
public:
  /**
   * Construct preallocated mailboxes and deterministic peer mappings.
   *
   * \param angle_sets Angle sets served by this communicator.
   * \param sweep_communicator Private sweep communication context.
   * \param max_message_bytes Exact maximum aggregate message size.
   * \param bounds Per-angle-set storage bounds and outgoing queue counts.
   */
  CBCD_AsynchronousCommunicator(const std::vector<AngleSet*>& angle_sets,
                                const SweepCommunicator& sweep_communicator,
                                std::size_t max_message_bytes,
                                const std::vector<AngleSetCommunicationBounds>& bounds);

  ~CBCD_AsynchronousCommunicator();

  void EnqueueOutgoing(std::size_t destination_index,
                       std::size_t worker_id,
                       std::size_t angle_set_id,
                       std::uint32_t destination_face_index,
                       const double* psi_values,
                       std::size_t num_psi_values)
  {
    assert(angle_set_id < destination_channels_by_angle_set_.size());
    const auto& channels = destination_channels_by_angle_set_[angle_set_id];
    assert(destination_index < channels.size());
    const auto channel = channels[destination_index];
    auto& queue = *destination_channels_[channel].worker_queues[worker_id];
    auto& record = queue.ReserveSlot();
    record.angle_set_id = angle_set_id;
    record.destination_face_index = destination_face_index;
    record.psi_values = psi_values;
    record.num_psi_values = num_psi_values;
    queue.PublishSlot();
  }

  template <typename Callback>
  bool ProcessIncoming(std::size_t angle_set_id, Callback callback)
  {
    return incoming_mailboxes_[angle_set_id]->ProcessReady(
             [&](const IncomingFaceBatch& batch)
             {
               callback(batch);
               receive_packets_->Release(batch.packet);
             }) > 0;
  }

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
    int destination_rank = 0;
    int mapped_rank = 0;
    /// One SPSC queue per scheduler worker; empty queues require no storage.
    std::vector<std::unique_ptr<OutgoingQueue>> worker_queues;
    /// Workers owning at least one outgoing face toward this destination.
    std::vector<std::size_t> active_workers;
    ByteArray reusable_packet;
  };

  struct InFlightSend
  {
    mpi::Request request;
    ByteArray data;
    std::size_t destination_channel_index = 0;
  };

  void CommThreadLoop();
  void ConfigureWorkerQueues(std::size_t num_workers);
  bool FlushDestination(std::size_t destination_channel_index);
  bool FlushOutgoing();
  bool ProbeAndReceive();
  bool PollInFlightSends();
  bool AllAngleSetsComplete() const;

  const SweepCommunicator& sweep_communicator_;
  std::size_t num_angle_sets_;
  std::vector<AngleSetCommunicationBounds> communication_bounds_;
  std::size_t num_workers_ = 0;
  int mpi_tag_;
  std::size_t message_limit_ = 0;
  /// Communicator ranks for the unique receive peers.
  std::vector<int> source_ranks_;
  std::vector<std::size_t> source_face_counts_;
  std::vector<std::size_t> remaining_source_faces_;
  /// Per-angle-set source indices indexed by this communicator's receive peer.
  std::vector<std::vector<std::uint32_t>> source_indices_by_angle_set_;
  std::vector<int> destination_ranks_;
  std::vector<DestinationChannel> destination_channels_;
  std::vector<std::vector<std::size_t>> destination_channels_by_angle_set_;
  /// Progress-thread-to-worker SPSC mailboxes indexed by angle-set ID.
  std::vector<std::unique_ptr<LockFreeSPSCSlotQueue<IncomingFaceBatch>>> incoming_mailboxes_;
  std::vector<std::vector<const OutgoingFaceRecord*>> pending_records_by_angle_set_;
  std::vector<std::size_t> active_angle_set_ids_;
  /// Packets shared by receive mailboxes, reclaimed only after worker-side placement.
  std::unique_ptr<CBCDReceivePacketPool> receive_packets_;
  /// Sends whose buffers remain MPI-owned.
  std::vector<InFlightSend> in_flight_sends_;
  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_complete_;
  bool sweep_active_ = false;
  SPMD_ThreadPool comm_pool_;
  /// Queues released after serialization.
  std::vector<std::pair<OutgoingQueue*, std::size_t>> pending_slot_releases_;
};

} // namespace opensn

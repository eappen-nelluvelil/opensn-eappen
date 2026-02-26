// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class MPICommunicatorSet;

/// An entry received from a remote location, destined for a specific angle set.
struct IncomingEntry
{
  uint64_t cell_global_id;
  unsigned int face_id;
  std::vector<double> psi_data;
};

/// Aggregated MPI communicator for threaded CBCD sweep.
///
/// A dedicated communication thread aggregates MPI sends/receives across all angle sets,
/// reducing the number of MPI calls by ~N× and eliminating MPI contention between worker threads.
class CBCD_AggregatedCommunicator
{
public:
  CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                               const MPICommunicatorSet& comm_set);

  ~CBCD_AggregatedCommunicator();

  // --- Worker thread interface ---

  /// Push outgoing psi for a cell-face to the per-destination queue.
  void EnqueueOutgoing(int dest_location,
                       size_t angle_set_id,
                       uint64_t cell_global_id,
                       unsigned int face_id,
                       std::vector<double>&& psi_data);

  /// Pull all received data for this angle set (swaps mailbox contents out).
  std::vector<IncomingEntry> DequeueIncoming(size_t angle_set_id);

  /// Signal that this angle set has no more outgoing data.
  void SignalAngleSetComplete(size_t angle_set_id);

  // --- Lifecycle (called by CBCDSweepChunk / scheduler) ---

  /// Launch the dedicated communication thread.
  void Start();

  /// Flush remaining sends, wait for completion, and join the communication thread.
  void Stop();

private:
  /// Main loop of the communication thread.
  void CommThreadLoop();

  /// Aggregate per-destination outgoing entries, serialize, and isend.
  void FlushOutgoing();

  /// Probe all dependencies, recv, deserialize, and dispatch to mailboxes.
  void ProbeAndReceive();

  /// Test pending isend completion and clean up completed sends.
  void PollPendingSends();

  /// Check if all angle sets are done and all outgoing queues are empty.
  bool AllWorkComplete() const;

  // --- Internal data structures ---

  struct OutgoingEntry
  {
    size_t angle_set_id;
    uint64_t cell_global_id;
    unsigned int face_id;
    std::vector<double> psi_data;
  };

  struct PerDestQueue
  {
    std::mutex mutex;
    std::vector<OutgoingEntry> entries;
  };

  struct PerAngleSetMailbox
  {
    std::mutex mutex;
    std::vector<IncomingEntry> entries;
  };

  struct PendingSend
  {
    mpi::Request request;
    ByteArray data;
  };

  // --- Data members ---

  const MPICommunicatorSet& comm_set_;
  size_t num_angle_sets_;

  /// One queue per destination location for outgoing data.
  std::map<int, PerDestQueue> outgoing_queues_;

  /// One mailbox per angle set for incoming data.
  std::vector<PerAngleSetMailbox> incoming_mailboxes_;

  /// Union of all location dependencies across all SPDS.
  std::set<int> all_location_dependencies_;

  /// Union of all location successors across all SPDS.
  std::set<int> all_location_successors_;

  /// MPI tag used by the aggregated communicator (set to num_angle_sets to avoid collisions).
  int aggregated_tag_;

  /// Tracks in-flight MPI_Isends.
  std::vector<PendingSend> pending_sends_;

  /// Shutdown signal for the communication thread.
  std::atomic<bool> stop_requested_{false};

  /// Per-angle-set completion flags.
  std::vector<std::atomic<bool>> angle_set_done_;

  /// The dedicated communication thread.
  std::thread comm_thread_;

  /// Flush interval for the communication thread.
  static constexpr size_t kFlushSizeThreshold = 1;
  static constexpr auto kFlushTimeInterval = std::chrono::microseconds(100);

  /// Timestamp of last flush.
  std::chrono::steady_clock::time_point last_flush_time_;
};

} // namespace opensn

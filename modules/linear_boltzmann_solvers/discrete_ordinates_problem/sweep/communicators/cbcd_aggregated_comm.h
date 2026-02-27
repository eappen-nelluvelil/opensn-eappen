// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
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
/// All inter-thread communication uses lock-free atomic operations (no mutexes).
class CBCD_AggregatedCommunicator
{
public:
  CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                               const MPICommunicatorSet& comm_set);

  ~CBCD_AggregatedCommunicator();

  // --- Worker thread interface ---

  /// Push outgoing psi for a cell-face to the per-destination queue (lock-free CAS).
  void EnqueueOutgoing(int dest_location,
                       size_t angle_set_id,
                       uint64_t cell_global_id,
                       unsigned int face_id,
                       std::vector<double>&& psi_data);

  /// Pull all received data for this angle set (lock-free atomic exchange).
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

  /// Drain per-destination outgoing queues, serialize, and isend.
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

  /// Lock-free singly-linked node for outgoing data (Treiber stack).
  struct OutgoingNode
  {
    OutgoingEntry entry;
    OutgoingNode* next;
  };

  /// Lock-free singly-linked node for incoming data batches (Treiber stack).
  struct IncomingNode
  {
    std::vector<IncomingEntry> entries;
    IncomingNode* next;
  };

  /// Lock-free per-destination outgoing queue.
  /// Workers push via CAS, comm thread drains via atomic exchange.
  struct PerDestQueue
  {
    alignas(64) std::atomic<OutgoingNode*> head{nullptr};
  };

  /// Lock-free per-angle-set incoming mailbox.
  /// Comm thread pushes via CAS, worker drains via atomic exchange.
  struct PerAngleSetMailbox
  {
    alignas(64) std::atomic<IncomingNode*> head{nullptr};
  };

  struct PendingSend
  {
    mpi::Request request;
    ByteArray data;
  };

  // --- Data members ---

  const MPICommunicatorSet& comm_set_;
  size_t num_angle_sets_;

  /// One lock-free queue per destination location for outgoing data.
  std::map<int, PerDestQueue> outgoing_queues_;

  /// One lock-free mailbox per angle set for incoming data.
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
};

} // namespace opensn

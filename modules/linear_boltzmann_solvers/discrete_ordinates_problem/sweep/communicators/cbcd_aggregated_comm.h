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
/// reducing the number of MPI calls and eliminating MPI contention between worker threads.
/// All inter-thread data handoff uses lock-free Treiber stacks (CAS push, atomic-exchange drain).
class CBCD_AggregatedCommunicator
{
public:
  CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                              const MPICommunicatorSet& comm_set);

  ~CBCD_AggregatedCommunicator();

  // --- Worker thread interface ---

  /// Push outgoing psi for a cell-face to the per-destination queue (lock-free).
  void EnqueueOutgoing(int dest_location,
                       size_t angle_set_id,
                       uint64_t cell_global_id,
                       unsigned int face_id,
                       std::vector<double>&& psi_data);

  /// Pull all received data for this angle set (lock-free).
  std::vector<IncomingEntry> DequeueIncoming(size_t angle_set_id);

  /// Signal that this angle set has no more outgoing data.
  void SignalAngleSetComplete(size_t angle_set_id);

  // --- Lifecycle ---

  /// Launch the dedicated communication thread.
  void Start();

  /// Flush remaining sends, wait for completion, and join the communication thread.
  void Stop();

private:
  void CommThreadLoop();
  void FlushOutgoing();
  void ProbeAndReceive();
  void PollPendingSends();
  bool AllWorkComplete() const;

  // --- Internal types ---

  struct OutgoingEntry
  {
    size_t angle_set_id;
    uint64_t cell_global_id;
    unsigned int face_id;
    std::vector<double> psi_data;
  };

  /// Lock-free node for the per-destination Treiber stack.
  struct OutgoingNode
  {
    OutgoingEntry entry;
    OutgoingNode* next;
  };

  /// Lock-free node for the per-angle-set Treiber stack.
  struct IncomingNode
  {
    std::vector<IncomingEntry> entries;
    IncomingNode* next;
  };

  /// Lock-free per-destination outgoing queue (Treiber stack).
  /// Workers push via CAS; comm thread drains via atomic exchange.
  struct PerDestQueue
  {
    alignas(64) std::atomic<OutgoingNode*> head{nullptr};

    /// CAS push — safe for concurrent producers, never blocks.
    void Push(OutgoingEntry&& entry);

    /// Atomic-exchange drain — returns all queued entries, deletes nodes.
    std::vector<OutgoingEntry> Drain();

    /// Read-only empty check (single atomic load, no cache-line write).
    bool Empty() const;
  };

  /// Lock-free per-angle-set incoming mailbox (Treiber stack).
  /// Comm thread pushes via CAS; worker drains via atomic exchange.
  struct PerAngleSetMailbox
  {
    alignas(64) std::atomic<IncomingNode*> head{nullptr};

    /// CAS push — used by comm thread to deposit received data.
    void Push(std::vector<IncomingEntry>&& batch);

    /// Atomic-exchange drain — returns all received entries, deletes nodes.
    std::vector<IncomingEntry> Drain();
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

  /// Union of location dependencies / successors across all SPDS.
  std::set<int> all_location_dependencies_;
  std::set<int> all_location_successors_;

  /// MPI tag (set to num_angle_sets to avoid collisions with per-angle-set tags).
  int aggregated_tag_;

  /// In-flight MPI_Isends.
  std::vector<PendingSend> pending_sends_;

  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_done_;
  std::thread comm_thread_;
};

} // namespace opensn

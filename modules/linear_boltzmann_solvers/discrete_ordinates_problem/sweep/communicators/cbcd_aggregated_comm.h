// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <thread>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class MPICommunicatorSet;

/// A single face's angular flux data received from a remote MPI rank.
/// Deserialized from the aggregated wire format and routed to the target angle set.
struct IncomingFaceData
{
  uint64_t cell_global_id;
  unsigned int face_id;
  std::vector<double> psi_data;
};

/// A single face's angular flux data queued for sending to a remote MPI rank.
/// Produced by worker threads and consumed by the communication thread for serialization.
struct OutgoingFaceData
{
  size_t angle_set_id;
  std::uint64_t cell_global_id;
  unsigned int face_id;
  std::vector<double> psi_data;
};

/// Lock-free Treiber stack for batched inter-thread communication.
/// Multiple producers push batches via CAS; single consumer drains all via atomic exchange.
template <typename T>
class LockFreeTreiberStack
{
  struct Node
  {
    T payload;
    Node* next;
  };
  alignas(std::hardware_destructive_interference_size) std::atomic<Node*> head{nullptr};

public:
  ~LockFreeTreiberStack()
  {
    auto* chain = head.exchange(nullptr, std::memory_order_acquire);
    while (chain)
    {
      auto* next = chain->next;
      delete chain;
      chain = next;
    }
  }

  /// CAS push — safe for concurrent producers, never blocks.
  void Push(T&& payload)
  {
    auto* node = new Node{std::move(payload), nullptr};
    auto* expected = head.load(std::memory_order_relaxed);
    do
    {
      node->next = expected;
    } while (not head.compare_exchange_weak(
      expected, node, std::memory_order_release, std::memory_order_relaxed));
  }

  /// Atomic-exchange drain — returns all queued entries, deletes nodes.
  std::vector<T> Drain()
  {
    auto* chain = head.exchange(nullptr, std::memory_order_acquire);
    std::vector<T> result;
    while (chain)
    {
      result.push_back(std::move(chain->payload));
      auto* next = chain->next;
      delete chain;
      chain = next;
    }
    return result;
  }

  /// Read-only empty check (single atomic load, no cache-line write).
  bool Empty() const { return head.load(std::memory_order_acquire) == nullptr; }
};


/**
 * Aggregated MPI communicator for threaded CBCD sweep.
 *
 * A dedicated communication thread aggregates MPI sends/receives across all angle sets,
 * reducing the number of MPI calls and eliminating MPI contention between worker threads.
 *
 * Both outgoing and incoming paths use lock-free Treiber stacks. Outgoing: batched per-destination
 * pushes (one push per destination per kernel completion). Incoming: one stack per angle set,
 * drained by worker threads via atomic exchange.
 *
 * Wire format for aggregated MPI messages (one message per destination per flush):
 *   [num_active_angle_sets : size_t]
 *   For each active angle set:
 *     [angle_set_id : size_t]
 *     [num_entries  : size_t]
 *     For each entry (one per outgoing face):
 *       [cell_global_id : uint64_t]
 *       [face_id        : unsigned int]
 *       [data_size      : size_t]         // number of doubles
 *       [psi_data       : double[data_size]]
 */
class CBCD_AggregatedCommunicator
{
public:
  CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                              const MPICommunicatorSet& comm_set,
                              size_t max_message_bytes);

  ~CBCD_AggregatedCommunicator();

  /// Return the queue index for a destination location.
  /// Returns -1 if not found.
  /// Allows callers to pre-resolve the destination once and batch efficiently.
  int GetQueueIndex(int dest_location) const;

  /// Push a batch of outgoing entries by pre-resolved queue index.
  /// One push per destination per kernel completion — eliminates per-face atomics.
  void EnqueueOutgoingBatchByIndex(int queue_index, std::vector<OutgoingFaceData>&& batch);

  /// Drain all received batches for this angle set (lock-free).
  /// Returns the raw batches to avoid flattening overhead.
  std::vector<std::vector<IncomingFaceData>> DequeueIncoming(size_t angle_set_id);


  /// Signal that this angle set has no more outgoing data.
  void SignalAngleSetComplete(size_t angle_set_id);

  /// Launch the dedicated communication thread.
  void Start();

  /// Flush remaining sends, wait for completion, and join the communication thread.
  void Stop();

private:
  /// Per-destination outgoing Treiber stack for batched face data.
  struct NeighborQueue
  {
    int dest_location;
    std::unique_ptr<LockFreeTreiberStack<std::vector<OutgoingFaceData>>> queue;
  };

  /// An in-flight MPI_Isend and its serialized message data.
  struct InFlightSend
  {
    mpi::Request request;
    ByteArray data;
  };

  // -- Communication thread methods ------------------------------------------

  void CommThreadLoop();
  bool FlushOutgoing(std::vector<std::vector<const OutgoingFaceData*>>& by_angle_set);
  bool ProbeAndReceive();
  bool PollInFlightSends();
  bool AllWorkComplete() const;

  // -- Configuration (immutable after construction) --------------------------

  const MPICommunicatorSet& comm_set_;
  size_t num_angle_sets_;

  /// MPI tag used for all aggregated messages (set to num_angle_sets to avoid
  /// collisions with per-angle-set tags used by other communicator types).
  int mpi_tag_;

  /// MPI ranks from which this rank receives face data (union over all angle sets).
  std::vector<int> source_ranks_;

  // -- Outgoing path (worker threads → comm thread) --------------------------

  /// One Treiber stack per destination MPI rank for outgoing face data batches.
  std::vector<NeighborQueue> outgoing_queues_;

  /// Maps destination MPI rank → index into outgoing_queues_.
  std::unordered_map<int, int> dest_to_queue_index_;

  // -- Incoming path (comm thread → worker threads) --------------------------

  /// One Treiber stack per angle set for incoming face data batches.
  std::vector<LockFreeTreiberStack<std::vector<IncomingFaceData>>> incoming_mailboxes_;

  // -- Communication thread state --------------------------------------------

  /// Pre-allocated receive buffer for MPI messages.
  ByteArray recv_buffer_;

  /// In-flight MPI_Isends awaiting completion.
  std::vector<InFlightSend> in_flight_sends_;

  // -- Synchronization -------------------------------------------------------

  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_done_;
  std::thread comm_thread_;
};

} // namespace opensn

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

/// An entry received from a remote location, destined for a specific angle set.
struct IncomingEntry
{
  uint64_t cell_global_id;
  unsigned int face_id;
  std::vector<double> psi_data;
};

/// Lock-free Treiber stack for single-consumer, multiple-producer communication 
/// between worker threads and the comm thread.
template <typename T>
class LockFreeTreiberStack
{
  struct Node
  {
    T payload;
    Node* next;
  };
  // Align head to a separate cache line to prevent false sharing between 
  // producers and consumer.
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

  /// Compare-and-swap (CAS) push
  /// Safe for concurrent producers, never blocks.
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

  /// Atomic-exchange drain
  /// Returns all queued entries, deletes nodes.
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
 * All inter-thread data handoff uses lock-free Treiber stacks.
 */
class CBCD_AggregatedCommunicator
{
public:
  /// Struct for outgoing data entries
  struct OutgoingEntry
  {
    size_t angle_set_id;
    std::uint64_t cell_global_id;
    unsigned int face_id;
    std::vector<double> psi_data;
  };

  CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                              const MPICommunicatorSet& comm_set,
                              size_t max_single_message_size_in_bytes = 0);

  ~CBCD_AggregatedCommunicator();

  /// Push all received batches for this angle set (lock-free).
  /// Returns the raw batches to avoid flattening overhead.
  std::vector<std::vector<IncomingEntry>> DequeueIncoming(size_t angle_set_id);

  /// Return the queue index for a destination location.
  /// Returns -1 if not found.
  /// Allows callers to pre-resolve the destination once and batch efficiently.
  int GetQueueIndex(int dest_location) const; 

  /// Push a batch of outgoing entries by pre-resolved queue index.
  void EnqueueOutgoingBatchByIndex(int queue_index, std::vector<OutgoingEntry>&& batch);

  /// Signal that this angle set has no more outgoing data.
  void SignalAngleSetComplete(size_t angle_set_id);

  /// Launch the dedicated communication thread.
  void Start();

  /// Flush remaining sends, wait for completion, and join the communication thread.
  void Stop();

private:
  /// Struct for per-destination queues of outgoing batches.
  struct NeighborQueue
  {
    int dest_location;
    std::unique_ptr<LockFreeTreiberStack<std::vector<OutgoingEntry>>> queue;
  };

  /// Struct for tracking in-flight MPI sends initiated by the comm thread.
  struct PendingSend
  {
    mpi::Request request;
    ByteArray data;
  };

  /// Main loop of the communication thread: 
  // - flush outgoing queues,
  // - probe/receive incoming messages, and
  // - track pending sends.
  void CommThreadLoop();

  bool FlushOutgoing(std::vector<std::vector<const OutgoingEntry*>>& by_angle_set);
  void ProbeAndReceive();
  void PollPendingSends();
  bool AllWorkComplete() const;

  const MPICommunicatorSet& comm_set_;
  size_t num_angle_sets_;

  /// Flat array of lock-free queues for outgoing data batches.
  std::vector<NeighborQueue> outgoing_queues_;

  /// Lookup map from destination location to outgoing queue index.
  std::unordered_map<int, int> dest_to_queue_index_;

  /// One lock-free mailbox per angle set for incoming data batches.
  std::vector<LockFreeTreiberStack<std::vector<IncomingEntry>>> incoming_mailboxes_;

  /// Contiguous array of location dependencies for cache-friendly polling.
  std::vector<int> location_dependencies_;

  /// Pre-allocated ByteArray for incoming MPI messages.
  ByteArray persistent_recv_buffer_;

  /// MPI tag (set to num_angle_sets to avoid collisions with per-angle-set tags).
  int aggregated_tag_;

  /// In-flight MPI_Isends.
  std::vector<PendingSend> pending_sends_;

  /// Flags to track completion status of angle sets and overall stop request.
  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_done_;

  /// Dedicated communication thread.
  std::thread comm_thread_;
};

} // namespace opensn
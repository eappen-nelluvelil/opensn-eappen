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

// IncomingFaceData and OutgoingFaceData have been eliminated: worker threads now pack the wire format
// directly into ByteArray buffers (see CopyOutgoingPsiBackToHost), avoiding
// per-face std::vector<double> heap allocations on the hot path.

/// Lock-free Treiber stack for batched inter-thread communication.
/// Multiple producers push batches via CAS; single consumer drains all via atomic exchange.
///
/// Uses an internal atomic free-list to recycle nodes, eliminating per-push `new` and
/// per-drain `delete` after the first cycle. This significantly reduces heap contention
/// on the hot path when multiple worker threads push concurrently.
template <typename T>
class LockFreeTreiberStack
{
  struct Node
  {
    T payload;
    Node* next;
  };
  alignas(std::hardware_destructive_interference_size) std::atomic<Node*> head{nullptr};
  alignas(std::hardware_destructive_interference_size) std::atomic<Node*> free_head{nullptr};

  /// Allocate a node — try free list first, fall back to heap.
  Node* AllocNode(T&& payload)
  {
    // Try to pop from free list (CAS loop).
    auto* node = free_head.load(std::memory_order_acquire);
    while (node)
    {
      if (free_head.compare_exchange_weak(
            node, node->next, std::memory_order_release, std::memory_order_acquire))
      {
        node->payload = std::move(payload);
        node->next = nullptr;
        return node;
      }
    }
    return new Node{std::move(payload), nullptr};
  }

  /// Return a chain of nodes to the free list in one atomic exchange.
  void ReturnChainToFreeList(Node* chain_head, Node* chain_tail)
  {
    auto* expected = free_head.load(std::memory_order_relaxed);
    do
    {
      chain_tail->next = expected;
    } while (not free_head.compare_exchange_weak(
      expected, chain_head, std::memory_order_release, std::memory_order_relaxed));
  }

  /// Delete all nodes in a chain.
  static void DeleteChain(std::atomic<Node*>& list)
  {
    auto* chain = list.exchange(nullptr, std::memory_order_acquire);
    while (chain)
    {
      auto* next = chain->next;
      delete chain;
      chain = next;
    }
  }

public:
  ~LockFreeTreiberStack()
  {
    DeleteChain(head);
    DeleteChain(free_head);
  }

  /// CAS push — safe for concurrent producers, never blocks.
  void Push(T&& payload)
  {
    auto* node = AllocNode(std::move(payload));
    auto* expected = head.load(std::memory_order_relaxed);
    do
    {
      node->next = expected;
    } while (not head.compare_exchange_weak(
      expected, node, std::memory_order_release, std::memory_order_relaxed));
  }

  /// Atomic-exchange drain — returns all queued entries, recycles nodes to free list.
  std::vector<T> Drain()
  {
    auto* chain = head.exchange(nullptr, std::memory_order_acquire);
    if (not chain)
      return {};
    std::vector<T> result;
    auto* first = chain;
    Node* last = nullptr;
    while (chain)
    {
      result.push_back(std::move(chain->payload));
      last = chain;
      chain = chain->next;
    }
    ReturnChainToFreeList(first, last);
    return result;
  }

  /// Atomic-exchange drain with in-place callback — avoids building a return vector.
  /// Returns true if any nodes were processed. Recycles drained nodes to free list.
  template <typename F>
  bool DrainAndProcess(F&& callback)
  {
    auto* chain = head.exchange(nullptr, std::memory_order_acquire);
    if (not chain)
      return false;
    auto* first = chain;
    Node* last = nullptr;
    while (chain)
    {
      callback(std::move(chain->payload));
      last = chain;
      chain = chain->next;
    }
    ReturnChainToFreeList(first, last);
    return true;
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
 *   [num_sections : size_t]              // may have duplicate angle_set_ids
 *   For each section (pre-packed by worker threads):
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

  /// Push a pre-packed wire-format ByteArray to a destination queue.
  /// The buffer contains one "section" of the wire format:
  ///   [angle_set_id : size_t][num_entries : size_t][entries...]
  /// The comm thread concatenates sections and prepends a count header before sending.
  void EnqueuePrepackedByIndex(int queue_index, ByteArray&& data);

  /// Drain all received batches for this angle set via in-place callback (lock-free).
  /// Avoids building and returning a vector of vectors.
  /// Returns true if any batches were processed.
  template <typename F>
  bool DrainIncoming(size_t angle_set_id, F&& callback)
  {
    assert(angle_set_id < num_angle_sets_);
    return incoming_mailboxes_[angle_set_id].DrainAndProcess(std::forward<F>(callback));
  }

  /// Signal that this angle set has no more outgoing data.
  void SignalAngleSetComplete(size_t angle_set_id);

  /// Launch the dedicated communication thread.
  void Start();

  /// Flush remaining sends, wait for completion, and join the communication thread.
  void Stop();

private:
  /// Per-destination outgoing Treiber stack for pre-packed wire-format sections.
  struct NeighborQueue
  {
    int dest_location;
    std::unique_ptr<LockFreeTreiberStack<ByteArray>> queue;
  };

  /// An in-flight MPI_Isend and its serialized message data.
  struct InFlightSend
  {
    mpi::Request request;
    ByteArray data;
  };

  // -- Communication thread methods ------------------------------------------

  void CommThreadLoop();
  bool FlushOutgoing();
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

  /// One Treiber stack per angle set for incoming wire-format sections (raw ByteArrays).
  /// Each ByteArray contains: [num_entries : size_t][entries...] where each entry is
  /// [cell_global_id : uint64_t][face_id : unsigned int][data_size : size_t][psi doubles].
  /// Deserialization is deferred to the worker thread (avoids per-face heap allocations).
  std::vector<LockFreeTreiberStack<ByteArray>> incoming_mailboxes_;

  // -- Communication thread state --------------------------------------------

  /// Pre-allocated receive buffer for MPI messages.
  ByteArray recv_buffer_;

  /// In-flight MPI_Isends awaiting completion.
  std::vector<InFlightSend> in_flight_sends_;

  // -- Send buffer pool (recycled ByteArrays retain allocated capacity) ------

  std::vector<ByteArray> send_buffer_pool_;

  /// Acquire a ByteArray from the pool (retains previous capacity) or create a new one.
  ByteArray AcquireSendBuffer()
  {
    if (not send_buffer_pool_.empty())
    {
      ByteArray buf = std::move(send_buffer_pool_.back());
      send_buffer_pool_.pop_back();
      return buf;
    }
    return ByteArray();
  }

  /// Return a ByteArray to the pool (clear content but retain allocated capacity).
  void ReleaseSendBuffer(ByteArray&& buf)
  {
    buf.Data().clear();
    send_buffer_pool_.push_back(std::move(buf));
  }

  // -- Synchronization -------------------------------------------------------

  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_done_;
  std::thread comm_thread_;
};

} // namespace opensn

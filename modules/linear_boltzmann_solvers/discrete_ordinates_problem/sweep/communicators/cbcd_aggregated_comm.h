// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cassert>
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

/**
 * Lock-free Treiber stack for batched inter-thread communication.
 *
 * Multiple producers Push via CAS; a single consumer drains all items via
 * atomic exchange (Drain / DrainAndProcess).  An internal atomic free-list
 * recycles nodes after drain, eliminating per-push `new` and per-drain
 * `delete` after the first cycle — significantly reducing heap contention
 * when multiple worker threads push concurrently.
 *
 * Call Preallocate() at construction to populate the free list, ensuring
 * that no heap allocations occur during the sweep hot path.
 *
 * ## ABA analysis for AllocNode's free-list CAS pop
 *
 * The CAS pop in AllocNode (load free_head → read node->next → CAS) is
 * theoretically vulnerable to the classic ABA problem: if node A is popped
 * from free_head by another thread, used in a Push, drained, and returned
 * to free_head — all between the current thread's load and CAS — the CAS
 * would succeed with a potentially stale next pointer.
 *
 * In the CBCD sweep context, this cannot occur in practice because the full
 * recycle cycle (AllocNode pop → Push to head_ → DrainAndProcess exchange →
 * ReturnChainToFreeList push) spans multiple cross-thread function calls and
 * atomic operations that collectively take hundreds of nanoseconds, while the
 * vulnerable window (between loading free_head and executing the CAS) is 1–3
 * CPU cycles (~1 ns).  Additionally:
 *  - HPC runtimes pin threads to cores, eliminating OS preemption.
 *  - Preallocate() ensures abundant free nodes, reducing recycle pressure.
 *  - incoming_mailboxes_ have a single producer (comm thread), so no
 *    concurrent AllocNode contention exists on those free lists.
 *  - The main-stack drain uses atomic exchange (not CAS), which is ABA-immune.
 */
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

  /// Pre-populate the free list with \p count nodes (call once during construction).
  /// Ensures that subsequent Push calls during the sweep never call `new`.
  void Preallocate(size_t count)
  {
    if (count == 0)
      return;
    // Build a linked chain in one batch.
    Node* chain_head = new Node{T{}, nullptr};
    Node* chain_tail = chain_head;
    for (size_t i = 1; i < count; ++i)
      chain_head = new Node{T{}, chain_head};
    ReturnChainToFreeList(chain_head, chain_tail);
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
 * Aggregated MPI communicator for the threaded CBCD sweep.
 *
 * A dedicated communication thread runs CommThreadLoop(), which aggregates
 * outgoing MPI sends across all angle sets and probes for incoming messages.
 * This eliminates MPI contention between worker threads and reduces the total
 * number of MPI calls (one message per destination per flush, rather than one
 * per angle-set per destination).
 *
 * Data flow:
 *   Worker threads → outgoing Treiber stacks (one per dest rank, lock-free push)
 *     → comm thread drains, concatenates sections, Isend
 *   Comm thread Iprobe/recv → incoming Treiber stacks (one per angle set)
 *     → worker threads drain via DrainIncoming (lock-free exchange)
 *
 * Lifecycle: Start() launches the comm thread; Stop() sets a flag, flushes
 * remaining sends, waits for in-flight Isends, and joins the thread.
 *
 * Wire format for aggregated MPI messages:
 *   [num_sections : size_t]
 *   For each section (pre-packed by worker threads):
 *     [angle_set_id : size_t]
 *     [num_entries  : size_t]
 *     For each entry (one per outgoing face):
 *       [cell_global_id : uint64_t]
 *       [face_id        : unsigned int]
 *       [data_size      : size_t]           // number of doubles
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
  /// The callback receives a const reference to each incoming ByteArray section;
  /// after the callback returns, the ByteArray is recycled to the incoming buffer
  /// pool (retaining allocated capacity for reuse in ProbeAndReceive).
  /// Returns true if any batches were processed.
  template <typename F>
  bool DrainIncoming(size_t angle_set_id, F&& callback)
  {
    assert(angle_set_id < num_angle_sets_);
    return incoming_mailboxes_[angle_set_id].DrainAndProcess(
      [this, &callback](ByteArray&& section)
      {
        callback(section);
        section.Data().clear(); // retain capacity
        incoming_recycler_.Push(std::move(section));
      });
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
    int dest_rank;
    std::unique_ptr<LockFreeTreiberStack<ByteArray>> queue;
  };

  struct SourceQueue
  {
    int source_location;
    int mapped_rank;
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

  /// Source locations and communicator-local ranks from which this rank receives face data.
  std::vector<SourceQueue> source_queues_;

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

  /// Lock-free recycler for consumed incoming ByteArrays.  Worker threads push
  /// used ByteArrays here (via the DrainIncoming wrapper); the comm thread drains
  /// them for reuse in ProbeAndReceive, retaining allocated capacity to avoid
  /// per-section heap allocations after the first sweep.
  LockFreeTreiberStack<ByteArray> incoming_recycler_;

  // -- Communication thread state --------------------------------------------

  /// Pre-allocated receive buffer for MPI messages.
  ByteArray recv_buffer_;

  /// Comm-thread-local cache of recycled incoming ByteArrays (retains heap capacity).
  /// Populated by draining incoming_recycler_ at the start of each ProbeAndReceive.
  std::vector<ByteArray> incoming_reuse_cache_;

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

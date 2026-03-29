// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <utility>
#include <vector>

namespace opensn
{

constexpr std::size_t LOCK_FREE_TREIBER_STACK_INTERFERENCE_SIZE =
#ifdef __cpp_lib_hardware_interference_size
  std::hardware_destructive_interference_size;
#else
  64;
#endif

/**
 * Recyclable Treiber stack with multi-producer push and single-consumer drain.
 *
 * The stack is used as a mailbox between CBCD worker threads and the aggregated
 * communicator thread. Producers push payloads with a lock-free CAS loop, and
 * the consumer drains the full list with one atomic exchange. A second lock-free
 * free list recycles nodes so the hot path avoids repeated heap traffic after
 * the initial preallocation.
 *
 * \tparam T Movable payload type.
 * \note Intended for the CBCD mailbox pattern: many producers, one drain site.
 */
template <class T>
class LockFreeTreiberStack
{
private:
  struct Node
  {
    /// Stored payload.
    T payload;
    /// Next pointer in the active or free list.
    Node* next = nullptr;
  };

  /// Head of the active stack.
  alignas(LOCK_FREE_TREIBER_STACK_INTERFERENCE_SIZE) std::atomic<Node*> head_{nullptr};
  /// Head of the recycled-node free list.
  alignas(LOCK_FREE_TREIBER_STACK_INTERFERENCE_SIZE) std::atomic<Node*> free_head_{nullptr};

  /// Allocate one node from the free list or the heap.
  Node* AllocNode(T&& payload)
  {
    auto* node = free_head_.load(std::memory_order_acquire);
    while (node)
    {
      if (free_head_.compare_exchange_weak(
            node, node->next, std::memory_order_release, std::memory_order_acquire))
      {
        node->payload = std::move(payload);
        node->next = nullptr;
        return node;
      }
    }

    return new Node{std::move(payload), nullptr};
  }

  /// Return a node chain to the recycled-node free list.
  void ReturnChainToFreeList(Node* chain_head, Node* chain_tail)
  {
    auto* expected = free_head_.load(std::memory_order_relaxed);
    do
    {
      chain_tail->next = expected;
    } while (not free_head_.compare_exchange_weak(
      expected, chain_head, std::memory_order_release, std::memory_order_relaxed));
  }

  /// Delete every node in the selected list.
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
  /**
   * Preallocate recycled nodes for later reuse.
   *
   * \param count Number of nodes to add to the free list.
   */
  void Preallocate(std::size_t count)
  {
    if (count == 0)
      return;

    Node* chain_head = new Node{};
    Node* chain_tail = chain_head;
    for (std::size_t i = 1; i < count; ++i)
      chain_head = new Node{T{}, chain_head};

    ReturnChainToFreeList(chain_head, chain_tail);
  }

  /**
   * Push one payload onto the active stack.
   *
   * \param payload Payload to enqueue.
   */
  void Push(T&& payload)
  {
    auto* node = AllocNode(std::move(payload));
    auto* expected = head_.load(std::memory_order_relaxed);
    do
    {
      node->next = expected;
    } while (not head_.compare_exchange_weak(
      expected, node, std::memory_order_release, std::memory_order_relaxed));
  }

  /**
   * Drain the stack into a vector.
   *
   * \return Moved payloads in LIFO order.
   */
  std::vector<T> Drain()
  {
    auto* chain = head_.exchange(nullptr, std::memory_order_acquire);
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

  /**
   * Drain the stack and process payloads in place.
   *
   * \tparam F Callback type.
   * \param callback Callable invoked once per drained payload.
   * \return `true` if any payload was processed.
   */
  template <class F>
  bool DrainAndProcess(F&& callback)
  {
    auto* chain = head_.exchange(nullptr, std::memory_order_acquire);
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

  /**
   * Check whether the active stack is empty.
   *
   * \return `true` if no payload is currently queued.
   */
  bool Empty() const { return head_.load(std::memory_order_acquire) == nullptr; }

  ~LockFreeTreiberStack()
  {
    DeleteChain(head_);
    DeleteChain(free_head_);
  }
};

} // namespace opensn

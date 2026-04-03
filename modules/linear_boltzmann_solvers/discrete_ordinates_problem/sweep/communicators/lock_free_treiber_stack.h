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

/// Recyclable Treiber stack with multi-producer push and single-consumer drain.
template <class T>
class LockFreeTreiberStack
{
private:
  struct Node
  {
    T payload;
    Node* next = nullptr;
  };

  alignas(LOCK_FREE_TREIBER_STACK_INTERFERENCE_SIZE) std::atomic<Node*> head_{nullptr};
  alignas(LOCK_FREE_TREIBER_STACK_INTERFERENCE_SIZE) std::atomic<Node*> free_head_{nullptr};

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

  void ReturnChainToFreeList(Node* chain_head, Node* chain_tail)
  {
    auto* expected = free_head_.load(std::memory_order_relaxed);
    do
    {
      chain_tail->next = expected;
    } while (not free_head_.compare_exchange_weak(
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

  bool Empty() const { return head_.load(std::memory_order_acquire) == nullptr; }

  ~LockFreeTreiberStack()
  {
    DeleteChain(head_);
    DeleteChain(free_head_);
  }
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <boost/lockfree/spsc_queue.hpp>
#include <boost/lockfree/stack.hpp>
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace opensn
{

/**
 * Bounded lock-free multi-producer, single-consumer ring buffer.
 *
 * Producers reserve slots through an atomic head counter and publish them with
 * a per-slot ready flag. The single consumer drains in FIFO order.
 */
template <typename T>
class LockFreeRingBuffer
{
public:
  struct Slot
  {
    T payload;
    std::atomic<bool> ready{false};
  };

  void Preallocate(const std::size_t capacity)
  {
    buffer_ = std::vector<Slot>(capacity);
  }

  Slot& ReserveSlot()
  {
    const auto idx = head_.fetch_add(1, std::memory_order_relaxed) % buffer_.size();
    while (buffer_[idx].ready.load(std::memory_order_acquire))
      std::this_thread::yield();
    return buffer_[idx];
  }

  void PublishSlot(Slot& slot) { slot.ready.store(true, std::memory_order_release); }

  void GetReadySlots(std::vector<Slot*>& out)
  {
    out.clear();
    if (buffer_.empty())
      return;

    const auto capacity = buffer_.size();
    auto current_tail = tail_;
    while (buffer_[current_tail % capacity].ready.load(std::memory_order_acquire))
    {
      out.push_back(&buffer_[current_tail % capacity]);
      ++current_tail;
    }
  }

  void FreeSlots(const std::size_t count)
  {
    const auto capacity = buffer_.size();
    for (std::size_t i = 0; i < count; ++i)
    {
      buffer_[tail_ % capacity].ready.store(false, std::memory_order_release);
      ++tail_;
    }
  }

  template <typename Callback>
  std::size_t ProcessReady(Callback&& cb)
  {
    if (buffer_.empty())
      return 0;

    const auto capacity = buffer_.size();
    std::size_t count = 0;
    while (true)
    {
      auto& slot = buffer_[tail_ % capacity];
      if (not slot.ready.load(std::memory_order_acquire))
        break;
      cb(slot.payload);
      slot.ready.store(false, std::memory_order_release);
      ++tail_;
      ++count;
    }
    return count;
  }

  bool Empty() const
  {
    if (buffer_.empty())
      return true;
    return not buffer_[tail_ % buffer_.size()].ready.load(std::memory_order_acquire);
  }

private:
  std::vector<Slot> buffer_;
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> head_{0};
  alignas(std::hardware_destructive_interference_size) std::size_t tail_{0};
};

/**
 * Recyclable lock-free Treiber stack.
 *
 * Wraps `boost::lockfree::stack` to match CBCD's multi-producer, single-consumer
 * handoff sites.
 */
template <class T>
class LockFreeTreiberStack
{
public:
  LockFreeTreiberStack() : stack_(0) {}

  /**
   * Reserve stack nodes for future pushes.
   *
   * \param count Number of nodes to reserve.
   */
  void Preallocate(std::size_t count)
  {
    if (count == 0)
      return;
    stack_.reserve_unsafe(count);
  }

  /**
   * Push one payload onto the stack.
   *
   * \param payload Payload to store.
   */
  void Push(T&& payload) { stack_.push(std::move(payload)); }

  /**
   * Drain all currently queued payloads and invoke a callback for each one.
   *
   * \tparam F Callable object invocable with `const T&`.
   * \param callback Callback applied to each drained payload.
   * \return True when at least one payload was drained.
   */
  template <class F>
  bool DrainAndProcess(F callback)
  {
    bool drained_any = false;
    T payload;
    while (stack_.pop(payload))
    {
      drained_any = true;
      callback(std::move(payload));
    }
    return drained_any;
  }

  /**
   * Drain and discard all currently queued payloads.
   *
   * \return True when at least one payload was drained.
   */
  bool DrainAndDiscard()
  {
    bool drained_any = false;
    T payload;
    while (stack_.pop(payload))
      drained_any = true;
    return drained_any;
  }

  /// Check whether the stack is currently empty.
  bool Empty() const { return stack_.empty(); }

private:
  boost::lockfree::stack<T> stack_;
};

/**
 * Bounded lock-free single-producer, single-consumer queue.
 *
 * Wraps `boost::lockfree::spsc_queue` behind the narrower CBCD access pattern:
 * one producer pushes fully formed payloads, and one consumer drains them in batches.
 * \note Using more than one producer or more than one consumer is invalid.
 */
template <class T>
class LockFreeSPSCQueue
{
public:
  explicit LockFreeSPSCQueue(std::size_t capacity)
    : queue_(std::make_unique<boost::lockfree::spsc_queue<T>>(std::max<std::size_t>(1, capacity)))
  {
  }

  LockFreeSPSCQueue(const LockFreeSPSCQueue&) = delete;
  LockFreeSPSCQueue& operator=(const LockFreeSPSCQueue&) = delete;
  LockFreeSPSCQueue(LockFreeSPSCQueue&&) noexcept = default;
  LockFreeSPSCQueue& operator=(LockFreeSPSCQueue&&) noexcept = default;

  /**
   * Push one payload onto the queue.
   *
   * \param payload Payload to store.
   * \return True when the payload was queued.
   */
  bool Push(T&& payload) { return queue_->push(std::move(payload)); }

  /**
   * Drain all currently queued payloads and invoke a callback for each one.
   *
   * \tparam F Callable object invocable with `T&&`.
   * \param callback Callback applied to each drained payload.
   * \return True when at least one payload was drained.
   */
  template <class F>
  bool DrainAndProcess(F callback)
  {
    bool drained_any = false;
    T payload;
    while (queue_->pop(payload))
    {
      drained_any = true;
      callback(std::move(payload));
    }
    return drained_any;
  }

  /**
   * Drain and discard all currently queued payloads.
   *
   * \return True when at least one payload was drained.
   */
  bool DrainAndDiscard()
  {
    bool drained_any = false;
    T payload;
    while (queue_->pop(payload))
      drained_any = true;
    return drained_any;
  }

  /// Check whether the queue is currently empty.
  bool Empty() const { return queue_->empty(); }

private:
  std::unique_ptr<boost::lockfree::spsc_queue<T>> queue_;
};

} // namespace opensn

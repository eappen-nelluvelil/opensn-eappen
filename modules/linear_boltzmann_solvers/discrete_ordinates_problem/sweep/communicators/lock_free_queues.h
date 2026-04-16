// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <boost/lockfree/spsc_queue.hpp>
#include <boost/lockfree/stack.hpp>
#include <cstddef>
#include <memory>
#include <utility>

namespace opensn
{

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

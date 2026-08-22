// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <new>
#include <thread>
#include <vector>

namespace opensn
{

/**
 * Race-free ownership state for a coalesced producer notification.
 *
 * Every producer publication calls Notify(), including publications made while a notification
 * is already outstanding. That unconditional read-modify-write gives Release() an acquire edge
 * from the latest pre-clear publication. A publication after Release() observes the cleared state
 * and must enqueue a new notification. The consumer may instead retain ownership with TryRetain()
 * when its queue recheck finds additional committed work.
 */
class CoalescedDoorbell
{
public:
  /// Publish producer ownership; return true only when a new notification must be enqueued.
  bool Notify() noexcept { return not outstanding_.exchange(true, std::memory_order_acq_rel); }

  /// Release consumer ownership and acquire every producer publication preceding this clear.
  bool Release() noexcept { return outstanding_.exchange(false, std::memory_order_acq_rel); }

  /// Retain consumer ownership if no producer has already reclaimed it.
  bool TryRetain() noexcept
  {
    bool expected = false;
    return outstanding_.compare_exchange_strong(
      expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  /// Report whether a notification is queued or owned by the consumer.
  bool IsOutstanding() const noexcept { return outstanding_.load(std::memory_order_acquire); }

private:
  std::atomic<bool> outstanding_{false};
};

/**
 * Bounded lock-free multi-producer, single-consumer ring buffer.
 *
 * Producers reserve slots through an atomic head counter and publish them with a per-slot
 * ready flag. The single consumer drains in FIFO order through the tail index. The queue
 * is bounded and reuses preallocated slots; it performs no dynamic allocation once the
 * storage has been initialized.
 *
 * In the CBCD aggregated communicator, LockFreeRingBuffer is the incoming per-angle-set
 * mailbox written by the communication thread and drained by the owning angle-set worker.
 * Outgoing traffic uses the committed SPSC queue below because static angle-set scheduling
 * gives each producer shard exactly one worker.
 *
 * LockFreeRingBuffer works under the following assumptions:
 * - producers reserve one slot, write the payload in place, and publish the slot exactly
 *   once
 * - the consumer drains published slots in FIFO order and returns them to the ring for
 *   reuse.
 *
 * This yields a fixed-capacity queue with explicit slot reuse.
 */
template <typename T>
class LockFreeRingBuffer
{
public:
  /// Slot payload with a publication flag.
  struct Slot
  {
    /// Stored payload.
    T payload;
    /// Publication flag visible to the single consumer.
    std::atomic<bool> ready{false};
  };

  /**
   * Allocate storage for the requested number of slots.
   *
   * \param capacity Number of ring-buffer slots.
   */
  void Preallocate(const std::size_t capacity) { buffer_ = std::vector<Slot>(capacity); }

  /**
   * Initialize every slot payload in place.
   *
   * \tparam Callback Callable invoked once per slot payload.
   * \param cb Initialization callback.
   */
  template <typename Callback>
  void InitializeSlots(Callback&& cb)
  {
    for (auto& slot : buffer_)
      cb(slot.payload);
  }

  /**
   * Reserve one slot for a producer.
   *
   * \return Writable slot reference.
   */
  Slot& ReserveSlot()
  {
    const auto capacity = buffer_.size();
    assert(capacity > 0);

    auto reservation = head_.load(std::memory_order_relaxed);
    while (true)
    {
      const auto consumed = consumed_tail_.load(std::memory_order_acquire);
      if (reservation - consumed >= capacity)
      {
        std::this_thread::yield();
        reservation = head_.load(std::memory_order_relaxed);
        continue;
      }
      if (head_.compare_exchange_weak(reservation,
                                      reservation + 1,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed))
        break;
    }

    const auto idx = reservation % capacity;
    while (buffer_[idx].ready.load(std::memory_order_acquire))
      std::this_thread::yield();
    return buffer_[idx];
  }

  /**
   * Publish one reserved slot to the consumer.
   *
   * \param slot Slot to publish.
   */
  void PublishSlot(Slot& slot) { slot.ready.store(true, std::memory_order_release); }

  /**
   * Gather currently ready slots without consuming them.
   *
   * \param out Output vector of ready slot pointers.
   */
  void GetReadySlots(std::vector<Slot*>& out,
                     const std::size_t max_count = static_cast<std::size_t>(-1))
  {
    out.clear();
    if (buffer_.empty())
      return;

    const auto capacity = buffer_.size();
    const auto count_limit = max_count < capacity ? max_count : capacity;
    auto current_tail = tail_;
    while (out.size() < count_limit and
           buffer_[current_tail % capacity].ready.load(std::memory_order_acquire))
    {
      out.push_back(&buffer_[current_tail % capacity]);
      ++current_tail;
    }
  }

  /**
   * Release the next `count` ready slots after they have been consumed.
   *
   * \param count Number of slots to free.
   */
  void FreeSlots(const std::size_t count)
  {
    const auto capacity = buffer_.size();
    for (std::size_t i = 0; i < count; ++i)
    {
      buffer_[tail_ % capacity].ready.store(false, std::memory_order_release);
      ++tail_;
    }
    consumed_tail_.store(tail_, std::memory_order_release);
  }

  /**
   * Consume all ready slots in FIFO order.
   *
   * \tparam Callback Callable invoked with each slot payload.
   * \param cb Consumer callback.
   * \return Number of consumed slots.
   */
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
    consumed_tail_.store(tail_, std::memory_order_release);
    return count;
  }

  /// Check whether the queue currently has no published slots.
  bool Empty() const
  {
    if (buffer_.empty())
      return true;
    return not buffer_[tail_ % buffer_.size()].ready.load(std::memory_order_acquire);
  }

private:
  /// Ring-buffer storage.
  std::vector<Slot> buffer_;
  /// Producer reservation index.
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> head_{0};
  /// Consumer drain index.
  alignas(std::hardware_destructive_interference_size) std::size_t tail_{0};
  /// Consumer-published reclamation index used to bound producer reservations.
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> consumed_tail_{0};
};

/**
 * Bounded single-producer, single-consumer queue with explicit batch commits.
 *
 * The producer may reserve and fill any number of slots, but none of those slots become
 * visible to the consumer until Commit() publishes the producer head. This is useful for
 * CBCD outgoing traffic: one angle-set worker fills all face records produced by a completed
 * device batch and then exposes the batch atomically to the communication thread.
 *
 * Unlike LockFreeRingBuffer, a slow producer cannot leave a globally visible publication
 * hole. The producer and consumer retain private monotonically increasing heads and exchange
 * only the committed and consumed positions.
 */
template <typename T>
class CommittedSPSCQueue
{
public:
  /// Queue slot owned by exactly one side at a time.
  struct Slot
  {
    T payload;
  };

  /// Allocate the fixed queue storage.
  void Preallocate(const std::size_t capacity) { buffer_ = std::vector<Slot>(capacity); }

  /// Initialize every slot payload in place.
  template <typename Callback>
  void InitializeSlots(Callback&& cb)
  {
    for (auto& slot : buffer_)
      cb(slot.payload);
  }

  /**
   * Reserve one producer slot.
   *
   * The returned slot remains invisible to the consumer until a subsequent Commit().
   */
  Slot& ReserveSlot()
  {
    const auto capacity = buffer_.size();
    assert(capacity > 0);
    while (producer_head_ - consumed_head_.load(std::memory_order_acquire) >= capacity)
      std::this_thread::yield();
    return buffer_[producer_head_++ % capacity];
  }

  /**
   * Publish every slot reserved since the preceding commit.
   *
   * \return `true` when this call publishes records not exposed by the preceding commit.
   */
  bool Commit()
  {
    const bool published_new_records =
      producer_head_ != committed_head_.load(std::memory_order_relaxed);
    committed_head_.store(producer_head_, std::memory_order_release);
    return published_new_records;
  }

  /// Gather the committed, not-yet-consumed prefix.
  void GetReadySlots(std::vector<Slot*>& out,
                     const std::size_t max_count = static_cast<std::size_t>(-1))
  {
    out.clear();
    if (buffer_.empty())
      return;

    const auto committed = committed_head_.load(std::memory_order_acquire);
    const auto capacity = buffer_.size();
    for (auto current = consumer_head_; current < committed and out.size() < max_count; ++current)
      out.push_back(&buffer_[current % capacity]);
  }

  /// Release a prefix returned by GetReadySlots().
  void FreeSlots(const std::size_t count)
  {
    consumer_head_ += count;
    consumed_head_.store(consumer_head_, std::memory_order_release);
  }

  /// Check whether no committed records await the consumer.
  bool Empty() const { return consumer_head_ == committed_head_.load(std::memory_order_acquire); }

private:
  std::vector<Slot> buffer_;
  /// Private producer reservation position.
  std::size_t producer_head_ = 0;
  /// Producer-published batch boundary.
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> committed_head_{0};
  /// Private consumer position.
  std::size_t consumer_head_ = 0;
  /// Consumer-published reclamation boundary.
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> consumed_head_{0};
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <limits>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace opensn
{

/**
 * Bounded lock-free multi-producer, single-consumer ring buffer.
 *
 * Producers reserve slots through an atomic head counter and publish them with a per-slot
 * ready flag. The single consumer drains in FIFO order through the tail index. The queue
 * is bounded and reuses preallocated slots; it performs no dynamic allocation once the
 * storage has been initialized.
 *
 * In the CBCD aggregated communicator, LockFreeRingBuffer serves two roles:
 * 1. an outgoing per-destination queue written by sweep worker threads and drained by the
 *    communication thread,
 * 2. an incoming per-angle-set queue written by the communication thread and drained by the
 *    owning angleset worker thread.
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
    const auto idx = head_.fetch_add(1, std::memory_order_relaxed) % buffer_.size();
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
};

/**
 * Bounded lock-free single-producer, single-consumer slot queue.
 *
 * The producer writes directly into a reserved slot, then publishes the new head
 * position. The consumer peeks published slots in FIFO order and releases them
 * after processing. The queue is bounded, performs no allocation after
 * `Preallocate`, and preserves slot payload capacity across publications.
 *
 * This queue is intended for CBCD ownership patterns in which:
 * - one worker thread is the sole producer for a shard,
 * - one communication thread is the sole consumer for that shard, or
 * - one communication thread is the sole producer for an incoming mailbox and
 *   one worker thread is the sole consumer.
 */
template <typename T>
class LockFreeSPSCSlotQueue
{
public:
  /// Slot payload stored in the bounded ring.
  struct Slot
  {
    T payload;
  };

  /**
   * Allocate storage for the requested number of slots.
   *
   * \param capacity Number of queue slots.
   */
  void Preallocate(const std::size_t capacity)
  {
    buffer_ = std::vector<Slot>(capacity);
    producer_head_ = 0;
    consumer_tail_ = 0;
    producer_tail_cache_ = 0;
    consumer_head_cache_ = 0;
    published_head_.store(0, std::memory_order_relaxed);
    consumed_tail_.store(0, std::memory_order_relaxed);
  }

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
   * Reserve one writable slot for the producer.
   *
   * \return Writable slot reference.
   */
  Slot& ReserveSlot()
  {
    assert(not buffer_.empty());
    const auto capacity = buffer_.size();
    while ((producer_head_ - producer_tail_cache_) >= capacity)
    {
      producer_tail_cache_ = consumed_tail_.load(std::memory_order_acquire);
      if ((producer_head_ - producer_tail_cache_) < capacity)
        break;
      std::this_thread::yield();
    }
    return buffer_[producer_head_ % capacity];
  }

  /// Publish the slot reserved by the last `ReserveSlot()` call.
  void PublishSlot()
  {
    ++producer_head_;
    published_head_.store(producer_head_, std::memory_order_release);
  }

  /**
   * Gather currently published slots without consuming them.
   *
   * \param out Output vector of ready slot pointers.
   * \param max_slots Maximum number of slots to gather.
   */
  void PeekReadySlots(std::vector<Slot*>& out,
                      const std::size_t max_slots = std::numeric_limits<std::size_t>::max())
  {
    out.clear();
    if (buffer_.empty())
      return;

    consumer_head_cache_ = published_head_.load(std::memory_order_acquire);
    const auto ready_count = std::min(consumer_head_cache_ - consumer_tail_, max_slots);
    out.reserve(ready_count);
    const auto capacity = buffer_.size();
    for (std::size_t i = 0; i < ready_count; ++i)
      out.push_back(&buffer_[(consumer_tail_ + i) % capacity]);
  }

  /**
   * Release the next `count` ready slots after they have been consumed.
   *
   * \param count Number of slots to release.
   */
  void ReleaseReadySlots(const std::size_t count)
  {
    if (count == 0)
      return;

    consumer_tail_ += count;
    consumed_tail_.store(consumer_tail_, std::memory_order_release);
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

    consumer_head_cache_ = published_head_.load(std::memory_order_acquire);
    const auto ready_count = consumer_head_cache_ - consumer_tail_;
    const auto capacity = buffer_.size();
    for (std::size_t i = 0; i < ready_count; ++i)
      cb(buffer_[(consumer_tail_ + i) % capacity].payload);
    ReleaseReadySlots(ready_count);
    return ready_count;
  }

  /// Check whether the queue currently has no published slots.
  bool Empty() const
  {
    if (buffer_.empty())
      return true;
    return published_head_.load(std::memory_order_acquire) == consumer_tail_;
  }

private:
  /// Ring storage reused across all publications.
  std::vector<Slot> buffer_;
  /// Producer-owned next unpublished sequence number.
  alignas(std::hardware_destructive_interference_size) std::size_t producer_head_ = 0;
  /// Consumer-owned next undisposed sequence number.
  alignas(std::hardware_destructive_interference_size) std::size_t consumer_tail_ = 0;
  /// Producer-side cached consumer tail used to avoid repeated atomic loads.
  std::size_t producer_tail_cache_ = 0;
  /// Consumer-side cached published head used to avoid repeated atomic loads.
  std::size_t consumer_head_cache_ = 0;
  /// Producer-published head visible to the consumer.
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> published_head_{0};
  /// Consumer-released tail visible to the producer.
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> consumed_tail_{0};
};

} // namespace opensn

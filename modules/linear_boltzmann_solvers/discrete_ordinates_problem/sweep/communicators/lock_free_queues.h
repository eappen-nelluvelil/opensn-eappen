// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

namespace opensn
{

inline constexpr std::size_t QueueHardwareInterferenceSize =
#ifdef __cpp_lib_hardware_interference_size
  std::hardware_destructive_interference_size;
#else
  64;
#endif

/**
 * Bounded lock-free single-producer, single-consumer slot queue.
 *
 * The producer writes directly into a reserved slot and then publishes the advanced
 * head sequence. The consumer peeks published slots in FIFO order and releases them
 * after processing. The queue is bounded, performs no allocation after `Preallocate`,
 * and preserves slot payload capacity across publications.
 *
 * This queue is intended for CBCD ownership patterns in which one worker thread is
 * the sole producer for an outgoing shard, or the communication thread is the sole
 * producer for an incoming mailbox.
 */
template <typename T>
class LockFreeSPSCSlotQueue
{
public:
  /// Slot payload stored in the bounded ring.
  struct Slot
  {
    /// Stored payload.
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
   * \param callback Initialization callback.
   */
  template <typename Callback>
  void InitializeSlots(Callback&& callback)
  {
    for (auto& slot : buffer_)
      callback(slot.payload);
  }

  /**
   * Reserve one writable slot for the producer.
   *
   * \return Writable slot reference.
   */
  Slot& ReserveSlot()
  {
    if (buffer_.empty())
      throw std::logic_error("SPSC queue: cannot reserve from an empty queue.");
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
    if (count > consumer_head_cache_ - consumer_tail_)
      throw std::logic_error("SPSC queue: release exceeds the published slot count.");

    consumer_tail_ += count;
    consumed_tail_.store(consumer_tail_, std::memory_order_release);
  }

  /**
   * Consume all ready slots in FIFO order.
   *
   * \tparam Callback Callable invoked with each slot payload.
   * \param callback Consumer callback.
   * \return Number of consumed slots.
   */
  template <typename Callback>
  std::size_t ProcessReady(Callback&& callback)
  {
    if (buffer_.empty())
      return 0;

    consumer_head_cache_ = published_head_.load(std::memory_order_acquire);
    const auto ready_count = consumer_head_cache_ - consumer_tail_;
    const auto capacity = buffer_.size();
    for (std::size_t i = 0; i < ready_count; ++i)
      callback(buffer_[(consumer_tail_ + i) % capacity].payload);
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
  alignas(QueueHardwareInterferenceSize) std::size_t producer_head_ = 0;
  /// Consumer-owned next unreleased sequence number.
  alignas(QueueHardwareInterferenceSize) std::size_t consumer_tail_ = 0;
  /// Producer-side cached consumer tail used to avoid repeated atomic loads.
  std::size_t producer_tail_cache_ = 0;
  /// Consumer-side cached published head used to avoid repeated atomic loads.
  std::size_t consumer_head_cache_ = 0;
  /// Producer-published head visible to the consumer.
  alignas(QueueHardwareInterferenceSize) std::atomic<std::size_t> published_head_{0};
  /// Consumer-released tail visible to the producer.
  alignas(QueueHardwareInterferenceSize) std::atomic<std::size_t> consumed_tail_{0};
};

} // namespace opensn

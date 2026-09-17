// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/utils/hardware_interference_size.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

namespace opensn
{

/** Bounded lock-free SPSC ring whose elements retain their allocated capacity. */
template <typename T>
class LockFreeSPSCSlotQueue
{
public:
  /** Allocates the ring storage and resets both endpoints. */
  void Preallocate(const std::size_t capacity)
  {
    buffer_ = std::vector<T>(capacity);
    producer_ = {};
    consumer_ = {};
    published_head_.store(0, std::memory_order_relaxed);
    consumed_tail_.store(0, std::memory_order_relaxed);
  }

  /** Applies `callback` once to each reusable element. */
  template <typename Callback>
  void InitializeSlots(Callback callback)
  {
    for (auto& element : buffer_)
      callback(element);
  }

  /** Returns the next producer-owned element, waiting only while the ring is full. */
  T& ReserveSlot()
  {
    const auto capacity = buffer_.size();
    assert(capacity > 0);
    while ((producer_.head - producer_.tail_cache) >= capacity)
    {
      producer_.tail_cache = consumed_tail_.load(std::memory_order_acquire);
      if ((producer_.head - producer_.tail_cache) < capacity)
        break;
      std::this_thread::yield();
    }
    return buffer_[producer_.head % capacity];
  }

  /** Makes the most recently reserved element visible to the consumer. */
  void PublishSlot()
  {
    ++producer_.head;
    published_head_.store(producer_.head, std::memory_order_release);
  }

  /** Returns pointers to the current contiguous logical prefix of ready elements. */
  void PeekReadySlots(std::vector<T*>& out,
                      const std::size_t max_slots = std::numeric_limits<std::size_t>::max())
  {
    out.clear();
    if (buffer_.empty())
      return;

    consumer_.head_cache = published_head_.load(std::memory_order_acquire);
    const auto ready_count = std::min(consumer_.head_cache - consumer_.tail, max_slots);
    out.reserve(ready_count);
    const auto capacity = buffer_.size();
    for (std::size_t i = 0; i < ready_count; ++i)
      out.push_back(&buffer_[(consumer_.tail + i) % capacity]);
  }

  /** Releases `count` elements previously returned by `PeekReadySlots`. */
  void ReleaseReadySlots(const std::size_t count)
  {
    if (count == 0)
      return;

    consumer_.tail += count;
    consumed_tail_.store(consumer_.tail, std::memory_order_release);
  }

  /** Processes and releases every element currently visible to the consumer. */
  template <typename Callback>
  std::size_t ProcessReady(Callback callback)
  {
    if (buffer_.empty())
      return 0;

    consumer_.head_cache = published_head_.load(std::memory_order_acquire);
    const auto ready_count = consumer_.head_cache - consumer_.tail;
    const auto capacity = buffer_.size();
    for (std::size_t i = 0; i < ready_count; ++i)
      callback(buffer_[(consumer_.tail + i) % capacity]);
    ReleaseReadySlots(ready_count);
    return ready_count;
  }

  /** Returns whether the consumer has no published elements. */
  bool Empty() const
  {
    if (buffer_.empty())
      return true;
    return published_head_.load(std::memory_order_acquire) == consumer_.tail;
  }

private:
  struct alignas(HardwareInterferenceSize) ProducerState
  {
    std::size_t head = 0;
    std::size_t tail_cache = 0;
  };
  struct alignas(HardwareInterferenceSize) ConsumerState
  {
    std::size_t tail = 0;
    std::size_t head_cache = 0;
  };

  ProducerState producer_;
  ConsumerState consumer_;
  alignas(HardwareInterferenceSize) std::atomic<std::size_t> published_head_{0};
  alignas(HardwareInterferenceSize) std::atomic<std::size_t> consumed_tail_{0};
  std::vector<T> buffer_;
};

} // namespace opensn

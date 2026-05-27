// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/utils/hardware_interference_size.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <limits>
#include <span>
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

  /** Returns the published prefix as at most two spans, valid until released. */
  std::array<std::span<const T>, 2>
  PeekReadySlots(const std::size_t max_slots = std::numeric_limits<std::size_t>::max())
  {
    if (buffer_.empty())
      return {};

    consumer_.head_cache = published_head_.load(std::memory_order_acquire);
    const auto ready_count = std::min(consumer_.head_cache - consumer_.tail, max_slots);
    if (ready_count == 0)
      return {};

    const auto capacity = buffer_.size();
    const auto first_index = consumer_.tail % capacity;
    const auto first_count = std::min(ready_count, capacity - first_index);
    const std::span<const T> buffer(buffer_);
    return {buffer.subspan(first_index, first_count), buffer.first(ready_count - first_count)};
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
    const auto ready = PeekReadySlots();
    const auto ready_count = ready[0].size() + ready[1].size();
    for (const auto span : ready)
      for (const auto& element : span)
        callback(element);
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

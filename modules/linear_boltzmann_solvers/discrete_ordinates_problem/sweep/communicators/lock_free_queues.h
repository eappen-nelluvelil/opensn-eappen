// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <vector>

namespace opensn
{

/**
 * Bounded single-producer, single-consumer queue with explicit batch commits.
 *
 * The producer may reserve and fill any number of slots, but none of those slots become
 * visible to the consumer until Commit() publishes the producer head. This is useful for
 * CBCD outgoing traffic: one angle-set worker fills all face records produced by a completed
 * device batch and then exposes the batch atomically to the communication thread. It also fits
 * incoming mailboxes, where the communication thread commits one received section at a time and
 * the angle-set owner consumes it. The producer and consumer retain private monotonically
 * increasing heads and exchange only the committed and consumed positions. CBCD sizes each queue
 * for its producer's complete topological record set for one sweep. Consequently, a valid
 * reservation never depends on consumer progress; exhaustion denotes an accounting error.
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
    if (capacity == 0)
      throw std::logic_error("CBCD SPSC queue: cannot reserve from an empty topology.");
    if (producer_head_ - consumed_head_.load(std::memory_order_acquire) >= capacity)
      throw std::logic_error("CBCD SPSC queue: exact topology capacity exhausted.");
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

  /** Consume the complete committed prefix in FIFO order. */
  template <typename Callback>
  std::size_t ProcessReady(Callback&& callback)
  {
    if (buffer_.empty())
      return 0;

    const auto committed = committed_head_.load(std::memory_order_acquire);
    const auto count = committed - consumer_head_;
    const auto capacity = buffer_.size();
    for (std::size_t i = 0; i < count; ++i)
      callback(buffer_[(consumer_head_ + i) % capacity].payload);
    FreeSlots(count);
    return count;
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

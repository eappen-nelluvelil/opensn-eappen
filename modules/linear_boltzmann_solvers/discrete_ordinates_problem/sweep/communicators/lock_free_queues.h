// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace opensn
{

/**
 * Fixed-universe, multi-producer/single-consumer ready-index set.
 *
 * Producers atomically mark indices whose private queues contain committed work. The sole
 * consumer takes a snapshot with atomic exchanges and services exactly the marked queues.
 * Repeated marks coalesce, which is safe because queue consumers drain the complete committed
 * prefix. A mark racing after an exchange remains set for the next snapshot; a mark racing before
 * an exchange is acquired by that exchange. Thus no separate ownership flag or bounded token ring
 * is needed, and an idle consumer examines ceil(num_indices / word_bits) atomic words rather than
 * every producer queue.
 */
class AtomicReadyIndexSet
{
public:
  /// Number of index bits represented by one atomic storage word.
  static constexpr std::size_t WORD_BITS = std::numeric_limits<std::uint64_t>::digits;
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                "CBCD ready-index words must be lock-free.");

  /// Allocate the exact number of words needed for `num_indices` possible indices.
  void Initialize(const std::size_t num_indices)
  {
    num_indices_ = num_indices;
    num_words_ = num_indices == 0 ? 0 : 1 + (num_indices - 1) / WORD_BITS;
    words_ = num_words_ == 0 ? nullptr : std::make_unique<ReadyWord[]>(num_words_);
    Clear();
  }

  /// Mark one index ready. Duplicate marks may coalesce until the consumer takes a snapshot.
  void Notify(const std::size_t index)
  {
    if (index >= num_indices_)
      throw std::out_of_range("CBCD ready-index notification is outside its fixed universe.");
    const auto word_index = index / WORD_BITS;
    const auto bit_index = index % WORD_BITS;
    words_[word_index].bits.fetch_or(std::uint64_t{1} << bit_index,
                                     std::memory_order_release);
  }

  /**
   * Atomically take every currently marked index.
   *
   * The returned order is increasing and each index occurs at most once in a snapshot.
   */
  std::size_t TakeReady(std::vector<std::size_t>& ready_indices)
  {
    ready_indices.clear();
    for (std::size_t word_index = 0; word_index < num_words_; ++word_index)
    {
      // An empty word is the common progress-loop state. Avoid an unnecessary read-modify-write
      // that would take exclusive ownership of its cache line and interfere with producers.
      if (words_[word_index].bits.load(std::memory_order_relaxed) == 0)
        continue;
      auto ready_bits = words_[word_index].bits.exchange(0, std::memory_order_acq_rel);
      while (ready_bits != 0)
      {
        const auto bit_index = static_cast<std::size_t>(std::countr_zero(ready_bits));
        const auto index = word_index * WORD_BITS + bit_index;
        if (index >= num_indices_)
          throw std::logic_error("CBCD ready-index set contains an invalid high bit.");
        ready_indices.push_back(index);
        ready_bits &= ready_bits - 1;
      }
    }
    return ready_indices.size();
  }

  /// Report whether no ready-index notification is outstanding.
  bool Empty() const noexcept
  {
    for (std::size_t i = 0; i < num_words_; ++i)
      if (words_[i].bits.load(std::memory_order_acquire) != 0)
        return false;
    return true;
  }

  /// Clear every notification. This is only valid while producers and the consumer are stopped.
  void Clear() noexcept
  {
    for (std::size_t i = 0; i < num_words_; ++i)
      words_[i].bits.store(0, std::memory_order_relaxed);
  }

private:
  /// Keep independently updated ready words off the same destructive-interference region.
  struct alignas(std::hardware_destructive_interference_size) ReadyWord
  {
    std::atomic<std::uint64_t> bits{0};
  };

  std::size_t num_indices_ = 0;
  std::size_t num_words_ = 0;
  std::unique_ptr<ReadyWord[]> words_;
};

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

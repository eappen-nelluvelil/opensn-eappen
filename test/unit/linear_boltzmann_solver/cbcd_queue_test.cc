// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace opensn
{

TEST(CBCDCoalescedDoorbellTest, CoversCommitBeforeAndAfterConsumerClear)
{
  CoalescedDoorbell doorbell;

  // The first publication needs a queue token. A second publication before the clear is
  // coalesced, but still writes the state that Release() acquires.
  EXPECT_TRUE(doorbell.Notify());
  EXPECT_FALSE(doorbell.Notify());
  EXPECT_TRUE(doorbell.Release());

  // When the committed-queue recheck is nonempty, the consumer can retain ownership without
  // publishing another token.
  EXPECT_TRUE(doorbell.TryRetain());
  EXPECT_TRUE(doorbell.IsOutstanding());
  EXPECT_TRUE(doorbell.Release());

  // A producer publication after the clear observes false and must enqueue a new token.
  EXPECT_TRUE(doorbell.Notify());
  EXPECT_FALSE(doorbell.TryRetain());
  EXPECT_TRUE(doorbell.Release());
  EXPECT_FALSE(doorbell.IsOutstanding());
}

TEST(CBCDCoalescedDoorbellTest, ConcurrentQueueCompositionDrainsEveryCommit)
{
  constexpr std::size_t num_values = 10000;
  CommittedSPSCQueue<std::size_t> queue;
  queue.Preallocate(num_values);
  LockFreeRingBuffer<std::size_t> notifications;
  notifications.Preallocate(1);
  CoalescedDoorbell doorbell;
  std::atomic<bool> producer_finished{false};

  std::thread producer(
    [&]
    {
      for (std::size_t value = 0; value < num_values; ++value)
      {
        queue.ReserveSlot().payload = value;
        queue.Commit();
        if (doorbell.Notify())
        {
          auto& slot = notifications.ReserveSlot();
          slot.payload = 0;
          notifications.PublishSlot(slot);
        }
      }
      producer_finished.store(true, std::memory_order_release);
    });

  std::vector<LockFreeRingBuffer<std::size_t>::Slot*> notification_slots;
  std::vector<CommittedSPSCQueue<std::size_t>::Slot*> ready;
  std::size_t expected_value = 0;
  bool fifo_order_preserved = true;
  bool ownership_preserved = true;
  while (expected_value < num_values or
         not producer_finished.load(std::memory_order_acquire) or
         doorbell.IsOutstanding() or not notifications.Empty())
  {
    notifications.GetReadySlots(notification_slots);
    if (notification_slots.empty())
    {
      std::this_thread::yield();
      continue;
    }

    // Match the communicator protocol: release the bounded notification slot before
    // servicing its producer so a racing producer can republish without blocking.
    notifications.FreeSlots(notification_slots.size());
    while (true)
    {
      queue.GetReadySlots(ready);
      for (const auto* slot : ready)
        fifo_order_preserved &= slot->payload == expected_value++;
      queue.FreeSlots(ready.size());

      ownership_preserved &= doorbell.Release();
      if (queue.Empty())
        break;
      if (not doorbell.TryRetain())
        break;
    }
  }
  producer.join();

  EXPECT_TRUE(fifo_order_preserved);
  EXPECT_TRUE(ownership_preserved);
  EXPECT_EQ(expected_value, num_values);
  EXPECT_TRUE(queue.Empty());
  EXPECT_TRUE(notifications.Empty());
  EXPECT_FALSE(doorbell.IsOutstanding());
}

TEST(CBCDCommittedSPSCQueueTest, PublishesOnlyCompleteBatches)
{
  CommittedSPSCQueue<int> queue;
  queue.Preallocate(4);
  std::vector<CommittedSPSCQueue<int>::Slot*> ready;

  queue.ReserveSlot().payload = 11;
  queue.ReserveSlot().payload = 12;
  queue.GetReadySlots(ready);
  EXPECT_TRUE(ready.empty());

  queue.Commit();
  queue.GetReadySlots(ready);
  ASSERT_EQ(ready.size(), 2);
  EXPECT_EQ(ready[0]->payload, 11);
  EXPECT_EQ(ready[1]->payload, 12);

  queue.FreeSlots(1);
  queue.GetReadySlots(ready);
  ASSERT_EQ(ready.size(), 1);
  EXPECT_EQ(ready[0]->payload, 12);
  queue.FreeSlots(1);
  EXPECT_TRUE(queue.Empty());
}

TEST(CBCDCommittedSPSCQueueTest, CommitReportsOnlyNewPublication)
{
  CommittedSPSCQueue<int> queue;
  queue.Preallocate(2);
  std::vector<CommittedSPSCQueue<int>::Slot*> ready;

  EXPECT_FALSE(queue.Commit());
  queue.ReserveSlot().payload = 11;
  EXPECT_TRUE(queue.Commit());
  EXPECT_FALSE(queue.Commit());

  queue.GetReadySlots(ready);
  ASSERT_EQ(ready.size(), 1);
  queue.FreeSlots(ready.size());
  EXPECT_FALSE(queue.Commit());
}

TEST(CBCDLockFreeRingBufferTest, BoundsConcurrentProducerReservations)
{
  constexpr std::size_t num_producers = 4;
  constexpr std::size_t values_per_producer = 500;
  constexpr std::size_t capacity = 7;
  LockFreeRingBuffer<std::size_t> queue;
  queue.Preallocate(capacity);

  std::vector<std::thread> producers;
  producers.reserve(num_producers);
  for (std::size_t producer = 0; producer < num_producers; ++producer)
    producers.emplace_back(
      [producer, &queue]
      {
        for (std::size_t value = 0; value < values_per_producer; ++value)
        {
          auto& slot = queue.ReserveSlot();
          slot.payload = producer * values_per_producer + value;
          queue.PublishSlot(slot);
        }
      });

  std::vector<std::uint8_t> seen(num_producers * values_per_producer, 0);
  std::size_t num_consumed = 0;
  while (num_consumed < seen.size())
  {
    num_consumed += queue.ProcessReady(
      [&seen](const std::size_t value)
      {
        ASSERT_LT(value, seen.size());
        EXPECT_EQ(seen[value], 0);
        seen[value] = 1;
      });
    if (num_consumed < seen.size())
      std::this_thread::yield();
  }
  for (auto& producer : producers)
    producer.join();

  EXPECT_TRUE(queue.Empty());
  for (const auto count : seen)
    EXPECT_EQ(count, 1);
}

TEST(CBCDLockFreeRingBufferTest, FullSnapshotStopsAtExactCapacity)
{
  constexpr std::size_t capacity = 4;
  LockFreeRingBuffer<std::size_t> queue;
  queue.Preallocate(capacity);
  for (std::size_t value = 0; value < capacity; ++value)
  {
    auto& slot = queue.ReserveSlot();
    slot.payload = value;
    queue.PublishSlot(slot);
  }

  std::vector<LockFreeRingBuffer<std::size_t>::Slot*> ready;
  queue.GetReadySlots(ready);
  ASSERT_EQ(ready.size(), capacity);
  for (std::size_t value = 0; value < capacity; ++value)
    EXPECT_EQ(ready[value]->payload, value);
  queue.FreeSlots(ready.size());
  EXPECT_TRUE(queue.Empty());
}

TEST(CBCDCommittedSPSCQueueTest, ReusesConsumedSlotsAcrossWrap)
{
  CommittedSPSCQueue<int> queue;
  queue.Preallocate(3);
  std::vector<CommittedSPSCQueue<int>::Slot*> ready;

  for (int value = 0; value < 3; ++value)
    queue.ReserveSlot().payload = value;
  queue.Commit();
  queue.GetReadySlots(ready);
  ASSERT_EQ(ready.size(), 3);
  queue.FreeSlots(3);

  for (int value = 3; value < 6; ++value)
    queue.ReserveSlot().payload = value;
  queue.Commit();
  queue.GetReadySlots(ready);
  ASSERT_EQ(ready.size(), 3);
  EXPECT_EQ(ready[0]->payload, 3);
  EXPECT_EQ(ready[1]->payload, 4);
  EXPECT_EQ(ready[2]->payload, 5);
  queue.FreeSlots(3);
  EXPECT_TRUE(queue.Empty());
}

TEST(CBCDCommittedSPSCQueueTest, UncommittedProducerCannotBlockAnotherShard)
{
  CommittedSPSCQueue<int> stalled_queue;
  CommittedSPSCQueue<int> progressing_queue;
  stalled_queue.Preallocate(2);
  progressing_queue.Preallocate(2);
  std::vector<CommittedSPSCQueue<int>::Slot*> ready;

  stalled_queue.ReserveSlot().payload = 1;
  progressing_queue.ReserveSlot().payload = 2;
  progressing_queue.Commit();

  stalled_queue.GetReadySlots(ready);
  EXPECT_TRUE(ready.empty());
  progressing_queue.GetReadySlots(ready);
  ASSERT_EQ(ready.size(), 1);
  EXPECT_EQ(ready[0]->payload, 2);
}

TEST(CBCDCommittedSPSCQueueTest, ConcurrentBatchPublicationPreservesFIFOOrder)
{
  constexpr std::size_t num_batches = 2000;
  constexpr std::size_t capacity = 127;
  CommittedSPSCQueue<std::size_t> queue;
  queue.Preallocate(capacity);

  std::size_t total_values = 0;
  for (std::size_t batch = 0; batch < num_batches; ++batch)
    total_values += 1 + batch % 17;

  std::atomic<bool> producer_finished{false};
  std::thread producer(
    [&]
    {
      std::size_t next_value = 0;
      for (std::size_t batch = 0; batch < num_batches; ++batch)
      {
        const auto batch_size = 1 + batch % 17;
        for (std::size_t i = 0; i < batch_size; ++i)
          queue.ReserveSlot().payload = next_value++;
        queue.Commit();
      }
      producer_finished.store(true, std::memory_order_release);
    });

  std::vector<CommittedSPSCQueue<std::size_t>::Slot*> ready;
  std::size_t expected_value = 0;
  bool fifo_order_preserved = true;
  while (expected_value < total_values or not producer_finished.load(std::memory_order_acquire))
  {
    queue.GetReadySlots(ready, 23);
    if (ready.empty())
    {
      std::this_thread::yield();
      continue;
    }
    for (const auto* slot : ready)
      fifo_order_preserved &= slot->payload == expected_value++;
    queue.FreeSlots(ready.size());
  }
  producer.join();

  EXPECT_TRUE(fifo_order_preserved);
  EXPECT_EQ(expected_value, total_values);
  EXPECT_TRUE(queue.Empty());
}

} // namespace opensn

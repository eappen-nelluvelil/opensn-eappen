// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include <array>
#include <atomic>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

namespace opensn
{

TEST(CBCDAtomicReadyIndexSetTest, CoalescesAndOrdersFixedUniverseIndices)
{
  const auto second_word_index = AtomicReadyIndexSet::WORD_BITS;
  AtomicReadyIndexSet ready;
  ready.Initialize(second_word_index + 2);
  std::vector<std::size_t> indices;

  EXPECT_TRUE(ready.Empty());
  ready.Notify(second_word_index + 1);
  ready.Notify(3);
  ready.Notify(3);
  EXPECT_FALSE(ready.Empty());

  EXPECT_EQ(ready.TakeReady(indices), 2);
  ASSERT_EQ(indices.size(), 2);
  EXPECT_EQ(indices[0], 3);
  EXPECT_EQ(indices[1], second_word_index + 1);
  EXPECT_TRUE(ready.Empty());
  EXPECT_THROW(ready.Notify(second_word_index + 2), std::out_of_range);
}

TEST(CBCDAtomicReadyIndexSetTest, ConcurrentMarksDoNotLoseCommittedShardWork)
{
  constexpr std::size_t num_producers = 8;
  constexpr std::size_t records_per_producer = 2000;
  AtomicReadyIndexSet ready;
  ready.Initialize(num_producers);

  std::vector<std::unique_ptr<CommittedSPSCQueue<std::size_t>>> queues;
  queues.reserve(num_producers);
  for (std::size_t producer = 0; producer < num_producers; ++producer)
  {
    auto queue = std::make_unique<CommittedSPSCQueue<std::size_t>>();
    queue->Preallocate(records_per_producer);
    queues.push_back(std::move(queue));
  }

  std::vector<std::thread> producers;
  producers.reserve(num_producers);
  for (std::size_t producer = 0; producer < num_producers; ++producer)
    producers.emplace_back(
      [producer, &ready, &queues]
      {
        for (std::size_t record = 0; record < records_per_producer; ++record)
        {
          queues[producer]->ReserveSlot().payload = record;
          if (queues[producer]->Commit())
            ready.Notify(producer);
        }
      });

  std::array<std::size_t, num_producers> next_expected{};
  std::vector<std::size_t> ready_indices;
  std::size_t records_consumed = 0;
  while (records_consumed != num_producers * records_per_producer)
  {
    ready.TakeReady(ready_indices);
    if (ready_indices.empty())
    {
      std::this_thread::yield();
      continue;
    }
    for (const auto producer : ready_indices)
      records_consumed += queues[producer]->ProcessReady(
        [producer, &next_expected](const std::size_t record)
        { EXPECT_EQ(record, next_expected[producer]++); });
  }

  for (auto& producer : producers)
    producer.join();

  // A producer mark can race after the consumer's snapshot while its preceding committed
  // record is nevertheless included in the queue drain. Consume that permitted redundant mark.
  ready.TakeReady(ready_indices);
  for (const auto producer : ready_indices)
    EXPECT_EQ(queues[producer]->ProcessReady([](const std::size_t) {}), 0);
  EXPECT_TRUE(ready.Empty());
  for (const auto expected : next_expected)
    EXPECT_EQ(expected, records_per_producer);
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

TEST(CBCDCommittedSPSCQueueTest, HoldsAndProcessesExactTopologyCapacity)
{
  constexpr std::size_t capacity = 4;
  CommittedSPSCQueue<std::size_t> queue;
  queue.Preallocate(capacity);
  for (std::size_t value = 0; value < capacity; ++value)
  {
    queue.ReserveSlot().payload = value;
    EXPECT_TRUE(queue.Commit());
  }

  std::size_t expected = 0;
  const auto processed =
    queue.ProcessReady([&expected](const std::size_t value) { EXPECT_EQ(value, expected++); });
  EXPECT_EQ(processed, capacity);
  EXPECT_EQ(expected, capacity);
  EXPECT_TRUE(queue.Empty());
}

TEST(CBCDCommittedSPSCQueueTest, RejectsTopologyOverflowWithoutWaiting)
{
  CommittedSPSCQueue<int> queue;
  queue.Preallocate(1);
  queue.ReserveSlot().payload = 7;
  queue.Commit();

  EXPECT_THROW(queue.ReserveSlot(), std::logic_error);
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
  std::size_t total_values = 0;
  for (std::size_t batch = 0; batch < num_batches; ++batch)
    total_values += 1 + batch % 17;

  // Match CBCD's topology-derived construction: the queue can hold the producer's complete
  // sweep even if the communication thread makes no progress while a batch is being formed.
  CommittedSPSCQueue<std::size_t> queue;
  queue.Preallocate(total_values);

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

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace opensn
{

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

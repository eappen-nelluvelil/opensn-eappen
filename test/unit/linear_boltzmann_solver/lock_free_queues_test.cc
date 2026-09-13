// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include <gtest/gtest.h>
#include <array>
#include <thread>

TEST(LockFreeSPSCSlotQueue, StagedPublication)
{
  opensn::LockFreeSPSCSlotQueue<int> queue;
  queue.Preallocate(3);
  queue.ReserveSlot() = 11;
  queue.StageSlot();
  queue.ReserveSlot() = 12;
  queue.StageSlot();
  EXPECT_TRUE(queue.Empty());

  queue.PublishStagedSlots();
  const auto ready = queue.PeekReadySlots();
  ASSERT_EQ(ready[0].size(), 2);
  EXPECT_TRUE(ready[1].empty());
  EXPECT_EQ(ready[0][0], 11);
  EXPECT_EQ(ready[0][1], 12);
  queue.ReleaseReadySlots(2);

  queue.ReserveSlot() = 13;
  queue.PublishSlot();
  EXPECT_EQ(queue.ProcessReady([](int value) { EXPECT_EQ(value, 13); }), 1);
  EXPECT_TRUE(queue.Empty());
}

TEST(LockFreeSPSCSlotQueue, WrappedSpansAndPartialRelease)
{
  opensn::LockFreeSPSCSlotQueue<int> queue;
  EXPECT_TRUE(queue.PeekReadySlots()[0].empty());
  EXPECT_EQ(queue.ProcessReady([](auto) { ADD_FAILURE(); }), 0);
  queue.Preallocate(3);
  EXPECT_TRUE(queue.PeekReadySlots()[0].empty());
  EXPECT_TRUE(queue.PeekReadySlots()[1].empty());
  for (int i = 0; i < 3; ++i)
  {
    queue.ReserveSlot() = i;
    queue.PublishSlot();
  }
  auto ready = queue.PeekReadySlots(2);
  ASSERT_EQ(ready[0].size(), 2);
  EXPECT_TRUE(ready[1].empty());
  queue.ReleaseReadySlots(2);
  for (int i = 3; i < 5; ++i)
  {
    queue.ReserveSlot() = i;
    queue.PublishSlot();
  }
  ready = queue.PeekReadySlots();
  ASSERT_EQ(ready[0].size(), 1);
  ASSERT_EQ(ready[1].size(), 2);
  EXPECT_EQ(ready[0][0], 2);
  EXPECT_EQ(ready[1][0], 3);
  EXPECT_EQ(ready[1][1], 4);
  EXPECT_TRUE(queue.PeekReadySlots(0)[0].empty());
  ready = queue.PeekReadySlots(2);
  ASSERT_EQ(ready[0].size(), 1);
  ASSERT_EQ(ready[1].size(), 1);
  queue.ReleaseReadySlots(2);
  EXPECT_EQ(queue.ProcessReady([](int value) { EXPECT_EQ(value, 4); }), 1);
  EXPECT_TRUE(queue.Empty());
}

TEST(LockFreeSPSCSlotQueue, StagedWraparound)
{
  // A full ring must publish its staged prefix before waiting for the consumer.
  for (const auto capacity : {1, 3, 17})
  {
    opensn::LockFreeSPSCSlotQueue<std::array<std::size_t, 3>> queue;
    queue.Preallocate(capacity);
    constexpr std::size_t count = 10000;
    bool valid = true;
    std::size_t received = 0;
    std::thread consumer(
      [&]
      {
        while (received < count)
        {
          queue.ProcessReady(
            [&](const auto& value)
            {
              valid &=
                value[0] == received && value[1] == received + 1 && value[2] == received * received;
              ++received;
            });
          std::this_thread::yield();
        }
      });
    for (std::size_t i = 0; i < count; ++i)
    {
      queue.ReserveSlot() = {i, i + 1, i * i};
      queue.StageSlot();
    }
    queue.PublishStagedSlots();
    consumer.join();
    EXPECT_TRUE(valid);
    EXPECT_EQ(received, count);
    EXPECT_TRUE(queue.Empty());
  }
}

TEST(LockFreeSPSCSlotQueue, SnapshotExcludesLaterPublication)
{
  opensn::LockFreeSPSCSlotQueue<int> queue;
  queue.Preallocate(3);
  queue.ReserveSlot() = 11;
  queue.PublishSlot();
  const auto ready = queue.PeekReadySlots();
  queue.ReserveSlot() = 12;
  queue.PublishSlot();
  ASSERT_EQ(ready[0].size(), 1);
  EXPECT_TRUE(ready[1].empty());
  EXPECT_EQ(ready[0][0], 11);
  queue.ReleaseReadySlots(1);
  EXPECT_EQ(queue.ProcessReady([](int value) { EXPECT_EQ(value, 12); }), 1);
}

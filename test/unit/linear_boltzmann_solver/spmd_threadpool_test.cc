// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/spmd_threadpool.h"
#include <gtest/gtest.h>
#include <atomic>
#include <thread>

TEST(SPMDThreadPool, ReusesSingleWorkerAcrossEpochs)
{
  opensn::SPMD_ThreadPool pool(1);
  std::thread::id worker;
  unsigned int input = 0;
  unsigned int output = 0;
  pool.AssignTask(
    [&](std::size_t id)
    {
      EXPECT_EQ(id, 0);
      if (input == 1)
        worker = std::this_thread::get_id();
      else
        EXPECT_EQ(std::this_thread::get_id(), worker);
      output = 3 * input;
    });
  for (input = 1; input <= 100; ++input)
  {
    pool.Run(0);
    pool.WaitAll();
    EXPECT_EQ(output, 3 * input);
  }
  EXPECT_NE(worker, std::this_thread::get_id());
  pool.Stop();
  pool.Stop();
}

TEST(SPMDThreadPool, DrainsProgressBeforeNextEpoch)
{
  opensn::SPMD_ThreadPool pool(1);
  std::atomic<bool> stop{false};
  std::atomic<bool> entered{false};
  unsigned int completed = 0;
  pool.AssignTask(
    [&](std::size_t)
    {
      entered.store(true, std::memory_order_release);
      while (not stop.load(std::memory_order_acquire))
        std::this_thread::yield();
      ++completed;
    });
  for (unsigned int epoch = 0; epoch < 100; ++epoch)
  {
    stop.store(false, std::memory_order_relaxed);
    entered.store(false, std::memory_order_relaxed);
    pool.Run(0);
    while (not entered.load(std::memory_order_acquire))
      std::this_thread::yield();
    stop.store(true, std::memory_order_release);
    pool.WaitAll();
    EXPECT_EQ(completed, epoch + 1);
  }
}

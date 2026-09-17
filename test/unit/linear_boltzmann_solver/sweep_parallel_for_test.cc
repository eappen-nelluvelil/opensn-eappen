// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep_parallel_for.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <memory>
#include <stdexcept>

TEST(SweepParallelFor, SerialUsesCallingThread)
{
  const auto caller = std::this_thread::get_id();
  for (const size_t count : {0, 1, 9})
  {
    size_t visited = 0;
    opensn::ParallelFor(count,
                        1,
                        [&](size_t i)
                        {
                          EXPECT_EQ(std::this_thread::get_id(), caller);
                          EXPECT_EQ(i, visited++);
                        });
    EXPECT_EQ(visited, count);
  }
}

TEST(SweepParallelFor, ParallelVisitsEachIndexOnce)
{
  std::array<std::atomic<unsigned int>, 9> visits{};
  opensn::ParallelFor(visits.size(), 3, [&](size_t i) { ++visits[i]; });
  for (const auto& count : visits)
    EXPECT_EQ(count.load(), 1U);
}

TEST(SweepParallelFor, PropagatesExceptions)
{
  for (const size_t threads : {1, 3})
    EXPECT_THROW(opensn::ParallelFor(9,
                                     threads,
                                     [](size_t i)
                                     {
                                       if (i == 2)
                                         throw std::runtime_error("sweep setup failed");
                                     }),
                 std::runtime_error);
}

TEST(SweepParallelFor, ReusesWorkersWithoutRetainingCaptures)
{
  opensn::ReleaseParallelForWorkers();
  std::array<std::thread::id, 3> workers{};
  std::array<unsigned int, 3> generations{};
  for (unsigned int batch = 0; batch < 8; ++batch)
  {
    auto owner = std::make_shared<int>(42);
    const std::weak_ptr<int> observer = owner;
    opensn::ParallelFor(3,
                        3,
                        [&, owner](size_t i)
                        {
                          thread_local unsigned int generation = 0;
                          generations[i] = ++generation;
                          if (batch == 0)
                            workers[i] = std::this_thread::get_id();
                          EXPECT_EQ(workers[i], std::this_thread::get_id());
                          EXPECT_EQ(*owner, 42);
                        });
    owner.reset();
    EXPECT_TRUE(observer.expired());
    for (const auto generation : generations)
      EXPECT_EQ(generation, batch + 1);
  }
  opensn::ReleaseParallelForWorkers();
}

TEST(SweepParallelFor, ReusesPoolAfterExceptionAndResize)
{
  EXPECT_THROW(opensn::ParallelFor(6,
                                   3,
                                   [](size_t i)
                                   {
                                     if (i == 1)
                                       throw std::runtime_error("setup failed");
                                   }),
               std::runtime_error);
  for (const size_t threads : {3, 2, 1, 4})
  {
    std::array<std::atomic<unsigned int>, 9> visits{};
    opensn::ParallelFor(visits.size(), threads, [&](size_t i) { ++visits[i]; });
    for (const auto& count : visits)
      EXPECT_EQ(count.load(), 1U);
  }
  opensn::ReleaseParallelForWorkers();
  opensn::ReleaseParallelForWorkers();
}

TEST(SweepParallelFor, NestedCallsUseIndependentPools)
{
  std::atomic<unsigned int> visits{0};
  opensn::ParallelFor(4, 2, [&](size_t) { opensn::ParallelFor(5, 2, [&](size_t) { ++visits; }); });
  EXPECT_EQ(visits.load(), 20U);
  opensn::ReleaseParallelForWorkers();
}

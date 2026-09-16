// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep_parallel_for.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
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

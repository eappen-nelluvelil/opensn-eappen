// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep_parallel_for.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstddef>
#include <limits>

namespace opensn
{

TEST(SweepParallelForTest, StaticPartitionsCoverWorkExactlyOnce)
{
  for (std::size_t count = 0; count <= 64; ++count)
    for (std::size_t num_partitions = 1; num_partitions <= 32; ++num_partitions)
    {
      std::size_t previous_end = 0;
      std::size_t minimum_size = std::numeric_limits<std::size_t>::max();
      std::size_t maximum_size = 0;
      for (std::size_t partition_id = 0; partition_id < num_partitions; ++partition_id)
      {
        const auto [begin, end] = GetStaticPartition(count, num_partitions, partition_id);
        EXPECT_EQ(begin, previous_end);
        ASSERT_GE(end, begin);
        const auto partition_size = end - begin;
        minimum_size = std::min(minimum_size, partition_size);
        maximum_size = std::max(maximum_size, partition_size);
        previous_end = end;
      }
      EXPECT_EQ(previous_end, count);
      EXPECT_LE(maximum_size - minimum_size, 1);
    }
}

TEST(SweepParallelForTest, StaticPartitionsBalanceTuoAngleSets)
{
  std::size_t total = 0;
  for (std::size_t worker_id = 0; worker_id < 20; ++worker_id)
  {
    const auto [begin, end] = GetStaticPartition(448, 20, worker_id);
    EXPECT_EQ(end - begin, worker_id < 8 ? 23 : 22);
    total += end - begin;
  }
  EXPECT_EQ(total, 448);

  std::size_t active_workers = 0;
  for (std::size_t worker_id = 0; worker_id < 192; ++worker_id)
  {
    const auto [begin, end] = GetStaticPartition(448, 192, worker_id);
    EXPECT_EQ(end - begin, worker_id < 64 ? 3 : 2);
    active_workers += end > begin;
  }
  EXPECT_EQ(active_workers, 192);
}

} // namespace opensn

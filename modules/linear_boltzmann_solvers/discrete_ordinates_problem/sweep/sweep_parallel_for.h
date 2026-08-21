// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

namespace opensn
{

/// Half-open range assigned to one participant in a static work partition.
struct WorkPartition
{
  std::size_t begin = 0;
  std::size_t end = 0;
};

/**
 * Partition `[0, count)` into contiguous ranges whose sizes differ by at most one.
 *
 * The first `count % num_partitions` ranges receive one additional item. This mapping is
 * shared by CBCD sweep workers and their SPSC producer shards; keeping those ownership maps
 * identical is required for the queues to remain single-producer.
 */
constexpr WorkPartition
GetStaticPartition(const std::size_t count,
                   const std::size_t num_partitions,
                   const std::size_t partition_id) noexcept
{
  assert(num_partitions > 0);
  assert(partition_id < num_partitions);
  const auto base_size = count / num_partitions;
  const auto remainder = count % num_partitions;
  const auto extra_before = partition_id < remainder ? partition_id : remainder;
  const auto begin = partition_id * base_size + extra_before;
  return {begin, begin + base_size + (partition_id < remainder ? 1 : 0)};
}

/// Run `function(i)` for i in [0, count) across `num_threads` threads, strided.
/// Exceptions thrown by any worker are propagated to the caller (first wins).
template <typename Function>
void
ParallelFor(std::size_t count, std::size_t num_threads, Function function)
{
  assert(num_threads > 0);
  std::vector<std::exception_ptr> exceptions(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  const auto join_workers = [&workers]() noexcept
  {
    for (auto& worker : workers)
      if (worker.joinable())
        worker.join();
  };

  try
  {
    for (std::size_t thread_id = 0; thread_id < num_threads; ++thread_id)
      workers.emplace_back(
        [&, thread_id]()
        {
          try
          {
            for (size_t i = thread_id; i < count; i += num_threads)
              function(i);
          }
          catch (...)
          {
            exceptions[thread_id] = std::current_exception();
          }
        });
  }
  catch (...)
  {
    join_workers();
    throw;
  }
  join_workers();

  for (const auto& exception : exceptions)
    if (exception)
      std::rethrow_exception(exception);
}

/// Run independent, nonuniform work items using dynamic scheduling.
template <typename Function>
void
ParallelForDynamic(std::size_t count, std::size_t num_threads, Function function)
{
  assert(num_threads > 0);
  std::atomic<std::size_t> next{0};
  ParallelFor(num_threads,
              num_threads,
              [&](const std::size_t)
              {
                while (true)
                {
                  const auto i = next.fetch_add(1, std::memory_order_relaxed);
                  if (i >= count)
                    break;
                  function(i);
                }
              });
}

} // namespace opensn

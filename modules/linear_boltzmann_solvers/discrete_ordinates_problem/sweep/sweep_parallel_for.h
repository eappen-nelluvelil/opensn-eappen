// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/utils/memory.h"
#include "framework/utils/parallel_for.h"
#include <exception>
#include <thread>
#include <vector>

namespace opensn
{

/// Run `function(i)` for i in [0, count) across `num_threads` threads, strided.
/// Exceptions thrown by any worker are propagated to the caller (first wins).
/// One worker executes inline. Larger pools are reused on subsequent calls.
template <typename Function>
void
ParallelFor(size_t count, size_t num_threads, Function function)
{
  if (count == 0)
    return;
  if (num_threads <= 1)
  {
    for (size_t i = 0; i < count; ++i)
      function(i);
    return;
  }

  std::vector<std::exception_ptr> exceptions(num_threads);
  TraceMemory("setup_batch.begin", 0, false, {{"threads", num_threads}, {"items", count}});
  RunParallelForWorkers(num_threads,
                        [&](size_t thread_id)
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
  TraceMemory("setup_batch.complete", 0, false, {{"threads", num_threads}, {"items", count}});

  for (const auto& exception : exceptions)
    if (exception)
      std::rethrow_exception(exception);
}

} // namespace opensn

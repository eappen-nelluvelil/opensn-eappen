// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/utils/parallel_for.h"
#include "framework/utils/spmd_threadpool.h"
#include <memory>
#include <utility>

namespace opensn
{
namespace
{
thread_local std::unique_ptr<SPMD_ThreadPool> parallel_for_workers;
}

void
RunParallelForWorkers(std::size_t num_threads, std::function<void(std::size_t)> task)
{
  if (not parallel_for_workers)
    parallel_for_workers = std::make_unique<SPMD_ThreadPool>();
  parallel_for_workers->Resize(num_threads);
  parallel_for_workers->ExecuteBatch(std::move(task));
  parallel_for_workers->AssignTask(nullptr);
}

void
ReleaseParallelForWorkers()
{
  parallel_for_workers.reset();
}

} // namespace opensn

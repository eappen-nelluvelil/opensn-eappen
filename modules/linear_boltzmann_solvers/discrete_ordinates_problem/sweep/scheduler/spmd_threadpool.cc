// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/spmd_threadpool.h"
#include <cassert>
#include <memory>
#include <utility>

namespace opensn
{
namespace
{
thread_local std::unique_ptr<SPMD_ThreadPool> parallel_for_workers;
}

SPMD_ThreadPool::SPMD_ThreadPool(std::size_t n)
{
  Resize(n);
}

SPMD_ThreadPool::~SPMD_ThreadPool()
{
  Stop();
}

void
SPMD_ThreadPool::Resize(std::size_t n)
{
  if (worker_threads_.size() == n)
    return;
  Stop();
  if (n > 0)
    Start(n);
}

void
SPMD_ThreadPool::Start(std::size_t n)
{
  if (workers_initialized_ || n == 0)
    return;

  worker_threads_.reserve(n);
  epoch_states_.assign(n, EpochState{0, 0});
  outstanding_ = 0;

  workers_initialized_ = true;
  try
  {
    for (std::size_t i = 0; i < n; ++i)
      worker_threads_.emplace_back(&SPMD_ThreadPool::InfiniteLoop, this, i);
  }
  catch (...)
  {
    Stop();
    throw;
  }
}

void
SPMD_ThreadPool::Stop()
{
  if (!workers_initialized_)
    return;

  {
    std::scoped_lock<std::mutex> lock(mutex_);
    stop_workers_ = true;
  }
  cv_start_.notify_all();
  cv_done_.notify_all();

  for (auto& t : worker_threads_)
    if (t.joinable())
      t.join();

  worker_threads_.clear();
  epoch_states_.clear();
  task_ = nullptr;

  workers_initialized_ = false;
  stop_workers_ = false;
  outstanding_ = 0;
}

void
SPMD_ThreadPool::Run(std::size_t thread_idx)
{
  assert(workers_initialized_);
  assert(thread_idx < worker_threads_.size());
  {
    std::scoped_lock<std::mutex> lock(mutex_);
    ++epoch_states_[thread_idx].request;
    ++outstanding_;
  }
  cv_start_.notify_all();
}

void
SPMD_ThreadPool::WaitAll()
{
  std::unique_lock<std::mutex> lock(mutex_);
  cv_done_.wait(lock, [this] { return outstanding_ == 0; });
}

void
SPMD_ThreadPool::InfiniteLoop(std::size_t thread_idx)
{
  while (true)
  {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_start_.wait(lock,
                     [&] { return stop_workers_ || epoch_states_[thread_idx].IsIncomplete(); });
      if (stop_workers_)
        return;
      ++epoch_states_[thread_idx].done;
    }
    task_(thread_idx);
    {
      std::scoped_lock<std::mutex> lock(mutex_);
      if (--outstanding_ == 0)
        cv_done_.notify_all();
    }
  }
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

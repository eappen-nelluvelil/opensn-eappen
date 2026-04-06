// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/spmd_threadpool.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "framework/utils/timer.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace opensn
{

template <typename CommonDataT>
void
DiscreteOrdinatesProblem::BuildCBCLikeFLUDSCommonDataInParallel(const std::string_view label)
{
  // Flatten per-quadrature SPDS lists into a single task vector. Pre-size each quadrature's
  // entry in `quadrature_fluds_commondata_map_` so worker threads can write distinct slots
  // concurrently without touching the map container itself.
  struct BuildTask
  {
    const SPDS* spds;
    std::unique_ptr<FLUDSCommonData>* slot;
  };
  std::vector<BuildTask> tasks;
  for (const auto& [quadrature, spds_list] : quadrature_spds_map_)
  {
    auto& commondata_vec = quadrature_fluds_commondata_map_[quadrature];
    commondata_vec.resize(spds_list.size());
    for (std::size_t i = 0; i < spds_list.size(); ++i)
      tasks.push_back({spds_list[i].get(), &commondata_vec[i]});
  }

  if (tasks.empty())
    return;

  const auto hardware_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  const auto num_workers = std::min(tasks.size(), hardware_threads);

  log.Log() << program_timer.GetTimeString() << " Building " << label
            << " FLUDS common data for " << tasks.size() << " SPDS using " << num_workers
            << " worker threads.\n";
  const auto start_time = std::chrono::steady_clock::now();

  SPMD_ThreadPool pool(num_workers);
  std::atomic<std::size_t> next_index{0};
  pool.ExecuteBatch(
    [&](std::size_t /*thread_id*/)
    {
      std::size_t index;
      // memory_order_relaxed is sufficient here: workers only need an atomic claim of a
      // unique task index, and no other state is shared across iterations.
      while ((index = next_index.fetch_add(1, std::memory_order_relaxed)) < tasks.size())
      {
        const auto& task = tasks[index];
        *task.slot = std::make_unique<CommonDataT>(
          *task.spds, grid_nodal_mappings_, *discretization_);
      }
    });

  const auto elapsed_seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

  std::size_t max_slots = 0;
  std::size_t min_slots = std::numeric_limits<std::size_t>::max();
  for (const auto& task : tasks)
  {
    const auto n = static_cast<const CommonDataT&>(**task.slot).GetNumLocalPsiFaceNodeSlots();
    max_slots = std::max(max_slots, n);
    min_slots = std::min(min_slots, n);
  }

  log.Log() << program_timer.GetTimeString() << " Finished building " << label
            << " FLUDS common data. Elapsed time: " << elapsed_seconds
            << " seconds. Max local psi face-node slots: " << max_slots
            << ". Min local psi face-node slots: " << min_slots << ".\n";
}

} // namespace opensn

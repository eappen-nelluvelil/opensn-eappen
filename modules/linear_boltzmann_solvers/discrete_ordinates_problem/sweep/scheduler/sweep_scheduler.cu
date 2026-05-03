// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/sweep_scheduler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "caribou/main.hpp"
#include "caliper/cali.h"
#include <algorithm>
#include <thread>
#include <vector>

namespace opensn
{

void
SweepScheduler::ScheduleAlgoAAO(SweepChunk& sweep_chunk)
{
  CALI_CXX_MARK_SCOPE("SweepScheduler::ScheduleAlgoAAO");

  // copy phi and src moments to device
  auto aah_sweep_chunk = static_cast<AAHDSweepChunk&>(sweep_chunk);
  aah_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();

  // reset dependency counters and pre-post receives for all anglesets
  std::size_t num_anglesets = angle_agg_.GetNumAngleSets();
  execution_order_.resize(num_anglesets);
  for (std::size_t i = 0; i < num_anglesets; ++i)
  {
    auto aahd_angle_set = static_cast<AAHD_AngleSet*>(angle_agg_[i].get());
    aahd_angle_set->ResetDependencyCounter();
    aahd_angle_set->PrepostReceives();
    execution_order_[i] = i;
  }

  // assign sweep task to thread pool (but not execution yet)
  pool_.AssignTask(
    [this, &sweep_chunk](std::size_t i)
    {
      auto* aahd = static_cast<AAHD_AngleSet*>(angle_agg_[i].get());
      aahd->AngleSetAdvance(sweep_chunk, AngleSetStatus::EXECUTE);
    });

  // poll for readiness and launch threads
  while (!execution_order_.empty())
  {
    for (auto it = execution_order_.begin(); it != execution_order_.end();)
    {
      auto* angle_set = static_cast<AAHD_AngleSet*>(angle_agg_[*it].get());
      if (angle_set->IsReady())
      {
        pool_.Run(*it);
        std::swap(*it, execution_order_.back());
        execution_order_.pop_back();
      }
      else
      {
        ++it;
      }
    }
  }
  pool_.WaitAll();

  // wait for sends and receive of delayed data
  for (auto& angle_set : angle_agg_)
  {
    auto aahd_angle_set = static_cast<AAHD_AngleSet*>(angle_set.get());
    aahd_angle_set->WaitForDownstreamAndDelayed();
  }

  // copy phi and outflow data back to host
  aah_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();

  // reset all anglesets
  for (auto& angle_set : angle_agg_)
    angle_set->ResetSweepBuffers();
}

void
SweepScheduler::ScheduleAlgoAsyncFIFO(SweepChunk& sweep_chunk)
{
  CALI_CXX_MARK_SCOPE("SweepScheduler::ScheduleAlgoAsyncFIFO");

  auto& cbcd_sweep_chunk = static_cast<CBCDSweepChunk&>(sweep_chunk);
  cbcd_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();
  cbcd_sweep_chunk.RefreshCachedKernelArgs();

  auto& angle_sets = cbcd_sweep_chunk.GetAngleSets();
  const auto num_angle_sets = angle_sets.size();
  for (auto* angle_set : angle_sets)
    angle_set->ResetDependencyCounter();

  const auto num_workers = pool_.GetSize();
  cbcd_sweep_chunk.StartCommunicator(num_workers);
  pool_.ExecuteBatch(
    [num_workers, num_angle_sets, &angle_sets, &cbcd_sweep_chunk](std::size_t worker_id)
    {
      const auto chunk_size = (num_angle_sets + num_workers - 1) / num_workers;
      const auto begin = worker_id * chunk_size;
      const auto end = std::min(begin + chunk_size, num_angle_sets);

      std::vector<std::size_t> active_angle_set_ids;
      active_angle_set_ids.reserve(end - begin);
      for (std::size_t i = begin; i < end; ++i)
        active_angle_set_ids.push_back(i);

      while (not active_angle_set_ids.empty())
      {
        bool any_work_done = false;
        for (std::size_t i = 0; i < active_angle_set_ids.size();)
        {
          auto* angle_set = angle_sets[active_angle_set_ids[i]];
          if (angle_set->IsExecuted())
          {
            active_angle_set_ids[i] = active_angle_set_ids.back();
            active_angle_set_ids.pop_back();
            continue;
          }

          if (not angle_set->IsInitialized())
          {
            if (not angle_set->TryInitialize(cbcd_sweep_chunk))
            {
              ++i;
              continue;
            }
            any_work_done = true;
          }

          any_work_done |= angle_set->TryAdvanceOneStep(cbcd_sweep_chunk, worker_id);

          if (angle_set->IsExecuted())
          {
            active_angle_set_ids[i] = active_angle_set_ids.back();
            active_angle_set_ids.pop_back();
            continue;
          }

          ++i;
        }
        if (not any_work_done)
          std::this_thread::yield();
      }
    });

  cbcd_sweep_chunk.StopCommunicator();

  cbcd_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();

  for (auto* angle_set : angle_sets)
    angle_set->ResetSweepBuffers();

  for (const auto& [bid, bndry] : angle_agg_.GetSimBoundaries())
    bndry->ResetAnglesReadyStatus();
}

} // namespace opensn

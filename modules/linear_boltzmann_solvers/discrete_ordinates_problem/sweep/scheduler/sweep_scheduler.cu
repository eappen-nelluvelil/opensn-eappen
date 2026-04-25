// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/sweep_scheduler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "caribou/main.hpp"
#include "caliper/cali.h"
#include <thread>

namespace opensn
{

void
SweepScheduler::ScheduleAlgoAAO(SweepChunk& sweep_chunk)
{
  CALI_CXX_MARK_SCOPE("SweepScheduler::ScheduleAlgoAAO");

  // copy phi and src moments to device
  auto aah_sweep_chunk = static_cast<AAHDSweepChunk&>(sweep_chunk);
  int groupset_id = angle_agg_.GetGroupsetID();
  aah_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();
  aah_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, true);

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

  // copy boundary, phi and outflow data back to host
  aah_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, false);
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
  const int groupset_id = angle_agg_.GetGroupsetID();
  cbcd_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();
  cbcd_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, true);
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

      bool all_done = false;
      while (not all_done)
      {
        all_done = true;
        bool any_work_done = false;
        for (std::size_t i = begin; i < end; ++i)
        {
          auto* angle_set = angle_sets[i];
          if (angle_set->IsExecuted())
            continue;
          all_done = false;
          if (not angle_set->IsInitialized())
          {
            any_work_done |= angle_set->TryInitialize(cbcd_sweep_chunk);
            continue;
          }
          any_work_done |= angle_set->TryAdvanceOneStep(cbcd_sweep_chunk, worker_id);
        }
        if ((not all_done) and (not any_work_done))
          std::this_thread::yield();
      }
    });

  cbcd_sweep_chunk.StopCommunicator();

  cbcd_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, false);
  cbcd_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();

  for (auto* angle_set : angle_sets)
    angle_set->ResetSweepBuffers();
}

} // namespace opensn

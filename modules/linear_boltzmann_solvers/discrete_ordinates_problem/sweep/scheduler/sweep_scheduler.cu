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
  // copy phi and src moments to device
  auto& aah_sweep_chunk = dynamic_cast<AAHDSweepChunk&>(sweep_chunk);
  int groupset_id = angle_agg_.GetGroupsetID();
  aah_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();
  aah_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, true);

  // reset dependency counters and pre-post receives for all anglesets
  std::size_t num_anglesets = angle_agg_.GetNumAngleSets();
  execution_order_.resize(num_anglesets);
  for (std::size_t i = 0; i < num_anglesets; ++i)
  {
    auto* aahd_angle_set = dynamic_cast<AAHD_AngleSet*>(angle_agg_[i].get());
    aahd_angle_set->ResetDependencyCounter();
    aahd_angle_set->PrepostReceives();
    execution_order_[i] = i;
  }

  // assign sweep task to thread pool (but not execution yet)
  pool_.AssignTask(
    [this, &sweep_chunk](std::size_t i)
    {
      auto* aahd = dynamic_cast<AAHD_AngleSet*>(angle_agg_[i].get());
      aahd->AngleSetAdvance(sweep_chunk, AngleSetStatus::EXECUTE);
    });

  // poll for readiness and launch threads
  while (!execution_order_.empty())
  {
    for (auto it = execution_order_.begin(); it != execution_order_.end();)
    {
      auto* angle_set = dynamic_cast<AAHD_AngleSet*>(angle_agg_[*it].get());
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
    auto* aahd_angle_set = dynamic_cast<AAHD_AngleSet*>(angle_set.get());
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

  auto& cbcd_sweep_chunk = dynamic_cast<CBCDSweepChunk&>(sweep_chunk);
  cbcd_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();
  cbcd_sweep_chunk.RefreshKernelArguments();

  auto& angle_sets = cbcd_sweep_chunk.GetAngleSets();
  const auto num_angle_sets = angle_sets.size();
  for (auto* angle_set : angle_sets)
    angle_set->ResetSweepDependencies();

  const auto num_workers = pool_.GetSize();
  auto* profiler = cbcd_sweep_chunk.GetProfiler();
  if (profiler)
    profiler->BeginSweep(num_workers);
  cbcd_sweep_chunk.StartCommunicator(num_workers);
  pool_.ExecuteBatch(
    [num_workers, num_angle_sets, &angle_sets, &cbcd_sweep_chunk, profiler](std::size_t worker_id)
    {
      if (profiler)
        profiler->RecordWorkerStart(worker_id, CBCDProfiler::Clock::now());
      std::vector<std::size_t> active_angle_set_ids;
      active_angle_set_ids.reserve((num_angle_sets + num_workers - 1) / num_workers);
      std::vector<CBCD_AngleSet*> ready_angle_sets;
      ready_angle_sets.reserve((num_angle_sets + num_workers - 1) / num_workers);
      std::vector<CBCD_AngleSet*> completed_angle_sets;
      completed_angle_sets.reserve((num_angle_sets + num_workers - 1) / num_workers);
      for (std::size_t i = worker_id; i < num_angle_sets; i += num_workers)
        active_angle_set_ids.push_back(i);

      while (not active_angle_set_ids.empty())
      {
        bool any_work_done = cbcd_sweep_chunk.PollWorkerDispatches(worker_id);
        ready_angle_sets.clear();
        completed_angle_sets.clear();
        for (std::size_t i = 0; i < active_angle_set_ids.size();)
        {
          auto* angle_set = angle_sets[active_angle_set_ids[i]];
          if (angle_set->IsSweepComplete())
          {
            active_angle_set_ids[i] = active_angle_set_ids.back();
            active_angle_set_ids.pop_back();
            continue;
          }

          if (not angle_set->IsSweepInitialized())
          {
            if (not angle_set->TryInitialize(cbcd_sweep_chunk))
            {
              ++i;
              continue;
            }
            any_work_done = true;
          }

          any_work_done |= angle_set->TryAdvanceOneStep(
            cbcd_sweep_chunk, cbcd_sweep_chunk.IsDispatchComplete(angle_set->GetID()));

          if (angle_set->IsSweepComplete())
          {
            active_angle_set_ids[i] = active_angle_set_ids.back();
            active_angle_set_ids.pop_back();
            continue;
          }

          if (angle_set->HasReadyBatch())
            ready_angle_sets.push_back(angle_set);
          if (angle_set->HasCompletedBatch())
            completed_angle_sets.push_back(angle_set);

          ++i;
        }
        for (auto* angle_set : completed_angle_sets)
        {
          CALI_CXX_MARK_SCOPE("CBCD_AngleSet::FlushBatch");
          any_work_done |= angle_set->PublishCompletedBatch(cbcd_sweep_chunk, worker_id);
        }
        any_work_done |= cbcd_sweep_chunk.DispatchReadyAngleSets(worker_id, ready_angle_sets);
        if (any_work_done)
        {
          if (profiler)
            profiler->RecordWorkerIdleEnd(worker_id, CBCDProfiler::Clock::now());
        }
        else
        {
          if (profiler)
          {
            profiler->RecordWorkerIdleStart(worker_id, CBCDProfiler::Clock::now());
            profiler->RecordWorkerYield(worker_id);
          }
          std::this_thread::yield();
        }
      }
      if (profiler)
        profiler->RecordWorkerStop(worker_id, CBCDProfiler::Clock::now());
    });

  CBCDProfiler::TimePoint communicator_stop_start;
  if (profiler)
    communicator_stop_start = CBCDProfiler::Clock::now();
  cbcd_sweep_chunk.StopCommunicator();
  if (profiler)
    profiler->RecordCommunicatorDrain(
      CBCDProfiler::ElapsedNanoseconds(communicator_stop_start, CBCDProfiler::Clock::now()));

  CBCDProfiler::TimePoint barrier_start;
  if (profiler)
    barrier_start = CBCDProfiler::Clock::now();
  opensn::mpi_comm.barrier();
  if (profiler)
    profiler->RecordEndBarrier(
      CBCDProfiler::ElapsedNanoseconds(barrier_start, CBCDProfiler::Clock::now()));

  cbcd_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();

  for (auto* angle_set : angle_sets)
    angle_set->ResetSweepBuffers();

  if (profiler)
    profiler->FinishSweep();
}

} // namespace opensn

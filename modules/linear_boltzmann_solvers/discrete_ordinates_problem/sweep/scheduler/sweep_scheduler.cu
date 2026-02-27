// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/sweep_scheduler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "caribou/main.hpp"
#include "caliper/cali.h"
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

  // allocate threads for each angleset
  std::size_t num_anglesets = angle_agg_.GetNumAngleSets();

  // set the latches for all anglesets
  for (auto& angle_set : angle_agg_)
  {
    auto aahd_angle_set = static_cast<AAHD_AngleSet*>(angle_set.get());
    aahd_angle_set->SetStartingLatch();
  }

  // launch threads
  pool_.run(
    [this, &sweep_chunk](std::size_t i)
    {
      auto* aahd = static_cast<AAHD_AngleSet*>(angle_agg_[i].get());
      aahd->AngleSetAdvance(sweep_chunk, AngleSetStatus::EXECUTE);
    });

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

  auto& cbcd_chunk = static_cast<CBCDSweepChunk&>(sweep_chunk);

  // Copy phi and source moments to device
  cbcd_chunk.GetProblem().CopyPhiAndSrcToDevice();

  const auto& angle_sets = cbcd_chunk.GetAngleSets();
  const size_t num_angle_sets = angle_sets.size();

  // Set sweep chunk reference and initialize latches for all angle sets
  for (auto* as : angle_sets)
  {
    as->SetSweepChunk(&cbcd_chunk);
    as->SetStartingLatch();
  }

  // Start aggregated communication thread
  cbcd_chunk.StartCommunicator();

  const auto num_workers = num_workers_;
  pool_.run(
    [&angle_sets, num_angle_sets, num_workers](std::size_t worker_id)
    {
      // Partition angle sets among workers (contiguous ranges)
      const size_t chunk_size = (num_angle_sets + num_workers - 1) / num_workers;
      const size_t begin = worker_id * chunk_size;
      const size_t end = std::min(begin + chunk_size, num_angle_sets);

      // Process until all assigned angle sets are finished
      bool all_done = false;
      while (not all_done)
      {
        all_done = true;
        bool any_work = false;

        for (size_t i = begin; i < end; ++i)
        {
          auto* as = angle_sets[i];

          if (as->IsFinished())
            continue;
          all_done = false;

          // Try initialization (non-blocking; waits for predecessor latches)
          if (not as->IsInitialized())
          {
            any_work |= as->TryInitialize();
            continue;
          }

          // One step of work: poll kernel, receive data, launch kernel
          any_work |= as->TryAdvanceOneStep();
        }

        if (not any_work and not all_done)
          std::this_thread::yield();
      }
    });

  // Flush + join communication thread
  cbcd_chunk.StopCommunicator();

  // Copy phi and outflow data back to host
  cbcd_chunk.GetProblem().CopyPhiAndOutflowBackToHost();
  opensn::mpi_comm.barrier();

  // Reset all angle sets
  for (auto* as : angle_sets)
    as->ResetSweepBuffers();

  for (const auto& [bid, bndry] : angle_agg_.GetSimBoundaries())
    bndry->ResetAnglesReadyStatus();
}

} // namespace opensn

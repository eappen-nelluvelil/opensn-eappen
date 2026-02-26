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

  // Set sweep chunk reference and initialize latches for all angle sets
  for (auto* as : angle_sets)
  {
    as->SetSweepChunk(&cbcd_chunk);
    as->SetStartingLatch();
  }

  // Start aggregated communication thread
  cbcd_chunk.StartCommunicator();

  // Launch one thread per angle set via SPMD_ThreadPool
  pool_.run(
    [&sweep_chunk, &angle_sets](std::size_t i)
    { angle_sets[i]->AngleSetAdvance(sweep_chunk, AngleSetStatus::EXECUTE); });

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

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include <memory>

namespace opensn
{

class CBCD_AggregatedCommunicator;
class CBCD_FLUDS;

/**
 * Sweep chunk for cell-by-cell device (CBCD) transport sweep.
 *
 * Owns the aggregated communicator and the cached GPU kernel launch parameters
 * for every angle set in the groupset.  The Sweep(angle_set, num_cells) method
 * issues a single GPU kernel launch with pre-computed arguments — no per-call
 * argument reconstruction.
 *
 * The constructor also computes the worst-case MPI receive buffer size by
 * walking all incoming non-local face data across angle sets, so the
 * aggregated communicator's receive buffer is right-sized without guessing.
 */
class CBCDSweepChunk : public SweepChunk
{
public:
  CBCDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);
  ~CBCDSweepChunk();

  DiscreteOrdinatesProblem& GetProblem() const { return problem_; }
  const LBSGroupset& GetGroupset() const { return groupset_; }
  unsigned int GetGroupsetGroupIndex() const { return groupset_.first_group; }

  /// Launch the GPU sweep kernel.  Cell IDs must already be in the FLUDS
  /// MappedHostVector (written by CBCD_AngleSet::TryAdvanceOneStep step C).
  using SweepChunk::Sweep;
  void Sweep(CBCD_AngleSet& angle_set, unsigned int num_ready_cells);

  const std::vector<CBCD_AngleSet*>& GetAngleSets() const { return angle_sets_; }

  void StartCommunicator();
  void StopCommunicator();
  CBCD_AggregatedCommunicator& GetAggregatedCommunicator();

private:
  /// Cached kernel launch parameters (constant after construction).
  struct CachedKernelParams
  {
    gpu_kernel::Arguments<gpu_kernel::SweepType::CBC> args;
    ::dim3 block_size;
    unsigned int grid_size_x;
    double* device_saved_psi;

    CachedKernelParams(gpu_kernel::Arguments<gpu_kernel::SweepType::CBC> a,
                       ::dim3 bs,
                       unsigned int gx,
                       double* sp)
      : args(a), block_size(bs), grid_size_x(gx), device_saved_psi(sp)
    {
    }
  };

  DiscreteOrdinatesProblem& problem_;
  std::vector<CBCD_AngleSet*> angle_sets_;
  std::vector<CBCD_FLUDS*> fluds_list_;
  std::unique_ptr<CBCD_AggregatedCommunicator> agg_comm_;
  std::vector<CachedKernelParams> cached_kernel_params_;
};

} // namespace opensn

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

/// CBC sweep chunk for device.
class CBCDSweepChunk : public SweepChunk
{
public:
  CBCDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  ~CBCDSweepChunk();

  DiscreteOrdinatesProblem& GetProblem() const { return problem_; }

  const LBSGroupset& GetGroupset() const { return groupset_; }

  unsigned int GetGroupsetGroupIndex() const { return groupset_.first_group; }

  /// Launch the GPU sweep kernel for the given angle set.
  /// Cell IDs must already be written to the FLUDS MappedHostVector by the caller.
  void GPUSweep(CBCD_AngleSet& angle_set, unsigned int num_ready_cells);

  const std::vector<CBCD_AngleSet*>& GetAngleSets() const { return angle_sets_; }

  /// Start the aggregated communication thread.
  void StartCommunicator();

  /// Stop the aggregated communication thread (flush + join).
  void StopCommunicator();

  /// Get a reference to the aggregated communicator.
  CBCD_AggregatedCommunicator& GetAggregatedCommunicator();

private:
  /// Pre-computed kernel launch parameters cached per angle set.
  struct CachedKernelParams
  {
    cbc_gpu_kernel::CBC_Arguments args;
    ::dim3 block_size;
    unsigned int grid_size_x;
    double* device_saved_psi;

    CachedKernelParams(cbc_gpu_kernel::CBC_Arguments a,
                       ::dim3 bs,
                       unsigned int gx,
                       double* sp)
      : args(a), block_size(bs), grid_size_x(gx), device_saved_psi(sp)
    {
    }
  };

  DiscreteOrdinatesProblem& problem_;
  std::vector<CBCD_AngleSet*> angle_sets_;
  std::unique_ptr<CBCD_AggregatedCommunicator> agg_comm_;
  /// Cached kernel arguments and launch dimensions per angle set (avoids re-construction each call).
  std::vector<CachedKernelParams> cached_kernel_params_;
};

} // namespace opensn

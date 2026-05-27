// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"
#include "caribou/main.hpp"

namespace crb = caribou;

namespace opensn
{

/// CBCD sweep chunk.
class CBCDSweepChunk : public SweepChunk
{
public:
  CBCDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  ~CBCDSweepChunk() override;

  DiscreteOrdinatesProblem& GetProblem() const { return problem_; }

  const LBSGroupset& GetGroupset() const { return groupset_; }

  unsigned int GetGroupsetGroupIndex() const { return groupset_.first_group; }

  const CellLBSView& GetCellTransportView(std::uint64_t cell_local_id) const
  {
    return cell_transport_views_[cell_local_id];
  }

  const std::vector<CBCD_AngleSet*>& GetAngleSets() const { return angle_sets_; }

  void StartCommunicator(std::size_t num_workers);

  void StopCommunicator();

  void RefreshCachedKernelArgs();

  using SweepChunk::Sweep;
  /// Launch one ready-cell batch.
  void Sweep(std::uint32_t num_ready_cells,
             std::size_t angle_set_id,
             const std::uint32_t* local_cell_ids);

private:
  struct CachedKernelParams
  {
    gpu_kernel::Arguments<SweepKind::CBC> args;
    crb::Dim3 block_size;
    unsigned int grid_size_x;
    CBCD_FLUDS* fluds;
    double* device_saved_psi;
  };
  DiscreteOrdinatesProblem& problem_;
  std::unique_ptr<CBCD_AsynchronousCommunicator> async_comm_;
  std::vector<CBCD_AngleSet*> angle_sets_;
  std::vector<CachedKernelParams> cached_params_;
};

} // namespace opensn

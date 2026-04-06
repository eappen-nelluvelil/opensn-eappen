// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"

namespace opensn
{

/**
 * Device-side CBC sweep chunk orchestrating GPU kernel dispatch and communication.
 *
 * Owns the aggregated CBCD_AsynchronousCommunicator and coordinates all
 * CBCD_AngleSet instances for a given groupset. Caches per-angle-set kernel
 * launch parameters (CachedKernelParams) for cache locality, avoiding
 * repeated parameter assembly on each kernel launch.
 */
class CBCDSweepChunk : public SweepChunk
{
public:
  CBCDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);
  ~CBCDSweepChunk() override;

  /// Return the owning discrete ordinates problem.
  DiscreteOrdinatesProblem& GetProblem() const { return problem_; }

  /// Return the groupset associated with this sweep chunk.
  const LBSGroupset& GetGroupset() const { return groupset_; }

  /// Return the first group index in the groupset.
  unsigned int GetGroupsetGroupIndex() const { return groupset_.first_group; }

  /// Return the cell transport view for a given local cell ID.
  const CellLBSView& GetCellTransportView(std::uint64_t cell_local_id) const
  {
    return cell_transport_views_[cell_local_id];
  }

  /// Return the list of angle sets managed by this chunk.
  const std::vector<CBCD_AngleSet*>& GetAngleSets() const { return angle_sets_; }

  /// Start the aggregated communicator thread.
  void StartCommunicator();

  /// Stop the aggregated communicator thread.
  void StopCommunicator();

  using SweepChunk::Sweep;

  /**
   * Launch the GPU sweep kernel for a batch of ready cells.
   *
   * \param num_ready_cells number of cells in the ready batch
   * \param angle_set_id index of the angle set to sweep
   */
  void Sweep(std::uint32_t num_ready_cells, size_t angle_set_id);

private:
  /**
   * Cached launch data for one angle set.
   *
   * Stored contiguously in cached_params_ for cache locality across
   * successive kernel launches.
   */
  struct CachedKernelParams
  {
    /// Pre-assembled GPU kernel arguments.
    gpu_kernel::Arguments<gpu_kernel::SweepType::CBC> args;
    /// CUDA block dimensions.
    ::dim3 block_size;
    /// CUDA grid X dimension.
    unsigned int grid_size_x;
    /// Pointer to the angle set's FLUDS.
    CBCD_FLUDS* fluds;
    /// Device pointer to saved angular flux storage.
    double* device_saved_psi;
  };

  /// Owning reference to the discrete ordinates problem.
  DiscreteOrdinatesProblem& problem_;
  /// Aggregated communicator owned by this sweep chunk.
  std::unique_ptr<CBCD_AsynchronousCommunicator> async_comm_;
  /// Angle sets managed by this sweep chunk.
  std::vector<CBCD_AngleSet*> angle_sets_;
  /// Per-angle-set cached kernel launch parameters.
  std::vector<CachedKernelParams> cached_params_;
};

} // namespace opensn

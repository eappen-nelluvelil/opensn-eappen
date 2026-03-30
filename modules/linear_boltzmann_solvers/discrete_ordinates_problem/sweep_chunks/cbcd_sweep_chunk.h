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
 * Cached GPU launch state and communicator owner for one CBCD groupset.
 *
 * The class owns the aggregated communicator and precomputed kernel-launch
 * arguments for each angle set. It also sizes the communicator receive buffer
 * from the grouped incoming-face topology so runtime communication avoids
 * guesswork.
 */
class CBCDSweepChunk : public SweepChunk
{
public:
  /// Construct the CBCD sweep chunk for one groupset.
  ///
  /// \param problem Owning discrete ordinates problem.
  /// \param groupset Groupset served by this sweep chunk.
  CBCDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);
  ~CBCDSweepChunk();

  DiscreteOrdinatesProblem& GetProblem() const { return problem_; }
  const LBSGroupset& GetGroupset() const { return groupset_; }
  unsigned int GetGroupsetGroupIndex() const { return groupset_.first_group; }

  /// Launch the CBCD GPU sweep kernel for a ready cell batch.
  ///
  /// \param angle_set Angle set to launch.
  /// \param num_ready_cells Number of ready cells in the FLUDS work list.
  using SweepChunk::Sweep;
  void Sweep(CBCD_AngleSet& angle_set, unsigned int num_ready_cells);

  const std::vector<CBCD_AngleSet*>& GetAngleSets() const { return angle_sets_; }

  /// Start the aggregated communicator thread.
  void StartCommunicator();
  /// Stop the aggregated communicator thread.
  void StopCommunicator();
  /// Queue saved-angular-flux downloads and scatter them into destination psi.
  void CopySavedPsiToDestinationPsi();
  /// Return the aggregated communicator.
  CBCD_AggregatedCommunicator& GetAggregatedCommunicator();

private:
  /// Cached launch data for one angle set.
  struct CachedKernelParams
  {
    /// Packed kernel arguments.
    gpu_kernel::Arguments<gpu_kernel::SweepType::CBC> args;
    /// CUDA/HIP thread-block dimensions.
    ::dim3 block_size;
    /// X grid dimension derived from the stride.
    unsigned int grid_size_x;
    /// Saved-psi device pointer.
    double* device_saved_psi;

    CachedKernelParams(gpu_kernel::Arguments<gpu_kernel::SweepType::CBC> a,
                       ::dim3 bs,
                       unsigned int gx,
                       double* sp)
      : args(a), block_size(bs), grid_size_x(gx), device_saved_psi(sp)
    {
    }
  };

  /// Owning discrete ordinates problem.
  DiscreteOrdinatesProblem& problem_;
  /// Angle sets served by this sweep chunk.
  std::vector<CBCD_AngleSet*> angle_sets_;
  /// FLUDS instances paired with angle_sets_.
  std::vector<CBCD_FLUDS*> fluds_list_;
  /// Aggregated communicator for this groupset.
  std::unique_ptr<CBCD_AggregatedCommunicator> agg_comm_;
  /// Cached launch data indexed by angle-set ID.
  std::vector<CachedKernelParams> cached_kernel_params_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/profiling/cbcd_profiler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include "caribou/main.hpp"

namespace crb = caribou;

namespace opensn
{

/// CBCD sweep chunk.
class CBCDSweepChunk : public SweepChunk
{
public:
  /// Build persistent kernel launches and the groupset-wide communicator.
  CBCDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  /// Stop the communicator before destroying angle-set storage.
  ~CBCDSweepChunk() override;

  /// Return the owning transport problem.
  DiscreteOrdinatesProblem& GetProblem() const { return problem_; }

  /// Return the active groupset.
  const LBSGroupset& GetGroupset() const { return groupset_; }

  /// Return the first global group index in the active groupset.
  unsigned int GetGroupsetGroupIndex() const { return groupset_.first_group; }

  /// Return transport metadata for one local cell.
  const CellLBSView& GetCellTransportView(std::uint64_t cell_local_id) const
  {
    return cell_transport_views_[cell_local_id];
  }

  /// Return CBCD angle sets in scheduler order.
  const std::vector<CBCD_AngleSet*>& GetAngleSets() const { return angle_sets_; }

  /// Start the MPI progress thread and configure worker-owned queues.
  void StartCommunicator(std::size_t num_workers);

  /// Drain and stop the MPI progress thread.
  void StopCommunicator();

  /// Exchange the newly computed lagged nonlocal fluxes after the DAG sweep.
  void ExchangeDelayedPsi();

  /// Refresh problem-dependent arguments cached for each angle set.
  void RefreshKernelArguments();

  /// Return optional rank-local CBCD instrumentation.
  CBCDProfiler* GetProfiler() const { return profiler_.get(); }

  using SweepChunk::Sweep;
  /// Launch one ready-cell batch.
  void Sweep(std::uint32_t num_ready_cells,
             std::size_t angle_set_id,
             const std::uint32_t* local_cell_ids);

private:
  struct KernelLaunch
  {
    /// Persistent kernel arguments refreshed when problem vectors change.
    gpu_kernel::Arguments<SweepKind::CBC> arguments;
    /// Fixed block geometry and stride-axis grid extent.
    crb::Dim3 threads_per_block;
    unsigned int num_stride_blocks;
    /// Owning FLUDS and optional saved-psi device storage.
    CBCD_FLUDS* fluds;
    double* device_saved_psi;
    /// Whether this launch needs lagged-bank pointer routing.
    bool has_delayed_fluxes;
  };

  struct DelayedPeerBuffer
  {
    struct Slice
    {
      CBCD_FLUDS* fluds = nullptr;
      std::size_t peer_index = 0;
      std::size_t offset = 0;
      std::size_t size = 0;
    };

    /// Global MPI partition and its rank in the applicable local communicator.
    int partition = -1;
    int communicator_rank = -1;
    /// Active angle-set slices into the header-free payload.
    std::vector<Slice> slices;
    /// Persistent payload storage.
    std::vector<double> values;
  };
  /// Owning problem and groupset-wide aggregated communicator.
  DiscreteOrdinatesProblem& problem_;
  std::unique_ptr<CBCDProfiler> profiler_;
  std::unique_ptr<CBCD_AsynchronousCommunicator> async_comm_;
  /// Angle sets and their persistent kernel launches in scheduler order.
  std::vector<CBCD_AngleSet*> angle_sets_;
  std::vector<KernelLaunch> kernel_launches_;
  /// Exact, persistent lagged exchange plans grouped by peer.
  std::vector<DelayedPeerBuffer> delayed_receive_peers_;
  std::vector<DelayedPeerBuffer> delayed_send_peers_;
  std::vector<mpi::Request> delayed_receive_requests_;
  std::vector<mpi::Request> delayed_send_requests_;
  int delayed_mpi_tag_ = 0;
  /// Whether any angle set requires a delayed-state epoch.
  bool has_delayed_fluxes_ = false;
};

} // namespace opensn

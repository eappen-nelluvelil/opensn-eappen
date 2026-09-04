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
#include "caribou/main.hpp"
#include <span>

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

  /// Refresh problem-dependent arguments cached for each angle set.
  void RefreshKernelArguments();

  /// Poll fused dispatches owned by one scheduler worker.
  bool PollWorkerDispatches(std::size_t worker_id);

  /// Return whether an angle set's most recent dispatch has completed.
  bool IsDispatchComplete(std::size_t angle_set_id) const;

  /// Launch compatible ready batches owned by one scheduler worker.
  bool DispatchReadyAngleSets(std::size_t worker_id, std::span<CBCD_AngleSet*> ready_angle_sets);

  /// Return optional rank-local CBCD instrumentation.
  CBCDProfiler* GetProfiler() const { return profiler_.get(); }

  using SweepChunk::Sweep;
  /// Launch one ready-cell batch.
  void
  Sweep(std::uint32_t num_ready_cells, std::size_t angle_set_id, std::uint32_t* local_cell_ids);

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
  };

  struct DispatchState
  {
    DispatchState(std::size_t stride, crb::Dim3 threads, unsigned int stride_blocks);

    std::size_t stride_size;
    crb::Dim3 threads_per_block;
    unsigned int num_stride_blocks;
    crb::Stream stream;
    crb::HostVector<gpu_kernel::CBCDBatchDescriptor> host_batches;
    crb::DeviceMemory<gpu_kernel::CBCDBatchDescriptor> device_batches;
    std::vector<CBCD_AngleSet*> ready_angle_sets;
    std::vector<std::size_t> active_angle_set_ids;
    std::size_t angle_set_capacity = 0;
    bool active = false;
  };

  enum class DispatchKind : std::uint8_t
  {
    NONE,
    SINGLE,
    FUSED
  };

  struct AngleSetDispatchStatus
  {
    DispatchKind kind = DispatchKind::NONE;
    bool complete = true;
  };

  void ConfigureWorkerDispatches(std::size_t num_workers);
  void LaunchSingleBatch(std::size_t worker_id,
                         std::size_t angle_set_id,
                         std::span<std::uint32_t> local_cell_ids);
  void LaunchFusedBatch(std::size_t worker_id, DispatchState& dispatch);
  /// Owning problem and groupset-wide aggregated communicator.
  DiscreteOrdinatesProblem& problem_;
  std::unique_ptr<CBCDProfiler> profiler_;
  std::unique_ptr<CBCD_AsynchronousCommunicator> async_comm_;
  /// Angle sets and their persistent kernel launches in scheduler order.
  std::vector<CBCD_AngleSet*> angle_sets_;
  std::vector<KernelLaunch> kernel_launches_;
  std::vector<gpu_kernel::CBCDLaunchData> host_launch_data_;
  crb::DeviceMemory<gpu_kernel::CBCDLaunchData> device_launch_data_;
  std::vector<std::unique_ptr<DispatchState>> dispatch_storage_;
  std::vector<std::vector<DispatchState*>> worker_dispatches_;
  std::vector<std::vector<std::size_t>> worker_angle_set_ids_;
  std::vector<DispatchState*> angle_set_dispatches_;
  std::vector<AngleSetDispatchStatus> angle_set_dispatch_status_;
  std::size_t configured_workers_ = 0;
};

} // namespace opensn

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
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/cbcd_batch.h"
#include "caribou/main.hpp"

namespace crb = caribou;

namespace opensn
{

/// Worker-owned descriptors remain immutable until their stream completes.
struct CBCDWorkerLaunch
{
  CBCDWorkerLaunch() = default;
  CBCDWorkerLaunch(const CBCDWorkerLaunch&) = delete;
  CBCDWorkerLaunch& operator=(const CBCDWorkerLaunch&) = delete;
  ~CBCDWorkerLaunch() { stream.synchronize(); }

  crb::Stream stream;
  crb::MappedHostVector<gpu_kernel::CBCDBatch> batches;
  bool in_flight = false;
  bool completed = false;

  void Poll()
  {
    completed = in_flight and stream.is_completed();
    if (completed)
    {
      in_flight = false;
      batches.clear();
    }
  }
};

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

  /// Return transport metadata for one local cell.
  const CellLBSView& GetCellTransportView(std::uint64_t cell_local_id) const
  {
    return cell_transport_views_[cell_local_id];
  }

  /// Return CBCD angle sets in scheduler order.
  const std::vector<CBCD_AngleSet*>& GetAngleSets() const { return angle_sets_; }

  /// Start a sweep on the reusable MPI progress thread and configure worker-owned queues.
  void StartCommunicator(std::size_t num_workers);

  /// Drain published work and wait for the MPI progress thread to become idle.
  void StopCommunicator();

  /// Refresh problem-dependent arguments cached for each angle set.
  void RefreshKernelArguments();

  /// Return optional rank-local CBCD instrumentation.
  CBCDProfiler* GetProfiler() const { return profiler_.get(); }

  CBCDWorkerLaunch* GetWorkerLaunch(std::size_t worker_id) const
  {
    return worker_launches_.empty() ? nullptr : worker_launches_[worker_id].get();
  }
  void LaunchWorkerBatch(std::size_t worker_id);

  using SweepChunk::Sweep;
  /// Launch one ready-cell batch.
  void Sweep(std::uint32_t num_ready_cells,
             std::size_t angle_set_id,
             const std::uint32_t* local_cell_ids,
             CBCDWorkerLaunch* worker_launch = nullptr);

private:
  struct KernelLaunch
  {
    /// Persistent kernel arguments refreshed when problem vectors change.
    gpu_kernel::Arguments<SweepKind::CBC> arguments;
    /// Fixed block geometry and stride-axis grid extent.
    crb::Dim3 threads_per_block;
    unsigned int num_stride_blocks = 0;
    /// Owning FLUDS and optional saved-psi device storage.
    CBCD_FLUDS* fluds = nullptr;
    double* device_saved_psi = nullptr;
  };
  /// Owning problem and groupset-wide aggregated communicator.
  DiscreteOrdinatesProblem& problem_;
  std::unique_ptr<CBCDProfiler> profiler_;
  std::unique_ptr<CBCD_AsynchronousCommunicator> async_comm_;
  /// Angle sets and their persistent kernel launches in scheduler order.
  std::vector<CBCD_AngleSet*> angle_sets_;
  std::vector<KernelLaunch> kernel_launches_;
  bool fuse_worker_launches_ = false;
  crb::HostVector<gpu_kernel::Arguments<SweepKind::CBC>> batch_arguments_;
  crb::DeviceMemory<gpu_kernel::Arguments<SweepKind::CBC>> device_batch_arguments_;
  std::vector<std::unique_ptr<CBCDWorkerLaunch>> worker_launches_;
};

} // namespace opensn

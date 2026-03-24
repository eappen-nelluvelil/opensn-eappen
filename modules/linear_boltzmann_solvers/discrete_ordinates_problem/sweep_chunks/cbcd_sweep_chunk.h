// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"
#include "caribou/main.hpp"
#include <memory>

namespace crb = caribou;

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

  const CellLBSView& GetCellTransportView(std::uint64_t cell_local_id) const
  {
    return cell_transport_views_[cell_local_id];
  }

  using SweepChunk::Sweep;
  void Sweep(CBCD_AngleSet& angle_set, const std::vector<std::uint32_t>& cell_local_ids);

  const std::vector<CBCD_AngleSet*>& GetAngleSets() const { return angle_sets_; }

  const std::vector<CBCD_FLUDS*>& GetFLUDSList() const { return fluds_list_; }

  const std::vector<crb::Stream*>& GetStreamsList() const { return streams_list_; }

  /// Set up per-worker aggregated communicators.
  /// Each worker thread in the SPMD_ThreadPool gets its own CBCD_AggregatedCommunicator
  /// handling only the angle sets assigned to that worker.
  /// Must be called before StartCommunicators() and after the pool size is known.
  /// Safe to call multiple times — only sets up on the first call, or if num_workers changes.
  void SetupPerWorkerCommunicators(size_t num_workers);

  /// Start all per-worker aggregated communication threads.
  void StartCommunicators();

  /// Stop all per-worker aggregated communication threads (flush + join).
  void StopCommunicators();

  /// Get a reference to the aggregated communicator for a given worker.
  CBCD_AggregatedCommunicator& GetAggregatedCommunicator(size_t worker_id);

private:
  DiscreteOrdinatesProblem& problem_;
  std::vector<CBCD_AngleSet*> angle_sets_;
  std::vector<CBCD_FLUDS*> fluds_list_;
  std::vector<crb::Stream*> streams_list_;
  std::vector<gpu_kernel::Arguments<gpu_kernel::SweepType::CBC>> kernel_args_list_;
  std::vector<::dim3> block_sizes_;
  std::vector<unsigned int> grid_size_x_list_;
  std::unique_ptr<CBCD_AggregatedCommunicator> agg_comm_;
  /// Per-worker aggregated communicators.
  std::vector<std::unique_ptr<CBCD_AggregatedCommunicator>> agg_comms_;

  /// Pre-computed worst-case receive buffer size (computed once in constructor).
  size_t max_message_size_ = 0;

  /// Number of workers for which communicators have been set up (0 = not yet set up).
  size_t setup_num_workers_ = 0;
};

} // namespace opensn

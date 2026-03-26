// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "caribou/main.hpp"
#include <atomic>
#include <cstdint>
#include <set>

namespace crb = caribou;

namespace opensn
{

class CBC_SPDS;
class CBCDSweepChunk;
class CBCD_AggregatedCommunicator;

/**
 * Cell-by-cell device (CBCD) angle set.
 *
 * Drives a single sweep direction (or small group of directions) through the
 * SPDS task DAG on the GPU.  Supports two execution modes:
 *
 *  1. **Blocking** — AngleSetAdvance() spins until all tasks complete.
 *  2. **Cooperative** — a bounded worker pool calls TryInitialize() and
 *     TryAdvanceOneStep() in a round-robin loop across many angle sets,
 *     overlapping GPU compute with MPI communication.
 *
 * Each call to TryAdvanceOneStep() performs up to five non-blocking steps:
 *   A. Poll the GPU stream for kernel completion → update task dependencies.
 *   B. Drain received MPI data from the aggregated communicator.
 *   C. Launch the next GPU kernel for newly-ready tasks.
 *   D. Pack and enqueue outgoing face data (overlapped with the GPU kernel).
 *   E. Check sweep completion and finalize.
 *
 * For reflecting-boundary problems, an early dependency countdown notifies
 * following angle sets as soon as all reflecting-boundary cells complete,
 * rather than waiting for the full sweep to finish.
 */
class CBCD_AngleSet : public AngleSet
{
public:
  CBCD_AngleSet(size_t id,
                size_t num_groups,
                const SPDS& spds,
                std::shared_ptr<FLUDS>& fluds,
                const std::vector<size_t>& angle_indices,
                std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                const MPICommunicatorSet& comm_set);

  ~CBCD_AngleSet();

  crb::Stream& GetStream() { return stream_; }
  std::uint32_t* GetDeviceAngleIndices() { return device_angle_indices_.get(); }

  void SetSweepChunk(CBCDSweepChunk* sweep_chunk) { cbcd_sweep_chunk_ = sweep_chunk; }
  void SetAggregatedCommunicator(CBCD_AggregatedCommunicator* agg_comm) { agg_comm_ = agg_comm; }
  CBCD_AggregatedCommunicator* GetAggregatedCommunicator() const { return agg_comm_; }

  void ResetDependencyCounter();
  void UpdateSweepDependencies(std::set<AngleSet*>& following_angle_sets) override;

  bool TryInitialize();
  bool TryAdvanceOneStep();

  bool IsFinished() const { return executed_; }
  bool IsInitialized() const { return initialized_; }

  AsynchronousCommunicator* GetCommunicator() override { return nullptr; }
  void InitializeDelayedUpstreamData() override {}
  int GetMaxBufferMessages() const override { return 0; }
  void SetMaxBufferMessages(int new_max) override {}

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus exec_status) override;
  AngleSetStatus FlushSendBuffers() override { return AngleSetStatus::MESSAGES_SENT; }
  void ResetSweepBuffers() override;
  bool ReceiveDelayedData() override { return true; }

  const double* PsiBoundary(uint64_t boundary_id,
                            unsigned int angle_num,
                            uint64_t cell_local_id,
                            unsigned int face_num,
                            unsigned int fi,
                            unsigned int g,
                            bool surface_source_active) override;

  double* PsiReflected(uint64_t boundary_id,
                       unsigned int angle_num,
                       uint64_t cell_local_id,
                       unsigned int face_num,
                       unsigned int fi) override;

private:
  void FinalizeSweep();

  /// Notify following angle sets that reflecting boundary data is ready.
  /// No-op if already notified or if the reflecting boundary count has not
  /// been reached.  Called from TryInitialize (zero-boundary fast path),
  /// TryAdvanceOneStep step D, and FinalizeSweep.
  void TryNotifyFollowers();

  crb::Stream stream_;
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;
  CBCD_FLUDS& cbcd_fluds_;
  CBCDSweepChunk* cbcd_sweep_chunk_;
  const CBC_SPDS& cbc_spds_;

  // -- CSR task DAG (built once, reused across sweeps) ----------------------

  std::vector<uint64_t> reference_ids_;      ///< Cell local ID per task.
  std::vector<uint32_t> successor_offsets_;   ///< CSR offset array (size N+1).
  std::vector<uint32_t> successor_data_;      ///< Flat successor task indices.
  std::vector<int> initial_deps_;             ///< Initial dependency counts per task.
  std::vector<int> remaining_deps_;           ///< Working copy, reset each sweep.
  std::vector<uint32_t> initial_ready_tasks_; ///< Zero-dependency tasks (constant).

  // -- Inter-angle-set synchronization --------------------------------------

  CBCD_AggregatedCommunicator* agg_comm_ = nullptr;
  std::size_t num_dependencies_ = 0;
  std::atomic<std::size_t> dependency_counter_{0};
  std::vector<CBCD_AngleSet*> following_angle_sets_;

  // -- Per-sweep working state ----------------------------------------------

  bool initialized_ = false;
  std::vector<uint32_t> ready_queue_;
  bool kernel_in_flight_ = false;
  std::vector<uint32_t> in_flight_task_indices_;
  std::vector<uint64_t> deferred_cell_ids_;
  size_t completed_count_ = 0;
  size_t total_tasks_ = 0;

  // -- Early dependency countdown for reflecting boundaries -----------------
  //
  // When this angle set has following angle sets (reflecting BCs), we track
  // how many reflecting-boundary tasks have completed.  Once all are done,
  // TryNotifyFollowers() fires the countdown so followers can begin init
  // before the full sweep finishes.
  //
  // is_reflecting_task_ is indexed by TASK index (not cell_local_id) and uses
  // vector<char> instead of vector<bool> to avoid bit-packing overhead.

  std::vector<char> is_reflecting_task_;        ///< 1 if task touches a reflecting boundary.
  size_t reflecting_tasks_completed_ = 0;       ///< Running count of completed reflecting tasks.
  size_t total_reflecting_tasks_ = 0;           ///< Total reflecting-boundary tasks (constant).
  bool followers_notified_ = false;             ///< True once TryNotifyFollowers() has fired.
};

} // namespace opensn

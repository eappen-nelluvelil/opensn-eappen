// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "caribou/main.hpp"
#include <latch>
#include <memory>
#include <set>
#include <unordered_set>

namespace crb = caribou;

namespace opensn
{

class CBC_SPDS;
class CBCDSweepChunk;
class CBCD_AggregatedCommunicator;

/// CBC angle set for device.
///
/// Supports two execution modes:
/// 1. **One-thread-per-angle-set** — call AngleSetAdvance() directly (blocking).
/// 2. **Cooperative scheduling** — a bounded worker pool calls TryInitialize() and
///    TryAdvanceOneStep() in a round-robin loop across many angle sets.
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

  std::vector<Task>& GetCurrentTaskList() { return current_task_list_; }

  void SetSweepChunk(CBCDSweepChunk* sweep_chunk) { cbcd_sweep_chunk_ = sweep_chunk; }

  /// Set the aggregated communicator pointer.
  void SetAggregatedCommunicator(CBCD_AggregatedCommunicator* agg_comm) { agg_comm_ = agg_comm; }

  /// Get the aggregated communicator pointer.
  CBCD_AggregatedCommunicator* GetAggregatedCommunicator() const { return agg_comm_; }

  /// Initialize the starting latch.
  void SetStartingLatch();

  /// Must be called after UpdateSweepDependencies and before launching threads.
  void UpdateSweepDependencies(std::set<AngleSet*>& following_angle_sets) override;

  /// Non-blocking initialization. Returns true when ready (latch satisfied).
  /// Returns false if waiting on predecessor angle sets (reflecting BCs).
  bool TryInitialize();

  /// Execute one step of work (poll kernel, receive data, launch kernel).
  /// Returns true if any work was done. Automatically finalizes when all tasks complete.
  bool TryAdvanceOneStep();

  /// Check if this angle set has completed all sweep tasks.
  bool IsFinished() const { return executed_; }

  /// Check if initialization has been done.
  bool IsInitialized() const { return initialized_; }

  AsynchronousCommunicator* GetCommunicator() override { return nullptr; }

  void InitializeDelayedUpstreamData() override {}

  int GetMaxBufferMessages() const override { return 0; }

  void SetMaxBufferMessages(int new_max) override {}

  /// Blocking sweep (backward-compatible with one-thread-per-angle-set scheduling).
  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus exec_status) override;

  AngleSetStatus FlushSendBuffers() override { return AngleSetStatus::MESSAGES_SENT; }

  /// Reset sweep buffers for next sweep.
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
  /// Signal completion, update boundaries, notify following angle sets, copy saved psi.
  void FinalizeSweep();

  /// Associated crb::Stream.
  crb::Stream stream_;
  /// Angle indices on GPU.
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;
  /// Reference to CBCD_FLUDS owned by this angleset.
  CBCD_FLUDS& cbcd_fluds_;
  /// Pointer to the sweep chunk.
  CBCDSweepChunk* cbcd_sweep_chunk_;
  /// Reference to the CBC SPDS (pulled up from CBC_AngleSet).
  const CBC_SPDS& cbc_spds_;
  /// Cell-by-cell task list (pulled up from CBC_AngleSet).
  std::vector<Task> current_task_list_;
  /// Pointer to the aggregated communicator (owned by CBCDSweepChunk).
  CBCD_AggregatedCommunicator* agg_comm_ = nullptr;
  /// Number of angle sets this one must wait for before starting.
  std::size_t num_dependencies_ = 0;
  /// A starting latch.
  /// Thread waits here until all predecessors count_down.
  /// A latch(0) is immediately released.
  std::unique_ptr<std::latch> starting_latch_;
  /// Anglesets whose latches this angleset counts down upon completion.
  std::vector<CBCD_AngleSet*> following_angle_sets_;
  /// Whether TryInitialize has completed successfully.
  bool initialized_ = false;
  /// Ready queue: task indices whose dependencies have been satisfied.
  std::vector<uint64_t> ready_queue_;
  /// Whether a GPU kernel is currently in-flight on this angle set's stream.
  bool kernel_in_flight_ = false;
  /// Tasks and cell IDs for the currently in-flight kernel batch.
  std::vector<Task*> in_flight_tasks_;
  std::vector<std::uint32_t> in_flight_cell_ids_;
  /// Number of completed tasks and total tasks for this sweep.
  size_t completed_count_ = 0;
  size_t total_tasks_ = 0;
  /// Cell local IDs that have outgoing reflecting boundary faces.
  std::unordered_set<uint64_t> reflecting_boundary_cells_;
  /// Number of reflecting boundary cells completed so far.
  size_t reflecting_boundary_completed_ = 0;
  /// Total reflecting boundary cells (set during init).
  size_t total_reflecting_boundary_cells_ = 0;
  /// Whether the latch has already been counted down early.
  bool latch_counted_down_ = false;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
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
class CBCD_FLUDS;

/**
 * Cooperative CBCD angle-set driver.
 *
 * The class advances one CBCD task graph on the GPU. It supports both the
 * legacy blocking `AngleSetAdvance()` path and the cooperative
 * `TryInitialize()` / `TryAdvanceOneStep()` path used by the threaded CBCD
 * scheduler.
 */
class CBCD_AngleSet : public AngleSet
{
public:
  /// Construct one CBCD angle set.
  ///
  /// \param id Angle-set identifier.
  /// \param num_groups Number of energy groups.
  /// \param spds Sweep ordering.
  /// \param fluds Angle-set FLUDS storage.
  /// \param angle_indices Angle indices in this set.
  /// \param boundaries Sweep-boundary map.
  CBCD_AngleSet(size_t id,
                size_t num_groups,
                const SPDS& spds,
                std::shared_ptr<FLUDS>& fluds,
                const std::vector<size_t>& angle_indices,
                std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries);

  ~CBCD_AngleSet();

  crb::Stream& GetStream() { return stream_; }
  std::uint32_t* GetDeviceAngleIndices() { return device_angle_indices_.get(); }

  void SetSweepChunk(CBCDSweepChunk* sweep_chunk) { cbcd_sweep_chunk_ = sweep_chunk; }
  void SetAggregatedCommunicator(CBCD_AggregatedCommunicator* agg_comm) { agg_comm_ = agg_comm; }
  CBCD_AggregatedCommunicator* GetAggregatedCommunicator() const { return agg_comm_; }

  /// Reset the inter-angle-set dependency counter for a new sweep.
  void ResetDependencyCounter();
  /// Register following angle sets for reflecting-boundary dependencies.
  void UpdateSweepDependencies(std::set<AngleSet*>& following_angle_sets) override;

  /// Perform sweep initialization when dependencies are satisfied.
  bool TryInitialize();
  /// Advance the sweep by one cooperative step.
  bool TryAdvanceOneStep();

  bool IsFinished() const { return executed_; }
  bool IsInitialized() const { return initialized_; }

  AsynchronousCommunicator* GetCommunicator() override { return nullptr; }
  void InitializeDelayedUpstreamData() override {}
  int GetMaxBufferMessages() const override { return 0; }
  void SetMaxBufferMessages(int new_max) override {}

  /// Run the angle set to completion in blocking mode.
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
  /// Finalize the sweep after all tasks complete.
  void FinalizeSweep();

  /// Notify following angle sets once reflecting data is ready.
  void TryNotifyFollowers();

  /// CUDA/HIP stream for this angle set.
  crb::Stream stream_;
  /// Device copy of the angle indices.
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;
  /// Per-angle-set FLUDS storage.
  CBCD_FLUDS& cbcd_fluds_;
  /// Owning sweep chunk.
  CBCDSweepChunk* cbcd_sweep_chunk_;
  /// CBC task graph.
  const CBC_SPDS& cbc_spds_;

  /// Cell local ID per task.
  std::vector<uint64_t> reference_ids_;
  /// CSR successor offsets.
  std::vector<uint32_t> successor_offsets_;
  /// Flat successor task indices.
  std::vector<uint32_t> successor_data_;
  /// Initial dependency counts per task.
  std::vector<int> initial_deps_;
  /// Per-sweep dependency counts.
  std::vector<int> remaining_deps_;
  /// Task indices with zero initial dependencies.
  std::vector<uint32_t> initial_ready_tasks_;

  /// Aggregated communicator used by this angle set.
  CBCD_AggregatedCommunicator* agg_comm_ = nullptr;
  /// Number of predecessor angle sets.
  std::size_t num_dependencies_ = 0;
  /// Remaining predecessor dependency count.
  std::atomic<std::size_t> dependency_counter_{0};
  /// Following angle sets for reflecting-boundary handoff.
  std::vector<CBCD_AngleSet*> following_angle_sets_;

  /// Initialization flag for the current sweep.
  bool initialized_ = false;
  /// Ready task queue for the next kernel launch.
  std::vector<uint32_t> ready_queue_;
  /// Kernel-in-flight flag.
  bool kernel_in_flight_ = false;
  /// Tasks currently represented by the in-flight kernel.
  std::vector<uint32_t> in_flight_task_indices_;
  /// Completed cell IDs awaiting outgoing-data handling.
  std::vector<uint64_t> deferred_cell_ids_;
  /// Number of completed tasks in the current sweep.
  size_t completed_count_ = 0;
  /// Total number of tasks in the sweep DAG.
  size_t total_tasks_ = 0;

  /// Per-task reflecting-boundary flag.
  std::vector<char> is_reflecting_task_;
  /// Number of completed reflecting tasks.
  size_t reflecting_tasks_completed_ = 0;
  /// Total number of reflecting tasks.
  size_t total_reflecting_tasks_ = 0;
  /// Reflecting-boundary notification flag.
  bool followers_notified_ = false;
};

} // namespace opensn

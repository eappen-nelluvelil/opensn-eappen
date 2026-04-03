// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "caribou/main.hpp"
#include <atomic>
#include <memory>
#include <set>

namespace crb = caribou;

namespace opensn
{

class CBC_SPDS;
class CBCDSweepChunk;
class CellFace;

/// CBC angle set for device.
class CBCD_AngleSet : public AngleSet
{
public:
  /// Construct a device CBC angle set.
  CBCD_AngleSet(size_t id,
                size_t num_groups,
                const SPDS& spds,
                std::shared_ptr<FLUDS>& fluds,
                const std::vector<size_t>& angle_indices,
                std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                const MPICommunicatorSet& comm_set);

  ~CBCD_AngleSet();

  /// Register following angle sets and initialize their dependency counts.
  void UpdateSweepDependencies(std::set<AngleSet*>& following_angle_sets) override;

  /// Reset the unresolved dependency counter before a sweep.
  void ResetDependencyCounter();

  /// Get the communicator associated with the angle set.
  AsynchronousCommunicator* GetCommunicator() override;

  /// Get the concrete communicator associated with the angle set.
  CBCD_AsynchronousCommunicator& GetAsyncCommunicator();

  /// Bind the angle set to the chunk-owned aggregated communicator.
  void SetCommunicator(CBCD_AsynchronousCommunicator& async_comm) { async_comm_ = &async_comm; }

  /// Get the communicator set used to construct the aggregated communicator.
  const MPICommunicatorSet& GetCommunicatorSet() const { return comm_set_; }

  void InitializeDelayedUpstreamData() override {}

  int GetMaxBufferMessages() const override { return 0; }

  void SetMaxBufferMessages(int new_max) override {}

  /// Initialize host-side sweep state once all dependencies are resolved.
  bool TryInitialize(CBCDSweepChunk& sweep_chunk);

  /// Advance the angle set by at most one ready-cell batch.
  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  AngleSetStatus FlushSendBuffers() override
  {
    return AngleSetStatus::MESSAGES_SENT;
  }

  void ResetSweepBuffers() override;

  bool ReceiveDelayedData() override { return true; }

  /// Get incoming boundary psi for a boundary face node.
  const double* PsiBoundary(uint64_t boundary_id,
                            unsigned int angle_num,
                            uint64_t cell_local_id,
                            unsigned int face_num,
                            unsigned int fi,
                            unsigned int g,
                            bool surface_source_active) override;

  /// Get outgoing reflected psi storage for a boundary face node.
  double* PsiReflected(uint64_t boundary_id,
                       unsigned int angle_num,
                       uint64_t cell_local_id,
                       unsigned int face_num,
                       unsigned int fi) override;

  /// Get the stream associated with the angle set.
  crb::Stream& GetStream() { return stream_; }

  /// Get the device angle-index array.
  std::uint32_t* GetDeviceAngleIndices() { return device_angle_indices_.get(); }

  /// Get the mutable task list for the current sweep.
  std::vector<Task>& GetCurrentTaskList() { return current_task_list_; }

  /// Check whether the angle set has completed its sweep.
  bool IsExecuted() const { return executed_; }

protected:
  /// Reference to the immutable CBC task graph.
  const CBC_SPDS& cbc_spds_;
  /// Communicator-set metadata for aggregated communicator construction.
  const MPICommunicatorSet& comm_set_;
  /// Mutable task state for the current sweep.
  std::vector<Task> current_task_list_;
  /// Chunk-owned aggregated communicator.
  CBCD_AsynchronousCommunicator* async_comm_ = nullptr;
  /// Associated device stream.
  crb::Stream stream_;
  /// Angle indices on the device.
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;
  /// Number of unresolved angle-set dependencies at startup.
  std::size_t num_dependencies_ = 0;
  /// Atomic counter for unresolved angle-set dependencies.
  std::atomic_size_t dependency_counter_ = 0;
  /// Following angle sets unblocked by this angle set.
  std::vector<CBCD_AngleSet*> following_angle_sets_;
  /// Ready tasks waiting for the next batch launch.
  std::vector<Task*> ready_queue_;
  /// Tasks in the current in-flight kernel batch.
  std::vector<Task*> in_flight_tasks_;
  /// Cell ids for the current in-flight kernel batch.
  std::vector<std::uint32_t> in_flight_cell_ids_;
  /// Cached reflecting-boundary task mask.
  std::vector<std::uint8_t> task_has_outgoing_reflecting_boundary_;
  /// Number of completed local tasks.
  std::size_t num_completed_tasks_ = 0;
  /// Initial number of tasks that produce reflecting-boundary data.
  std::size_t initial_reflecting_task_count_ = 0;
  /// Remaining number of tasks that still need to produce reflecting-boundary data.
  std::size_t pending_reflecting_tasks_ = 0;
  /// Flag indicating boundary data has been copied for this sweep.
  bool boundary_data_initialized_ = false;
  /// Flag indicating following angle sets have been notified.
  bool following_angle_sets_notified_ = false;
  /// Flag indicating a kernel batch is in flight.
  bool kernel_in_flight_ = false;

private:
  /// Initialize the immutable reflecting-boundary task mask.
  void InitializeReflectingTaskMask();
  /// Check whether a cell face is an outgoing reflecting boundary face.
  bool IsOutgoingReflectingFace(const CellFace& face,
                                std::uint64_t cell_local_id,
                                std::size_t face_id) const;
  /// Initialize mutable task state for a new sweep.
  void InitializeTaskState();
  /// Decrement following angle-set dependency counters once all reflecting data is ready.
  void NotifyFollowingAngleSets();
  /// Notify following angle sets once all reflecting-boundary producers have completed.
  void TryNotifyFollowingAngleSets();
};

} // namespace opensn

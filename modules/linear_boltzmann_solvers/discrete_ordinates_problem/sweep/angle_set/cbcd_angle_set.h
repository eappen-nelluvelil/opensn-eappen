// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "caribou/main.hpp"
#include <array>
#include <atomic>
#include <unordered_map>

namespace crb = caribou;

namespace opensn
{

class CBCD_FLUDS;
class CBC_SPDS;
class CBCDSweepChunk;
class CellFace;
class LBSGroupset;

/// Device CBCD angle set with task-graph-driven batched execution.
class CBCD_AngleSet : public AngleSet
{
public:
  struct BatchState
  {
    std::uint8_t ready_buffer_index = 0;
    std::uint8_t launch_buffer_index = 0;
    std::uint8_t completed_buffer_index = 0;
    std::array<std::uint8_t, 3> free_buffer_indices = {1, 2, 0};
    std::uint8_t num_free_buffers = 2;
    std::uint32_t launch_count = 0;
    std::uint32_t completed_count = 0;
    bool kernel_in_flight = false;
    bool completed_batch_pending = false;

    void Reset()
    {
      ready_buffer_index = 0;
      launch_buffer_index = 0;
      completed_buffer_index = 0;
      free_buffer_indices = {1, 2, 0};
      num_free_buffers = 2;
      launch_count = 0;
      completed_count = 0;
      kernel_in_flight = false;
      completed_batch_pending = false;
    }

    std::uint8_t AcquireFreeBuffer()
    {
      return free_buffer_indices[--num_free_buffers];
    }

    void ReleaseBuffer(const std::uint8_t buffer_index)
    {
      free_buffer_indices[num_free_buffers++] = buffer_index;
    }
  };

  CBCD_AngleSet(size_t id,
                const LBSGroupset& groupset,
                const SPDS& spds,
                std::shared_ptr<FLUDS>& fluds,
                const std::vector<size_t>& angle_indices,
                std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                const MPICommunicatorSet& comm_set);

  ~CBCD_AngleSet() override = default;

  void ResetDependencyCounter();
  void RebindBoundaryData();
  AsynchronousCommunicator* GetCommunicator() override { return nullptr; }
  void SetCommunicator(CBCD_AsynchronousCommunicator& async_comm) { async_comm_ = &async_comm; }
  const MPICommunicatorSet& GetCommunicatorSet() const { return comm_set_; }

  void InitializeDelayedUpstreamData() override {}

  int GetMaxBufferMessages() const override { return 0; }
  void SetMaxBufferMessages(int) override {}

  /// Initialize the angle set after its upstream angle-set dependencies are resolved.
  bool TryInitialize(CBCDSweepChunk& sweep_chunk);

  /// Advance the angle set by one scheduler step.
  bool TryAdvanceOneStep(CBCDSweepChunk& sweep_chunk, std::size_t worker_id);

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  AngleSetStatus FlushSendBuffers() override { return AngleSetStatus::MESSAGES_SENT; }
  void ResetSweepBuffers() override;
  bool ReceiveDelayedData() override { return true; }
  crb::Stream& GetStream() { return stream_; }
  std::uint32_t* GetDeviceAngleIndices() { return device_angle_indices_.get(); }
  bool IsExecuted() const { return executed_; }
  bool IsInitialized() const { return boundary_data_initialized_; }

private:
  const CBC_SPDS& cbc_spds_;
  const MPICommunicatorSet& comm_set_;
  CBCD_FLUDS& cbcd_fluds_;
  CBCD_AsynchronousCommunicator* async_comm_ = nullptr;
  crb::Stream stream_;
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;
  std::vector<std::uint32_t> successor_offsets_;
  std::vector<std::uint32_t> successor_data_;
  std::vector<int> initial_deps_;
  std::vector<int> remaining_deps_;
  std::vector<std::uint32_t> initial_ready_cell_ids_;
  std::size_t num_tasks_ = 0;
  /// Atomic counterpart of `AngleSet::num_dependencies_` for concurrent follower release.
  std::atomic<std::size_t> dependency_counter_;
  std::unordered_map<std::uint64_t, SweepBoundary*> boundary_ptrs_;
  BatchState batch_state_;
  std::vector<std::uint8_t> cell_has_outgoing_reflecting_boundary_;
  std::size_t num_completed_tasks_ = 0;
  std::size_t initial_reflecting_task_count_ = 0;
  std::size_t pending_reflecting_tasks_ = 0;
  bool boundary_data_initialized_ = false;
  bool following_angle_sets_notified_ = false;

  void InitializeReflectingTaskMask();
  void InitializeTaskGraphData();
  bool IsOutgoingReflectingFace(const CellFace& face,
                                std::uint64_t cell_local_id,
                                std::size_t face_id) const;
  void InitializeTaskState();
  bool TryRetireCompletedBatch();
  bool TryLaunchReadyBatch(CBCDSweepChunk& sweep_chunk);
  void FlushCompletedBatch(CBCDSweepChunk& sweep_chunk, std::size_t worker_id);
  void TryNotifyFollowingAngleSets();
};

} // namespace opensn

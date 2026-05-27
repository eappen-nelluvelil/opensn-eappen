// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "caribou/main.hpp"
#include <array>
#include <atomic>

namespace crb = caribou;

namespace opensn
{

class CBCD_FLUDS;
class CBC_SPDS;
class CBCDSweepChunk;
struct CBCDWorkerLaunch;
class CellFace;
class LBSGroupset;

class CBCD_AngleSet : public AngleSet
{
public:
  CBCD_AngleSet(size_t id,
                const LBSGroupset& groupset,
                const SPDS& spds,
                std::shared_ptr<FLUDS>& fluds,
                const std::vector<size_t>& angle_indices,
                std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                const SweepCommunicator& sweep_communicator);

  ~CBCD_AngleSet() override = default;

  void ResetSweepDependencies();
  /// Refresh device angle data and reflecting metadata after a runtime rebuild.
  void RefreshDeviceData();
  AsynchronousCommunicator* GetCommunicator() override { return nullptr; }
  void SetCommunicator(CBCD_AsynchronousCommunicator& async_comm) { async_comm_ = &async_comm; }
  const SweepCommunicator& GetSweepCommunicator() const { return sweep_communicator_; }

  void InitializeDelayedUpstreamData() override {}

  int GetMaxBufferMessages() const override { return 0; }
  void SetMaxBufferMessages(int /*max_buffer_messages*/) override {}

  /// Initialize the angle set after its upstream angle-set dependencies are resolved.
  bool TryInitialize(CBCDSweepChunk& sweep_chunk);

  bool TryAdvanceOneStep(CBCDSweepChunk& sweep_chunk,
                         std::size_t worker_id,
                         CBCDWorkerLaunch& worker_launch);

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  AngleSetStatus FlushSendBuffers() override { return AngleSetStatus::MESSAGES_SENT; }
  void ResetSweepBuffers() override;
  bool ReceiveDelayedData() override { return true; }
  crb::Stream& GetStream() { return stream_; }
  std::uint32_t* GetDeviceAngleIndices() { return device_angle_indices_.get(); }
  bool IsSweepComplete() const { return executed_; }
  bool IsSweepInitialized() const { return sweep_initialized_; }

private:
  /// Triple-buffer state for ready, executing, and completed cell batches.
  struct BatchPipeline
  {
    /// Buffers currently collecting, executing, and awaiting publication.
    std::uint8_t ready_buffer = 0;
    std::uint8_t launch_buffer = 0;
    std::uint8_t completed_buffer = 0;
    /// Stack of buffers not owned by a pipeline stage.
    std::array<std::uint8_t, 3> free_buffers = {1, 2, 0};
    std::uint8_t num_free_buffers = 2;
    /// Cell counts also encode whether launch/completed stages are occupied.
    std::uint32_t launch_count = 0;
    std::uint32_t completed_count = 0;

    void Reset() { *this = {}; }

    bool HasKernelInFlight() const { return launch_count != 0; }
    bool HasCompletedBatch() const { return completed_count != 0; }

    std::uint8_t AcquireFreeBuffer() { return free_buffers[--num_free_buffers]; }

    void ReleaseBuffer(const std::uint8_t buffer) { free_buffers[num_free_buffers++] = buffer; }
  };

  const CBC_SPDS& cbc_spds_;
  const SweepCommunicator& sweep_communicator_;
  CBCD_FLUDS& cbcd_fluds_;
  CBCD_AsynchronousCommunicator* async_comm_ = nullptr;
  crb::Stream stream_;
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;
  std::vector<std::uint32_t> cell_successor_offsets_;
  std::vector<std::uint32_t> cell_successors_;
  std::vector<unsigned int> initial_cell_dependencies_;
  std::vector<unsigned int> remaining_cell_dependencies_;
  /// Local cell tasks ready before any nonlocal messages arrive.
  std::vector<std::uint32_t> initial_ready_cell_ids_;
  /// Atomic counterpart of the base dependency counter for concurrent follower release.
  std::atomic<std::size_t> unresolved_sweep_dependencies_{0};
  BatchPipeline batch_pipeline_;
  std::vector<std::uint8_t> has_outgoing_reflecting_face_;
  std::size_t num_completed_cells_ = 0;
  std::size_t num_reflecting_cells_ = 0;
  std::size_t pending_reflecting_cells_ = 0;
  bool sweep_initialized_ = false;
  bool followers_released_ = false;

  void BuildReflectingCellMask();
  /// Flatten the local cell-task DAG into persistent CSR storage.
  void BuildCellTaskGraph();
  bool IsOutgoingReflectingFace(const CellFace& face,
                                std::uint64_t cell_local_id,
                                std::size_t face_id) const;
  void InitializeSweepState();
  /// Retire the batch after this worker has observed stream completion.
  void RetireCompletedBatch();
  void AddReadyBatch(CBCDSweepChunk& sweep_chunk, CBCDWorkerLaunch& worker_launch);
  void PublishCompletedBatch(CBCDSweepChunk& sweep_chunk, std::size_t worker_id);
  /// Release follower angle sets after all reflecting writes are visible.
  void TryReleaseFollowers();
};

} // namespace opensn

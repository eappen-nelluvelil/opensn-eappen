// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "caribou/main.hpp"
#include <array>
#include <atomic>
#include <cassert>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace crb = caribou;

namespace opensn
{

class CBCD_FLUDS;
class CBC_SPDS;
class CBCDSweepChunk;
class CellFace;
class LBSGroupset;

/**
 * CBCD angle set with task-graph-driven batched execution.
 *
 * Manages the host-side state machine for one device-resident CBCD angle set.
 * The angle set waits for upstream dependencies, launches ready-cell batches on
 * its stream, drains received non-local face data, and flushes completed outgoing
 * data through the aggregated communicator.
 */
class CBCD_AngleSet : public AngleSet
{
public:
  /// Per-sweep launch/completion state for the current kernel batch.
  struct BatchState
  {
    /// Buffer receiving newly ready local cell IDs.
    std::uint8_t ready_buffer_index = 0;
    /// Buffer backing the currently running kernel launch.
    std::uint8_t launch_buffer_index = 0;
    /// Buffer holding the most recently completed kernel batch until it is flushed.
    std::uint8_t completed_buffer_index = 0;
    /// Indices of currently free mapped-host cell-ID buffers.
    std::array<std::uint8_t, 3> free_buffer_indices = {1, 2, 0};
    /// Number of free mapped-host cell-ID buffers.
    std::uint8_t num_free_buffers = 2;
    /// Number of local cells in the currently running kernel launch.
    std::uint32_t launch_count = 0;
    /// Number of local cells in the completed batch waiting to be flushed.
    std::uint32_t completed_count = 0;
    /// Flag indicating whether a kernel launch is currently outstanding.
    bool kernel_in_flight = false;
    /// Flag indicating whether a completed batch is waiting to be flushed.
    bool completed_batch_pending = false;

    /// Reset the batch state between sweeps.
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

    /// Acquire one free mapped-host cell-ID buffer.
    std::uint8_t AcquireFreeBuffer()
    {
      if (num_free_buffers == 0)
        throw std::logic_error("CBCD angle set: no free launch buffer.");
      return free_buffer_indices[--num_free_buffers];
    }

    /**
     * Return one mapped-host cell-ID buffer to the free list.
     *
     * \param buffer_index Buffer index to release.
     */
    void ReleaseBuffer(const std::uint8_t buffer_index)
    {
      if (num_free_buffers >= free_buffer_indices.size())
        throw std::logic_error("CBCD angle set: launch buffer released more than once.");
      free_buffer_indices[num_free_buffers++] = buffer_index;
    }
  };

  /**
   * Construct one CBCD angle set.
   *
   * \param id Angle-set ID.
   * \param groupset Groupset that owns this angle set; supplies the group count and id.
   * \param spds Sweep plane data structure for this angle set.
   * \param fluds Device FLUDS for this angle set.
   * \param angle_indices Global angle indices represented by this angle set.
   * \param boundaries Sweep-boundary table indexed by boundary ID.
   * \param comm_set MPI communicator set used to build the aggregated communicator.
   */
  CBCD_AngleSet(size_t id,
                const LBSGroupset& groupset,
                const SPDS& spds,
                std::shared_ptr<FLUDS>& fluds,
                const std::vector<size_t>& angle_indices,
                std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                const MPICommunicatorSet& comm_set);

  ~CBCD_AngleSet() override = default;

  /// Reset the unresolved angle-set dependency counter before a sweep.
  void ResetDependencyCounter();

  /// Return the delayed-data communicator for this angle set.
  AsynchronousCommunicator* GetCommunicator() override;

  /// Bind the angle set to the sweep-chunk-owned aggregated communicator.
  void SetCommunicator(CBCD_AsynchronousCommunicator& async_comm) { async_comm_ = &async_comm; }

  /// Return the communicator set used to construct the aggregated communicator.
  const MPICommunicatorSet& GetCommunicatorSet() const { return comm_set_; }

  void InitializeDelayedUpstreamData() override {}

  /// Return the buffered-message limit used by the scheduler.
  int GetMaxBufferMessages() const override { return 0; }

  /// Set the buffered-message limit used by the scheduler.
  void SetMaxBufferMessages(int) override {}

  /**
   * Initialize the angle set once all upstream angle-set dependencies are resolved.
   *
   * Copies incoming boundary data to the device, resets per-sweep task state, and
   * marks the angle set ready for batched execution.
   *
   * \param sweep_chunk Owning CBCD sweep chunk.
   * \return True when initialization was performed on this call.
   */
  bool TryInitialize(CBCDSweepChunk& sweep_chunk);

  /**
   * Advance the angle set by at most one scheduler step.
   *
   * One step may retire a completed batch, drain newly received faces, launch the
   * next ready batch, flush completed outgoing data, or finalize the angle set.
   *
   * \param sweep_chunk Owning CBCD sweep chunk.
   * \param worker_id Worker ID that owns this angle set in the current sweep schedule.
   * \return True when any forward progress was made.
   */
  bool TryAdvanceOneStep(CBCDSweepChunk& sweep_chunk, std::size_t worker_id);

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  /// Flush buffered sends for this angle set.
  AngleSetStatus FlushSendBuffers() override { return AngleSetStatus::MESSAGES_SENT; }

  /// Reset per-sweep state and buffers.
  void ResetSweepBuffers() override;

  /// Report whether delayed upstream data has been received.
  bool ReceiveDelayedData() override { return true; }

  /// Return the stream associated with this angle set.
  crb::Stream& GetStream() { return stream_; }

  /// Return the device pointer to the angle-index table.
  std::uint32_t* GetDeviceAngleIndices() { return device_angle_indices_.get(); }

  /// Check whether the angle set has completed its sweep.
  bool IsExecuted() const { return executed_; }
  /// Check whether the angle set has been initialized for the current sweep.
  bool IsInitialized() const { return boundary_data_initialized_; }

private:
  const CBC_SPDS& cbc_spds_;
  /// Communicator-set metadata for aggregated communicator construction.
  const MPICommunicatorSet& comm_set_;
  /// Per-angle-set FLUDS.
  CBCD_FLUDS& cbcd_fluds_;
  /// Sweep chunk-owned aggregated communicator.
  CBCD_AsynchronousCommunicator* async_comm_ = nullptr;
  /// Associated crb::Stream.
  crb::Stream stream_;
  /// Angle indices on GPU.
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;
  /// Successor offsets indexed by local cell ID.
  std::vector<std::uint32_t> successor_offsets_;
  /// Successor local cell IDs stored in CSR order.
  std::vector<std::uint32_t> successor_data_;
  /// Initial dependency counts per local cell.
  std::vector<std::uint32_t> initial_deps_;
  /// Per-sweep dependency counts per local cell.
  std::vector<std::uint32_t> remaining_deps_;
  /// Local cell IDs with zero initial dependencies.
  std::vector<std::uint32_t> initial_ready_cell_ids_;
  /// Cached total number of local cells/tasks in task graph.
  std::size_t num_tasks_ = 0;
  /// Atomic counter for unresolved angle-set dependencies.  Shadows the base's non-atomic
  /// counter because CBCD's worker-batch scheduler decrements followers' counters from
  /// concurrent worker threads, which requires atomic read-modify-write semantics.
  std::atomic<std::size_t> dependency_counter_;
  /// Cached boundary lookup table.
  std::unordered_map<std::uint64_t, SweepBoundary*> boundary_ptrs_;
  /// Explicit launch/completion state for the current sweep batch.
  BatchState batch_state_;
  /// Cached reflecting-boundary producer mask by local cell ID.
  std::vector<std::uint8_t> cell_has_outgoing_reflecting_boundary_;
  /// Number of completed local tasks.
  std::size_t num_completed_tasks_ = 0;
  /// Initial number of local cells that produce reflecting boundary data.
  std::size_t initial_reflecting_task_count_ = 0;
  /// Remaining number of local cells that still need to produce reflecting boundary data.
  std::size_t pending_reflecting_tasks_ = 0;
  /// Flag indicating if incoming boundary data has been copied to the device.
  bool boundary_data_initialized_ = false;
  /// Flag indicating if following anglesets have been notified of completion.
  bool following_angle_sets_notified_ = false;
  /// Flag indicating that local work is complete and delayed-output markers were published.
  bool local_completion_signaled_ = false;

  /// Build the reflecting-boundary producer mask from the CBC task graph.
  void InitializeReflectingTaskMask();

  /// Flatten the CBC task graph into lookup tables.
  void InitializeTaskGraphData();

  /// Check whether a cell face is an outgoing reflecting boundary face.
  bool IsOutgoingReflectingFace(const CellFace& face,
                                std::uint64_t cell_local_id,
                                std::size_t face_id) const;

  /// Reset mutable task state for a new sweep.
  void InitializeTaskState();

  /// Retire the completed kernel batch and update successor dependency state.
  bool TryRetireCompletedBatch();

  /// Launch the next ready-cell batch when the current stream is idle.
  bool TryLaunchReadyBatch(CBCDSweepChunk& sweep_chunk);

  /// Pack and send deferred outgoing data for the completed batch.
  void FlushCompletedBatch(CBCDSweepChunk& sweep_chunk, std::size_t worker_id);

  /// Notify following angle sets once all reflecting-boundary producers have completed.
  void TryNotifyFollowingAngleSets();
};

} // namespace opensn

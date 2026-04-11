// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "caribou/main.hpp"
#include <atomic>
#include <set>
#include <unordered_map>

namespace crb = caribou;

namespace opensn
{

class CBC_SPDS;
class CBCD_FLUDS;
class CBCDSweepChunk;
class CellFace;

/**
 * Device-side CBC angle set with task-graph-driven batched kernel dispatch.
 *
 * Manages the per-angle-set sweep state for the CBCD algorithm: dependency
 * tracking via CSR-format successor lists, ready-queue management, batched
 * GPU kernel launches, and deferred outgoing-data handling. Each angle set
 * maintains its own device stream for asynchronous kernel execution.
 *
 * ## Sweep lifecycle
 *
 * 1. **Initialization:** TryInitialize copies boundary psi to the device and
 *    seeds the ready queue with zero-dependency root tasks.
 * 2. **Batched execution:** AngleSetAdvance repeatedly drains the ready queue,
 *    launches a GPU kernel batch, processes deferred outgoing data from the
 *    previous batch (overlapping compute with host-side packing), and receives
 *    incoming nonlocal data from the aggregated communicator.
 * 3. **Completion:** Once all tasks are complete, reflecting-boundary data is
 *    copied, following angle sets are notified, and saved angular fluxes are
 *    transferred to the host.
 *
 * ## Deferred outgoing data
 *
 * Outgoing angular flux is not packed immediately after a kernel batch.
 * Instead, the next kernel batch is launched first, and the previous batch's
 * outgoing data is packed while the GPU computes, hiding the host-side
 * memcpy/packing latency behind kernel execution.
 */
class CBCD_AngleSet : public AngleSet
{
public:
  struct BatchState
  {
    std::vector<std::uint32_t> launch_cell_ids;
    std::vector<std::uint32_t> completed_cell_ids;
    bool kernel_in_flight = false;

    void Reserve(const std::size_t num_tasks)
    {
      launch_cell_ids.reserve(num_tasks);
      completed_cell_ids.reserve(num_tasks);
    }

    void Reset()
    {
      launch_cell_ids.clear();
      completed_cell_ids.clear();
      kernel_in_flight = false;
    }
  };

  /**
   * Construct a device CBC angle set.
   *
   * \param id unique angle set identifier
   * \param num_groups number of energy groups
   * \param spds sweep-plane data structure (must be a CBC_SPDS)
   * \param fluds shared FLUDS storage (must be a CBCD_FLUDS)
   * \param angle_indices quadrature angle indices for this set
   * \param boundaries boundary condition map
   * \param comm_set MPI communicator set for aggregated communicator construction
   */
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
  bool TryAdvanceOneStep();

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

  /// Check whether the angle set has completed its sweep.
  bool IsExecuted() const { return executed_; }
  bool IsInitialized() const { return boundary_data_initialized_; }

protected:
  /// Reference to the immutable CBC task graph.
  const CBC_SPDS& cbc_spds_;
  /// Communicator-set metadata for aggregated communicator construction.
  const MPICommunicatorSet& comm_set_;
  /// Per-angle-set FLUDS storage (cached to avoid per-step virtual dispatch).
  CBCD_FLUDS& cbcd_fluds_;
  /// Chunk-owned aggregated communicator.
  CBCD_AsynchronousCommunicator* async_comm_ = nullptr;
  /// Owning sweep chunk for the current sweep.
  CBCDSweepChunk* sweep_chunk_ = nullptr;
  /// Associated device stream.
  crb::Stream stream_;
  /// Angle indices on the device.
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;
  /// Flat successor offsets.
  std::vector<std::uint32_t> successor_offsets_;
  /// Flat successor local cell IDs.
  std::vector<std::uint32_t> successor_data_;
  /// Initial dependency counts per local cell.
  std::vector<int> initial_deps_;
  /// Per-sweep dependency counts per local cell.
  std::vector<int> remaining_deps_;
  /// Local cell IDs with zero initial dependencies.
  std::vector<std::uint32_t> initial_ready_cell_ids_;
  /// Cached total number of local cells/tasks in the CBC graph.
  std::size_t num_tasks_ = 0;
  /// Number of unresolved angle-set dependencies at startup.
  std::size_t num_dependencies_ = 0;
  /// Atomic counter for unresolved angle-set dependencies.
  std::atomic_size_t dependency_counter_ = 0;
  /// Following angle sets unblocked by this angle set.
  std::vector<CBCD_AngleSet*> following_angle_sets_;
  /// Cached boundary lookup table to avoid repeated ordered-map access.
  std::unordered_map<std::uint64_t, SweepBoundary*> boundary_ptrs_;
  /// Reflecting boundaries touched by this angle set.
  std::vector<SweepBoundary*> reflecting_boundaries_;
  /// Ready local cell IDs waiting for the next batch launch.
  std::vector<std::uint32_t> ready_cell_ids_;
  /// Explicit launch/completion state for the current sweep batch pipeline.
  BatchState batch_state_;
  /// Cached reflecting-boundary producer mask by local cell ID.
  std::vector<std::uint8_t> cell_has_outgoing_reflecting_boundary_;
  /// Number of completed local tasks.
  std::size_t num_completed_tasks_ = 0;
  /// Initial number of local cells that produce reflecting-boundary data.
  std::size_t initial_reflecting_task_count_ = 0;
  /// Remaining number of local cells that still need to produce reflecting-boundary data.
  std::size_t pending_reflecting_tasks_ = 0;
  /// Cached flag indicating whether any following angle sets actually wait on this set.
  bool has_following_angle_sets_ = false;
  /// Flag indicating boundary data has been copied for this sweep.
  bool boundary_data_initialized_ = false;
  /// Flag indicating following angle sets have been notified.
  bool following_angle_sets_notified_ = false;

private:
  /// Initialize the immutable reflecting-boundary task mask.
  void InitializeReflectingTaskMask();
  /// Initialize immutable task-graph lookup tables.
  void InitializeTaskGraphData();
  /// Check whether a cell face is an outgoing reflecting boundary face.
  bool IsOutgoingReflectingFace(const CellFace& face,
                                std::uint64_t cell_local_id,
                                std::size_t face_id) const;
  /// Initialize mutable task state for a new sweep.
  void InitializeTaskState();
  /// Retire the completed kernel batch and update successor/deferred state.
  bool TryRetireCompletedBatch();
  /// Launch the next ready-cell batch if the stream is idle.
  bool TryLaunchReadyBatch(CBCDSweepChunk& sweep_chunk);
  /// Pack and send deferred outgoing data for the completed batch.
  void FlushCompletedBatch(CBCDSweepChunk& sweep_chunk);
  /// Decrement following angle-set dependency counters once all reflecting data is ready.
  void NotifyFollowingAngleSets();
  /// Notify following angle sets once all reflecting-boundary producers have completed.
  void TryNotifyFollowingAngleSets();
};

} // namespace opensn

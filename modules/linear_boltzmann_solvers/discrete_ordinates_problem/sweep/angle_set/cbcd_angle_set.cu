// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/sweep_boundary.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

namespace opensn
{

CBCD_AngleSet::CBCD_AngleSet(size_t id,
                             const LBSGroupset& groupset,
                             const SPDS& spds,
                             std::shared_ptr<FLUDS>& fluds,
                             const std::vector<size_t>& angle_indices,
                             std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                             const MPICommunicatorSet& comm_set)
  : AngleSet(id, groupset, spds, fluds, angle_indices, boundaries),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds)),
    comm_set_(comm_set),
    cbcd_fluds_(static_cast<CBCD_FLUDS&>(*fluds_)),
    stream_(),
    device_angle_indices_(angles_.size())
{
  boundary_ptrs_.reserve(boundaries_.size());
  for (auto& [bid, bndry] : boundaries_)
    boundary_ptrs_.emplace(bid, bndry.get());

  crb::MemoryPinningManager angle_indices_pinner_(angles_);
  crb::copy(device_angle_indices_, angle_indices_pinner_, angles_.size(), 0, 0, stream_);
  cbcd_fluds_.GetStream() = stream_;
  cbcd_fluds_.AllocateLocalAndSavedPsi();
  cbcd_fluds_.AllocateDelayedPsiBanks();
  cbcd_fluds_.InitializeReflectingBoundaryNodes(boundaries_);
  InitializeTaskGraphData();
  InitializeReflectingTaskMask();
}

AsynchronousCommunicator*
CBCD_AngleSet::GetCommunicator()
{
  return nullptr;
}

void
CBCD_AngleSet::ResetDependencyCounter()
{
  // The base `AngleSet::UpdateSweepDependencies` (invoked once per angle set during
  // `AngleAggregation` setup) populated `following_angle_sets_` and incremented this
  // angle set's `num_dependencies_`.  Seed the CBCD atomic counter from that value so
  // worker threads can decrement it in `TryNotifyFollowingAngleSets` without locking.
  dependency_counter_.store(num_dependencies_, std::memory_order_relaxed);
}

bool
CBCD_AngleSet::IsOutgoingReflectingFace(const CellFace& face,
                                        const std::uint64_t cell_local_id,
                                        const std::size_t face_id) const
{
  if ((face.has_neighbor) or
      (cbc_spds_.GetCellFaceOrientations()[cell_local_id][face_id] != FaceOrientation::OUTGOING))
    return false;
  const auto boundary_it = boundary_ptrs_.find(face.neighbor_id);
  return ((boundary_it != boundary_ptrs_.end()) and (boundary_it->second->IsReflecting()));
}

void
CBCD_AngleSet::InitializeReflectingTaskMask()
{
  const auto& task_list = cbc_spds_.GetTaskList();
  cell_has_outgoing_reflecting_boundary_.assign(task_list.size(), 0);

  for (std::size_t task_idx = 0; task_idx < task_list.size(); ++task_idx)
  {
    const auto& cell = *task_list[task_idx].cell_ptr;
    bool has_outgoing_reflecting_face = false;
    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      if (IsOutgoingReflectingFace(cell.faces[f], cell.local_id, f))
      {
        has_outgoing_reflecting_face = true;
        break;
      }
    }

    if (has_outgoing_reflecting_face)
    {
      cell_has_outgoing_reflecting_boundary_[task_idx] = 1;
      ++initial_reflecting_task_count_;
    }
  }
}

void
CBCD_AngleSet::InitializeTaskGraphData()
{
  if (not initial_deps_.empty())
    return;

  const auto& task_list = cbc_spds_.GetTaskList();
  num_tasks_ = task_list.size();

  initial_deps_.resize(num_tasks_);
  remaining_deps_.resize(num_tasks_);
  successor_offsets_.assign(num_tasks_ + 1, 0);
  initial_ready_cell_ids_.clear();
  initial_ready_cell_ids_.reserve(num_tasks_);

  for (std::size_t task_idx = 0; task_idx < task_list.size(); ++task_idx)
  {
    const auto& task = task_list[task_idx];
    initial_deps_[task_idx] = task.num_dependencies;
    successor_offsets_[task_idx + 1] = static_cast<std::uint32_t>(task.successors.size());
    if (task.num_dependencies == 0)
      initial_ready_cell_ids_.push_back(static_cast<std::uint32_t>(task_idx));
  }

  for (std::size_t task_idx = 0; task_idx < num_tasks_; ++task_idx)
    successor_offsets_[task_idx + 1] += successor_offsets_[task_idx];

  successor_data_.resize(successor_offsets_.back());
  for (std::size_t task_idx = 0; task_idx < task_list.size(); ++task_idx)
  {
    const auto& task = task_list[task_idx];
    std::copy(task.successors.begin(),
              task.successors.end(),
              successor_data_.begin() + successor_offsets_[task_idx]);
  }
}

void
CBCD_AngleSet::InitializeTaskState()
{
  std::copy(initial_deps_.begin(), initial_deps_.end(), remaining_deps_.begin());
  batch_state_.Reset();
  auto& ready_cell_ids = cbcd_fluds_.GetLocalCellIDs(batch_state_.ready_buffer_index);
  ready_cell_ids.clear();
  ready_cell_ids.insert(
    ready_cell_ids.end(), initial_ready_cell_ids_.begin(), initial_ready_cell_ids_.end());
  num_completed_tasks_ = 0;
  pending_reflecting_tasks_ = following_angle_sets_.empty() ? 0 : initial_reflecting_task_count_;
}

bool
CBCD_AngleSet::TryRetireCompletedBatch()
{
  if ((not batch_state_.kernel_in_flight) or (not stream_.is_completed()))
    return false;

  auto& completed_cell_ids = cbcd_fluds_.GetLocalCellIDs(batch_state_.launch_buffer_index);
  for (std::uint32_t i = 0; i < batch_state_.launch_count; ++i)
  {
    const auto cell_local_id = completed_cell_ids[i];
    const auto succ_begin = successor_offsets_[cell_local_id];
    const auto succ_end = successor_offsets_[cell_local_id + 1];
    for (auto succ_i = succ_begin; succ_i < succ_end; ++succ_i)
    {
      const auto successor = successor_data_[succ_i];
      auto& remaining = remaining_deps_[successor];
      if (remaining == 0)
        throw std::logic_error("CBCD angle set: local dependency counter underflow.");
      if (--remaining == 0)
        cbcd_fluds_.GetLocalCellIDs(batch_state_.ready_buffer_index).push_back(successor);
    }

    if ((not following_angle_sets_.empty()) and (not following_angle_sets_notified_) and
        (cell_has_outgoing_reflecting_boundary_[cell_local_id] != 0))
    {
      if (pending_reflecting_tasks_ == 0)
        throw std::logic_error("CBCD angle set: reflecting task counter underflow.");
      --pending_reflecting_tasks_;
    }
  }

  if (num_completed_tasks_ > num_tasks_ or
      batch_state_.launch_count > num_tasks_ - num_completed_tasks_)
    throw std::logic_error("CBCD angle set: completed task count exceeds the task graph.");
  num_completed_tasks_ += batch_state_.launch_count;
  batch_state_.completed_buffer_index = batch_state_.launch_buffer_index;
  batch_state_.completed_count = batch_state_.launch_count;
  batch_state_.completed_batch_pending = true;
  batch_state_.launch_count = 0;
  batch_state_.kernel_in_flight = false;
  return true;
}

bool
CBCD_AngleSet::TryLaunchReadyBatch(CBCDSweepChunk& sweep_chunk)
{
  auto& ready_cell_ids = cbcd_fluds_.GetLocalCellIDs(batch_state_.ready_buffer_index);
  if (batch_state_.kernel_in_flight or ready_cell_ids.empty())
    return false;

  if (ready_cell_ids.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::length_error("CBCD angle set: ready batch exceeds the kernel count range.");
  const auto launch_count = static_cast<std::uint32_t>(ready_cell_ids.size());
  batch_state_.launch_buffer_index = batch_state_.ready_buffer_index;
  batch_state_.launch_count = launch_count;
  batch_state_.ready_buffer_index = batch_state_.AcquireFreeBuffer();
  cbcd_fluds_.GetLocalCellIDs(batch_state_.ready_buffer_index).clear();
  sweep_chunk.Sweep(launch_count, GetID(), ready_cell_ids.data());
  batch_state_.kernel_in_flight = true;
  return true;
}

void
CBCD_AngleSet::FlushCompletedBatch(CBCDSweepChunk& sweep_chunk, const std::size_t worker_id)
{
  if (not batch_state_.completed_batch_pending)
    return;

  auto& completed_cell_ids = cbcd_fluds_.GetLocalCellIDs(batch_state_.completed_buffer_index);
  cbcd_fluds_.CopyOutgoingPsiBackToHost(
    sweep_chunk,
    async_comm_,
    worker_id,
    GetID(),
    GetAngleIndices(),
    {completed_cell_ids.data(), static_cast<std::size_t>(batch_state_.completed_count)});
  completed_cell_ids.clear();
  batch_state_.ReleaseBuffer(batch_state_.completed_buffer_index);
  batch_state_.completed_buffer_index = 0;
  batch_state_.completed_count = 0;
  batch_state_.completed_batch_pending = false;
  TryNotifyFollowingAngleSets();
}

void
CBCD_AngleSet::TryNotifyFollowingAngleSets()
{
  if (following_angle_sets_notified_)
    return;

  if (following_angle_sets_.empty())
  {
    following_angle_sets_notified_ = true;
    return;
  }

  if (pending_reflecting_tasks_ != 0)
    return;

  // Cycles-4 manages reflecting-boundary lifecycle through the per-groupset
  // `CopyDelayedAngularFluxOldToNew`/`CopyDelayedAngularFluxNewToOld` rotation instead of
  // the previous per-angle-set ready-status protocol, so no extra notification of the
  // reflecting boundaries is required here.
  //
  // `following_angle_sets_` (inherited from `AngleSet`) is populated by the base's
  // `UpdateSweepDependencies`.  Each follower is a `CBCD_AngleSet`, so the static_cast
  // is safe and gives access to the atomic dependency counter.
  for (auto* angle_set : following_angle_sets_)
  {
    auto* cbcd_angle_set = static_cast<CBCD_AngleSet*>(angle_set);
    auto remaining = cbcd_angle_set->dependency_counter_.load(std::memory_order_relaxed);
    do
    {
      if (remaining == 0)
        throw std::logic_error("CBCD angle set: angle-set dependency counter underflow.");
    } while (not cbcd_angle_set->dependency_counter_.compare_exchange_weak(
      remaining, remaining - 1, std::memory_order_release, std::memory_order_relaxed));
  }
  following_angle_sets_notified_ = true;
}

bool
CBCD_AngleSet::TryInitialize(CBCDSweepChunk& sweep_chunk)
{
  if (boundary_data_initialized_)
    return false;
  if (dependency_counter_.load(std::memory_order_acquire) != 0)
    return false;

  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::TryInitialize");

  cbcd_fluds_.CopyDelayedPsiToDevice();
  cbcd_fluds_.CopyIncomingBoundaryPsiToDevice(sweep_chunk, this);
  InitializeTaskState();
  boundary_data_initialized_ = true;
  return true;
}

bool
CBCD_AngleSet::TryAdvanceOneStep(CBCDSweepChunk& cbcd_sweep_chunk, const std::size_t worker_id)
{
  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::TryAdvanceOneStep");

  if (executed_ or (not boundary_data_initialized_))
    return false;

  auto& ready_cell_ids = cbcd_fluds_.GetLocalCellIDs(batch_state_.ready_buffer_index);
  const bool kernel_completed = batch_state_.kernel_in_flight and stream_.is_completed();
  const bool has_incoming = async_comm_ != nullptr and async_comm_->HasIncoming(GetID());
  const bool local_work_complete = (num_completed_tasks_ == num_tasks_) and
                                   (not batch_state_.kernel_in_flight) and
                                   (not batch_state_.completed_batch_pending);
  const bool can_signal_completion = local_work_complete and (not local_completion_signaled_);
  const bool can_finalize =
    local_work_complete and local_completion_signaled_ and
    (async_comm_ == nullptr or (async_comm_->AreDelayedReceivesComplete(GetID()) and
                                (not async_comm_->HasIncoming(GetID()))));

  if ((not kernel_completed) and (not batch_state_.completed_batch_pending) and
      ready_cell_ids.empty() and (not has_incoming) and (not can_signal_completion) and
      (not can_finalize))
    return false;

  bool work_done = false;

  // Retire a completed kernel batch before processing new arrivals.
  if (kernel_completed)
  {
    CALI_CXX_MARK_SCOPE("CBCD_AngleSet::RetireBatch");
    work_done |= TryRetireCompletedBatch();
  }

  // Consume any newly received non-local face data and release newly ready cells.
  if (has_incoming)
  {
    CALI_CXX_MARK_SCOPE("CBCD_AngleSet::ProcessIncoming");
    work_done |= async_comm_->ProcessIncoming(
      GetID(),
      [this](const IncomingFaceBatch& batch)
      {
        const auto* psi_base = batch.psi_data.data();
        if (batch.kind == CBCDMessageKind::NORMAL_FACE_PSI)
        {
          // Normal traffic populates the active task graph: each face payload unblocks the
          // owning cell once its incoming dependency count reaches zero.
          for (const auto& entry : batch.entries)
          {
            const auto cell_local_id =
              cbcd_fluds_.ScatterReceivedFaceData(batch.source_slot,
                                                  entry.source_face_index,
                                                  psi_base + entry.payload_offset,
                                                  entry.payload_size);
            auto& remaining = remaining_deps_[cell_local_id];
            if (remaining == 0)
              throw std::logic_error("CBCD angle set: nonlocal dependency counter underflow.");
            if (--remaining == 0)
              cbcd_fluds_.GetLocalCellIDs(batch_state_.ready_buffer_index)
                .push_back(static_cast<std::uint32_t>(cell_local_id));
          }
        }
        else
        {
          // Delayed traffic only populates the lagged-incoming `_new` bank. It must not
          // touch dependency counters because the corresponding edge is, by construction,
          // not part of the current-iteration task graph.
          assert(batch.kind == CBCDMessageKind::DELAYED_FACE_PSI);
          for (const auto& entry : batch.entries)
            cbcd_fluds_.ScatterReceivedDelayedFaceData(batch.source_slot,
                                                       entry.source_face_index,
                                                       psi_base + entry.payload_offset,
                                                       entry.payload_size);
        }
      });
  }

  // Launch the next batch once the stream is idle.
  if ((not batch_state_.kernel_in_flight) and (not ready_cell_ids.empty()))
  {
    CALI_CXX_MARK_SCOPE("CBCD_AngleSet::LaunchBatch");
    work_done |= TryLaunchReadyBatch(cbcd_sweep_chunk);
  }

  // Flush the completed batch after launching the next one so host packing
  // overlaps with device execution when another batch is ready.
  if (batch_state_.completed_batch_pending)
  {
    CALI_CXX_MARK_SCOPE("CBCD_AngleSet::FlushBatch");
    FlushCompletedBatch(cbcd_sweep_chunk, worker_id);
    work_done = true;
  }

  // Publish delayed-output completion as soon as local work is done. This must precede
  // waiting for delayed input; otherwise a distributed cycle can make every rank wait
  // before any rank emits the marker that breaks the wait.
  const bool all_local_work_complete = (num_completed_tasks_ == num_tasks_) and
                                       (not batch_state_.kernel_in_flight) and
                                       (not batch_state_.completed_batch_pending);
  if (all_local_work_complete and (not local_completion_signaled_))
  {
    if (async_comm_ != nullptr)
      async_comm_->SignalAngleSetComplete(GetID(), worker_id);
    local_completion_signaled_ = true;
    work_done = true;
  }

  // Keep the angle set assigned to its worker until every delayed payload has been
  // published before its source's completion marker and drained from this mailbox. The
  // completion acquire followed by the mailbox check closes the publish/observe ordering.
  const bool communication_complete =
    async_comm_ == nullptr or
    (async_comm_->AreDelayedReceivesComplete(GetID()) and (not async_comm_->HasIncoming(GetID())));
  if (all_local_work_complete and local_completion_signaled_ and communication_complete)
  {
    CALI_CXX_MARK_SCOPE("CBCD_AngleSet::FinalizeCompletion");
    TryNotifyFollowingAngleSets();
    executed_ = true;
    cbcd_fluds_.CopyDelayedLocalPsiFromDevice();
    cbcd_fluds_.CopySavedPsiFromDevice();
    cbcd_fluds_.SynchronizeHostData();
    cbcd_fluds_.CopySavedPsiToDestinationPsi(cbcd_sweep_chunk, this);
    return true;
  }

  return work_done;
}

AngleSetStatus
CBCD_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  auto& cbcd_sweep_chunk = static_cast<CBCDSweepChunk&>(sweep_chunk);
  if (not boundary_data_initialized_)
  {
    if (not TryInitialize(cbcd_sweep_chunk))
      return AngleSetStatus::NOT_FINISHED;
  }

  while (not executed_)
  {
    if (TryAdvanceOneStep(cbcd_sweep_chunk, 0))
      continue;
    std::this_thread::yield();
  }

  return AngleSetStatus::FINISHED;
}

void
CBCD_AngleSet::ResetSweepBuffers()
{
  batch_state_.Reset();
  for (std::size_t i = 0; i < 3; ++i)
    cbcd_fluds_.GetLocalCellIDs(i).clear();
  cbcd_fluds_.ClearLocalAndReceivePsi();
  num_completed_tasks_ = 0;
  pending_reflecting_tasks_ = 0;
  boundary_data_initialized_ = false;
  following_angle_sets_notified_ = false;
  local_completion_signaled_ = false;
  ResetDependencyCounter();
  executed_ = false;
}

} // namespace opensn

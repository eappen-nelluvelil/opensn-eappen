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
#include "framework/utils/error.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cstring>

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
    device_angle_indices_(angles_.size()),
    device_queue_state_(1),
    device_completed_count_(1),
    device_publication_count_(1)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  use_device_closure_ = true;
#endif
  cbcd_fluds_.GetStream() = stream_;
  cbcd_fluds_.AllocateLocalAndSavedPsi();
  BuildCellTaskGraph();
}

void
CBCD_AngleSet::RefreshDeviceData()
{
  crb::MemoryPinningManager angle_indices_pinner(angles_);
  crb::copy(device_angle_indices_, angle_indices_pinner, angles_.size(), 0, 0, stream_);
  stream_.synchronize();

  BuildReflectingCellMask();
  BuildPublicationMask();
  cbcd_fluds_.BuildReflectingBoundaryPlans(boundaries_);
}

void
CBCD_AngleSet::ResetSweepDependencies()
{
  unresolved_sweep_dependencies_.store(num_dependencies_, std::memory_order_relaxed);
}

bool
CBCD_AngleSet::IsOutgoingReflectingFace(const CellFace& face,
                                        const std::uint64_t cell_local_id,
                                        const std::size_t face_id) const
{
  if ((face.has_neighbor) or
      (cbc_spds_.GetCellFaceOrientations()[cell_local_id][face_id] != FaceOrientation::OUTGOING))
    return false;
  const auto boundary_it = boundaries_.find(face.neighbor_id);
  return boundary_it != boundaries_.end() and boundary_it->second->IsReflecting();
}

void
CBCD_AngleSet::BuildReflectingCellMask()
{
  const auto& task_list = cbc_spds_.GetTaskList();
  has_outgoing_reflecting_face_.assign(task_list.size(), 0);
  num_reflecting_cells_ = 0;

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
      has_outgoing_reflecting_face_[task_idx] = 1;
      ++num_reflecting_cells_;
    }
  }
}

void
CBCD_AngleSet::BuildPublicationMask()
{
  const auto num_cells = cbc_spds_.GetTaskList().size();
  const auto& common_data = cbcd_fluds_.GetCommonData();
  for (std::size_t cell_local_id = 0; cell_local_id < num_cells; ++cell_local_id)
    if (has_outgoing_reflecting_face_[cell_local_id] != 0 or
        not common_data.GetOutgoingNonlocalFaces(cell_local_id).empty())
      cell_flags_[cell_local_id] |= CBCD_CELL_REQUIRES_PUBLICATION;

  if (use_device_closure_ and std::any_of(cell_flags_.begin(),
                                          cell_flags_.end(),
                                          [](const auto flags) { return flags != 0; }))
  {
    device_cell_flags_ = crb::DeviceMemory<std::uint8_t>(num_cells);
    crb::copy(device_cell_flags_, cell_flags_, num_cells);
  }
}

void
CBCD_AngleSet::BuildCellTaskGraph()
{
  const auto& task_list = cbc_spds_.GetTaskList();
  const auto num_cells = task_list.size();

  initial_cell_dependencies_.resize(num_cells);
  remaining_cell_dependencies_.resize(num_cells);
  initial_local_dependencies_.assign(num_cells, 0);
  initial_remote_dependencies_.assign(num_cells, 0);
  remaining_remote_dependencies_.assign(num_cells, 0);
  locally_ready_.assign(num_cells, 0);
  cell_flags_.assign(num_cells, 0);
  cell_successor_offsets_.assign(num_cells + 1, 0);
  initial_ready_cell_ids_.clear();
  initial_ready_cell_ids_.reserve(num_cells);

  for (std::size_t task_idx = 0; task_idx < task_list.size(); ++task_idx)
  {
    const auto& task = task_list[task_idx];
    initial_cell_dependencies_[task_idx] = task.num_dependencies;
    cell_successor_offsets_[task_idx + 1] = static_cast<std::uint32_t>(task.successors.size());
    if (task.num_dependencies == 0)
      initial_ready_cell_ids_.push_back(static_cast<std::uint32_t>(task_idx));
  }

  for (std::size_t task_idx = 0; task_idx < num_cells; ++task_idx)
    cell_successor_offsets_[task_idx + 1] += cell_successor_offsets_[task_idx];

  cell_successors_.resize(cell_successor_offsets_.back());
  for (std::size_t task_idx = 0; task_idx < task_list.size(); ++task_idx)
  {
    const auto& task = task_list[task_idx];
    std::copy(task.successors.begin(),
              task.successors.end(),
              cell_successors_.begin() + cell_successor_offsets_[task_idx]);
    for (const auto successor : task.successors)
      ++initial_local_dependencies_[successor];
  }

  for (std::size_t cell_local_id = 0; cell_local_id < num_cells; ++cell_local_id)
  {
    initial_remote_dependencies_[cell_local_id] =
      initial_cell_dependencies_[cell_local_id] - initial_local_dependencies_[cell_local_id];
    if (initial_remote_dependencies_[cell_local_id] != 0)
      cell_flags_[cell_local_id] |= CBCD_CELL_HAS_REMOTE_PREDECESSOR;
  }

  if (use_device_closure_)
  {
    device_remaining_local_dependencies_ = crb::DeviceMemory<std::uint32_t>(num_cells);
    device_cell_successor_offsets_ =
      crb::DeviceMemory<std::uint32_t>(cell_successor_offsets_.size());
    device_cell_successors_ = crb::DeviceMemory<std::uint32_t>(cell_successors_.size());
    device_cell_queue_ = crb::DeviceMemory<std::uint32_t>(num_cells);
    crb::copy(
      device_cell_successor_offsets_, cell_successor_offsets_, cell_successor_offsets_.size());
    if (not cell_successors_.empty())
      crb::copy(device_cell_successors_, cell_successors_, cell_successors_.size());
  }
}

void
CBCD_AngleSet::InitializeSweepState()
{
  std::copy(initial_cell_dependencies_.begin(),
            initial_cell_dependencies_.end(),
            remaining_cell_dependencies_.begin());
  std::copy(initial_remote_dependencies_.begin(),
            initial_remote_dependencies_.end(),
            remaining_remote_dependencies_.begin());
  for (std::size_t cell_local_id = 0; cell_local_id < locally_ready_.size(); ++cell_local_id)
    locally_ready_[cell_local_id] = initial_local_dependencies_[cell_local_id] == 0;
  if (use_device_closure_)
    crb::copy(device_remaining_local_dependencies_,
              initial_local_dependencies_,
              initial_local_dependencies_.size(),
              0,
              0,
              stream_);
  batch_pipeline_.Reset();
  auto& ready_cell_ids = cbcd_fluds_.GetCellBatchBuffer(batch_pipeline_.ready_buffer);
  std::copy(initial_ready_cell_ids_.begin(), initial_ready_cell_ids_.end(), ready_cell_ids.begin());
  batch_pipeline_.ready_count = static_cast<std::uint32_t>(initial_ready_cell_ids_.size());
  num_completed_cells_ = 0;
  pending_reflecting_cells_ = following_angle_sets_.empty() ? 0 : num_reflecting_cells_;
}

bool
CBCD_AngleSet::TryRetireCompletedBatch(CBCDSweepChunk& sweep_chunk, const bool dispatch_completed)
{
  if ((not batch_pipeline_.HasKernelInFlight()) or (not dispatch_completed))
    return false;

  auto& completed_cell_ids = cbcd_fluds_.GetCellBatchBuffer(batch_pipeline_.launch_buffer);
  auto& ready_cell_ids = cbcd_fluds_.GetCellBatchBuffer(batch_pipeline_.ready_buffer);
  const auto completed_count =
    use_device_closure_ ? device_completed_count_.front() : batch_pipeline_.launch_count;
  const auto publication_count =
    use_device_closure_ ? device_publication_count_.front() : completed_count;
  for (std::uint32_t i = 0; i < publication_count; ++i)
  {
    const auto cell_local_id = completed_cell_ids[i];
    if (not use_device_closure_)
    {
      const auto succ_begin = cell_successor_offsets_[cell_local_id];
      const auto succ_end = cell_successor_offsets_[cell_local_id + 1];
      for (auto succ_i = succ_begin; succ_i < succ_end; ++succ_i)
      {
        if (--remaining_cell_dependencies_[cell_successors_[succ_i]] == 0)
          ready_cell_ids[batch_pipeline_.ready_count++] = cell_successors_[succ_i];
      }
    }

    if ((not following_angle_sets_.empty()) and (not followers_released_) and
        (has_outgoing_reflecting_face_[cell_local_id] != 0))
    {
      --pending_reflecting_cells_;
    }
  }

  if (auto* profiler = sweep_chunk.GetProfiler())
    profiler->RecordKernelLaunch(GetID(), completed_count);
  num_completed_cells_ += completed_count;
  batch_pipeline_.completed_buffer = batch_pipeline_.launch_buffer;
  batch_pipeline_.completed_count = completed_count;
  batch_pipeline_.publication_count = publication_count;
  batch_pipeline_.launch_count = 0;
  return true;
}

bool
CBCD_AngleSet::HasReadyBatch() const
{
  return (not batch_pipeline_.HasKernelInFlight()) and batch_pipeline_.ready_count != 0;
}

std::span<std::uint32_t>
CBCD_AngleSet::PrepareReadyBatch()
{
  auto& ready_cell_ids = cbcd_fluds_.GetCellBatchBuffer(batch_pipeline_.ready_buffer);
  const auto launch_count = batch_pipeline_.ready_count;
  batch_pipeline_.launch_buffer = batch_pipeline_.ready_buffer;
  batch_pipeline_.launch_count = launch_count;
  batch_pipeline_.ready_buffer = batch_pipeline_.AcquireFreeBuffer();
  batch_pipeline_.ready_count = 0;
  if (use_device_closure_)
  {
    device_completed_count_.front() = 0;
    device_publication_count_.front() = 0;
#if defined(__NVCC__) || defined(__HIPCC__)
    crb::impl::copy_h2d(device_cell_queue_.get(), ready_cell_ids.data(), launch_count, stream_);
#endif
  }
  return {ready_cell_ids.data(), launch_count};
}

void
CBCD_AngleSet::PublishCompletedBatch(CBCDSweepChunk& sweep_chunk, const std::size_t worker_id)
{
  if (not batch_pipeline_.HasCompletedBatch())
    return;

  auto& completed_cell_ids = cbcd_fluds_.GetCellBatchBuffer(batch_pipeline_.completed_buffer);
  cbcd_fluds_.PublishOutgoingPsi(
    sweep_chunk,
    *async_comm_,
    worker_id,
    GetID(),
    GetAngleIndices(),
    {completed_cell_ids.data(), static_cast<std::size_t>(batch_pipeline_.publication_count)});
  batch_pipeline_.ReleaseBuffer(batch_pipeline_.completed_buffer);
  batch_pipeline_.completed_count = 0;
  batch_pipeline_.publication_count = 0;
  TryReleaseFollowers();
}

void
CBCD_AngleSet::TryReleaseFollowers()
{
  if (followers_released_)
    return;

  if (following_angle_sets_.empty())
  {
    followers_released_ = true;
    return;
  }

  if (pending_reflecting_cells_ != 0)
    return;

  // Publish reflecting-boundary writes before releasing follower dependencies.
  for (auto* angle_set : following_angle_sets_)
  {
    auto* cbcd_angle_set = static_cast<CBCD_AngleSet*>(angle_set);
    cbcd_angle_set->unresolved_sweep_dependencies_.fetch_sub(1, std::memory_order_release);
  }
  followers_released_ = true;
}

bool
CBCD_AngleSet::TryInitialize(CBCDSweepChunk& sweep_chunk)
{
  if (sweep_initialized_)
    return false;
  if (unresolved_sweep_dependencies_.load(std::memory_order_acquire) != 0)
    return false;

  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::TryInitialize");

  cbcd_fluds_.LoadIncomingBoundaryPsi(sweep_chunk, *this);
  InitializeSweepState();
  sweep_initialized_ = true;
  return true;
}

bool
CBCD_AngleSet::TryAdvanceOneStep(CBCDSweepChunk& cbcd_sweep_chunk,
                                 const std::size_t worker_id,
                                 const bool dispatch_completed)
{
  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::TryAdvanceOneStep");

  if (executed_ or (not sweep_initialized_))
    return false;

  auto& ready_cell_ids = cbcd_fluds_.GetCellBatchBuffer(batch_pipeline_.ready_buffer);
  const bool kernel_completed = batch_pipeline_.HasKernelInFlight() and dispatch_completed;
  const bool has_incoming = async_comm_->HasIncoming(GetID());
  const bool can_finalize = (num_completed_cells_ == initial_cell_dependencies_.size()) and
                            (not batch_pipeline_.HasKernelInFlight()) and
                            (not batch_pipeline_.HasCompletedBatch());

  if ((not kernel_completed) and (not batch_pipeline_.HasCompletedBatch()) and
      batch_pipeline_.ready_count == 0 and (not has_incoming) and (not can_finalize))
    return false;

  bool work_done = false;

  if (kernel_completed)
  {
    CALI_CXX_MARK_SCOPE("CBCD_AngleSet::RetireBatch");
    work_done |= TryRetireCompletedBatch(cbcd_sweep_chunk, dispatch_completed);
  }

  if (has_incoming and (not batch_pipeline_.HasKernelInFlight()))
  {
    CALI_CXX_MARK_SCOPE("CBCD_AngleSet::ProcessIncoming");
    work_done |= async_comm_->ProcessIncoming(
      GetID(),
      [this, &ready_cell_ids](const IncomingFaceBatch& batch)
      {
        const auto* psi_base = batch.psi_values.data();
        for (const auto& face : batch.faces)
        {
          const auto cell_local_id = cbcd_fluds_.StoreIncomingFace(
            batch.source_partition_index, face.incoming_face_index, psi_base + face.psi_offset);
          if (use_device_closure_)
          {
            if (--remaining_remote_dependencies_[cell_local_id] == 0 and
                locally_ready_[cell_local_id] != 0)
              ready_cell_ids[batch_pipeline_.ready_count++] = cell_local_id;
          }
          else if (--remaining_cell_dependencies_[cell_local_id] == 0)
            ready_cell_ids[batch_pipeline_.ready_count++] = cell_local_id;
        }
      });
  }

  // Pack after retirement while the scheduler assembles the next dispatch.
  if (batch_pipeline_.HasCompletedBatch())
  {
    CALI_CXX_MARK_SCOPE("CBCD_AngleSet::FlushBatch");
    PublishCompletedBatch(cbcd_sweep_chunk, worker_id);
    work_done = true;
  }

  if (num_completed_cells_ == initial_cell_dependencies_.size() and
      (not batch_pipeline_.HasKernelInFlight()) and (not batch_pipeline_.HasCompletedBatch()))
  {
    CALI_CXX_MARK_SCOPE("CBCD_AngleSet::FinalizeCompletion");
    async_comm_->SignalAngleSetComplete(GetID());
    TryReleaseFollowers();
    executed_ = true;
    cbcd_fluds_.CopySavedPsiFromDevice();
    cbcd_fluds_.StoreSavedPsi(cbcd_sweep_chunk, *this);
    return true;
  }

  return work_done;
}

AngleSetStatus
CBCD_AngleSet::AngleSetAdvance(SweepChunk&, AngleSetStatus)
{
  OpenSnLogicalError("Device CBC angle sets are advanced only by the ASYNC_FIFO scheduler.");
}

void
CBCD_AngleSet::ResetSweepBuffers()
{
  batch_pipeline_.Reset();
  cbcd_fluds_.ClearLocalAndReceivePsi();
  num_completed_cells_ = 0;
  pending_reflecting_cells_ = 0;
  sweep_initialized_ = false;
  followers_released_ = false;
  ResetSweepDependencies();
  executed_ = false;
}

CBCDDeviceScheduler
CBCD_AngleSet::GetDeviceScheduler()
{
  return {device_queue_state_.get(),
          device_cell_queue_.get(),
          device_remaining_local_dependencies_.get(),
          device_cell_successor_offsets_.get(),
          device_cell_successors_.get(),
          device_cell_flags_.get(),
          locally_ready_.data(),
          remaining_remote_dependencies_.data(),
          device_completed_count_.data(),
          device_publication_count_.data()};
}

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cassert>

namespace opensn
{

CBCD_AngleSet::CBCD_AngleSet(size_t id,
                             size_t num_groups,
                             const SPDS& spds,
                             std::shared_ptr<FLUDS>& fluds,
                             const std::vector<size_t>& angle_indices,
                             std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                             const MPICommunicatorSet& comm_set)
  : AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds)),
    comm_set_(comm_set),
    stream_(crb::Stream::create()),
    device_angle_indices_(angles_.size(), stream_)
{
  crb::MemoryPinningManager angle_indices_pinner_(angles_);
  crb::copy(device_angle_indices_, angle_indices_pinner_, angles_.size(), 0, 0, stream_);
  // Set CBCD_FLUDS stream and asynchronously allocate storage for local psi
  auto* cbcd_fluds = std::static_pointer_cast<CBCD_FLUDS>(fluds_).get();
  cbcd_fluds->GetStream() = stream_;
  cbcd_fluds->AllocateLocalAndSavedPsi();
  cbcd_fluds->InitializeReflectingBoundaryNodes(boundaries_);
  InitializeTaskGraphData();
  AllocateDeviceTaskState();
  cbc_spds_.CopyTaskGraphDataOnDevice();
  InitializeReflectingTaskMask();
}

CBCD_AngleSet::~CBCD_AngleSet()
{
  if (device_remaining_deps_.get())
    device_remaining_deps_.async_free(stream_);
  if (device_remaining_successors_to_retire_.get())
    device_remaining_successors_to_retire_.async_free(stream_);
  if (device_ready_task_indices_.get())
    device_ready_task_indices_.async_free(stream_);
  if (device_ready_task_count_.get())
    device_ready_task_count_.async_free(stream_);
  device_angle_indices_.async_free(stream_);
}

AsynchronousCommunicator*
CBCD_AngleSet::GetCommunicator()
{
  return nullptr;
}

CBCD_AsynchronousCommunicator&
CBCD_AngleSet::GetAsyncCommunicator()
{
  OpenSnLogicalErrorIf(async_comm_ == nullptr, "CBCD angle set communicator has not been bound");
  return *async_comm_;
}

void
CBCD_AngleSet::UpdateSweepDependencies(std::set<AngleSet*>& following_angle_sets)
{
  std::transform(following_angle_sets.begin(),
                 following_angle_sets.end(),
                 std::back_inserter(following_angle_sets_),
                 [](AngleSet* as) { return static_cast<CBCD_AngleSet*>(as); });
  for (auto* following_angle_set : following_angle_sets_)
    ++(following_angle_set->num_dependencies_);
}

void
CBCD_AngleSet::ResetDependencyCounter()
{
  dependency_counter_.store(num_dependencies_, std::memory_order_relaxed);
}

bool
CBCD_AngleSet::IsOutgoingReflectingFace(const CellFace& face,
                                        const std::uint64_t cell_local_id,
                                        const std::size_t face_id) const
{
  if (face.has_neighbor)
    return false;
  if (cbc_spds_.GetCellFaceOrientations()[cell_local_id][face_id] != FaceOrientation::OUTGOING)
    return false;

  const auto boundary_it = boundaries_.find(face.neighbor_id);
  return boundary_it != boundaries_.end() and boundary_it->second->IsReflecting();
}

void
CBCD_AngleSet::InitializeReflectingTaskMask()
{
  const auto& task_list = cbc_spds_.GetTaskList();
  task_has_outgoing_reflecting_boundary_.assign(task_list.size(), 0);

  for (std::size_t task_idx = 0; task_idx < task_list.size(); ++task_idx)
  {
    const auto& cell = *task_list[task_idx].cell_ptr;
    const bool has_outgoing_reflecting_face = std::any_of(
      cell.faces.begin(),
      cell.faces.end(),
      [this, cell_local_id = cell.local_id, face_id = std::size_t{0}](const CellFace& face) mutable
      {
        const bool is_outgoing_reflecting_face =
          IsOutgoingReflectingFace(face, cell_local_id, face_id);
        ++face_id;
        return is_outgoing_reflecting_face;
      });

    if (has_outgoing_reflecting_face)
    {
      task_has_outgoing_reflecting_boundary_[task_idx] = 1;
      ++initial_reflecting_task_count_;
    }
  }
}

void
CBCD_AngleSet::InitializeTaskGraphData()
{
  if (not reference_ids_.empty())
    return;

  const auto& task_list = cbc_spds_.GetTaskList();
  const auto num_tasks = task_list.size();

  reference_ids_.resize(num_tasks);
  initial_deps_.resize(num_tasks);
  initial_successors_to_retire_.resize(num_tasks);
  successor_offsets_.assign(num_tasks + 1, 0);
  predecessor_offsets_.assign(num_tasks + 1, 0);
  initial_ready_tasks_.clear();
  initial_ready_tasks_.reserve(num_tasks);

  for (std::size_t task_idx = 0; task_idx < num_tasks; ++task_idx)
  {
    const auto& task = task_list[task_idx];
    reference_ids_[task_idx] = static_cast<std::uint32_t>(task.reference_id);
    initial_deps_[task_idx] = static_cast<int>(task.num_dependencies);
    initial_successors_to_retire_[task_idx] = static_cast<std::uint32_t>(task.successors.size());
    successor_offsets_[task_idx + 1] = static_cast<std::uint32_t>(task.successors.size());
    predecessor_offsets_[task_idx + 1] = static_cast<std::uint32_t>(task.predecessors.size());
    if (task.num_dependencies == 0)
      initial_ready_tasks_.push_back(static_cast<std::uint32_t>(task_idx));
  }

  for (std::size_t task_idx = 0; task_idx < num_tasks; ++task_idx)
  {
    successor_offsets_[task_idx + 1] += successor_offsets_[task_idx];
    predecessor_offsets_[task_idx + 1] += predecessor_offsets_[task_idx];
  }

  successor_data_.resize(successor_offsets_.back());
  predecessor_data_.resize(predecessor_offsets_.back());
  for (std::size_t task_idx = 0; task_idx < num_tasks; ++task_idx)
  {
    const auto& task = task_list[task_idx];
    std::copy(task.successors.begin(),
              task.successors.end(),
              successor_data_.begin() + successor_offsets_[task_idx]);
    std::copy(task.predecessors.begin(),
              task.predecessors.end(),
              predecessor_data_.begin() + predecessor_offsets_[task_idx]);
  }
}

void
CBCD_AngleSet::InitializeTaskState()
{
  remaining_deps_ = initial_deps_;
  remaining_successors_to_retire_ = initial_successors_to_retire_;
  ready_queue_ = initial_ready_tasks_;
  in_flight_task_indices_.clear();
  in_flight_cell_ids_.clear();
  num_completed_tasks_ = 0;
  pending_reflecting_tasks_ = initial_reflecting_task_count_;
  ResetDeviceTaskState();
}

void
CBCD_AngleSet::AllocateDeviceTaskState()
{
  const auto num_tasks = reference_ids_.size();
  if (num_tasks == 0 or device_remaining_deps_.get() != nullptr)
    return;

  device_remaining_deps_ = crb::DeviceMemory<int>(num_tasks, stream_);
  device_remaining_successors_to_retire_ = crb::DeviceMemory<std::uint32_t>(num_tasks, stream_);
  device_ready_task_indices_ = crb::DeviceMemory<std::uint32_t>(num_tasks, stream_);
  device_ready_task_count_ = crb::DeviceMemory<std::uint32_t>(1, stream_);

  device_task_state_.remaining_dependencies = device_remaining_deps_.get();
  device_task_state_.remaining_successors_to_retire = device_remaining_successors_to_retire_.get();
  device_task_state_.ready_task_indices = device_ready_task_indices_.get();
  device_task_state_.ready_task_count = device_ready_task_count_.get();
  device_task_state_.num_tasks = static_cast<std::uint32_t>(num_tasks);

  ResetDeviceTaskState();
}

void
CBCD_AngleSet::ResetDeviceTaskState()
{
  if (reference_ids_.empty())
    return;

  crb::HostVector<int> host_initial_deps(initial_deps_.begin(), initial_deps_.end());
  crb::HostVector<std::uint32_t> host_initial_successors_to_retire(
    initial_successors_to_retire_.begin(), initial_successors_to_retire_.end());
  crb::HostVector<std::uint32_t> host_initial_ready_tasks(initial_ready_tasks_.begin(),
                                                          initial_ready_tasks_.end());
  crb::HostVector<std::uint32_t> host_ready_task_count(1);
  host_ready_task_count.front() = static_cast<std::uint32_t>(initial_ready_tasks_.size());

  crb::copy(device_remaining_deps_, host_initial_deps, host_initial_deps.size(), 0, 0, stream_);
  crb::copy(device_remaining_successors_to_retire_,
            host_initial_successors_to_retire,
            host_initial_successors_to_retire.size(),
            0,
            0,
            stream_);
  if (not host_initial_ready_tasks.empty())
    crb::copy(device_ready_task_indices_,
              host_initial_ready_tasks,
              host_initial_ready_tasks.size(),
              0,
              0,
              stream_);
  crb::copy(device_ready_task_count_, host_ready_task_count, 1, 0, 0, stream_);
}

void
CBCD_AngleSet::NotifyFollowingAngleSets()
{
  for (auto* following_angle_set : following_angle_sets_)
  {
    const auto old_value =
      following_angle_set->dependency_counter_.fetch_sub(1, std::memory_order_acq_rel);
    assert(old_value > 0);
  }
}

void
CBCD_AngleSet::TryNotifyFollowingAngleSets()
{
  if (following_angle_sets_notified_ or pending_reflecting_tasks_ != 0)
    return;

  for (auto& [bid, boundary] : boundaries_)
    boundary->UpdateAnglesReadyStatus(angles_);
  NotifyFollowingAngleSets();
  following_angle_sets_notified_ = true;
}

bool
CBCD_AngleSet::TryInitialize(CBCDSweepChunk& sweep_chunk)
{
  if (boundary_data_initialized_)
    return false;
  if (dependency_counter_.load(std::memory_order_acquire) != 0)
    return false;

  sweep_chunk_ = &sweep_chunk;
  InitializeTaskGraphData();
  auto* cbcd_fluds = static_cast<CBCD_FLUDS*>(fluds_.get());
  cbcd_fluds->CopyIncomingBoundaryPsiToDevice(sweep_chunk, this);
  InitializeTaskState();
  boundary_data_initialized_ = true;
  return true;
}

bool
CBCD_AngleSet::TryAdvanceOneStep()
{
  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::TryAdvanceOneStep");

  if (executed_ or not boundary_data_initialized_)
    return false;

  auto& cbcd_sweep_chunk = *sweep_chunk_;
  auto* cbcd_fluds = static_cast<CBCD_FLUDS*>(fluds_.get());

  bool work_done = false;

  if (kernel_in_flight_ and stream_.is_completed())
  {
    cbcd_fluds->CopyOutgoingPsiBackToHost(cbcd_sweep_chunk, this, in_flight_cell_ids_);

    for (const auto task_idx : in_flight_task_indices_)
    {
      const auto succ_begin = successor_offsets_[task_idx];
      const auto succ_end = successor_offsets_[task_idx + 1];
      for (auto succ_i = succ_begin; succ_i < succ_end; ++succ_i)
      {
        const auto succ = successor_data_[succ_i];
        if (--remaining_deps_[succ] == 0)
          ready_queue_.push_back(succ);
      }

      const auto pred_begin = predecessor_offsets_[task_idx];
      const auto pred_end = predecessor_offsets_[task_idx + 1];
      for (auto pred_i = pred_begin; pred_i < pred_end; ++pred_i)
      {
        const auto pred = predecessor_data_[pred_i];
        assert(remaining_successors_to_retire_[pred] > 0);
        --remaining_successors_to_retire_[pred];
      }

      if (task_has_outgoing_reflecting_boundary_[task_idx] != 0)
      {
        assert(pending_reflecting_tasks_ > 0);
        --pending_reflecting_tasks_;
      }
    }
    num_completed_tasks_ += in_flight_task_indices_.size();
    in_flight_task_indices_.clear();
    in_flight_cell_ids_.clear();
    kernel_in_flight_ = false;
    work_done = true;
    TryNotifyFollowingAngleSets();
  }

  work_done = async_comm_->DrainIncoming(
                GetID(),
                [this, cbcd_fluds](const CBCD_AsynchronousCommunicator::IncomingSection& section)
                {
                  const auto* ptr = section.Data();
                  const size_t num_entries = CBCD_AsynchronousCommunicator::Wire::LoadSize(ptr);
                  for (size_t e = 0; e < num_entries; ++e)
                  {
                    const auto entry_header =
                      CBCD_AsynchronousCommunicator::Wire::LoadEntryHeader(ptr);
                    const auto* psi_data = reinterpret_cast<const double*>(ptr);
                    ptr += entry_header.data_size * sizeof(double);

                    const auto task_id = cbcd_fluds->ScatterReceivedFaceData(
                      entry_header.cell_global_id, entry_header.face_id, psi_data);
                    if (--remaining_deps_[task_id] == 0)
                      ready_queue_.push_back(static_cast<std::uint32_t>(task_id));
                  }
                }) ||
              work_done;

  if ((not kernel_in_flight_) and (not ready_queue_.empty()))
  {
    in_flight_task_indices_ = std::move(ready_queue_);
    ready_queue_.clear();
    in_flight_cell_ids_.clear();
    in_flight_cell_ids_.reserve(in_flight_task_indices_.size());
    for (const auto task_idx : in_flight_task_indices_)
      in_flight_cell_ids_.push_back(reference_ids_[task_idx]);

    cbcd_fluds->CopyIncomingNonlocalPsiToDevice(this, in_flight_cell_ids_);
    cbcd_sweep_chunk.Sweep(in_flight_cell_ids_, GetID());
    kernel_in_flight_ = true;
    work_done = true;
  }

  const bool all_done = num_completed_tasks_ == reference_ids_.size();
  if (all_done and (not kernel_in_flight_))
  {
    async_comm_->SignalAngleSetComplete(GetID());
    TryNotifyFollowingAngleSets();
    executed_ = true;
    cbcd_fluds->CopySavedPsiFromDevice();
    cbcd_fluds->CopySavedPsiToDestinationPsi(cbcd_sweep_chunk, this);
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
  if (not boundary_data_initialized_)
    return AngleSetStatus::NOT_FINISHED;

  return TryAdvanceOneStep() ? AngleSetStatus::READY_TO_EXECUTE : AngleSetStatus::NOT_FINISHED;
}

void
CBCD_AngleSet::ResetSweepBuffers()
{
  ready_queue_.clear();
  in_flight_task_indices_.clear();
  in_flight_cell_ids_.clear();
  fluds_->ClearLocalAndReceivePsi();
  num_completed_tasks_ = 0;
  pending_reflecting_tasks_ = 0;
  boundary_data_initialized_ = false;
  following_angle_sets_notified_ = false;
  kernel_in_flight_ = false;
  ResetDependencyCounter();
  executed_ = false;
}

const double*
CBCD_AngleSet::PsiBoundary(uint64_t boundary_id,
                           unsigned int angle_num,
                           uint64_t cell_local_id,
                           unsigned int face_num,
                           unsigned int fi,
                           unsigned int g,
                           bool surface_source_active)
{
  if (boundaries_[boundary_id]->IsReflecting())
    return boundaries_[boundary_id]->PsiIncoming(cell_local_id, face_num, fi, angle_num, g);

  if (not surface_source_active)
    return boundaries_[boundary_id]->ZeroFlux(g);

  return boundaries_[boundary_id]->PsiIncoming(cell_local_id, face_num, fi, angle_num, g);
}

double*
CBCD_AngleSet::PsiReflected(uint64_t boundary_id,
                            unsigned int angle_num,
                            uint64_t cell_local_id,
                            unsigned int face_num,
                            unsigned int fi)
{
  return boundaries_[boundary_id]->PsiOutgoing(cell_local_id, face_num, fi, angle_num);
}

} // namespace opensn

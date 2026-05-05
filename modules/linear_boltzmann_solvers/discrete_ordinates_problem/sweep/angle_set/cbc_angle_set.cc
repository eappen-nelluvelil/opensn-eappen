// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "framework/mesh/cell/cell.h"
#include "caliper/cali.h"

namespace opensn
{

bool
CBC_AngleSet::IncomingBoundaryFacesReady(const Cell& cell) const
{
  const auto& face_orientations = cbc_spds_.GetCellFaceOrientations()[cell.local_id];
  for (std::size_t f = 0; f < cell.faces.size(); ++f)
  {
    const auto& face = cell.faces[f];
    if (face.has_neighbor or face_orientations[f] != FaceOrientation::INCOMING)
      continue;

    if (not boundaries_.at(face.neighbor_id)->CheckAnglesReadyStatus(angles_))
      return false;
  }

  return true;
}

CBC_AngleSet::CBC_AngleSet(size_t id,
                           unsigned int num_groups,
                           const SPDS& spds,
                           std::shared_ptr<FLUDS>& fluds,
                           const std::vector<size_t>& angle_indices,
                           std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                           int max_mpi_message_size,
                           const MPICommunicatorSet& comm_set)
  : AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds_)),
    ready_tasks_(),
    async_comm_(id, *fluds, max_mpi_message_size, comm_set)
{
}

AsynchronousCommunicator*
CBC_AngleSet::GetCommunicator()
{
  return &async_comm_;
}

AngleSetStatus
CBC_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBC_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  if (task_list_ == nullptr)
  {
    task_list_ = &cbc_spds_.GetTaskList();
    remaining_dependencies_.resize(task_list_->size());
    completed_tasks_.assign(task_list_->size(), 0);
    ready_tasks_.clear();
    ready_tasks_.reserve(task_list_->size());

    for (std::size_t i = 0; i < task_list_->size(); ++i)
    {
      const auto num_dependencies = (*task_list_)[i].num_dependencies;
      remaining_dependencies_[i] = num_dependencies;
      if (num_dependencies == 0)
        ready_tasks_.push_back(static_cast<std::uint32_t>(i));
    }
  }

  sweep_chunk.SetAngleSet(*this);

  async_comm_.ReceiveData(received_task_buffer_);

  for (const auto task_number : received_task_buffer_)
    if ((--remaining_dependencies_[task_number] == 0) and (not completed_tasks_[task_number]))
      ready_tasks_.push_back(task_number);

  if (async_comm_.HasPendingCommunication())
    async_comm_.SendData();

  if (permission != AngleSetStatus::EXECUTE)
  {
    const bool all_tasks_completed = (num_completed_tasks_ == task_list_->size());
    const bool all_messages_sent =
      (not async_comm_.HasPendingCommunication()) or async_comm_.SendData();
    if (all_tasks_completed and all_messages_sent)
    {
      for (const auto& boundary_entry : boundaries_)
        boundary_entry.second->UpdateAnglesReadyStatus(angles_);
      executed_ = true;
      return AngleSetStatus::FINISHED;
    }

    return ready_tasks_.empty() ? AngleSetStatus::NOT_FINISHED : AngleSetStatus::READY_TO_EXECUTE;
  }

  std::vector<std::uint32_t> boundary_blocked_tasks;
  boundary_blocked_tasks.reserve(ready_tasks_.size());

  while (not ready_tasks_.empty())
  {
    const auto task_idx = ready_tasks_.back();
    ready_tasks_.pop_back();
    const auto& cell_task = (*task_list_)[task_idx];

    if (not IncomingBoundaryFacesReady(*cell_task.cell_ptr))
    {
      boundary_blocked_tasks.push_back(task_idx);
      continue;
    }

    sweep_chunk.SetCell(cell_task.cell_ptr, *this);
    sweep_chunk.Sweep(*this);

    for (const auto& local_task_num : cell_task.successors)
      if ((--remaining_dependencies_[local_task_num] == 0) and
          (not completed_tasks_[local_task_num]))
        ready_tasks_.push_back(local_task_num);

    completed_tasks_[task_idx] = 1;
    ++num_completed_tasks_;
    if (async_comm_.HasPendingCommunication())
      async_comm_.SendData();
  }

  ready_tasks_.insert(
    ready_tasks_.end(), boundary_blocked_tasks.begin(), boundary_blocked_tasks.end());

  const bool all_tasks_completed = (num_completed_tasks_ == task_list_->size());
  const bool all_messages_sent =
    (not async_comm_.HasPendingCommunication()) or async_comm_.SendData();

  if (all_tasks_completed and all_messages_sent)
  {
    // Update boundary readiness
    for (const auto& boundary_entry : boundaries_)
      boundary_entry.second->UpdateAnglesReadyStatus(angles_);
    executed_ = true;
    return AngleSetStatus::FINISHED;
  }

  return AngleSetStatus::NOT_FINISHED;
}

void
CBC_AngleSet::ResetSweepBuffers()
{
  task_list_ = nullptr;
  remaining_dependencies_.clear();
  completed_tasks_.clear();
  ready_tasks_.clear();
  received_task_buffer_.clear();
  num_completed_tasks_ = 0;
  async_comm_.Reset();
  fluds_->ClearLocalAndReceivePsi();
  executed_ = false;
}

const double*
CBC_AngleSet::PsiBoundary(uint64_t boundary_id,
                          unsigned int angle_num,
                          uint64_t cell_local_id,
                          unsigned int face_num,
                          unsigned int fi,
                          unsigned int g,
                          bool surface_source_active)
{
  if ((not boundaries_[boundary_id]->IsReflecting()) and (not surface_source_active))
    return boundaries_[boundary_id]->ZeroFlux(g);
  return boundaries_[boundary_id]->PsiIncoming(cell_local_id, face_num, fi, angle_num, g);
}

double*
CBC_AngleSet::PsiReflected(uint64_t boundary_id,
                           unsigned int angle_num,
                           uint64_t cell_local_id,
                           unsigned int face_num,
                           unsigned int fi)
{
  return boundaries_[boundary_id]->PsiOutgoing(cell_local_id, face_num, fi, angle_num);
}

} // namespace opensn

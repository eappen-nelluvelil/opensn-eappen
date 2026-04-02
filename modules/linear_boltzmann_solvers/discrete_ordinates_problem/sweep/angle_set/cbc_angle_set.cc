// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/data_types/range.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"

namespace opensn
{

CBC_AngleSet::CBC_AngleSet(size_t id,
                           unsigned int num_groups,
                           const SPDS& spds,
                           std::shared_ptr<FLUDS>& fluds,
                           const std::vector<size_t>& angle_indices,
                           std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                           const MPICommunicatorSet& comm_set)
  : AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds_)),
    ready_tasks_(),
    async_comm_(id, *fluds, comm_set),
    cbc_fluds_(dynamic_cast<CBC_FLUDS&>(*fluds_))
{
  const auto num_tasks = cbc_spds_.GetTaskList().size();
  remaining_dependencies_.resize(num_tasks);
  num_satisfied_successors_.resize(num_tasks);
  completed_tasks_.resize(num_tasks);
  ready_tasks_.reserve(num_tasks);
  ResetTaskState();
}

AsynchronousCommunicator*
CBC_AngleSet::GetCommunicator()
{
  return static_cast<AsynchronousCommunicator*>(&async_comm_);
}

AngleSetStatus
CBC_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBC_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  const auto& task_list = cbc_spds_.GetTaskList();
  sweep_chunk.SetAngleSet(*this);

  const auto tasks_who_received_data = async_comm_.ReceiveData();

  for (const std::uint64_t task_number : tasks_who_received_data)
  {
    if ((--remaining_dependencies_[task_number] == 0) and not completed_tasks_[task_number])
      ready_tasks_.push_back(task_number);
  }

  async_comm_.SendData();

  // Check if boundaries allow for execution
  for (auto& [bid, boundary] : boundaries_)
    if (not boundary->CheckAnglesReadyStatus(angles_))
      return AngleSetStatus::NOT_FINISHED;

  while (not ready_tasks_.empty())
  {
    const auto task_idx = ready_tasks_.back();
    ready_tasks_.pop_back();
    if (completed_tasks_[task_idx])
      continue;

    const auto& cell_task = task_list[task_idx];

    cbc_fluds_.AllocateSlot(cell_task.cell_ptr->local_id);
    sweep_chunk.SetCell(cell_task.cell_ptr, *this);
    sweep_chunk.Sweep(*this);

    for (const auto& local_task_num : cell_task.successors)
    {
      if ((--remaining_dependencies_[local_task_num] == 0) and not completed_tasks_[local_task_num])
        ready_tasks_.push_back(local_task_num);
    }

    completed_tasks_[task_idx] = 1;
    ++num_completed_tasks;
    async_comm_.SendData();

    for (const auto predecessor : cell_task.predecessors)
    {
      const auto& predecessor_task = task_list[predecessor];
      auto& num_satisfied = num_satisfied_successors_[predecessor];
      ++num_satisfied;

      if (num_satisfied >= predecessor_task.successors.size())
        cbc_fluds_.DeallocateSlot(predecessor_task.cell_ptr->local_id);
    }

    if (cell_task.successors.empty())
      cbc_fluds_.DeallocateSlot(cell_task.cell_ptr->local_id);
  }

  const bool all_tasks_completed = (num_completed_tasks == task_list.size());
  const bool all_messages_sent = async_comm_.SendData();

  if (all_tasks_completed and all_messages_sent)
  {
    // Update boundary readiness
    for (auto& [bid, boundary] : boundaries_)
      boundary->UpdateAnglesReadyStatus(angles_);
    executed_ = true;
    return AngleSetStatus::FINISHED;
  }

  return AngleSetStatus::NOT_FINISHED;
}

void
CBC_AngleSet::ResetSweepBuffers()
{
  ResetTaskState();
  async_comm_.Reset();
  fluds_->ClearLocalAndReceivePsi();
  executed_ = false;
}

void
CBC_AngleSet::ResetTaskState()
{
  const auto& task_list = cbc_spds_.GetTaskList();

  std::fill(num_satisfied_successors_.begin(), num_satisfied_successors_.end(), 0);
  std::fill(completed_tasks_.begin(), completed_tasks_.end(), 0);
  ready_tasks_.clear();
  num_completed_tasks = 0;

  for (std::uint32_t task_idx = 0; task_idx < task_list.size(); ++task_idx)
  {
    remaining_dependencies_[task_idx] = task_list[task_idx].num_dependencies;
    if (remaining_dependencies_[task_idx] == 0)
      ready_tasks_.push_back(task_idx);
  }
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
  if (boundaries_[boundary_id]->IsReflecting())
    return boundaries_[boundary_id]->PsiIncoming(cell_local_id, face_num, fi, angle_num, g);

  if (not surface_source_active)
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

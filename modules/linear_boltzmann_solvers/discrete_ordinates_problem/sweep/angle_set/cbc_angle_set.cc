// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
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
    async_comm_(id, *fluds, comm_set)
{
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

  if (task_list_ptr_ == nullptr)
  {
    task_list_ptr_ = &cbc_spds_.GetTaskList();
    const auto& tasks = *task_list_ptr_;
    const size_t n = tasks.size();

    // Build shadow array of mutable state only, avoiding a full copy
    // of the task list (which would heap-allocate each successors vector).
    task_states_.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
      task_states_[i].num_dependencies = tasks[i].num_dependencies;
      task_states_[i].completed = false;
    }

    // Build initial ready queue: tasks with no dependencies
    ready_tasks_.reserve(n);
    for (uint64_t i = 0; i < n; ++i)
      if (task_states_[i].num_dependencies == 0)
        ready_tasks_.push_back(i);
  }

  sweep_chunk.SetAngleSet(*this);

  auto tasks_who_received_data = async_comm_.ReceiveData();

  for (const uint64_t task_number : tasks_who_received_data)
    if (--task_states_[task_number].num_dependencies == 0)
      ready_tasks_.push_back(task_number);

  async_comm_.SendData();

  // Check if boundaries allow for execution
  for (auto& [bid, boundary] : boundaries_)
    if (not boundary->CheckAnglesReadyStatus(angles_))
      return AngleSetStatus::NOT_FINISHED;

  // Process ready tasks
  const auto& tasks = *task_list_ptr_;
  while (not ready_tasks_.empty())
  {
    const uint64_t task_idx = ready_tasks_.back();
    ready_tasks_.pop_back();
    const auto& cell_task = tasks[task_idx];

    sweep_chunk.SetCell(cell_task.cell_ptr, *this);
    sweep_chunk.Sweep(*this);

    for (uint64_t local_task_num : cell_task.successors)
      if (--task_states_[local_task_num].num_dependencies == 0)
        ready_tasks_.push_back(local_task_num);

    task_states_[task_idx].completed = true;
    ++num_completed_tasks_;
    async_comm_.SendData();
  }

  const bool all_tasks_completed = (num_completed_tasks_ == tasks.size());
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
  task_list_ptr_ = nullptr;
  task_states_.clear();
  ready_tasks_.clear();
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

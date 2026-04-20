// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/data_types/range.h"
#include "caliper/cali.h"
#include <cassert>

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
    cbc_fluds_(dynamic_cast<CBC_FLUDS&>(*fluds)),
    is_cylindrical_(spds.GetGrid()->GetCoordinateSystem() == CoordinateSystemType::CYLINDRICAL)
{

  const auto& task_list = cbc_spds_.GetTaskList();
  const auto num_tasks = task_list.size();
  initial_dependencies_.resize(num_tasks);
  task_order_.assign(num_tasks, std::numeric_limits<std::uint32_t>::max());
  remaining_dependencies_.resize(num_tasks);
  task_completed_.resize(num_tasks, 0);
  initial_ready_tasks_.reserve(num_tasks);
  ready_tasks_.reserve(num_tasks);

  const auto& local_subgrid = cbc_spds_.GetLocalSubgrid();
  cylindrical_task_sequence_.assign(local_subgrid.begin(), local_subgrid.end());
  for (std::uint32_t order = 0; order < local_subgrid.size(); ++order)
    task_order_[local_subgrid[order]] = order;

  for (std::uint32_t task_idx = 0; task_idx < num_tasks; ++task_idx)
  {
    const auto& task = task_list[task_idx];
    const auto num_dependencies = task.num_dependencies;
    initial_dependencies_[task_idx] = num_dependencies;
    if (num_dependencies == 0)
      initial_ready_tasks_.push_back(task_idx);
  }

  ResetTaskState();
}

AsynchronousCommunicator*
CBC_AngleSet::GetCommunicator()
{
  return static_cast<AsynchronousCommunicator*>(&async_comm_);
}

void
CBC_AngleSet::InitializeDelayedUpstreamData()
{
  async_comm_.InitializeDelayedUpstreamData();
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

  if (is_cylindrical_)
    return AngleSetAdvanceCylindrical(sweep_chunk, permission, tasks_who_received_data);

  for (const auto& task_number : tasks_who_received_data)
  {
    assert(remaining_dependencies_[task_number] > 0);
    if (--remaining_dependencies_[task_number] == 0)
      ready_tasks_.push_back(task_number);
  }

  async_comm_.SendData();

  // Check if boundaries allow for execution
  for (auto& [bid, boundary] : boundaries_)
    if (not boundary->CheckAnglesReadyStatus(angles_))
      return AngleSetStatus::NOT_FINISHED;

  if (permission != AngleSetStatus::EXECUTE)
    return ready_tasks_.empty() ? AngleSetStatus::NOT_FINISHED : AngleSetStatus::READY_TO_EXECUTE;

  while (not ready_tasks_.empty())
  {
    const auto task_idx = PopNextReadyTask();
    if (task_completed_[task_idx] != 0)
      continue;
    const auto& cell_task = task_list[task_idx];

    sweep_chunk.SetCell(cell_task.cell_ptr, *this);
    sweep_chunk.Sweep(*this);

    for (const auto& local_task_num : cell_task.successors)
    {
      assert(remaining_dependencies_[local_task_num] > 0);
      if (--remaining_dependencies_[local_task_num] == 0)
        ready_tasks_.push_back(local_task_num);
    }

    task_completed_[task_idx] = 1;
    ++num_completed_tasks_;
    async_comm_.SendData();
  }

  const bool all_tasks_completed = (num_completed_tasks_ == task_list.size());
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

AngleSetStatus
CBC_AngleSet::AngleSetAdvanceCylindrical(SweepChunk& sweep_chunk,
                                         AngleSetStatus permission,
                                         const std::vector<std::uint64_t>& tasks_who_received_data)
{
  const auto& task_list = cbc_spds_.GetTaskList();

  for (const auto& task_number : tasks_who_received_data)
  {
    assert(remaining_dependencies_[task_number] > 0);
    --remaining_dependencies_[task_number];
  }

  async_comm_.SendData();

  for (auto& [bid, boundary] : boundaries_)
    if (not boundary->CheckAnglesReadyStatus(angles_))
      return AngleSetStatus::NOT_FINISHED;

  while (next_cylindrical_task_ < cylindrical_task_sequence_.size())
  {
    const auto task_idx = cylindrical_task_sequence_[next_cylindrical_task_];
    if (task_completed_[task_idx] != 0)
    {
      ++next_cylindrical_task_;
      continue;
    }

    if (remaining_dependencies_[task_idx] != 0)
      return permission == AngleSetStatus::EXECUTE ? AngleSetStatus::NOT_FINISHED
                                                   : AngleSetStatus::READY_TO_EXECUTE;

    if (permission != AngleSetStatus::EXECUTE)
      return AngleSetStatus::READY_TO_EXECUTE;

    const auto& cell_task = task_list[task_idx];
    sweep_chunk.SetCell(cell_task.cell_ptr, *this);
    sweep_chunk.Sweep(*this);

    for (const auto& local_task_num : cell_task.successors)
    {
      assert(remaining_dependencies_[local_task_num] > 0);
      --remaining_dependencies_[local_task_num];
    }

    task_completed_[task_idx] = 1;
    ++num_completed_tasks_;
    ++next_cylindrical_task_;
    async_comm_.SendData();
  }

  const bool all_tasks_completed = (num_completed_tasks_ == task_list.size());
  const bool all_messages_sent = async_comm_.SendData();

  if (all_tasks_completed and all_messages_sent)
  {
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
  std::copy(
    initial_dependencies_.begin(), initial_dependencies_.end(), remaining_dependencies_.begin());
  ready_tasks_ = initial_ready_tasks_;
  std::fill(task_completed_.begin(), task_completed_.end(), 0);
  num_completed_tasks_ = 0;
  next_cylindrical_task_ = 0;
}

std::uint32_t
CBC_AngleSet::PopNextReadyTask()
{
  if (not is_cylindrical_)
  {
    const auto task_idx = ready_tasks_.back();
    ready_tasks_.pop_back();
    return task_idx;
  }

  auto best_it = ready_tasks_.begin();
  auto best_order = task_order_[*best_it];
  for (auto it = std::next(ready_tasks_.begin()); it != ready_tasks_.end(); ++it)
  {
    const auto order = task_order_[*it];
    if (order < best_order)
    {
      best_it = it;
      best_order = order;
    }
  }

  const auto task_idx = *best_it;
  ready_tasks_.erase(best_it);
  return task_idx;
}

bool
CBC_AngleSet::ReceiveDelayedData()
{
  return async_comm_.ReceiveDelayedData();
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

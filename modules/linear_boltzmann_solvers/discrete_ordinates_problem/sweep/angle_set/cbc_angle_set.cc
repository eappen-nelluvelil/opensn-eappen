// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"

namespace opensn
{

CBC_AngleSet::CBC_AngleSet(std::size_t id,
                           const LBSGroupset& groupset,
                           const SPDS& spds,
                           std::shared_ptr<FLUDS>& fluds,
                           const std::vector<std::size_t>& angle_indices,
                           std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                           int max_mpi_message_size,
                           const MPICommunicatorSet& comm_set)
  : AngleSet(id, groupset, spds, fluds, angle_indices, boundaries),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds_)),
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
  auto status = ProgressCommunication();
  if (status == AngleSetStatus::FINISHED)
    return status;

  if (permission != AngleSetStatus::EXECUTE)
    return HasReadyTasks() ? AngleSetStatus::READY_TO_EXECUTE : status;

  return HasReadyTasks() ? AdvanceReadyTasks(sweep_chunk) : status;
}

void
CBC_AngleSet::InitializeSweep()
{
  if (task_list_ == nullptr)
  {
    task_list_ = &cbc_spds_.GetTaskList();
    remaining_dependencies_ = cbc_spds_.GetInitialTaskDependencies();
    ready_tasks_ = cbc_spds_.GetInitialReadyTasks();
  }
}

AngleSetStatus
CBC_AngleSet::ProgressCommunication()
{
  if (executed_)
    return AngleSetStatus::FINISHED;

  InitializeSweep();
  async_comm_.ReceiveData(received_task_buffer_);
  for (const auto task_number : received_task_buffer_)
    if (--remaining_dependencies_[task_number] == 0)
      ready_tasks_.push_back(task_number);

  async_comm_.ProgressNormalSends();
  if (not IsDependencyResolved())
    return AngleSetStatus::RECEIVING;

  if (FinishIfReady())
    return AngleSetStatus::FINISHED;

  return HasReadyTasks() ? AngleSetStatus::READY_TO_EXECUTE : AngleSetStatus::RECEIVING;
}

AngleSetStatus
CBC_AngleSet::AdvanceReadyTasks(SweepChunk& sweep_chunk)
{
  if (executed_)
    return AngleSetStatus::FINISHED;

  InitializeSweep();
  if (not IsDependencyResolved())
    return AngleSetStatus::RECEIVING;

  if (ready_tasks_.empty())
  {
    async_comm_.FlushNormalSendBuffers();
    return FinishIfReady() ? AngleSetStatus::FINISHED : AngleSetStatus::RECEIVING;
  }

  sweep_chunk.SetAngleSet(*this);
  ++statistics_.ready_bursts;

  while (not ready_tasks_.empty())
  {
    const auto task_idx = ready_tasks_.back();
    ready_tasks_.pop_back();
    const auto& cell_task = (*task_list_)[task_idx];

    sweep_chunk.SetCell(cell_task.cell_ptr);
    sweep_chunk.Sweep(*this);

    for (const auto& local_task_num : cell_task.successors)
      if (--remaining_dependencies_[local_task_num] == 0)
        ready_tasks_.push_back(local_task_num);

    ++num_completed_tasks_;
    ++statistics_.ready_tasks;
    if (async_comm_.NormalFlushRequired())
      break;
  }

  async_comm_.FlushNormalSendBuffers();

  if (FinishIfReady())
    return AngleSetStatus::FINISHED;

  return HasReadyTasks() ? AngleSetStatus::READY_TO_EXECUTE : AngleSetStatus::NOT_FINISHED;
}

bool
CBC_AngleSet::FinishIfReady()
{
  if (num_completed_tasks_ != task_list_->size() or async_comm_.HasPendingCommunication())
    return false;

  for (auto* angle_set : following_angle_sets_)
    angle_set->DecrementCounter();
  executed_ = true;
  return true;
}

void
CBC_AngleSet::ResetSweepBuffers()
{
  task_list_ = nullptr;
  remaining_dependencies_.clear();
  ready_tasks_.clear();
  received_task_buffer_.clear();
  num_completed_tasks_ = 0;
  async_comm_.Reset();
  fluds_->ClearLocalAndReceivePsi();
  executed_ = false;
}

} // namespace opensn

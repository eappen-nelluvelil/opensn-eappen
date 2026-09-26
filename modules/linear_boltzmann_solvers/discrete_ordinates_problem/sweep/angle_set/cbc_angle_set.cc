// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "caliper/cali.h"

namespace opensn
{

CBC_AngleSet::CBC_AngleSet(std::size_t id,
                           const LBSGroupset& groupset,
                           const SPDS& spds,
                           std::shared_ptr<FLUDS>& fluds,
                           const std::vector<std::size_t>& angle_indices,
                           std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                           int max_mpi_message_size,
                           const SweepCommunicator& sweep_communicator,
                           std::shared_ptr<const SweepCommunicator> event_communicator)
  : AngleSet(id, groupset, spds, fluds, angle_indices, boundaries),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds_)),
    event_communicator_(std::move(event_communicator)),
    async_comm_(groupset.id,
                id,
                *fluds,
                max_mpi_message_size,
                event_communicator_ ? *event_communicator_ : sweep_communicator)
{
}

AsynchronousCommunicator*
CBC_AngleSet::GetCommunicator()
{
  return &async_comm_;
}

void
CBC_AngleSet::InitializeTasks()
{
  if (task_list_ == nullptr)
  {
    task_list_ = &cbc_spds_.GetTaskList();
    remaining_dependency_counts_ = cbc_spds_.GetInitialTaskDependencyCounts();
    ready_tasks_ = cbc_spds_.GetInitialReadyTasks();
  }
}

void
CBC_AngleSet::UpdateReceivedDependencies()
{
  for (const auto task_number : received_task_buffer_)
    if (--remaining_dependency_counts_[task_number] == 0)
      ready_tasks_.push_back(task_number);
}

AngleSetStatus
CBC_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  if (executed_)
    return AngleSetStatus::FINISHED;
  InitializeTasks();
  async_comm_.ReceiveData(received_task_buffer_);
  UpdateReceivedDependencies();
  return AdvanceReadyTasks(sweep_chunk, permission);
}

AngleSetStatus
CBC_AngleSet::AdvanceReadyTasks(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  if (executed_)
    return AngleSetStatus::FINISHED;
  CALI_CXX_MARK_SCOPE("AngleSetAdvance");
  InitializeTasks();

  if (async_comm_.HasPendingCommunication())
    async_comm_.SendData();

  if (not IsDependencyResolved())
    return AngleSetStatus::RECEIVING;

  if (permission != AngleSetStatus::EXECUTE)
    return AngleSetStatus::READY_TO_EXECUTE;

  const bool had_ready_tasks = not ready_tasks_.empty();
  if (had_ready_tasks)
    sweep_chunk.SetAngleSet(*this);

  while (not ready_tasks_.empty())
  {
    const auto task_idx = ready_tasks_.back();
    ready_tasks_.pop_back();
    const auto& cell_task = (*task_list_)[task_idx];

    sweep_chunk.SetCell(cell_task.cell_ptr);
    sweep_chunk.Sweep(*this);

    for (const auto& local_task_num : cell_task.successors)
      if (--remaining_dependency_counts_[local_task_num] == 0)
        ready_tasks_.push_back(local_task_num);

    ++num_completed_tasks_;
    // Send when the existing packet limit closes a normal message. Otherwise
    // coalesce consecutive records until no more local tasks are ready; the
    // exit path starts every remaining partial packet before waiting for data.
    if (async_comm_.HasClosedNormalPacket())
      async_comm_.SendData();
  }

  const bool all_tasks_completed = (num_completed_tasks_ == task_list_->size());
  // An idle visit already polled sends above and cannot have created new packets.
  // Productive visits must also start every remaining partial packet before returning.
  if (had_ready_tasks and async_comm_.HasPendingCommunication())
    async_comm_.SendData();

  // The communicator owns copies of outgoing flux until the final send drain.
  // Local dependents need completed flux, not remote MPI send completion.
  if (all_tasks_completed)
  {
    for (auto* angle_set : following_angle_sets_)
      angle_set->DecrementCounter();
    executed_ = true;
    return AngleSetStatus::FINISHED;
  }

  return AngleSetStatus::NOT_FINISHED;
}

bool
CBC_AngleSet::ReceivePacket(int source_rank, std::span<const char> packet)
{
  InitializeTasks();
  received_task_buffer_.clear();
  async_comm_.DecodePacket(source_rank, packet, received_task_buffer_);
  UpdateReceivedDependencies();
  return IsDependencyResolved() and not ready_tasks_.empty();
}

void
CBC_AngleSet::ResetSweepBuffers()
{
  task_list_ = nullptr;
  remaining_dependency_counts_.clear();
  ready_tasks_.clear();
  received_task_buffer_.clear();
  num_completed_tasks_ = 0;
  async_comm_.Reset();
  fluds_->ClearLocalAndReceivePsi();
  executed_ = false;
}

} // namespace opensn

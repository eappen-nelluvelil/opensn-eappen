// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
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
                           size_t num_groups,
                           const SPDS& spds,
                           std::shared_ptr<FLUDS>& fluds,
                           const std::vector<size_t>& angle_indices,
                           std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                           const MPICommunicatorSet& comm_set,
                           bool use_gpu)
  : AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries, use_gpu),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds_)),
    async_comm_(id, *fluds, comm_set),
    cbc_fluds_(dynamic_cast<CBC_FLUDS&>(*fluds_)),
    use_gpus_(use_gpu)
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
  if (use_gpus_)
    return GPUAngleSetAdvance(sweep_chunk, permission);
  else
    return CPUAngleSetAdvance(sweep_chunk, permission);
}

AngleSetStatus
CBC_AngleSet::CPUAngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBC_AngleSet::CPUAngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  if (current_task_list_.empty())
    current_task_list_ = cbc_spds_.GetTaskList();

  sweep_chunk.SetAngleSet(*this);

  auto tasks_who_received_data = async_comm_.ReceiveData();

  for (const uint64_t task_number : tasks_who_received_data)
    --current_task_list_[task_number].num_dependencies;

  async_comm_.SendData();

  // Check if boundaries allow for execution
  for (auto& [bid, boundary] : boundaries_)
    if (not boundary->CheckAnglesReadyStatus(angles_))
      return AngleSetStatus::NOT_FINISHED;

  bool all_tasks_completed = true;
  bool a_task_executed = true;

  // int num_ready_tasks_per_while_loop = 0;

  while (a_task_executed)
  {
    a_task_executed = false;
    for (auto& cell_task : current_task_list_)
    {
      if (not cell_task.completed)
        all_tasks_completed = false;
      if (cell_task.num_dependencies == 0 and not cell_task.completed)
      {
        // ++num_ready_tasks_per_while_loop;

        cbc_fluds_.Allocate(cell_task.cell_ptr->local_id);

        sweep_chunk.SetCell(cell_task.cell_ptr, *this);
        sweep_chunk.Sweep(*this);

        for (uint64_t local_task_num : cell_task.local_successors)
          --current_task_list_[local_task_num].num_dependencies;

        cell_task.completed = true;
        a_task_executed = true;
        async_comm_.SendData();

        // Update predecessor dependency consumption counts
        for (uint64_t local_task_num : cell_task.local_predecessors)
        {
          ++current_task_list_[local_task_num].num_satisfied_downwind_deps;

          if (current_task_list_[local_task_num].num_satisfied_downwind_deps >=
              current_task_list_[local_task_num].local_successors.size())
            cbc_fluds_.Deallocate(current_task_list_[local_task_num].cell_ptr->local_id);
        }

        // Deallocate if cell has no local successors
        if (cell_task.local_successors.empty())
          cbc_fluds_.Deallocate(cell_task.cell_ptr->local_id);
      }
    } // for cell_task

    // opensn::log.Log() << "CBC_AngleSet::CPUAngleSetAdvance - "
    //                   << "Number of ready tasks this iteration: " << num_ready_tasks_per_while_loop
    //                   << "\n";
    // num_ready_tasks_per_while_loop = 0;

    async_comm_.SendData();
  }

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
CBC_AngleSet::GPUAngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBC_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  if (current_task_list_.empty())
    current_task_list_ = cbc_spds_.GetTaskList();

  // Cast SweepChunk to CBCSweepChunk to access GPU-specific methods
  auto& cbc_sweep_chunk = dynamic_cast<CBCSweepChunk&>(sweep_chunk);

  cbc_sweep_chunk.SetAngleSet(*this);

  auto tasks_who_received_data = async_comm_.ReceiveData();

  for (const uint64_t task_number : tasks_who_received_data)
    --current_task_list_[task_number].num_dependencies;

  async_comm_.SendData();

  // Check if boundaries allow for execution
  for (auto& [bid, boundary] : boundaries_)
    if (not boundary->CheckAnglesReadyStatus(angles_))
      return AngleSetStatus::NOT_FINISHED;

  if (not tasks_who_received_data.empty())
    return AngleSetStatus::NOT_FINISHED;

  // Update boundary data
  // This is really only needed for reflecting boundaries
  if (not has_set_boundary_data_)
  {
    dynamic_cast<CBC_FLUDS&>(*fluds_).SetBoundaryPsiData(sweep_chunk, *this);
    has_set_boundary_data_ = true;
  }
  // dynamic_cast<CBC_FLUDS&>(*fluds_).SetBoundaryPsiData(sweep_chunk, *this);

  std::vector<Task*> ready_tasks;

  bool all_tasks_completed = true;
  bool tasks_were_executed = true;
  while (tasks_were_executed)
  {
    tasks_were_executed = false;

    for (auto& cell_task : current_task_list_)
    {
      if (not cell_task.completed)
      {
        all_tasks_completed = false;
        if (cell_task.num_dependencies == 0)
          ready_tasks.push_back(&cell_task);
      }
    }

    // Note that if any tasks received data this iteration, we skip execution
    // This is to make sure that we can pass in as many ready tasks as possible to the GPU,
    // rather than executing a smaller batch, then receiving data, and having more ready tasks
    // if (not ready_tasks.empty() and tasks_who_received_data.empty())
    if (not ready_tasks.empty())
    {

      cbc_sweep_chunk.SetTaskList(ready_tasks);
      // cbc_sweep_chunk.GPUSweep(*this);
      // cbc_sweep_chunk.Sweep(*this);
      cbc_sweep_chunk.GPUSweep_With_CBCD_FLUDS(*this);

      for (auto* cell_task : ready_tasks)
      {
        for (uint64_t local_task_num : cell_task->local_successors)
          --current_task_list_[local_task_num].num_dependencies;

        cell_task->completed = true;
        async_comm_.SendData(); // Need to play around with when to send data
      }

      tasks_were_executed = true;
      ready_tasks.clear();
    }

    async_comm_.SendData();
  }

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
  current_task_list_.clear();
  async_comm_.Reset();
  fluds_->ClearLocalAndReceivePsi();
  executed_ = false;
  has_set_boundary_data_ = false;
}

const double*
CBC_AngleSet::PsiBoundary(uint64_t boundary_id,
                          unsigned int angle_num,
                          uint64_t cell_local_id,
                          unsigned int face_num,
                          unsigned int fi,
                          int g,
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
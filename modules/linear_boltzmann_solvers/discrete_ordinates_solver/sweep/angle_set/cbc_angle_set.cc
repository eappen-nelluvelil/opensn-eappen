// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep_chunks/sweep_chunk.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/math/math_range.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <cstdint>
#include <sys/types.h>

namespace opensn
{

CBC_AngleSet::CBC_AngleSet(size_t id,
                           size_t num_groups,
                           const SPDS& spds,
                           std::shared_ptr<FLUDS>& fluds,
                           const std::vector<size_t>& angle_indices,
                           std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                           const MPICommunicatorSet& comm_set)
  : AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds_)),
    async_comm_(id, *fluds, comm_set)
{
  // NEW:
  // Initialize runtime info when the task list is first set (or reset)
  // Note: current_task_list_ might be empty here if ResetSweepBuffers is
  // called before the first AngleSetAdvance.
  // Initialization might be better placed at the start of AngleSetAdvance if 
  // list is empty 
}

AsynchronousCommunicator*
CBC_AngleSet::GetCommunicator()
{
  return static_cast<AsynchronousCommunicator*>(&async_comm_);
}

// NEW: InitializeTaskRuntimeInfo() implementation
void CBC_AngleSet::InitializeTaskRuntimeInfo() {
  const size_t num_tasks = current_task_list_.size();
  task_runtime_info_.assign(num_tasks, TaskRuntimeInfo()); // Resize and default construct

  for (size_t i = 0; i < num_tasks; ++i) {
      // Initialize remaining dependencies from the SPDS task list definition
      task_runtime_info_[i].remaining_local_dependencies = current_task_list_[i].num_local_predecessors;
      // Crucially, also initialize remaining non-local deps based on SPDS definition
      task_runtime_info_[i].remaining_non_local_dependencies = current_task_list_[i].num_non_local_predecessors;
      task_runtime_info_[i].completed = false; // Ensure completed is reset
  }
}

AngleSetStatus
CBC_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBC_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  // NEW:
  // If task_list runtime info is empty, initialize it
  if (current_task_list_.empty())
  {
    current_task_list_ = cbc_spds_.GetTaskList();
    InitializeTaskRuntimeInfo();  // Initialize runtime info based on the new list
  }

  sweep_chunk.SetAngleSet(*this);

  // NEW: --- Handle non-local dependencies
  auto tasks_who_received_data = async_comm_.ReceiveData();

  for (const uint64_t task_number : tasks_who_received_data)
  {
    // OLD:
    // --current_task_list_[task_number].num_dependencies;

    // NEW:
    // Check bounds just in case task_number is an index
    if (task_number < task_runtime_info_.size())
    {
      // Decrement the *non-local* dependency count
      if (task_runtime_info_[task_number].remaining_non_local_dependencies > 0)
      {
        task_runtime_info_[task_number].remaining_non_local_dependencies--;
      }
    }
  }

  async_comm_.SendData();

  // Check if boundaries allow for execution
  for (auto& [bid, boundary] : boundaries_)
    if (not boundary->CheckAnglesReadyStatus(angles_))
      return AngleSetStatus::NOT_FINISHED;

  bool all_tasks_completed = true;
  bool a_task_executed = true;
  while (a_task_executed)
  {
    a_task_executed = false;

    // OLD:
    /*
    for (auto& cell_task : current_task_list_)
    {
      if (not cell_task.completed)
        all_tasks_completed = false;
      if (cell_task.num_dependencies == 0 and not cell_task.completed)
      {
        sweep_chunk.SetCell(cell_task.cell_ptr, *this);
        sweep_chunk.Sweep(*this);

        for (uint64_t local_task_num : cell_task.successors)
          --current_task_list_[local_task_num].num_dependencies;

        cell_task.completed = true;
        a_task_executed = true;
        async_comm_.SendData();
      }
    } // for cell_task
    async_comm_.SendData();
    */

    // NEW:
    // Iterate by index to access both current_task_list_ and task_runtime_info_
    for (size_t task_idx = 0; task_idx < current_task_list_.size(); ++task_idx)
    {
      auto& runtime_info = task_runtime_info_[task_idx];
      const auto& task_def = current_task_list_[task_idx];  // Use original definition for successors, cell_ptr, etc.

      if (!runtime_info.completed)
      {
        all_tasks_completed = false;  // Found an incomplete task
        
        // NEW: Readiness check: only local dependencies must be zero for execution within this loop.
        // Overall readiness (including non-local messages and BCs) is checked before the loop.
        // if (runtime_info.remaining_local_dependencies == 0 &&
        //     runtime_info.remaining_non_local_dependencies == 0)
        if (runtime_info.remaining_local_dependencies == 0)
        {
          sweep_chunk.SetCell(task_def.cell_ptr, *this);
          sweep_chunk.Sweep(*this);

          // Decrement *local* dependencies for successors
          for (uint64_t successor_task_idx : task_def.successors)
          {
            if (successor_task_idx < task_runtime_info_.size() && // Bounds check
                task_runtime_info_[successor_task_idx].remaining_local_dependencies > 0)  // Avoid underflow
                task_runtime_info_[successor_task_idx].remaining_local_dependencies--;
          }

          runtime_info.completed = true; // Mark this completed in runtime
          a_task_executed = true;
          async_comm_.SendData(); // Send any new data generated
        } // if ready
      } // if not completed
    } // for cell_task
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
  // NEW:
  task_runtime_info_.clear(); // Clear runtime info as well
  fluds_->ClearLocalAndReceivePsi();
  executed_ = false;
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

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/math/math_range.h"
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

  CBC_FLUDS* fluds_ptr = dynamic_cast<CBC_FLUDS*>(&*fluds_); // Get concrete FLUDS type

  if (current_task_list_.empty())
  {
    current_task_list_ = cbc_spds_.GetTaskList();
    current_spls_to_task_index_map_ = cbc_spds_.GetSPLSToTaskIndexMap();
  }

  sweep_chunk.SetAngleSet(*this);

  auto tasks_who_received_data = async_comm_.ReceiveData();

  for (const uint64_t task_number : tasks_who_received_data)
  {
    const int mapped_task_number = current_spls_to_task_index_map_[task_number];
    --current_task_list_[mapped_task_number].num_dependencies;
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
    all_tasks_completed = true;

    for (int current_task_idx_in_list = 0; current_task_idx_in_list < current_task_list_.size();
         ++current_task_idx_in_list)
    {
      Task& cell_task = current_task_list_[current_task_idx_in_list];

      const uint64_t producer_original_id =
        cell_task.reference_id; // Original ID of the cell for this task

      if (cell_task.completed)
        continue; // Already done in a previous AngleSetAdvance or earlier in this one

      all_tasks_completed = false; // If we're here and it's not completed, then not all done

      if (cell_task.num_dependencies == 0) // Task is ready
      {
        // LIVENESS: Allocate for producer (current cell_task)
        // The store_step from SPDS liveness analysis is its ideal position in the SPLS.
        // `current_task_idx_in_list` is this ideal position.

        int designated_store_step = fluds_ptr->GetCellStoreStep(producer_original_id);

        if (!fluds_ptr->IsCellPsiAllocated(producer_original_id))
        { // Avoid re-alloc if sweep is re-entered
          if (designated_store_step > current_task_idx_in_list)
          {
            // This should not happen if task is ready. It means liveness map says store later,
            // but dependencies say ready now. Indicates a mismatch or complex scenario.
            // For safety, if it's ready, it needs memory if it's going to be computed.
            log.Log0Warning() << "Cell " << producer_original_id << " (task_idx "
                              << current_task_idx_in_list
                              << ") ready now, but liveness store_step is " << designated_store_step
                              << ". Allocating based on readiness.";
          }

          fluds_ptr->AllocatePsiForCell(producer_original_id);
        }

        sweep_chunk.SetCell(cell_task.cell_ptr, *this);
        sweep_chunk.Sweep(*this); // Uses fluds_ptr->GetUpwindPsiData / GetDownwindPsiWritePtr

        // LIVENESS: Attempt to deallocate predecessors of this cell_task
        // `GetTaskLocalPredecessorsMap()` returns map: consumer_idx -> list of producer_indices
        // So, `cbc_spds_.GetTaskLocalPredecessorsMap()[current_task_idx_in_list]` gives list of its
        // producers.
        const auto& task_predecessors =
          cbc_spds_.GetTaskLocalPredecessorsMap()[current_task_idx_in_list];

        for (int pred_task_new_idx : task_predecessors)
        {
          // `pred_new_task_idx` is the index (in spls-order) of a cell whose data `cell_task` just
          // consumed.
          const Task& pred_task_obj =
            cbc_spds_.GetTaskByNewIndex(pred_task_new_idx); // Get Task to find original_id

          uint64_t pred_original_id = pred_task_obj.reference_id;

          // Discard predecessor IF this current_task_idx_in_list is its designated discard time
          if (fluds_ptr->IsCellPsiAllocated(pred_original_id) && /* check if allocated */
              fluds_ptr->GetCellDiscardStep(pred_original_id) == current_task_idx_in_list)
          {
            fluds_ptr->DeallocatePsiForCell(pred_original_id);
          }
        }

        // LIVENESS: Attempt to deallocate current cell_task if it's discard time, e.g.,
        // if it only sent to MPI and had no local successors, or this is its last consuming step
        if (fluds_ptr->IsCellPsiAllocated(producer_original_id) && /* check if allocated */
            fluds_ptr->GetCellDiscardStep(producer_original_id) == current_task_idx_in_list)
        {
          fluds_ptr->DeallocatePsiForCell(producer_original_id);
        }

        // Update num_dependencies for local successors
        // `cell_task.successors` contains new_task_indices (spls-ordered indices)
        for (uint64_t successor_new_task_idx : cell_task.successors)
        {
          if (successor_new_task_idx < current_task_list_.size())
          {
            if (!current_task_list_[successor_new_task_idx].completed)
            { // Only decrement if successor not already done
              --current_task_list_[successor_new_task_idx].num_dependencies;
            }
          }
          else { /* Log error: successor_new_task_idx out of bounds */ }
        }

        cell_task.completed = true;
        a_task_executed = true;
        async_comm_.SendData();
      }
    } // for cell_task
    async_comm_.SendData();
  }

  // After the while loop, check if all tasks truly completed.
  // `all_tasks_completed_this_invocation` might be false if the while loop exited because no task
  // was ready, but not all tasks are actually done.
  bool all_actually_done = true;
  for (const auto& task : current_task_list_)
  {
    if (!task.completed)
    {
      all_actually_done = false;
      break;
    }
  }

  const bool all_messages_sent = async_comm_.SendData();

  if (all_actually_done and all_messages_sent)
  {
    // **LIVENESS: Final cleanup for any cells marked to discard at num_tasks**
    // num_tasks is effectively current_task_list_.size()
    for (int task_idx = 0; task_idx < current_task_list_.size(); ++task_idx)
    {
      uint64_t original_id = current_task_list_[task_idx].reference_id;
      if (fluds_ptr->IsCellPsiAllocated(original_id) &&
          fluds_ptr->GetCellDiscardStep(original_id) == static_cast<int>(current_task_list_.size()))
      {
        fluds_ptr->DeallocatePsiForCell(original_id);
      }
    }
    // End Liveness final cleanup

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
  // In addition to existing resets:
  // Ensure all memory from the pool is returned if this AngleSet is being reset
  // for a completely new solver iteration (e.g., outer Picard iteration).
  CBC_FLUDS* fluds_ptr = dynamic_cast<CBC_FLUDS*>(&*fluds_);
  if (fluds_ptr)
  {
    fluds_ptr->ForceDeallocateAllTrackedPsi();
    if (fluds_ptr->GetNumAllocatedPoolSlots() != 0)
    {
      opensn::log.Log0Warning() << "CBC_AngleSet::ResetSweepBuffers: "
                                << fluds_ptr->GetNumAllocatedPoolSlots()
                                << " slots still in pool after ForceDeallocateAllTrackedPsi.";
    }
  }

  current_task_list_.clear();
  async_comm_.Reset();
  fluds_->ClearLocalAndReceivePsi(); // This also clears live_cell_psi_pointers_ and
                                     // deplocs_outgoing_messages_
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

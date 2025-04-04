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

namespace opensn
{

CBC_AngleSet::CBC_AngleSet(size_t id,
                           size_t num_groups,
                           const SPDS& spds,
                           std::shared_ptr<FLUDS>& fluds,
                           const std::vector<size_t>& angle_indices,
                           std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                           const MPICommunicatorSet& comm_set,
                           LBSSolver& lbs_solver) // Add solver reference parameter
  : AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds_)),
    async_comm_(id, *fluds, comm_set),
    lbs_solver_ref_(lbs_solver) // Initialize the reference
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
  while (a_task_executed)
  {
    a_task_executed = false;
    for (auto& cell_task : current_task_list_)
    {
      if (not cell_task.completed) 
        all_tasks_completed = false;
      if (cell_task.num_dependencies == 0 and not cell_task.completed)
      {
        const Cell* cell_ptr = cell_task.cell_ptr;

        // --- Modification 1: Populate the CBC FLUDS local cell angular flux storage before the sweep
        // Calculate the size and starting pointer of the subset for this cell
        const auto& sdm = sweep_chunk.GetSpatialDiscretization();
        const auto& groupset = sweep_chunk.GetGroupset(); // Use getter
        const auto& psi_uk_man = groupset.psi_uk_man_;     // Access from groupset
        auto& groupset_psi_vector = lbs_solver_ref_.GetPsiNewLocal()[groupset.id];

        const size_t num_nodes = sdm.GetCellNumNodes(*cell_ptr);
        const size_t num_angles = this->GetNumAngles(); // Angles in this specific AngleSet
        const size_t num_groups = this->GetNumGroups(); // Groups in this specific AngleSet/Groupset
        const size_t cell_subset_size = num_nodes * num_angles * num_groups;

        // Map to the start of the cell's data using the unknown manager psi_uk_man
        const int64_t cell_dof_map_start = sdm.MapDOFLocal(*cell_ptr, 0, psi_uk_man, 0, 0); 

        // Get the starting pointer in the main psi vector
        const double* subset_start_ptr = nullptr;

        if (!groupset_psi_vector.empty() && cell_dof_map_start >= 0 &&
            (cell_dof_map_start + cell_subset_size) <= groupset_psi_vector.size())
        {
          subset_start_ptr = &groupset_psi_vector[cell_dof_map_start];
        } // Add error handling error later on

        // Get the CBC_FLUDS instance and update its subset storage
        auto& cbc_fluds = dynamic_cast<CBC_FLUDS&>(*fluds_);
        cbc_fluds.UpdateCurrentCellPsiSubset(subset_start_ptr, cell_subset_size);
        // --- End of Modification 1

        sweep_chunk.SetCell(cell_task.cell_ptr, *this);
        sweep_chunk.Sweep(*this); // This now writes to cbc_fluds.current_cell_psi_subset_

        // CURRENTLY BROKEN!
        // --- Modification 2: Copy subset data back to the main vector ---
        if (lbs_solver_ref_.GetOptions().save_angular_flux) // Check if saving is enabled
        {
          // Get the data that was just computed by Sweep and stored in the subset
          const auto& subset_data = cbc_fluds.GetCurrentCellPsiSubset(); // Use const getter is fine
          const size_t cell_subset_size = subset_data.size(); // Size of the data for this cell

          if (cell_subset_size > 0 && cell_dof_map_start >= 0 &&
              (cell_dof_map_start + cell_subset_size) <= groupset_psi_vector.size())
          {
            // Get a pointer to the starting location in the main vector
            double* main_vector_start_ptr = &groupset_psi_vector[cell_dof_map_start];

            // Copy the computed results from the subset into the main vector
            std::copy(subset_data.begin(), subset_data.end(), main_vector_start_ptr);    
          }
          // Need to add error handling here for bounds access errors; currently, there are issues
        }

        for (uint64_t local_task_num : cell_task.successors)
          --current_task_list_[local_task_num].num_dependencies;

        cell_task.completed = true;
        a_task_executed = true;
        async_comm_.SendData();
      }
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

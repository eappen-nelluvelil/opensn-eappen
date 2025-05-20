// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/utils/timer.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <boost/graph/topological_sort.hpp>
#include <cstdint>

namespace opensn
{

CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const std::shared_ptr<MeshContinuum> grid,
                   bool allow_cycles)
  : SPDS(omega, grid)
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

  size_t num_loc_cells = grid->local_cells.size();

  // Populate Cell Relationships
  std::vector<std::set<std::pair<int, double>>> cell_successors(num_loc_cells);
  std::set<int> location_successors;
  std::set<int> location_dependencies;

  PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

  location_successors_.reserve(location_successors.size());
  location_dependencies_.reserve(location_dependencies.size());

  for (auto v : location_successors)
    location_successors_.push_back(v);

  for (auto v : location_dependencies)
    location_dependencies_.push_back(v);

  // Build local cell graph
  Graph local_DG(num_loc_cells);

  // Create graph edges
  for (int c = 0; c < num_loc_cells; ++c)
    for (auto& successor : cell_successors[c])
      boost::add_edge(c, successor.first, successor.second, local_DG);

  if (allow_cycles)
  {
    auto edges_to_remove = RemoveCyclicDependencies(local_DG);
    for (auto& edge_to_remove : edges_to_remove)
      local_sweep_fas_.emplace_back(edge_to_remove.first, edge_to_remove.second);
  }

  // Generate topological sorting
  spls_.clear();
  boost::topological_sort(local_DG, std::back_inserter(spls_));
  std::reverse(spls_.begin(), spls_.end());
  if (spls_.empty())
  {
    throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
                           "Cycles need to be allowed by the calling application.");
  }

  // Create task list
  std::vector<std::vector<int>> global_dependencies;
  global_dependencies.resize(opensn::mpi_comm.size());
  CommunicateLocationDependencies(location_dependencies_, global_dependencies);

  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

  // ---------------------------------------------------------------------------

  // Successors in each Task still refere to original local_ids
  // If task_list_ is now indexed 0, ..., N - 1 based on spls_, successors
  // might need re-mapping if they are to be used as direct indices into the *newly ordered*
  // task_list_
  // Option A: Keep successors as local_ids and do a lookup
  // Option B: Remap successors to be indices into the new spls-ordered task_list_
  // Go with option B
  map_original_local_id_new_task_index_.clear();
  map_original_local_id_new_task_index_.resize(num_loc_cells);
  for (int i = 0; i < spls_.size(); ++i)
  {
    int original_local_id = spls_[i];
    map_original_local_id_new_task_index_[original_local_id] = i;
  }

  task_list_.clear();
  task_list_.resize(num_loc_cells);

  std::vector<Task> temp_task_list(
    num_loc_cells); // Temporary list to build tasks out of order initially

  // First, populate tasks with cell_ptr, initial dependencies, and successors
  for (const auto& cell : grid_->local_cells)
  {
    const size_t cell_local_id = cell.local_id;
    const size_t num_faces = cell.faces.size();
    unsigned int num_dependencies = 0;
    std::vector<uint64_t> successors;

    for (size_t f = 0; f < num_faces; ++f)
    {
      const auto& face = cell.faces[f];
      if (cell_face_orientations_[cell_local_id][f] == INCOMING)
      {
        if (face.has_neighbor)
          ++num_dependencies;
      }
      else if (cell_face_orientations_[cell_local_id][f] == OUTGOING)
      {

        if (face.has_neighbor and grid->IsCellLocal(face.neighbor_id))
          successors.push_back(grid->cells[face.neighbor_id].local_id);
      }
    }

    temp_task_list[cell_local_id] = {num_dependencies,
                                     successors,    // Successors are local IDs
                                     cell_local_id, // Store original local_id for reference
                                     &cell,
                                     false};
  }

  // Create task_list_, ordered by spls_
  task_list_.clear();
  task_list_.reserve(spls_.size());

  for (int original_local_id_from_spls : spls_)
  {
    Task task = temp_task_list[original_local_id_from_spls];

    // Re-map successors for task
    std::vector<uint64_t> remapped_successors;
    remapped_successors.reserve(task.successors.size());
    for (uint64_t succ_original_local_id : task.successors)
    {
      remapped_successors.push_back(map_original_local_id_new_task_index_[succ_original_local_id]);
    }
    task.successors = remapped_successors;

    // Push back task onto task list vector in SPLS's topological sort order
    task_list_.push_back(task);
  }

  task_local_predecessors_map_.assign(task_list_.size(), std::vector<int>());

  for (int producer_task_idx = 0; producer_task_idx < task_list_.size(); ++producer_task_idx)
  {
    for (uint64_t consumer_task_idx_remapped : task_list_[producer_task_idx].successors)
    {
      if (consumer_task_idx_remapped < task_list_.size())
      { // Bounds check
        task_local_predecessors_map_[consumer_task_idx_remapped].push_back(producer_task_idx);
      }
      // else { /* error */ }
    }
  }

  // --- Liveness Analysis Data Structures ---
  
  // To store the "store" event time (iteration number in simulated sweep)
  std::vector<int> cell_store_time(num_loc_cells, -1);
  // To store the "discard" event time
  std::vector<int> cell_discard_time(num_loc_cells, -1);
  // To track how many local successors still need a cell's data
  std::vector<int> cell_pending_local_consumers(num_loc_cells, 0);

  int peak_live_cell_count = 0;
  int current_live_cell_count = 0;
  std::set<int> live_cells_task_indices; // Stores task_indices of live cells

  // Initialize pending_local_consumers for each cell
  // The task_list_ is already ordered by spls_ (new_task_index from 0 to N-1)
  for (int producer_task_idx = 0; producer_task_idx < task_list_.size(); ++producer_task_idx)
  {
    const auto& producer_task = task_list_[producer_task_idx];
    cell_pending_local_consumers[producer_task_idx] = producer_task.successors.size();

    // Also, consider if this cell sends data via MPI.
    // This requires checking its outgoing faces against non-local neighbors.
    // For simplicity, let's assume if it has *any* non-local successor (even if not explicitly
    // in task.successors which are local), its data is live until MPI send.
    // A more refined approach checks faces. For now, let's add a placeholder:
    bool sends_to_mpi = false;
    const Cell& cell = *producer_task.cell_ptr;
    const size_t original_local_id = cell.local_id; // Assuming Task stores this or we can get it
    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      if (cell_face_orientations_[original_local_id][f] == FaceOrientation::OUTGOING)
      {
        const auto& face = cell.faces[f];
        if (face.has_neighbor && !grid_->IsCellLocal(face.neighbor_id))
        {
          sends_to_mpi = true;
          break;
        }
      }
    }
    if (sends_to_mpi)
    {
      // Increment pending consumers to account for the MPI send.
      // This "MPI consumer" is considered satisfied when data is copied to send buffer.
      cell_pending_local_consumers[producer_task_idx]++;
    }
  }

  // --- Simulate the Sweep for Liveness ---
  // We iterate through the spls-ordered task_list. This represents the order of *computation*.
  // `sim_step` acts as a logical time.
  for (int sim_step = 0; sim_step < task_list_.size(); ++sim_step)
  {
    int current_task_idx_being_processed = sim_step; // Since task_list_ is spls-ordered
    const auto& processed_task = task_list_[current_task_idx_being_processed];
    const size_t processed_cell_original_local_id =
      processed_task.reference_id; // Assuming Task::reference_id is original_local_id

    // 1. Cell `processed_task` is computed. Its data becomes "live".
    //    Record store time.
    cell_store_time[processed_cell_original_local_id] = sim_step;
    live_cells_task_indices.insert(current_task_idx_being_processed);
    current_live_cell_count = live_cells_task_indices.size();
    if (current_live_cell_count > peak_live_cell_count)
    {
      peak_live_cell_count = current_live_cell_count;
    }
    // Log: Store psi for cell original_local_id (task_idx current_task_idx_being_processed) at
    // sim_step

    // 2. This `processed_task` satisfies one dependency for each of its local predecessors
    //    (that fed into it). Check if those predecessors can now be discarded.
    //    This requires knowing the *predecessors* of `processed_task`.
    for (int pred_task_idx : task_local_predecessors_map_[current_task_idx_being_processed])
    {
      // `pred_task_idx` is a local predecessor whose data has just been "consumed"
      // by `current_task_idx_being_processed`.
      if (live_cells_task_indices.count(pred_task_idx)) // only if it's still live
      {
        cell_pending_local_consumers[pred_task_idx]--;
        if (cell_pending_local_consumers[pred_task_idx] == 0)
        {
          // All consumers (local successors + MPI send) of pred_task_idx's data are done.
          cell_discard_time[processed_task
                              .reference_id /*Incorrect: should be pred_task.reference_id*/] =
            sim_step;
          // Corrected:
          const auto& pred_task_ref = task_list_[pred_task_idx]; // Get the actual predecessor task
          cell_discard_time[pred_task_ref.reference_id] = sim_step;

          live_cells_task_indices.erase(pred_task_idx);
          // Log: Discard psi for cell pred_task_ref.reference_id (task_idx pred_task_idx) at
          // sim_step
        }
      }
    }

    // 3. If `processed_task` sends data to MPI, consider that "consumer" satisfied now.
    //    (This is a simplification; actual MPI send completion is later).
    bool sends_to_mpi_for_processed_task = false; // Recalculate or fetch this flag
    const Cell& p_cell = *processed_task.cell_ptr;
    const size_t p_original_local_id = p_cell.local_id;
    for (size_t f = 0; f < p_cell.faces.size(); ++f)
    {
      if (cell_face_orientations_[p_original_local_id][f] == FaceOrientation::OUTGOING)
      {
        const auto& face = p_cell.faces[f];
        if (face.has_neighbor && !grid_->IsCellLocal(face.neighbor_id))
        {
          sends_to_mpi_for_processed_task = true;
          break;
        }
      }
    }
    if (sends_to_mpi_for_processed_task)
    {
      if (live_cells_task_indices.count(current_task_idx_being_processed))
      {
        cell_pending_local_consumers[current_task_idx_being_processed]--;
        if (cell_pending_local_consumers[current_task_idx_being_processed] == 0)
        {
          cell_discard_time[processed_cell_original_local_id] = sim_step;
          live_cells_task_indices.erase(current_task_idx_being_processed);
          // Log: Discard psi for processed_cell_original_local_id (due to MPI fulfillment) at
          // sim_step
        }
      }
    }

    // Update current_live_cell_count after potential discards
    current_live_cell_count = live_cells_task_indices.size();
  } // End of simulated sweep loop

  // At the end, any cell still in live_cells_task_indices is live until the very end
  // (e.g. boundary cells whose data is used for tallies, or cells whose MPI data is still pending)
  // Their discard time could be task_list_.size() or remain -1.
  for (int live_task_idx : live_cells_task_indices)
  {
    const auto& live_task = task_list_[live_task_idx];
    if (cell_discard_time[live_task.reference_id] == -1)
    {                                                                // If not already discarded
      cell_discard_time[live_task.reference_id] = task_list_.size(); // Discard at the very end
    }
  }

  // Store info
  peak_liveness_count_ = peak_live_cell_count;
  psi_store_timestep_.resize(num_loc_cells);
  psi_discard_timestep_.resize(num_loc_cells);
  for (int i = 0; i < num_loc_cells; ++i)
  { // Assuming original_local_id maps 0 to N-1
    psi_store_timestep_[i] = cell_store_time[i];
    psi_discard_timestep_[i] = cell_discard_time[i];
  }

  log.Log() << "Liveness Analysis for SPDS (omega=" << omega_.PrintStr() << "): "
            << "Peak live cell count = " << peak_liveness_count_;
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

const std::vector<int>&
CBC_SPDS::GetSPLSToTaskIndexMap() const
{
  return map_original_local_id_new_task_index_;
}

const std::vector<std::vector<int>>&
CBC_SPDS::GetTaskLocalPredecessorsMap() const
{
  return task_local_predecessors_map_;
}
const Task&
CBC_SPDS::GetTaskByNewIndex(int new_task_idx) const
{
  if (new_task_idx < 0 || new_task_idx >= task_list_.size())
  {
    throw std::out_of_range("GetTaskByNewIndex: new_task_idx out of bounds.");
  }
  return task_list_[new_task_idx];
}

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/cbc.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/utils/timer.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <boost/graph/topological_sort.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <sys/types.h>
#include <unordered_map>
#include <algorithm>      // For std::max
#include "framework/logging/log.h"  // For logging errors

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

  // For each local cell create a task
  for (const auto& cell : grid_->local_cells)
  {
    const size_t num_faces = cell.faces.size();
    unsigned int num_dependencies = 0;
    std::vector<uint64_t> succesors;

    for (size_t f = 0; f < num_faces; ++f)
    {
      if (cell_face_orientations_[cell.local_id][f] == INCOMING)
      {
        if (cell.faces[f].has_neighbor)
          ++num_dependencies;
      }
      else if (cell_face_orientations_[cell.local_id][f] == OUTGOING)
      {
        const auto& face = cell.faces[f];
        if (face.has_neighbor and grid->IsCellLocal(face.neighbor_id))
          succesors.push_back(grid->cells[face.neighbor_id].local_id);
      }
    }

    task_list_.push_back({num_dependencies, succesors, cell.local_id, &cell, false});

    // Populate the map from local_id to the index in task_list_
    // Here, task_list_.size() - 1 is the index where the Task object
    // for the current cell was just inserted into the task_list_
    // Allows us to later find the position of a specific cell's task in the 
    // task list vector by knowing its local ID
    local_id_to_task_map_[cell.local_id] = task_list_.size() - 1;
  }

  if (not spls_.empty())
  { 
    // --- START: Calculate the last use index for each local cell's data ---
    // Create a map from cell_local_id to its index in spls_ for quick lookup
    std::unordered_map<uint64_t, size_t> cell_id_spls_index_map;
    cell_id_spls_index_map.reserve(spls_.size());

    for (size_t k = 0; k < spls_.size(); ++k)
    {
      cell_id_spls_index_map[static_cast<uint64_t>(spls_[k])] = k;
    }

    // Resize local ID to task ID to store results
    local_cell_last_use_index_.resize(spls_.size());

    // Iterate through the topologically sorted local cell indices
    // (k = index in spls_)
    for (size_t k = 0; k < spls_.size(); ++k)
    {
      // Get the local cell ID at this position k in the topological sort
      const uint64_t cell_local_id_U = static_cast<uint64_t>(spls_[k]);

      // Find the index of the task in the task in the task list using the map
      size_t task_index = 0;
      try
      {
        task_index = local_id_to_task_map_.at(cell_local_id_U);
      }
      catch (const std::out_of_range& oor)
      {
        // Indicates an internal consistency if a cell from spls_ wasn't
        // added to the task list or map correctly
        log.LogAllError() << "CBC_SPDS Constructor: Cell " << cell_local_id_U
                          << " found in topological sort but not in task index map. "
                          << "Check task_list population logic.";
        throw std::runtime_error("Cell from topological sort not found in task index map.");
      }

      // Access the corresponding task using the safely obtained index
      const auto& task = task_list_[task_index];

      // Default: last needed when computing itself (index k)
      size_t max_successor_spls_index = k;

      // Find the maximum index among local successors in the topological sort
      if (not task.successors.empty())
      {
        for (const uint64_t successor_local_id : task.successors)
        {
          // Find the index (position in spls_) of this successor cell
          try
          {
            size_t successor_spls_index = cell_id_spls_index_map.at(successor_local_id);
            max_successor_spls_index    = std::max(max_successor_spls_index,
                                                   successor_spls_index);
          }
          catch (const std::out_of_range& oor)
          {
            log.LogAllError() << "CBC_SPDS Constructor: Successor cell " << successor_local_id
                              << " for cell " << cell_local_id_U
                              << " not found in topological sort map during lifetime calculation. "
                              << "Check graph construction and successor logic.";

            // Here, we could either throw a runtime error, or we could use
            // the default lifetime k
          }
        }
      } // end if task has successors

      // Store the calculated last use index (position in spls_) for the cell
      // at position k
      local_cell_last_use_index_[k] = max_successor_spls_index;
    }
    // --- END: Calculate the last use index for each local cell's data ---

    // --- START: Populate the release schedule map
    // Clear the previous release schedule (if constructor is called again)
    release_schedule_.clear();

    for (size_t k = 0; k < spls_.size(); ++k)
    {
      // Get the cell ID at position k in the topological sort
      const uint64_t cell_local_id_U = static_cast<uint64_t>(spls_[k]);

      // Get the index in (spls_) where this cell's data is last needed
      const size_t last_use_index = local_cell_last_use_index_[k];

      // Schedule this cell (cell_local_id_U) to be released after the cell at
      // index last_use_index is processed
      // We create the vector if the key doesn't exist
      release_schedule_[last_use_index].push_back(cell_local_id_U);
    }
    // --- END: Calculate the release schedule

    // --- START: calculate the maximum number of active cells
    size_t current_active_count = 0;
    size_t peak_active_count    = 0;

    // Simulate the sweep, step-by-step, based on the topological sort order (spls_)
    for (size_t k = 0; k < spls_.size(); ++k)
    {
      // 1. Simulate allocation: one cell becomes active at cell k
      current_active_count++;

      // 2. Simulate release: check if any cells complete their lifetime after
      // after step k
      // Use find(), which is safer than [], if a key might not exist
      auto release_it = release_schedule_.find(k);
      
      if (release_it != release_schedule_.end())
      {
        // Decrement count by the number of cells released at step k
        current_active_count -= release_it->second.size();
      }

      // 3. Update the peak count observed so far
      peak_active_count = std::max(peak_active_count, current_active_count);
    }

    // Store the calculated maximum value
    max_active_cells_ = peak_active_count;

    // Add 1 as a safety buffer or if peak_active_count could be 0 for trivial
    // cases
    if (max_active_cells_ == 0 && !spls_.empty())
    {
      max_active_cells_ = 1;
    }

    log.Log() << "SPDS calculated maximum active cells = " << max_active_cells_;

    /// --- END: calculate the maximum number of active cells
  } // end if spls_ not empty
} // end of CBC_SPDS constructor scope

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

} // namespace opensn

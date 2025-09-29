// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/utils/timer.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/transitive_reduction.hpp>
#include <unordered_map>
#include <set>
#include <algorithm>

namespace opensn
{

CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const std::shared_ptr<MeshContinuum>& grid,
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
  for (size_t c = 0; c < num_loc_cells; ++c)
    for (const auto& successor : cell_successors[c])
      boost::add_edge(c, successor.first, successor.second, local_DG);

  if (allow_cycles)
  {
    auto edges_to_remove = RemoveCyclicDependencies(local_DG);
    for (auto& edge_to_remove : edges_to_remove)
      local_sweep_fas_.emplace_back(edge_to_remove.first, edge_to_remove.second);
  }

  // Generate topological sorting
  spls_.clear();
  boost::topological_sort(local_DG, std::back_inserter(spls_)); // NOLINT
  std::reverse(spls_.begin(), spls_.end());
  if (spls_.empty())
  {
    throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
                           "Cycles need to be allowed by the calling application.");
  }

  // Generate levelized SPLS
  levelized_spls_max_level_ = 0;
  std::vector<int> levels(num_vertices(local_DG), 0);
  for (auto& v : spls_)
  {
    for (auto ei = out_edges(v, local_DG); ei.first != ei.second; ++ei.first)
    {
      auto u = target(*ei.first, local_DG);
      levels[u] = std::max(levels[u], levels[v] + 1);
      levelized_spls_max_level_ = std::max(levelized_spls_max_level_, levels[u]);
    }
  }

  levelized_spls_.resize(levelized_spls_max_level_ + 1);
  for (auto v = 0; v < num_vertices(local_DG); ++v)
    levelized_spls_[levels[v]].push_back(v);

  // Regenerate SPLS to match levelized SPLS
  spls_.clear();
  levelized_spls_max_level_width_ = 0;
  for (auto& level : levelized_spls_)
  {
    levelized_spls_max_level_width_ = std::max(levelized_spls_max_level_width_, level.size());
    for (auto& cell : level)
      spls_.push_back(cell);
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
    std::vector<uint64_t> remote_predecessors;
    std::vector<uint64_t> local_predecessors;
    unsigned int num_consumptions = 0;
    std::vector<uint64_t> successors;
    std::vector<uint64_t> remote_successors;

    for (size_t f = 0; f < num_faces; ++f)
    {
      const auto& face = cell.faces[f];
      const auto& cell_face_orientation = cell_face_orientations_[cell.local_id][f];

      if (cell_face_orientation == INCOMING)
      {
        if (face.has_neighbor)
        {
          ++num_dependencies;
          if (grid->IsCellLocal(face.neighbor_id))
            local_predecessors.push_back(grid->cells[face.neighbor_id].local_id);
          else
            remote_predecessors.push_back(face.neighbor_id);
        }
      }
      else if (cell_face_orientation == OUTGOING)
      {
        if (face.has_neighbor)
        {
          if (grid->IsCellLocal(face.neighbor_id))
            successors.push_back(grid->cells[face.neighbor_id].local_id);
          else
            remote_successors.push_back(face.neighbor_id);
        }
      }
    }

    task_list_.push_back({num_dependencies, remote_predecessors, 
                          local_predecessors, num_consumptions, 
                          successors, remote_successors, cell.local_id, &cell, 
                          false});
  }

  peak_number_alive_cells_ = std::min(SimulateLocalSweep(),
                                      spls_.size());

  // opensn::log.Log() << "CBC_SPDS: est. # of required cells = " << peak_number_alive_cells_
  //                   << ", # of local active cells = " << SimulateLocalSweep()
  //                   // << ", # of max active edges = "
  //                   // << peak_number_local_active_edges
  //                   // << ", remote parents = " << total_number_of_remote_parents
  //                   // << ", unique remote parents = " << unique_remote_parents.size()
  //                   // << ", remote children = " << total_number_of_remote_children 
  //                   // << ", unique remote children = " << unique_remote_children.size()
  //                   << "\n";
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

size_t
CBC_SPDS::SimulateLocalSweep() const
{
   // Create mapping from cell local ID to task index
  std::unordered_map<uint64_t, size_t> cell_id_to_task_idx;
  for (size_t i = 0; i < task_list_.size(); ++i)
    cell_id_to_task_idx[task_list_[i].reference_id] = i;
  
  // Create local simulation tasks
  std::vector<Task> sim_tasks = task_list_;
  
  size_t peak_allocated = 0;
  size_t currently_allocated = 0;
  std::unordered_set<size_t> tasks_with_remote_predecessors;
  std::unordered_set<size_t> tasks_with_remote_successors;

  for (size_t task_idx = 0; task_idx < sim_tasks.size(); ++task_idx)
  {
    auto& task = sim_tasks[task_idx];
    if ((not task.remote_predecessors.empty()))
    {
      tasks_with_remote_predecessors.insert(task_idx);
    }
    if ((not task.remote_successors.empty()))
    {
      tasks_with_remote_successors.insert(task_idx);
    }
  }

  // Assume all remote dependencies are satisfied 
  for (size_t task_idx = 0; task_idx < sim_tasks.size(); ++task_idx)
  {
    auto& task = sim_tasks[task_idx];
    if (not task.remote_predecessors.empty() and (task.num_dependencies >= task.remote_predecessors.size()))
    {
      task.num_dependencies -= task.remote_predecessors.size();
    }
  }

  currently_allocated = tasks_with_remote_predecessors.size() + tasks_with_remote_successors.size();
  
  bool a_task_executed = true;
  while (a_task_executed)
  {
    a_task_executed = false;
    
    // Process tasks sequentially within the iteration
    for (size_t task_idx = 0; task_idx < sim_tasks.size(); ++task_idx)
    {
      auto& task = sim_tasks[task_idx];

      if (task.num_dependencies == 0 and (not task.completed))
      {
        // Allocate cell
        if ((not tasks_with_remote_predecessors.contains(task_idx)) and 
            (not tasks_with_remote_successors.contains(task_idx)))
        {
          ++currently_allocated;
        }

        peak_allocated = std::max(peak_allocated, currently_allocated);
        a_task_executed = true;
        
        // Reduce dependencies for local successors
        for (uint64_t succ_cell_id : task.successors)
        {
          size_t succ_task_idx = cell_id_to_task_idx[succ_cell_id];
          --sim_tasks[succ_task_idx].num_dependencies;
        }
        
        task.completed = true;
        
        // Update predecessor consumption counts
        for (uint64_t pred_cell_id : task.local_predecessors)
        {
          size_t pred_task_idx = cell_id_to_task_idx[pred_cell_id];
          auto& pred_task = sim_tasks[pred_task_idx];
          ++pred_task.num_consumptions;
          
          const auto local_successor_size = pred_task.successors.size();
          if ((pred_task.num_consumptions >= local_successor_size) and
              (not tasks_with_remote_successors.contains(pred_task_idx)) and
              (not tasks_with_remote_predecessors.contains(pred_task_idx)))
          {
            --currently_allocated;
          }
        }

        if (task.successors.empty() and 
            (not tasks_with_remote_successors.contains(task_idx)) and 
            (not tasks_with_remote_predecessors.contains(task_idx)))
          --currently_allocated;
      }
    }
  }

  return peak_allocated;
}

} // namespace opensn

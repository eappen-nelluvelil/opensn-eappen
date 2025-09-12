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

  // Create task list
  std::vector<std::vector<int>> global_dependencies;
  global_dependencies.resize(opensn::mpi_comm.size());
  CommunicateLocationDependencies(location_dependencies_, global_dependencies);

  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

  // IDEA: Also account for remote predecessors and successors in sweep
  std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, std::string>>> local_children_to_remote_parent_map;
  std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, std::string>>> remote_parent_to_local_children_map;

  std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, std::string>>> remote_children_to_local_parent_map;
  std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, std::string>>> local_parent_to_remote_children_map;

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
          {
            remote_predecessors.push_back(face.neighbor_id);
            local_children_to_remote_parent_map[cell.local_id].push_back(std::make_pair(face.neighbor_id, "remote"));
            remote_parent_to_local_children_map[face.neighbor_id].push_back(std::make_pair(cell.local_id, "local"));
          }
        }
      }
      else if (cell_face_orientation == OUTGOING)
      {
        if (face.has_neighbor)
        {
          if (grid->IsCellLocal(face.neighbor_id))
            successors.push_back(grid->cells[face.neighbor_id].local_id);
          else
          {
            remote_successors.push_back(face.neighbor_id);
            local_parent_to_remote_children_map[cell.local_id].push_back(std::make_pair(face.neighbor_id, "remote"));
            remote_children_to_local_parent_map[face.neighbor_id].push_back(std::make_pair(cell.local_id, "local"));
          }
        }
      }
    }

    task_list_.push_back({num_dependencies, remote_predecessors, 
                          local_predecessors, num_consumptions, 
                          successors, remote_successors, cell.local_id, &cell, false});
  }

  // Generate levelized SPLS
  int max_level = 0;
  std::vector<int> levels(num_vertices(local_DG), 0);
  for (auto& v : spls_)
  {
    for (auto ei = out_edges(v, local_DG); ei.first != ei.second; ++ei.first)
    {
      auto successor = target(*ei.first, local_DG);
      levels[successor] = std::max(levels[successor], levels[v] + 1);
      max_level = std::max(max_level, levels[successor]);
    }
  }
  levelized_spls_.resize(max_level + 1);
  for (auto v = 0; v < num_vertices(local_DG); ++v)
    levelized_spls_[levels[v]].push_back(v);

  // Regenerate spls to match levelized spls
  spls_.clear();
  for (auto& level : levelized_spls_)
    for (auto& cell : level)
      spls_.push_back(cell);

  // ---------------------------------------------------------------------------
  // Simulate local sweep
  // ---------------------------------------------------------------------------

  std::vector<std::pair<uint64_t, std::string>> active_cells;
  std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, std::string>>> children_map;
  std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, std::string>>> parent_map;

  for (auto it = levelized_spls_.begin(); it != levelized_spls_.end(); ++it)
  {
    for (auto cell : *it)
    {
      // Populate children map
      for (auto ei = out_edges(cell, local_DG); ei.first != ei.second; ++ei.first)
      {
        auto succ = target(*ei.first, local_DG);
        children_map[cell].push_back(std::make_pair(succ, "local"));
      }

      // Populate parent map
      for (auto ei = in_edges(cell, local_DG); ei.first != ei.second; ++ei.first)
      {
        auto pred = source(*ei.first, local_DG);
        parent_map[cell].push_back(std::make_pair(pred, "local"));
      }
    }
  }

  size_t peak_number_of_active_cells = 0;
  for (auto cell : spls_)
  {
    // Add current cell to active set
    active_cells.push_back(std::make_pair(cell, "local"));

    // Add current cell's remote parents to active set
    for (const auto& remote_parent : remote_parent_to_local_children_map[cell])
      active_cells.push_back(remote_parent);

    for (const auto& remote_children : local_parent_to_remote_children_map[cell])
      active_cells.push_back(remote_children);

    peak_number_of_active_cells = std::max(peak_number_of_active_cells, active_cells.size());

    // Check if any local parents can be removed from active set
    for (const auto& parent : parent_map[cell])
    {
      bool all_children_processed = true;
      for (const auto& child : children_map[parent.first])
      {
        auto it = std::find(active_cells.begin(), active_cells.end(), child);
        if (it == active_cells.end())
        {
          all_children_processed = false;
          break;
        }
      }

      // Remove parent from active set
      if (all_children_processed)
      {
        auto it = std::find(active_cells.begin(), active_cells.end(), parent);
        if (it != active_cells.end())
          active_cells.erase(it);
      }
    }

    // peak_number_of_active_cells = std::max(peak_number_of_active_cells, active_cells.size());

    // IDEA: Adding remote parents and remote children to the active set and never removing
    // them during the simulated sweep could lead to an overestimate of the 
    // number of blocks needed for the allocator
    // I could remove remote parents as they're no longer needed and local cells
    // once they've satisfied their downwind dependencies

    // Check if any remote parents can be removed from active set
    // for (const auto& remote_parent : local_children_to_remote_parent_map[cell])
    // {
    //   bool all_children_processed = true;
    //   for (const auto& local_child : remote_parent_to_local_children_map[remote_parent.first])
    //   {
    //     auto it = std::find(active_cells.begin(), active_cells.end(), local_child);
    //     if (it == active_cells.end())
    //     {
    //       all_children_processed = false;
    //       break;
    //     }
    //   }

    //   if (all_children_processed)
    //   {
    //     auto it = std::find(active_cells.begin(), active_cells.end(), remote_parent);
    //     if (it != active_cells.end())
    //       active_cells.erase(it);
    //   }
    // }

    // peak_number_of_active_cells = std::max(peak_number_of_active_cells, active_cells.size());

    // Check if any local cells can be removed from the active set
    // for (const auto& local_parent : remote_children_to_local_parent_map[cell])
    // {
    //   bool all_children_processed = true;
    //   for (const auto& remote_child : local_parent_to_remote_children_map[local_parent.first])
    //   {
    //     auto it = std::find(active_cells.begin(), active_cells.end(), remote_child);
    //     if (it == active_cells.end())
    //     {
    //       all_children_processed = false;
    //       break;
    //     }
    //   }

    //   if (all_children_processed)
    //   {
    //     auto it = std::find(active_cells.begin(), active_cells.end(), local_parent);
    //     if (it != active_cells.end())
    //       active_cells.erase(it);
    //   }
    // }

    // peak_number_of_active_cells = std::max(peak_number_of_active_cells, active_cells.size());
  }

  peak_number_alive_cells_ = std::min(peak_number_of_active_cells, spls_.size());

  opensn::log.Log() << "\nCBC_SPDS: Peak number of active local and remote cells during simulated sweep = " << peak_number_alive_cells_;

}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

void CBC_SPDS::SimulateLocalSweep()
{
  
}

// void
// CBC_SPDS::SimulateLocalSweep()
// {
//   std::vector<Task> sim_task_list = task_list_;
//   const size_t num_local_tasks = sim_task_list.size();

//   uint64_t currently_allocated_blocks = 0;
//   uint64_t peak_allocated_blocks = 0;

//   // Simluate that all remote dependencies have been met
//   // for (auto& task : sim_task_list)
//   // {
//   //   const auto& cell = *task.cell_ptr;
//   //   unsigned int remote_deps = 0;
//   //   for (size_t f = 0; f < cell.faces.size(); ++f)
//   //     if (cell_face_orientations_[cell.local_id][f] == FaceOrientation::INCOMING)
//   //     {
//   //       const auto& face = cell.faces[f];
//   //       if (face.has_neighbor and not grid_->IsCellLocal(face.neighbor_id))
//   //         ++remote_deps;
//   //     }
//   //   task.num_dependencies -= remote_deps;
//   // }

//   for (auto& task : sim_task_list)
//     if (not task.remote_predecessors.empty())
//     {
//       // task.num_dependencies -= task.remote_predecessors.size();
//       task.num_dependencies = 0;
//     }

//   // Simulate the local sweep execution, which mirrors AngleSetAdvance
//   std::vector<bool> has_allocated_block;
//   has_allocated_block.assign(num_local_tasks, false);

//   bool a_task_executed = true;
//   while (a_task_executed)
//   {
//     a_task_executed = false;
//     for (auto& task : sim_task_list)
//     {
//       if (task.num_dependencies == 0 and not task.completed)
//       {
//         // Simulate allocation for the current task
//         if (not has_allocated_block[task.reference_id])
//         {
//           has_allocated_block[task.reference_id] = true;
//           ++currently_allocated_blocks;
//           peak_allocated_blocks = std::max(peak_allocated_blocks, currently_allocated_blocks);
//         }
        
//         // Mark task as complete and update its successors
//         task.completed = true;
//         a_task_executed = true;

//         for (const uint64_t succ_idx : task.successors)
//           --sim_task_list[succ_idx].num_dependencies;

//         // Simulate deallocation for predecessors whose data is now fully consumed
//         for (const uint64_t pred_idx : task.local_predecessors)
//         {
//           auto& predecessor_task = sim_task_list[pred_idx];
//           ++predecessor_task.num_consumptions;

//           if ((predecessor_task.num_consumptions == predecessor_task.successors.size()))
//           {
//             --currently_allocated_blocks;
//           }
//         }

//         // If this task is a sink (no local successors), its memory
//         // is deallocated immediately after execution
//         if (task.successors.empty())
//         {
//           --currently_allocated_blocks;
//         }
//       }
//     } // for task
//   }   // while a_task_executed

//   // IDEA: 9/3
//   // A conservative estimate of the number of peak alive cells is to sum the
//   // max width of the topologically sorted TDG + the number of unique predecessors
//   // that each of the corresponding cells at the max width level
//   peak_number_alive_cells_ = peak_allocated_blocks;
// }

} // namespace opensn

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

  // Estimate peak number of alive cells during sweep
  // SimulateLocalSweep();

  size_t total_number_of_remote_parents = 0;
  size_t total_number_of_remote_children = 0;

  std::unordered_set<uint64_t> unique_remote_parents;
  std::unordered_set<uint64_t> unique_remote_children;
  
  for (const auto& task : task_list_)
  {
    total_number_of_remote_parents += task.remote_predecessors.size();
    total_number_of_remote_children += task.remote_successors.size();
    
    for (const auto& remote_predecessor : task.remote_predecessors)
      unique_remote_parents.insert(static_cast<uint64_t>(remote_predecessor));

    for (const auto& remote_successor : task.remote_successors)
      unique_remote_children.insert(static_cast<uint64_t>(remote_successor));
  }

  size_t number_of_unique_remote_parents = unique_remote_parents.size();
  size_t number_of_unique_remote_children = unique_remote_children.size();

  size_t peak_number_local_active_edges = 0;
  std::set<std::pair<uint64_t, uint64_t>> local_active_edges;

  for (int i = levelized_spls_max_level_; i >= 0; --i)
  {
    for (const auto& cell : levelized_spls_[i])
    {
      // Remove edges going out of cell
      const auto& successors = task_list_[cell].successors;

      for (const auto& successor : successors)
      {
        const auto& cell_to_successor_edge = std::make_pair(static_cast<uint64_t>(cell), static_cast<uint64_t>(successor));
        auto it = local_active_edges.find(cell_to_successor_edge);
        if (it != local_active_edges.end())
          local_active_edges.erase(cell_to_successor_edge);
      }

      // Add edges going into cell
      const auto& predecessors = task_list_[cell].local_predecessors;
      for (const auto& predecessor : predecessors)
      {
        const auto& predecessor_to_cell_edge = std::make_pair(static_cast<uint64_t>(predecessor), static_cast<uint64_t>(cell));
        auto it = local_active_edges.find(predecessor_to_cell_edge);
        if (it == local_active_edges.end())
          local_active_edges.insert(predecessor_to_cell_edge);
      }
    }

    peak_number_local_active_edges = std::max(peak_number_local_active_edges, local_active_edges.size());
  }

  // This is a fudge factor for 
  if (levelized_spls_max_level_width_ == 1)
    ++peak_number_local_active_edges;

  size_t estimated_number_of_peak_active_edges = 
    peak_number_local_active_edges + number_of_unique_remote_parents + number_of_unique_remote_children;

  peak_number_alive_cells_ = std::min(estimated_number_of_peak_active_edges, spls_.size());

  opensn::log.Log() << "CBC_SPDS: Peak number of alive cells during sweep: "
                     << peak_number_alive_cells_ << "\n";
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

// void
// CBC_SPDS::SimulateLocalSweep()
// {
//   std::vector<std::pair<uint64_t, std::string>> active_cells;

//   size_t peak = 0;
//   for (const auto& cell : spls_)
//   {
//     // Add current cell to active cell list
//     active_cells.push_back(std::make_pair(cell, std::string("local")));

//     // Add current cell's remote predecessors to active cell list
//     for (const auto& remote_predecessor : local_children_to_remote_parent_map_[cell])
//       active_cells.push_back(remote_predecessor);

//     // Add current cell's remote successors to active cell list
//     for (const auto& remote_successor : local_parent_to_remote_children_map_[cell])
//       active_cells.push_back(remote_successor);

//     peak = std::max(peak, active_cells.size());

//     // Check if any local predecessors can be removed from active cell list
//     for (const auto& local_predecessor : local_children_to_local_parent_map_[cell])
//     {
//       bool all_local_successors_processed = true;
//       for (const auto& local_successor : local_parent_to_local_children_map_[local_predecessor.first])
//       {
//         auto it = std::find(active_cells.begin(), active_cells.end(),
//                               local_successor);
//         if (it == active_cells.end())
//         {
//           all_local_successors_processed = false;
//           break;
//         }
//       }

//       if (all_local_successors_processed)
//       {
//         auto it = std::find(active_cells.begin(), active_cells.end(),
//                              local_predecessor);
//         if (it != active_cells.end())
//           active_cells.erase(it);
//       }
//     }
//   }

//   peak_number_alive_cells_ = std::min(peak, spls_.size());
// }

void
SimulateLocalSweep()
{
  // size_t total_number_of_remote_parents = 0;
  // size_t total_number_of_remote_children = 0;

  // std::unordered_set<uint64_t> unique_remote_parents;
  // std::unordered_set<uint64_t> unique_remote_children;
  
  // for (const auto& task : task_list_)
  // {
  //   total_number_of_remote_parents += task.remote_predecessors.size();
  //   total_number_of_remote_children += task.remote_successors.size();
    
  //   for (const auto& remote_predecessor : task.remote_predecessors)
  //     unique_remote_parents.insert(static_cast<uint64_t>(remote_predecessor));

  //   for (const auto& remote_successor : task.remote_successors)
  //     unique_remote_children.insert(static_cast<uint64_t>(remote_successor));
  // }

  // size_t number_of_unique_remote_parents = unique_remote_parents.size();
  // size_t number_of_unique_remote_children = unique_remote_children.size();

  // size_t peak_number_local_active_edges = 0;

  // for (int i = levelized_spls_max_level_; i >= 0; --i)
  // {
  //   for (const auto& cell : levelized_spls_[i])
  //   {
  //     // Remove edges going out of cell
  //     const auto& successors = task_list_[cell].successors;

  //     for (const auto& successor : successors)
  //     {
  //       const auto& cell_to_successor_edge = std::make_pair(static_cast<uint64_t>(cell), static_cast<uint64_t>(successor));
  //       auto it = local_active_edges.find(cell_to_successor_edge);
  //       if (it != local_active_edges.end())
  //         local_active_edges.erase(cell_to_successor_edge);
  //     }

  //     // Add edges going into cell
  //     const auto& predecessors = task_list_[cell].local_predecessors;
  //     for (const auto& predecessor : predecessors)
  //     {
  //       const auto& predecessor_to_cell_edge = std::make_pair(static_cast<uint64_t>(predecessor), static_cast<uint64_t>(cell));
  //       auto it = local_active_edges.find(predecessor_to_cell_edge);
  //       if (it == local_active_edges.end())
  //         local_active_edges.insert(predecessor_to_cell_edge);
  //     }
  //   }

  //   peak_number_local_active_edges = std::max(peak_number_local_active_edges, local_active_edges.size());
  // }

  // // This is a fudge factor for 
  // if (levelized_spls_max_level_width_ == 1)
  //   ++peak_number_local_active_edges;

  // size_t estimated_number_of_peak_active_edges = 
  //   peak_number_local_active_edges + number_of_unique_remote_parents + number_of_unique_remote_children;

  // peak_number_alive_cells_ = std::min(estimated_number_of_peak_active_edges, spls_.size());
}

} // namespace opensn

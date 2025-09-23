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
  size_t total_number_of_remote_parents = 0;
  size_t total_number_of_remote_children = 0;

  std::unordered_set<uint64_t> unique_remote_parents;
  std::unordered_set<uint64_t> unique_remote_children;

  for (const auto& task : task_list_)
  {
    total_number_of_remote_parents += task.remote_predecessors.size();
    total_number_of_remote_children += task.remote_successors.size();

    for (const auto& remote_parent : task.remote_predecessors)
      unique_remote_parents.insert(remote_parent);
    for (const auto& remote_child : task.remote_successors)
      unique_remote_children.insert(remote_child);
  }

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

  // This is a fudge factor for transport_1d_1.py because the levelized SPLS
  // has a max width of 1 but the actual sweep needs 1 block for each cell
  // and 1 block for the successor cell
  // This is only the case for running with 1 MPI rank
  // if (levelized_spls_max_level_width_ == 1)
  //   ++peak_number_local_active_edges;

  // size_t estimated_number_of_peak_active_edges = 
  //   peak_number_local_active_edges + total_number_of_remote_parents + total_number_of_remote_children;

  // peak_number_alive_cells_ = std::min(estimated_number_of_peak_active_edges, spls_.size());

  // opensn::log.Log() << "CBC_SPDS: # of local active edges = " << peak_number_local_active_edges << "\n";
  // peak_number_alive_cells_ = std::min(ComputePeakActiveEdgesExtendedGraph() + 
  //                                     unique_remote_parents.size() +
  //                                     unique_remote_children.size(), 
  //                                     spls_.size());

  peak_number_alive_cells_ = std::min(ComputePeakActiveEdgesExtendedGraph(),
                                      spls_.size());

  // if (unique_remote_parents.size() != total_number_of_remote_parents ||
  //     unique_remote_children.size() != total_number_of_remote_children)
  // {
  //   opensn::log.Log() << "CBC_SPDS: Warning: Duplicate remote parents/children detected. "
  //                            << "Unique remote parents = " << unique_remote_parents.size()
  //                            << ", total remote parents = " << total_number_of_remote_parents
  //                            << ", unique remote children = " << unique_remote_children.size()
  //                            << ", total remote children = " << total_number_of_remote_children
  //                            << "\n";
  // }

  opensn::log.Log() << "CBC_SPDS: est. # of required cells = " << peak_number_alive_cells_
                    << ", # of local active cells = " << ComputePeakActiveEdgesExtendedGraph()
                    // << ", # of max active edges = "
                    // << peak_number_local_active_edges
                    // << ", remote parents = " << total_number_of_remote_parents
                    // << ", unique remote parents = " << unique_remote_parents.size()
                    // << ", remote children = " << total_number_of_remote_children 
                    // << ", unique remote children = " << unique_remote_children.size()
                    << "\n";

  // opensn::log.Log() << "CBC_SPDS: # of max active edges from full sweep graph = "
  //                   << ComputePeakActiveEdgesExtendedGraph() << "\n";
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

void
SimulateLocalSweep()
{
}

/*
size_t CBC_SPDS::ComputePeakActiveEdgesExtendedGraph() const
{
  // Collect all cell IDs (local + remote)
  std::set<uint64_t> all_cell_ids;
  std::unordered_map<uint64_t, size_t> global_to_graph_id;

  // Add local cell IDs
  for (const auto& cell : grid_->local_cells)
    all_cell_ids.insert(cell.global_id);

  // Add remote cell IDs from task dependencies
  for (const auto& task : task_list_)
  {
    for (const auto& remote_pred : task.remote_predecessors)
      all_cell_ids.insert(remote_pred);
    for (const auto& remote_succ : task.remote_successors)
      all_cell_ids.insert(remote_succ);
  }

  // Create mapping from global cell ID to graph vertex ID
  size_t vertex_counter = 0;
  for (const auto& cell_id : all_cell_ids)
    global_to_graph_id[cell_id] = vertex_counter++;

  // Create extended graph with all cells
  Graph extended_graph(all_cell_ids.size());

  // Add edges from task dependencies
  for (const auto& task : task_list_)
  {
    uint64_t current_cell_global_id = grid_->local_cells[task.reference_id].global_id;
    size_t current_vertex = global_to_graph_id[current_cell_global_id];
    
    // Add edges from remote predecessors to current cell
    for (const auto& remote_pred : task.remote_predecessors)
    {
      size_t pred_vertex = global_to_graph_id[remote_pred];
      boost::add_edge(pred_vertex, current_vertex, 1.0, extended_graph);
    }
    
    // Add edges from local predecessors to current cell
    for (const auto& local_pred : task.local_predecessors)
    {
      uint64_t pred_global_id = grid_->local_cells[local_pred].global_id;
      size_t pred_vertex = global_to_graph_id[pred_global_id];
      boost::add_edge(pred_vertex, current_vertex, 1.0, extended_graph);
    }
    
    // Add edges from current cell to local successors
    for (const auto& successor : task.successors)
    {
      uint64_t succ_global_id = grid_->local_cells[successor].global_id;
      size_t succ_vertex = global_to_graph_id[succ_global_id];
      boost::add_edge(current_vertex, succ_vertex, 1.0, extended_graph);
    }
    
    // Add edges from current cell to remote successors
    for (const auto& remote_succ : task.remote_successors)
    {
      size_t succ_vertex = global_to_graph_id[remote_succ];
      boost::add_edge(current_vertex, succ_vertex, 1.0, extended_graph);
    }
  }

  // Generate levelized structure for extended graph
  std::vector<int> extended_levels(num_vertices(extended_graph), 0);
  int extended_max_level = 0;

  // Compute levels using topological ordering
  std::vector<size_t> extended_topo_order;
  boost::topological_sort(extended_graph, std::back_inserter(extended_topo_order));
  std::reverse(extended_topo_order.begin(), extended_topo_order.end());

  for (auto& v : extended_topo_order)
  {
    for (auto ei = out_edges(v, extended_graph); ei.first != ei.second; ++ei.first)
    {
      auto u = target(*ei.first, extended_graph);
      extended_levels[u] = std::max(extended_levels[u], extended_levels[v] + 1);
      extended_max_level = std::max(extended_max_level, extended_levels[u]);
    }
  }

  // Create levelized structure for extended graph
  std::vector<std::vector<uint64_t>> extended_levelized_spls(extended_max_level + 1);
  std::unordered_map<size_t, uint64_t> graph_id_to_global;
  for (const auto& [global_id, graph_id] : global_to_graph_id)
  {
    graph_id_to_global[graph_id] = global_id;
    extended_levelized_spls[extended_levels[graph_id]].push_back(global_id);
  }

  // Regenerate SPLS to match levelized SPLS
  extended_topo_order.clear();
  size_t extended_levelized_spls_max_level_width = 0;
  for (auto& level : extended_levelized_spls)
  {
    extended_levelized_spls_max_level_width = std::max(extended_levelized_spls_max_level_width, level.size());
    for (auto& cell : level)
      extended_topo_order.push_back(cell);
  }

  // Simulate sweep using levelized approach
  size_t peak_number_extended_active_edges = 0;
  std::set<std::pair<uint64_t, uint64_t>> extended_active_edges;

  for (int i = extended_max_level; i >= 0; --i)
  {
    for (const auto& cell_global_id : extended_levelized_spls[i])
    {
      size_t cell_vertex = global_to_graph_id[cell_global_id];
      
      // Remove edges going out of cell
      for (auto ei = out_edges(cell_vertex, extended_graph); ei.first != ei.second; ++ei.first)
      {
        auto successor_vertex = target(*ei.first, extended_graph);
        uint64_t successor_global_id = graph_id_to_global[successor_vertex];
        
        auto edge = std::make_pair(cell_global_id, successor_global_id);
        extended_active_edges.erase(edge);
      }
      
      // Add edges going into cell
      for (auto ei = in_edges(cell_vertex, extended_graph); ei.first != ei.second; ++ei.first)
      {
        auto predecessor_vertex = source(*ei.first, extended_graph);
        uint64_t predecessor_global_id = graph_id_to_global[predecessor_vertex];
        
        auto edge = std::make_pair(predecessor_global_id, cell_global_id);
        extended_active_edges.insert(edge);
      }
    }
    
    peak_number_extended_active_edges = std::max(peak_number_extended_active_edges, 
                                                 extended_active_edges.size());
  }

  if (extended_levelized_spls_max_level_width == 1)
    ++peak_number_extended_active_edges;

  return std::max(peak_number_extended_active_edges, static_cast<size_t>(1));
}
*/

/*
size_t CBC_SPDS::ComputePeakActiveEdgesExtendedGraph() const
{
  // Create mapping from cell local ID to task index
  std::unordered_map<uint64_t, size_t> cell_id_to_task_idx;
  for (size_t i = 0; i < task_list_.size(); ++i)
    cell_id_to_task_idx[task_list_[i].reference_id] = i;
  
  // Create local simulation tasks
  std::vector<Task> sim_tasks = task_list_;  // Copy the real task list
  
  size_t peak_allocated = 0;
  size_t currently_allocated = 0;
  
  bool a_task_executed = true;
  while (a_task_executed)
  {
    a_task_executed = false;
    
    // Process tasks sequentially within the iteration (this is the key!)
    for (size_t task_idx = 0; task_idx < sim_tasks.size(); ++task_idx)
    {
      auto& task = sim_tasks[task_idx];
      
      if (task.num_dependencies == 0 && !task.completed)
      {
        // Allocate cell
        currently_allocated++;
        peak_allocated = std::max(peak_allocated, currently_allocated);
        a_task_executed = true;
        
        // Reduce dependencies for successors (can enable tasks later in THIS iteration)
        for (uint64_t succ_cell_id : task.successors)
        {
          size_t succ_task_idx = cell_id_to_task_idx[succ_cell_id];
          sim_tasks[succ_task_idx].num_dependencies--;
        }
        
        task.completed = true;
        
        // Update predecessor consumption counts (can deallocate in THIS iteration)
        for (uint64_t pred_cell_id : task.local_predecessors)
        {
          size_t pred_task_idx = cell_id_to_task_idx[pred_cell_id];
          auto& pred_task = sim_tasks[pred_task_idx];
          pred_task.num_consumptions++;
          
          // Deallocate if all successors consumed
          if (pred_task.num_consumptions >= pred_task.successors.size())
            currently_allocated--;
        }
        
        // Deallocate if no successors
        if (task.successors.empty())
          currently_allocated--;
      }
    }
  }
  
  // Apply fudge factor
  if (levelized_spls_max_level_width_ == 1)
    ++peak_allocated;
    
  return std::max(peak_allocated, static_cast<size_t>(1));
}
*/

// size_t CBC_SPDS::ComputePeakActiveEdgesExtendedGraph() const
// {
//   // Create mapping from cell local ID to task index
//   std::unordered_map<uint64_t, size_t> cell_id_to_task_idx;
//   for (size_t i = 0; i < task_list_.size(); ++i)
//     cell_id_to_task_idx[task_list_[i].reference_id] = i;
  
//   // Create local simulation tasks
//   std::vector<Task> sim_tasks = task_list_;  // Copy the real task list
  
//   // MODIFICATION: Assume all remote dependencies are satisfied
//   // This reduces each task's dependency count by the number of remote predecessors
//   for (auto& task : sim_tasks)
//   {
//     if (task.num_dependencies >= task.remote_predecessors.size())
//     {
//       task.num_dependencies -= task.remote_predecessors.size();
//     }
//     // else
//     //   task.num_dependencies = 0;  // Safety check in case of inconsistent data
//   }
  
//   size_t peak_allocated = 0;
//   size_t currently_allocated = 0;
  
//   bool a_task_executed = true;
//   while (a_task_executed)
//   {
//     a_task_executed = false;
    
//     // Process tasks sequentially within the iteration
//     for (size_t task_idx = 0; task_idx < sim_tasks.size(); ++task_idx)
//     {
//       auto& task = sim_tasks[task_idx];
      
//       if (task.num_dependencies == 0 && !task.completed)
//       {
//         // Allocate cell
//         currently_allocated++;
//         peak_allocated = std::max(peak_allocated, currently_allocated);
//         a_task_executed = true;
        
//         // Reduce dependencies for local successors
//         for (uint64_t succ_cell_id : task.successors)
//         {
//           size_t succ_task_idx = cell_id_to_task_idx[succ_cell_id];
//           if (sim_tasks[succ_task_idx].num_dependencies > 0)
//             sim_tasks[succ_task_idx].num_dependencies--;
//         }
        
//         task.completed = true;
        
//         // Update predecessor consumption counts (local predecessors only)
//         for (uint64_t pred_cell_id : task.local_predecessors)
//         {
//           size_t pred_task_idx = cell_id_to_task_idx[pred_cell_id];
//           auto& pred_task = sim_tasks[pred_task_idx];
//           pred_task.num_consumptions++;
          
//           // Calculate total successors (local + remote)
//           // size_t total_successors = pred_task.successors.size() + pred_task.remote_successors.size();
//           size_t local_successors = pred_task.successors.size();
          
//           // Deallocate only when ALL successors (local + remote) have consumed
//           // if (pred_task.num_consumptions >= total_successors)
//           if (pred_task.num_consumptions >= local_successors)
//             --currently_allocated;
//         }
        
//         // For the current task, consider remote successors in deallocation logic
//         // size_t total_current_successors = task.successors.size() + task.remote_successors.size();
//         size_t local_current_successors = task.successors.size();
        
//         // Deallocate if no successors at all
//         if (local_current_successors == 0)
//           --currently_allocated;
//         // NOTE: If task has remote successors, it stays allocated 
//         // (this contributes to higher peak memory usage in multi-rank)
//       }
//     }
//   }
  
//   // Apply fudge factor
//   if (levelized_spls_max_level_width_ == 1)
//     ++peak_allocated;
    
//   return std::max(peak_allocated, static_cast<size_t>(1));
// }

size_t 
CBC_SPDS::ComputePeakActiveEdgesExtendedGraph() const
{
  // Create mapping from cell local ID to task index
  std::unordered_map<uint64_t, size_t> cell_id_to_task_idx;
  for (size_t i = 0; i < task_list_.size(); ++i)
    cell_id_to_task_idx[task_list_[i].reference_id] = i;
  
  // Create local simulation tasks
  std::vector<Task> sim_tasks = task_list_;  // Copy the real task list

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
      
      if (task.num_dependencies == 0 && !task.completed)
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
        
        // Update predecessor consumption counts (local predecessors only)
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
      }
    }
  }
  
  if (levelized_spls_max_level_width_ == 1)
    ++peak_allocated;

  return std::max(peak_allocated, static_cast<size_t>(1));
}

} // namespace opensn

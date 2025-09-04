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
  // NOTE: I'm fairly sure a levelized SPLS is the same thing as the transitive 
  // reduction of a DAG
  // Double-check with Daryl on this
  int max_level = 0;
  std::vector<int> levels(num_vertices(local_DG), 0);
  for (auto& v : spls_)
  {
    for (auto ei = out_edges(v, local_DG); ei.first != ei.second; ++ei.first)
    {
      auto u = target(*ei.first, local_DG);
      levels[u] = std::max(levels[u], levels[v] + 1);
      max_level = std::max(max_level, levels[u]);
    }
  }

  levelized_spls_.resize(max_level + 1);
  for (auto v = 0; v < num_vertices(local_DG); ++v)
    levelized_spls_[levels[v]].push_back(v);

  // Regenerate SPLS to match levelized SPLS
  spls_.clear();
  std::vector<size_t> num_cells_per_level(levelized_spls_.size(), 0);
  unsigned int max_wavefront_size = 0;
  unsigned int max_predecessor_wavefront_size = 0;
  int level_idx = 0;
  for (auto& level : levelized_spls_)
  {
    for (auto& cell : level)
    {
      spls_.push_back(cell);
      ++num_cells_per_level[level_idx];
    }
    max_predecessor_wavefront_size = max_wavefront_size;
    max_wavefront_size = std::max(max_wavefront_size, 
                                  static_cast<unsigned int>(level.size()));

    // opensn::log.Log() << "CBC_SPDS: Level " << level_idx << ", size = " << level.size();
    ++level_idx;
  }

  std::sort(num_cells_per_level.begin(), num_cells_per_level.end());
  size_t max_wavefront_size_2 = num_cells_per_level.back();
  size_t max_predecessor_wavefront_size_2 = num_cells_per_level.at(num_cells_per_level.size() - 2);

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
        if (face.has_neighbor and grid->IsCellLocal(face.neighbor_id))
          successors.push_back(grid->cells[face.neighbor_id].local_id);
      }
    }

    task_list_.push_back({num_dependencies, remote_predecessors, 
                          local_predecessors, num_consumptions, 
                          successors, cell.local_id, &cell, false});
  }

  // Get peak number of alive cells during local sweep
  // SimulateLocalSweep();
  // opensn::log.Log() << "CBC_SPDS: Max wavefront size: " << max_wavefront_size_2
  //                   << ", Max predecessor wavefront size: " << max_predecessor_wavefront_size_2;
  peak_number_alive_cells_ = max_wavefront_size_2 + max_predecessor_wavefront_size_2;
  opensn::log.Log() << "CBC_SPDS::CBC_SPDS: Peak alive cells = " << peak_number_alive_cells_;
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

void
CBC_SPDS::SimulateLocalSweep()
{
  std::vector<Task> sim_task_list = task_list_;
  const size_t num_local_tasks = sim_task_list.size();

  uint64_t currently_allocated_blocks = 0;
  uint64_t peak_allocated_blocks = 0;

  // Simluate that all remote dependencies have been met
  // for (auto& task : sim_task_list)
  // {
  //   const auto& cell = *task.cell_ptr;
  //   unsigned int remote_deps = 0;
  //   for (size_t f = 0; f < cell.faces.size(); ++f)
  //     if (cell_face_orientations_[cell.local_id][f] == FaceOrientation::INCOMING)
  //     {
  //       const auto& face = cell.faces[f];
  //       if (face.has_neighbor and not grid_->IsCellLocal(face.neighbor_id))
  //         ++remote_deps;
  //     }
  //   task.num_dependencies -= remote_deps;
  // }

  for (auto& task : sim_task_list)
    if (not task.remote_predecessors.empty())
    {
      // task.num_dependencies -= task.remote_predecessors.size();
      task.num_dependencies = 0;
    }

  // Simulate the local sweep execution, which mirrors AngleSetAdvance
  std::vector<bool> has_allocated_block;
  has_allocated_block.assign(num_local_tasks, false);

  bool a_task_executed = true;
  while (a_task_executed)
  {
    a_task_executed = false;
    for (auto& task : sim_task_list)
    {
      if (task.num_dependencies == 0 and not task.completed)
      {
        // Simulate allocation for the current task
        if (not has_allocated_block[task.reference_id])
        {
          has_allocated_block[task.reference_id] = true;
          ++currently_allocated_blocks;
          peak_allocated_blocks = std::max(peak_allocated_blocks, currently_allocated_blocks);
        }
        
        // Mark task as complete and update its successors
        task.completed = true;
        a_task_executed = true;

        for (const uint64_t succ_idx : task.successors)
          --sim_task_list[succ_idx].num_dependencies;

        // Simulate deallocation for predecessors whose data is now fully consumed
        for (const uint64_t pred_idx : task.local_predecessors)
        {
          auto& predecessor_task = sim_task_list[pred_idx];
          ++predecessor_task.num_consumptions;

          if ((predecessor_task.num_consumptions == predecessor_task.successors.size()))
          {
            --currently_allocated_blocks;
          }
        }

        // If this task is a sink (no local successors), its memory
        // is deallocated immediately after execution
        if (task.successors.empty())
        {
          --currently_allocated_blocks;
        }
      }
    } // for task
  }   // while a_task_executed

  // IDEA: 9/3
  // A conservative estimate of the number of peak alive cells is to sum the
  // max width of the topologically sorted TDG + the number of unique predecessors
  // that each of the corresponding cells at the max width level
  peak_number_alive_cells_ = peak_allocated_blocks;
}

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/utils/timer.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"
#include "caliper/cali.h"

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
  std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
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
  for (size_t c = 0; c < num_loc_cells; ++c) // NOLINT
    for (const auto& successor : cell_successors[c])
      boost::add_edge(c, successor.first, successor.second, local_DG);

  if (allow_cycles) // NOLINT
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

  // For each local cell create a task
  for (const auto& cell : grid_->local_cells)
  {
    const size_t num_faces = cell.faces.size();
    unsigned int num_dependencies = 0;
    std::vector<std::uint32_t> predecessors;
    std::vector<std::uint32_t> successors;
    unsigned int num_satisfied_successors = 0;

    for (size_t f = 0; f < num_faces; ++f)
    {
      const auto& face = cell.faces[f];
      const auto& face_orientation = cell_face_orientations_[cell.local_id][f];
      if (face_orientation == INCOMING)
      {
        if (face.has_neighbor)
        {
          ++num_dependencies;
          if (grid->IsCellLocal(face.neighbor_id))
            predecessors.push_back(grid->cells[face.neighbor_id].local_id);
        }
      }
      else if (face_orientation == OUTGOING)
      {
        if (face.has_neighbor and grid->IsCellLocal(face.neighbor_id))
          successors.push_back(grid->cells[face.neighbor_id].local_id);
      }
    }
    task_list_.push_back({num_dependencies,
                          predecessors,
                          successors,
                          num_satisfied_successors,
                          cell.local_id,
                          &cell,
                          false});
  }
  // SimulateLocalSweep();
  // opensn::log.Log() << "CBC_SPDS constructed with " << task_list_.size() << " tasks and
  // max_num_pool_allocator_slots_ = " << max_num_pool_allocator_slots_ << ".\n";
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

void
CBC_SPDS::SimulateLocalSweep()
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::SimulateLocalSweep");

  const auto num_tasks = task_list_.size();
  if (num_tasks == 0)
    return;

  // Construct reflexive, transitive closure of the local sweep dependency graph
  std::vector<boost::dynamic_bitset<>> reachability_matrix(num_tasks,
                                                           boost::dynamic_bitset<>(num_tasks));
  for (auto it = spls_.rbegin(); it != spls_.rend(); ++it)
  {
    const auto u = *it;
    const auto& task = task_list_[u];
    reachability_matrix[u].set(u); // Reflexive closure
    for (const auto& successor : task.successors)
      reachability_matrix[u] |= reachability_matrix[successor]; // Transitive closure
  }

  // Solve matching on implicit graph
  ImplicitHopcroftKarp matcher(num_tasks, task_list_, reachability_matrix);
  size_t matching_size = matcher.Solve();

  // Use Dilworth's theorem to compute maximum antichain size = minimum path cover size = num_tasks
  // - matching_size
  if (matcher.VerifyMatching())
    max_num_pool_allocator_slots_ = num_tasks - matching_size;
  else
  {
    max_num_pool_allocator_slots_ =
      num_tasks; // Fallback to worst case if matching verification fails
    opensn::log.Log0Warning() << "CBC_SPDS::SimulateLocalSweep: Matching verification failed. "
                                 "Falling back to worst case for max_num_pool_allocator_slots_.\n";
  }
}

} // namespace opensn

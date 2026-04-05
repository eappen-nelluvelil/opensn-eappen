// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc_slot_planner.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"
#include <boost/graph/topological_sort.hpp>
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>

namespace opensn
{

void
CBC_SPDS::BuildTaskGraph()
{
  constexpr auto incoming = FaceOrientation::INCOMING;
  constexpr auto outgoing = FaceOrientation::OUTGOING;

  const auto num_loc_cells = grid_->local_cells.size();
  task_list_.assign(num_loc_cells, Task{});

  for (const auto& cell : grid_->local_cells)
  {
    unsigned int num_dependencies = 0;
    std::vector<std::uint32_t> predecessors;
    std::vector<std::uint32_t> successors;
    predecessors.reserve(cell.faces.size());
    successors.reserve(cell.faces.size());

    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = cell_face_orientations_[cell.local_id][f];

      if (orientation == incoming and face.has_neighbor)
      {
        ++num_dependencies;
        if (face.IsNeighborLocal(grid_.get()))
          predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
      }
      else if (orientation == outgoing and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
        successors.push_back(grid_->cells[face.neighbor_id].local_id);
    }

    task_list_[cell.local_id] = Task{
      0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
  }
}

CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const std::shared_ptr<MeshContinuum>& grid,
                   const bool allow_cycles)
  : SPDS(omega, grid)
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

  const auto num_loc_cells = grid->local_cells.size();

  std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
  std::set<int> location_successors;
  std::set<int> location_dependencies;

  PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

  location_successors_.reserve(location_successors.size());
  for (const auto loc : location_successors)
    location_successors_.push_back(loc);

  location_dependencies_.reserve(location_dependencies.size());
  for (const auto loc : location_dependencies)
    location_dependencies_.push_back(loc);

  Graph local_dg(num_loc_cells);
  for (std::size_t c = 0; c < num_loc_cells; ++c)
    for (const auto& successor : cell_successors[c])
      boost::add_edge(c, successor.first, successor.second, local_dg);

  if (allow_cycles)
  {
    const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
    for (const auto& [u, v] : edges_to_remove)
      local_sweep_fas_.emplace_back(u, v);
  }

  spls_.clear();
  boost::topological_sort(local_dg, std::back_inserter(spls_));
  std::reverse(spls_.begin(), spls_.end());
  if (spls_.empty())
  {
    throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
                           "Cycles need to be allowed by the calling application.");
  }

  topo_order_.reserve(spls_.size());
  for (const auto v : spls_)
    topo_order_.push_back(static_cast<std::uint32_t>(v));

  BuildTaskGraph();

  // Initialize slot outputs to the safe identity assignment (one slot per cell); a subsequent
  // ComputeMaxNumLocalPsiSlots() call refines them via the reuse planner.
  max_num_local_psi_slots_ = num_loc_cells;
  num_static_local_psi_slots_ = num_loc_cells;
  task_slot_ids_.resize(num_loc_cells);
  std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), 0);
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const noexcept
{
  return task_list_;
}

void
CBC_SPDS::ComputeMaxNumLocalPsiSlots()
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMaxNumLocalPsiSlots");

  const auto num_tasks = static_cast<std::uint32_t>(task_list_.size());
  if (num_tasks == 0)
  {
    max_num_local_psi_slots_ = 0;
    num_static_local_psi_slots_ = 0;
    task_slot_ids_.clear();
    return;
  }

  thread_local detail::CBCSlotPlannerWorkspace workspace;
  detail::CBCDenseHopcroftKarp allocator(
    num_tasks, task_list_, topo_order_, task_slot_ids_, workspace);
  const auto slot_plan = allocator.Solve();
  max_num_local_psi_slots_ = slot_plan.num_dynamic_slots;
  num_static_local_psi_slots_ = slot_plan.num_static_slots;
}

} // namespace opensn

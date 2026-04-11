// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc_slot_planner.h"
#include "framework/logging/log.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <numeric>
#include <stdexcept>
#include <boost/graph/topological_sort.hpp>

namespace opensn
{

void
CBC_SPDS::BuildTaskGraph()
{
  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

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

      if (orientation == INCOMING and face.has_neighbor)
      {
        ++num_dependencies;
        if (face.IsNeighborLocal(grid_.get()))
          predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
      }
      else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
        successors.push_back(grid_->cells[face.neighbor_id].local_id);
    }

    task_list_[cell.local_id] = Task{
      0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
  }
}

void
CBC_SPDS::BuildLocalFaceTaskGraph()
{
  const auto num_loc_cells = grid_->local_cells.size();
  cell_face_offsets_.assign(num_loc_cells + 1, 0);
  std::size_t total_num_faces = 0;
  for (const auto& cell : grid_->local_cells)
  {
    cell_face_offsets_[cell.local_id] = static_cast<std::uint32_t>(total_num_faces);
    total_num_faces += cell.faces.size();
  }
  cell_face_offsets_.back() = static_cast<std::uint32_t>(total_num_faces);
  outgoing_local_face_task_ids_.assign(total_num_faces, INVALID_LOCAL_FACE_TASK_ID);
  incoming_local_face_task_ids_.assign(total_num_faces, INVALID_LOCAL_FACE_TASK_ID);
  std::vector<std::uint32_t> topo_rank(num_loc_cells, 0);
  for (std::size_t rank = 0; rank < topo_order_.size(); ++rank)
    topo_rank[topo_order_[rank]] = static_cast<std::uint32_t>(rank);

  producer_cell_face_offsets_.assign(num_loc_cells + 1, 0);
  local_face_producer_ranks_.clear();
  local_face_consumer_ranks_.clear();
  max_local_face_node_count_ = 0;

  for (std::size_t producer_rank = 0; producer_rank < topo_order_.size(); ++producer_rank)
  {
    producer_cell_face_offsets_[producer_rank] =
      static_cast<std::uint32_t>(local_face_producer_ranks_.size());

    const auto producer_cell_local_id = topo_order_[producer_rank];
    const auto& cell = grid_->local_cells[producer_cell_local_id];
    const auto& face_orientations = cell_face_orientations_[producer_cell_local_id];

    for (std::size_t face_id = 0; face_id < cell.faces.size(); ++face_id)
    {
      const auto& face = cell.faces[face_id];
      if (face_orientations[face_id] != FaceOrientation::OUTGOING or
          not face.IsNeighborLocal(grid_.get()))
        continue;

      const auto consumer_cell_local_id = face.GetNeighborLocalID(grid_.get());
      const auto consumer_face_id =
        static_cast<std::uint16_t>(face.GetNeighborAdjacentFaceIndex(grid_.get()));
      const auto num_face_nodes = static_cast<std::uint16_t>(face.vertex_ids.size());
      max_local_face_node_count_ = std::max(max_local_face_node_count_,
                                            static_cast<std::size_t>(num_face_nodes));

      const auto face_task_id = static_cast<std::uint32_t>(local_face_producer_ranks_.size());
      local_face_producer_ranks_.push_back(static_cast<std::uint32_t>(producer_rank));
      local_face_consumer_ranks_.push_back(topo_rank[consumer_cell_local_id]);
      outgoing_local_face_task_ids_[cell_face_offsets_[producer_cell_local_id] + face_id] =
        face_task_id;
      incoming_local_face_task_ids_[cell_face_offsets_[consumer_cell_local_id] + consumer_face_id] =
        face_task_id;
    }
  }

  producer_cell_face_offsets_.back() = static_cast<std::uint32_t>(local_face_producer_ranks_.size());
  local_face_slot_ids_.resize(local_face_producer_ranks_.size());
  std::iota(local_face_slot_ids_.begin(), local_face_slot_ids_.end(), std::uint32_t{0});
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

  location_successors_.assign(location_successors.begin(), location_successors.end());
  location_dependencies_.assign(location_dependencies.begin(), location_dependencies.end());

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

  topo_order_.assign(spls_.begin(), spls_.end());

  std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
  CommunicateLocationDependencies(location_dependencies_, global_dependencies);
  BuildTaskGraph();
  BuildLocalFaceTaskGraph();

  // Safe identity assignment: one slot per local directed face. ComputeMaxNumLocalPsiSlots()
  // refines this to the optimal face-slot count if called subsequently.
  max_num_local_psi_slots_ = local_face_producer_ranks_.size();
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
    local_face_slot_ids_.clear();
    return;
  }

  thread_local detail::ThreadLocalWorkspace workspace;
  detail::BuildCBCReachability(num_tasks, task_list_, topo_order_, workspace);

  detail::DenseLocalFaceHopcroftKarp face_allocator(local_face_producer_ranks_,
                                                    local_face_consumer_ranks_,
                                                    producer_cell_face_offsets_,
                                                    local_face_slot_ids_,
                                                    workspace);
  const auto face_result = face_allocator.Solve();
  max_num_local_psi_slots_ = face_result.slot_count;
  if (face_result.verifier_rejected)
    opensn::log.LogAllWarning()
      << "CBC_SPDS::ComputeMaxNumLocalPsiSlots: local-face slot-assignment verifier rejected "
      << "the planner output; falling back to the identity assignment "
         "(one slot per local directed face).";
}

std::uint32_t
CBC_SPDS::GetOutgoingLocalFaceTaskID(const std::uint32_t cell_local_id,
                                     const unsigned int face_id) const noexcept
{
  return outgoing_local_face_task_ids_[cell_face_offsets_[cell_local_id] + face_id];
}

std::uint32_t
CBC_SPDS::GetIncomingLocalFaceTaskID(const std::uint32_t cell_local_id,
                                     const unsigned int face_id) const noexcept
{
  return incoming_local_face_task_ids_[cell_face_offsets_[cell_local_id] + face_id];
}

} // namespace opensn

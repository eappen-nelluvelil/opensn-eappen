// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/utils/timer.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <boost/graph/topological_sort.hpp>
#include <algorithm>

namespace opensn
{

CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const std::shared_ptr<MeshContinuum>& grid,
                   bool allow_cycles)
  : SPDS(omega, grid), allow_cycles_(allow_cycles)
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

  // Identify delayed local faces from the local FAS
  // For each FAS edge (src → dst), find the faces connecting them
  for (const auto& [src_lid, dst_lid] : local_sweep_fas_)
  {
    const auto& src_cell = grid_->local_cells[src_lid];
    for (size_t f = 0; f < src_cell.faces.size(); ++f)
    {
      const auto& face = src_cell.faces[f];
      if (face.has_neighbor and grid_->IsCellLocal(face.neighbor_id))
      {
        auto neighbor_lid = grid_->cells[face.neighbor_id].local_id;
        if (neighbor_lid == dst_lid and
            cell_face_orientations_[src_lid][f] == FaceOrientation::OUTGOING)
        {
          delayed_local_outgoing_faces_.insert(
            {static_cast<uint32_t>(src_lid), static_cast<uint32_t>(f)});
        }
      }
    }

    const auto& dst_cell = grid_->local_cells[dst_lid];
    for (size_t f = 0; f < dst_cell.faces.size(); ++f)
    {
      const auto& face = dst_cell.faces[f];
      if (face.has_neighbor and grid_->IsCellLocal(face.neighbor_id))
      {
        auto neighbor_lid = grid_->cells[face.neighbor_id].local_id;
        if (neighbor_lid == src_lid and
            cell_face_orientations_[dst_lid][f] == FaceOrientation::INCOMING)
        {
          delayed_local_incoming_faces_.insert(
            {static_cast<uint32_t>(dst_lid), static_cast<uint32_t>(f)});
        }
      }
    }
  }

  // Generate location-to-location dependencies
  global_dependencies_.resize(opensn::mpi_comm.size());
  CommunicateLocationDependencies(location_dependencies_, global_dependencies_);

  // Build the task list
  // Note: task list construction must account for local FAS edges and delayed
  // non-local dependencies. Global cycle breaking (BuildGlobalSweepFAS/TDG)
  // will be called later by DiscreteOrdinatesProblem, which will call
  // RebuildTaskList() to update the task list.
  BuildTaskList();
}

void
CBC_SPDS::BuildGlobalSweepFAS()
{
  assert(not global_dependencies_.empty());

  CALI_CXX_MARK_SCOPE("CBC_SPDS::BuildGlobalSweepFAS");

  const int comm_size = opensn::mpi_comm.size();
  Graph global_tdg(comm_size);

  for (int loc = 0; loc < comm_size; ++loc)
  {
    for (auto dep : global_dependencies_[loc])
    {
      double weight = 1.0;
      if (not global_edge_weights_.empty())
      {
        const int idx = dep * comm_size + loc;
        if (idx < static_cast<int>(global_edge_weights_.size()) and
            global_edge_weights_[idx] > 0.0)
          weight = global_edge_weights_[idx];
      }
      boost::add_edge(dep, loc, weight, global_tdg);
    }
  }

  if (allow_cycles_)
  {
    auto edges_to_remove = RemoveCyclicDependencies(global_tdg);
    for (const auto& [e0, e1] : edges_to_remove)
    {
      global_sweep_fas_.emplace_back(e0);
      global_sweep_fas_.emplace_back(e1);
    }
  }
}

void
CBC_SPDS::BuildGlobalSweepTDG()
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::BuildGlobalSweepTDG");

  // Create graph
  Graph global_tdg(opensn::mpi_comm.size());

  for (int loc = 0; loc < opensn::mpi_comm.size(); ++loc)
    for (auto dep : global_dependencies_[loc])
      boost::add_edge(dep, loc, 1.0, global_tdg);

  // De-serialize FAS edges
  std::vector<std::pair<int, int>> edges_to_remove;
  edges_to_remove.resize(global_sweep_fas_.size() / 2, std::make_pair(0, 0));
  int i = 0;
  for (auto& edge : edges_to_remove)
  {
    edge.first = global_sweep_fas_[i++];
    edge.second = global_sweep_fas_[i++];
  }

  // Remove edges and update dependency lists
  for (auto& edge_to_remove : edges_to_remove)
  {
    auto rlocI = edge_to_remove.first;
    auto locI = edge_to_remove.second;

    boost::remove_edge(rlocI, locI, global_tdg);

    if (locI == opensn::mpi_comm.rank())
    {
      auto dependent_location =
        std::find(location_dependencies_.begin(), location_dependencies_.end(), rlocI);
      if (dependent_location != location_dependencies_.end())
        location_dependencies_.erase(dependent_location);
      delayed_location_dependencies_.push_back(rlocI);
    }

    if (rlocI == opensn::mpi_comm.rank())
      delayed_location_successors_.push_back(locI);
  }

  // Identify delayed non-local incoming faces
  // For each local cell, check if any incoming non-local faces come from a delayed dependency
  std::set<int> delayed_dep_set(delayed_location_dependencies_.begin(),
                                delayed_location_dependencies_.end());
  for (const auto& cell : grid_->local_cells)
  {
    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      if (cell_face_orientations_[cell.local_id][f] == FaceOrientation::INCOMING)
      {
        const auto& face = cell.faces[f];
        if (face.has_neighbor and not grid_->IsCellLocal(face.neighbor_id))
        {
          const auto& neighbor = grid_->cells[face.neighbor_id];
          if (delayed_dep_set.count(neighbor.partition_id) > 0)
          {
            delayed_nonlocal_incoming_faces_.insert(
              {static_cast<uint32_t>(cell.local_id), static_cast<uint32_t>(f)});
          }
        }
      }
    }
  }

  // Rebuild task list to account for delayed non-local dependencies
  BuildTaskList();
}

std::vector<double>
CBC_SPDS::ComputeLocalLocationEdgeWeights() const
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeLocalLocationEdgeWeights");

  const int comm_size = opensn::mpi_comm.size();
  std::vector<double> row(comm_size, 0.0);

  constexpr double tolerance = 1.0e-16;

  for (const auto& cell : grid_->local_cells)
  {
    const auto& face_orientations = cell_face_orientations_[cell.local_id];
    std::size_t f = 0;
    for (const auto& face : cell.faces)
    {
      if (face.has_neighbor and not face.IsNeighborLocal(grid_.get()) and
          face_orientations[f] == FaceOrientation::OUTGOING)
      {
        const double mu = omega_.Dot(face.normal);
        if (mu > tolerance)
        {
          const auto& adj_cell = grid_->cells[face.neighbor_id];
          const int to_loc = adj_cell.partition_id;
          row[to_loc] += mu * mu * face.area;
        }
      }
      ++f;
    }
  }

  return row;
}

void
CBC_SPDS::BuildTaskList()
{
  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

  task_list_.clear();

  for (const auto& cell : grid_->local_cells)
  {
    const size_t num_faces = cell.faces.size();
    unsigned int num_dependencies = 0;
    std::vector<std::uint32_t> successors;

    for (size_t f = 0; f < num_faces; ++f)
    {
      if (cell_face_orientations_[cell.local_id][f] == INCOMING)
      {
        if (cell.faces[f].has_neighbor)
        {
          // Skip delayed local incoming faces (local FAS edges)
          if (IsDelayedLocalIncomingFace(cell.local_id, static_cast<uint32_t>(f)))
            continue;

          // Skip delayed non-local incoming faces (global cycle edges)
          if (IsDelayedNonlocalIncomingFace(cell.local_id, static_cast<uint32_t>(f)))
            continue;

          ++num_dependencies;
        }
      }
      else if (cell_face_orientations_[cell.local_id][f] == OUTGOING)
      {
        const auto& face = cell.faces[f];
        if (face.has_neighbor and grid_->IsCellLocal(face.neighbor_id))
        {
          auto neighbor_lid = grid_->cells[face.neighbor_id].local_id;

          // Skip delayed local outgoing faces (local FAS edges)
          if (IsDelayedLocalOutgoingFace(cell.local_id, static_cast<uint32_t>(f)))
            continue;

          successors.push_back(neighbor_lid);
        }
      }
    }

    task_list_.push_back({num_dependencies, successors, cell.local_id, &cell, false});
  }
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

} // namespace opensn

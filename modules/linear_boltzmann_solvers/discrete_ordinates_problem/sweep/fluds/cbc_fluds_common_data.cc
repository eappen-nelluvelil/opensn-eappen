// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/cell/cell.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include <algorithm>
#include <unordered_set>

namespace opensn
{

namespace
{
std::uint64_t PackEdge(std::uint32_t upwind_local_id, std::uint32_t downwind_local_id) noexcept
{
  return (static_cast<std::uint64_t>(upwind_local_id) << 32) |
          static_cast<std::uint64_t>(downwind_local_id);
}
} // namespace

CBC_FLUDSCommonData::CBC_FLUDSCommonData(
  const SPDS& spds, const std::vector<CellFaceNodalMapping>& grid_nodal_mappings)
  : FLUDSCommonData(spds, grid_nodal_mappings)
{
  const auto& grid = *spds.GetGrid();
  const auto& face_orientations = spds.GetCellFaceOrientations();
  const auto& delayed_loc_deps = spds.GetDelayedLocationDependencies();

  delayed_local_incoming_faces_.resize(grid.local_cells.size());
  delayed_local_outgoing_faces_.resize(grid.local_cells.size());
  delayed_nonlocal_incoming_faces_.resize(grid.local_cells.size());
  delayed_nonlocal_face_info_by_cell_.resize(grid.local_cells.size());
  delayed_preloc_face_node_count_.assign(delayed_loc_deps.size(), 0);

  std::unordered_set<std::uint64_t> delayed_local_edges;
  for (const auto& [u, v] : spds.GetLocalSweepFAS())
    delayed_local_edges.insert(PackEdge(u, v));

  has_delayed_local_dependencies_ = not delayed_local_edges.empty();

  for (const auto& cell : grid.local_cells)
  {
    delayed_local_incoming_faces_[cell.local_id].assign(cell.faces.size(), 0);
    delayed_local_outgoing_faces_[cell.local_id].assign(cell.faces.size(), 0);
    delayed_nonlocal_incoming_faces_[cell.local_id].assign(cell.faces.size(), 0);
    delayed_nonlocal_face_info_by_cell_[cell.local_id].resize(cell.faces.size());

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = face_orientations[cell.local_id][f];

      if (not face.has_neighbor)
        continue;

      if (face.IsNeighborLocal(&grid))
      {
        const auto& adj_cell = grid.cells[face.neighbor_id];

        if (orientation == FaceOrientation::INCOMING and
            delayed_local_edges.contains(PackEdge(adj_cell.local_id, cell.local_id)))
          delayed_local_incoming_faces_[cell.local_id][f] = 1;

        if (orientation == FaceOrientation::OUTGOING and
            delayed_local_edges.contains(PackEdge(cell.local_id, adj_cell.local_id)))
          delayed_local_outgoing_faces_[cell.local_id][f] = 1;

        continue;
      }

      const int neighbor_loc = face.GetNeighborPartitionID(&grid);
      const auto delayed_it =
        std::find(delayed_loc_deps.begin(), delayed_loc_deps.end(), neighbor_loc);
      const bool is_delayed_nonlocal = delayed_it != delayed_loc_deps.end();

      if (orientation == FaceOrientation::INCOMING)
      {
        if (is_delayed_nonlocal)
        {
          const auto prelocI =
            static_cast<std::uint32_t>(std::distance(delayed_loc_deps.begin(), delayed_it));
          const auto num_face_nodes =
            static_cast<std::uint32_t>(grid_nodal_mappings[cell.local_id][f].face_node_mapping_.size());

          DelayedNonlocalFaceInfo info{
            prelocI, delayed_preloc_face_node_count_[prelocI], num_face_nodes};

          delayed_nonlocal_incoming_faces_[cell.local_id][f] = 1;
          delayed_nonlocal_face_info_by_cell_[cell.local_id][f] = info;
          delayed_nonlocal_face_info_.emplace(
            FLUDSCommonData::CellFaceKey{cell.global_id, static_cast<unsigned int>(f)}, info);

          delayed_preloc_face_node_count_[prelocI] += num_face_nodes;
        }
        else
          ++num_incoming_nonlocal_faces_;
      }
      else if (orientation == FaceOrientation::OUTGOING)
        ++num_outgoing_nonlocal_faces_;
    }
  }
}

bool
CBC_FLUDSCommonData::IsDelayedLocalIncomingFace(std::uint32_t cell_local_id,
                                                unsigned int face_id) const noexcept
{
  return delayed_local_incoming_faces_[cell_local_id][face_id] != 0;
}

bool
CBC_FLUDSCommonData::IsDelayedLocalOutgoingFace(std::uint32_t cell_local_id,
                                                unsigned int face_id) const noexcept
{
  return delayed_local_outgoing_faces_[cell_local_id][face_id] != 0;
}

bool
CBC_FLUDSCommonData::IsDelayedNonlocalIncomingFace(std::uint32_t cell_local_id,
                                                    unsigned int face_id) const noexcept
{
  return delayed_nonlocal_incoming_faces_[cell_local_id][face_id] != 0;
}

const CBC_FLUDSCommonData::DelayedNonlocalFaceInfo&
CBC_FLUDSCommonData::GetDelayedNonlocalFaceInfo(std::uint32_t cell_local_id,
                                                unsigned int face_id) const noexcept
{
  return delayed_nonlocal_face_info_by_cell_[cell_local_id][face_id];
}

bool
CBC_FLUDSCommonData::TryGetDelayedNonlocalFaceInfo(std::uint64_t cell_global_id,
                                                    unsigned int face_id,
                                                    DelayedNonlocalFaceInfo& info) const noexcept
{
  const auto it = delayed_nonlocal_face_info_.find({cell_global_id, face_id});
  if (it == delayed_nonlocal_face_info_.end())
    return false;
  info = it->second;
  return true;
}

} // namespace opensn
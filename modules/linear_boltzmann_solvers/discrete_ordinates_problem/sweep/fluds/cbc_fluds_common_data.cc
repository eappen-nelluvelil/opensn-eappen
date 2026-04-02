// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/cell/cell.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"

namespace opensn
{

CBC_FLUDSCommonData::CBC_FLUDSCommonData(
  const SPDS& spds, const std::vector<CellFaceNodalMapping>& grid_nodal_mappings)
  : FLUDSCommonData(spds, grid_nodal_mappings),
    num_incoming_nonlocal_faces_(0),
    num_incoming_nonlocal_face_nodes_(0),
    num_outgoing_nonlocal_faces_(0)
{
  const auto& grid = *spds.GetGrid();
  const auto& face_orientations = spds.GetCellFaceOrientations();
  incoming_nonlocal_face_info_by_cell_.resize(grid.local_cells.size());

  for (const auto& cell : grid.local_cells)
  {
    incoming_nonlocal_face_info_by_cell_[cell.local_id].resize(cell.faces.size());

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = face_orientations[cell.local_id][f];

      if ((not face.has_neighbor) or (face.IsNeighborLocal(&grid)))
        continue;

      if (orientation == FaceOrientation::INCOMING)
      {
        ++num_incoming_nonlocal_faces_;
        const auto num_face_nodes = static_cast<std::uint32_t>(
          grid_nodal_mappings[cell.local_id][f].face_node_mapping_.size());

        IncomingNonlocalFaceInfo info{static_cast<std::uint32_t>(num_incoming_nonlocal_face_nodes_),
                                      num_face_nodes};

        incoming_nonlocal_face_info_by_cell_[cell.local_id][f] = info;
        incoming_nonlocal_face_info_.emplace(
          CellFaceKey{cell.global_id, static_cast<unsigned int>(f)}, info);
        num_incoming_nonlocal_face_nodes_ += num_face_nodes;
      }
      else if (orientation == FaceOrientation::OUTGOING)
        ++num_outgoing_nonlocal_faces_;
    }
  }
}

const CBC_FLUDSCommonData::IncomingNonlocalFaceInfo&
CBC_FLUDSCommonData::GetIncomingNonlocalFaceInfo(const std::uint32_t cell_local_id,
                                                 const unsigned int face_id) const noexcept
{
  return incoming_nonlocal_face_info_by_cell_[cell_local_id][face_id];
}

bool
CBC_FLUDSCommonData::TryGetIncomingNonlocalFaceInfo(const std::uint64_t cell_global_id,
                                                    const unsigned int face_id,
                                                    IncomingNonlocalFaceInfo& info) const noexcept
{
  const auto it = incoming_nonlocal_face_info_.find({cell_global_id, face_id});
  if (it == incoming_nonlocal_face_info_.end())
    return false;

  info = it->second;
  return true;
}

} // namespace opensn

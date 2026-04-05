// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/cell/cell.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include <cassert>

namespace opensn
{

CBC_FLUDSCommonData::CBC_FLUDSCommonData(
  const SPDS& spds,
  const std::vector<CellFaceNodalMapping>& grid_nodal_mappings,
  const SpatialDiscretization& sdm)
  : FLUDSCommonData(spds, grid_nodal_mappings),
    num_incoming_nonlocal_faces_(0),
    num_incoming_nonlocal_face_nodes_(0),
    num_outgoing_nonlocal_faces_(0)
{
  const auto& grid = *spds.GetGrid();
  const auto& face_orientations = spds.GetCellFaceOrientations();
  outgoing_nonlocal_face_counts_.assign(spds.GetLocationSuccessors().size(), 0);
  outgoing_nonlocal_face_node_counts_.assign(spds.GetLocationSuccessors().size(), 0);
  cell_face_offsets_.resize(grid.local_cells.size() + 1, 0);
  cell_node_offsets_.resize(grid.local_cells.size() + 1, 0);
  size_t total_num_faces = 0;
  size_t total_num_cell_nodes = 0;
  for (const auto& cell : grid.local_cells)
  {
    cell_face_offsets_[cell.local_id] = total_num_faces;
    total_num_faces += cell.faces.size();
    cell_node_offsets_[cell.local_id] = static_cast<std::uint32_t>(total_num_cell_nodes);
    total_num_cell_nodes += sdm.GetCellMapping(cell).GetNumNodes();
  }
  cell_face_offsets_.back() = total_num_faces;
  cell_node_offsets_.back() = static_cast<std::uint32_t>(total_num_cell_nodes);
  local_outgoing_node_indices_.assign(total_num_cell_nodes, std::numeric_limits<std::uint32_t>::max());
  incoming_nonlocal_face_info_.resize(total_num_faces);
  outgoing_nonlocal_face_info_.resize(total_num_faces);

  for (const auto& cell : grid.local_cells)
  {
    const size_t face_offset = cell_face_offsets_[cell.local_id];
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const auto cell_node_offset = cell_node_offsets_[cell.local_id];
    std::uint32_t next_local_outgoing_node = 0;

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = face_orientations[cell.local_id][f];
      const size_t face_storage_index = face_offset + f;

      if (orientation == FaceOrientation::OUTGOING and face.has_neighbor and
          face.IsNeighborLocal(&grid))
      {
        const auto num_face_nodes = cell_mapping.GetNumFaceNodes(f);
        for (size_t fi = 0; fi < num_face_nodes; ++fi)
        {
          const auto cell_node = static_cast<std::uint32_t>(cell_mapping.MapFaceNode(f, fi));
          auto& compact_index = local_outgoing_node_indices_[cell_node_offset + cell_node];
          if (compact_index == std::numeric_limits<std::uint32_t>::max())
            compact_index = next_local_outgoing_node++;
        }
      }

      if ((not face.has_neighbor) or (face.IsNeighborLocal(&grid)))
        continue;

      if (orientation == FaceOrientation::INCOMING)
      {
        ++num_incoming_nonlocal_faces_;
        const auto num_face_nodes = static_cast<std::uint32_t>(
          grid_nodal_mappings[cell.local_id][f].face_node_mapping_.size());

        IncomingNonlocalFaceInfo info{static_cast<std::uint32_t>(num_incoming_nonlocal_face_nodes_),
                                      num_face_nodes};

        incoming_nonlocal_face_info_[face_storage_index] = info;
        incoming_nonlocal_face_info_by_key_.emplace(
          CellFaceKey{cell.global_id, static_cast<unsigned int>(f)}, face_storage_index);
        num_incoming_nonlocal_face_nodes_ += num_face_nodes;
      }
      else if (orientation == FaceOrientation::OUTGOING)
      {
        ++num_outgoing_nonlocal_faces_;
        const auto deplocI =
          static_cast<std::size_t>(spds.MapLocJToDeplocI(face.GetNeighborPartitionID(&grid)));
        ++outgoing_nonlocal_face_counts_[deplocI];
        outgoing_nonlocal_face_node_counts_[deplocI] +=
          grid_nodal_mappings[cell.local_id][f].face_node_mapping_.size();
        outgoing_nonlocal_face_info_[face_storage_index] = OutgoingNonlocalFaceInfo{
          face.GetNeighborPartitionID(&grid),
          face.neighbor_id,
          static_cast<unsigned int>(grid_nodal_mappings[cell.local_id][f].associated_face_),
          static_cast<std::uint32_t>(
            grid_nodal_mappings[cell.local_id][f].face_node_mapping_.size())};
      }
    }

    max_local_outgoing_node_count_ =
      std::max(max_local_outgoing_node_count_, static_cast<size_t>(next_local_outgoing_node));
  }
}

const CBC_FLUDSCommonData::IncomingNonlocalFaceInfo&
CBC_FLUDSCommonData::GetIncomingNonlocalFaceInfo(const std::uint32_t cell_local_id,
                                                 const unsigned int face_id) const noexcept
{
  return incoming_nonlocal_face_info_[cell_face_offsets_[cell_local_id] + face_id];
}

const CBC_FLUDSCommonData::IncomingNonlocalFaceInfo&
CBC_FLUDSCommonData::GetIncomingNonlocalFaceInfoByKey(const std::uint64_t cell_global_id,
                                                      const unsigned int face_id) const noexcept
{
  return incoming_nonlocal_face_info_[GetIncomingNonlocalFaceStorageIndexByKey(cell_global_id,
                                                                               face_id)];
}

const CBC_FLUDSCommonData::IncomingNonlocalFaceInfo&
CBC_FLUDSCommonData::GetIncomingNonlocalFaceInfoByStorageIndex(
  const std::size_t storage_index) const noexcept
{
  return incoming_nonlocal_face_info_[storage_index];
}

std::size_t
CBC_FLUDSCommonData::GetIncomingNonlocalFaceStorageIndexByKey(
  const std::uint64_t cell_global_id, const unsigned int face_id) const noexcept
{
  const auto it = incoming_nonlocal_face_info_by_key_.find({cell_global_id, face_id});
  assert(it != incoming_nonlocal_face_info_by_key_.end());
  return it->second;
}

const CBC_FLUDSCommonData::OutgoingNonlocalFaceInfo&
CBC_FLUDSCommonData::GetOutgoingNonlocalFaceInfo(const std::uint32_t cell_local_id,
                                                 const unsigned int face_id) const noexcept
{
  return outgoing_nonlocal_face_info_[cell_face_offsets_[cell_local_id] + face_id];
}

std::uint32_t
CBC_FLUDSCommonData::GetLocalOutgoingCompactNodeIndex(const std::uint32_t cell_local_id,
                                                      const std::uint32_t cell_node) const noexcept
{
  const auto compact_index = local_outgoing_node_indices_[cell_node_offsets_[cell_local_id] + cell_node];
  assert(compact_index != std::numeric_limits<std::uint32_t>::max());
  return compact_index;
}

} // namespace opensn

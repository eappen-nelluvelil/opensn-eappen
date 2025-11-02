// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include <sys/types.h>

namespace opensn
{

CBCD_FLUDS::CBCD_FLUDS(CBC_FLUDS& cbc_fluds,
                       size_t num_total_faces,
                       size_t incoming_boundary_psi_buffer_size,
                       const std::vector<uint64_t>& face_neighbor_local_ids,
                       const std::vector<unsigned int>& face_neighbor_cell_node_map,
                       const std::vector<int>& cell_to_local_face_offset_map,
                       const std::vector<int>& boundary_psi_map)
{
  device_buffer_ = crb::DeviceMemory<double>(cbc_fluds.GetGPULocalPsiDataSize());

  cbc_fluds.BuildDeviceCellDOFMap();
  cell_dof_map_storage_ = Storage<size_t>(cbc_fluds.GetCellDOFMap().size());
  cell_dof_map_storage_.Copy(cbc_fluds.GetCellDOFMap().begin(), cbc_fluds.GetCellDOFMap().end());

  cell_id_storage_ = Storage<uint64_t>(cbc_fluds.GetNumLocalCells());
  cell_face_offset_storage_ = Storage<int>(cbc_fluds.GetNumLocalCells());

  // Auxiliary maps for dealing with cells with purely local dependencies
  face_neighbor_cell_node_map_storage_ = Storage<unsigned int>(num_total_faces * cbc_max_face_dofs);
  face_neighbor_cell_node_map_storage_.Copy(face_neighbor_cell_node_map.begin(), face_neighbor_cell_node_map.end());
  face_neighbor_local_ids_storage_ = Storage<uint64_t>(num_total_faces);
  face_neighbor_local_ids_storage_.Copy(face_neighbor_local_ids.begin(), face_neighbor_local_ids.end());

  // Boundary
  boundary_psi_buffer_ = Storage<double>(incoming_boundary_psi_buffer_size);
  boundary_psi_map_storage_ = Storage<int>(num_total_faces);
  boundary_psi_map_storage_.Copy(boundary_psi_map.begin(), boundary_psi_map.end());

  cell_to_local_face_offset_storage_ = Storage<int>(cell_to_local_face_offset_map.size());
  cell_to_local_face_offset_storage_.Copy(cell_to_local_face_offset_map.begin(), cell_to_local_face_offset_map.end());
}

void
CBC_FLUDS::Prepare_CBCD_FLUDS(AngleSet& angle_set)
{
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();

  size_t num_total_faces = 0;
  for (const auto& cell : grid->local_cells)
    num_total_faces += cell.faces.size();

  std::vector<int> cell_to_local_face_offset_map(grid->local_cells.size());
  size_t cell_face_stride = 0;
  for (const auto& cell : grid->local_cells)
  {
    cell_to_local_face_offset_map[cell.local_id] = cell_face_stride;
    cell_face_stride += cell.faces.size();
  }

  std::vector<unsigned int> face_neighbor_cell_node_map(num_total_faces * cbc_max_face_dofs, 0);
  std::vector<uint64_t> face_neighbor_local_ids(num_total_faces, 0);

  std::vector<int> boundary_psi_map(num_total_faces, -1);
  size_t incoming_boundary_psi_buffer_size = 0;
  size_t face_offset_stride = 0;

  for (const auto& cell : grid->local_cells)
  {
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const FaceNodalMapping& face_nodal_mapping = GetCommonData().GetFaceNodalMapping(cell.local_id, f);
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;

      if ((face_orientations[f] != FaceOrientation::INCOMING))
        continue;

      const size_t current_face_offset = face_offset_stride + f;

      if (is_local_face)
      {
        face_neighbor_local_ids[current_face_offset] = face.GetNeighborLocalID(sdm_.GetGrid().get());
        const size_t neighbor_node_map_offset = current_face_offset * cbc_max_face_dofs;
        for (size_t fj = 0; fj < num_face_nodes; ++fj)
          face_neighbor_cell_node_map[neighbor_node_map_offset + fj] =
            face_nodal_mapping.cell_node_mapping_[fj];
      }
      else if (not is_boundary_face)
      {

      }
      else
      {
        boundary_psi_map[current_face_offset] = incoming_boundary_psi_buffer_size;
        incoming_boundary_psi_buffer_size += (num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups());
      }
    }
    face_offset_stride += cell.faces.size();
  }

  num_total_faces_ = num_total_faces;
  incoming_boundary_psi_buffer_size_ = incoming_boundary_psi_buffer_size;
  face_neighbor_local_ids_ = face_neighbor_local_ids;
  face_neighbor_cell_node_map_ = face_neighbor_cell_node_map;
  cell_to_local_face_offset_map_ = cell_to_local_face_offset_map;
  boundary_psi_map_ = boundary_psi_map;
}

std::vector<double>
CBC_FLUDS::GetBoundaryPsiData(SweepChunk& sweep_chunk,
                              AngleSet& angle_set)
{
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();
  const auto& angle_indices = cbc_angle_set.GetAngleIndices();


  std::vector<double> boundary_psi_buffer(incoming_boundary_psi_buffer_size_, 0.0);
  size_t boundary_buffer_offset = 0;

  for (const auto& cell : grid->local_cells)
  {
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;

      if (face_orientations[f] != FaceOrientation::INCOMING or is_local_face or not is_boundary_face)
        continue;

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
        {
          const auto direction_num = angle_indices[as_ss_idx];
          const double* psi_in = cbc_angle_set.PsiBoundary(
            face.neighbor_id,
            direction_num,
            cell.local_id,
            f,
            fj,
            sweep_chunk.GetGroupSetGroupIndex(),
            sweep_chunk.IsSurfaceSourceActive()
          );

          if (psi_in)
            std::copy(psi_in,
                      psi_in + angle_set.GetNumGroups(),
                      &boundary_psi_buffer[boundary_buffer_offset +
                                           (fj * angle_set.GetNumAngles() + as_ss_idx) * angle_set.GetNumGroups()]);
        }
      }
      boundary_buffer_offset += (num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups());
    }
  }
  return boundary_psi_buffer;
}

void
CBC_FLUDS::UpdateBoundaryPsiData(SweepChunk& sweep_chunk,
                                 AngleSet& angle_set)
{
  const auto& boundary_psi_buffer =
      GetBoundaryPsiData(sweep_chunk, angle_set);
  bool boundary_psi_data_changed = (incoming_boundary_psi_buffer_ != boundary_psi_buffer);

  if (boundary_psi_data_changed)
  {
    incoming_boundary_psi_buffer_ = boundary_psi_buffer;
    SetBoundaryPsiData(incoming_boundary_psi_buffer_);
  }
}

void
CBC_FLUDS::Create_CBCD_FLUDS()
{
  if (cbcd_fluds_ == nullptr)
  {
    CBCD_FLUDS* cbcd_fluds = new CBCD_FLUDS(*this,
                                            num_total_faces_,
                                            incoming_boundary_psi_buffer_size_,
                                            face_neighbor_local_ids_,
                                            face_neighbor_cell_node_map_,
                                            cell_to_local_face_offset_map_,
                                            boundary_psi_map_);
    cbcd_fluds_ = cbcd_fluds;

    incoming_boundary_psi_buffer_.resize(incoming_boundary_psi_buffer_size_, 0.0);
  }
}

void
CBC_FLUDS::SetBoundaryPsiData(const std::vector<double>& boundary_psi)
{
  reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->boundary_psi_buffer_.Copy(boundary_psi.begin(), boundary_psi.end());
}

void
CBC_FLUDS::Destroy_CBCD_FLUDS()
{
  if (cbcd_fluds_)
  {
    CBCD_FLUDS* cbcd_fluds = reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_);
    delete cbcd_fluds;
    cbcd_fluds_ = nullptr;
  }
}

} // namespace opensn
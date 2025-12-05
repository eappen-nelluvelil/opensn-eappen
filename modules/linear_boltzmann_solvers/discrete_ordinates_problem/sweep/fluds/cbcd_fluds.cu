// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include <sys/types.h>

namespace opensn
{

CBCD_FLUDS::CBCD_FLUDS(CBC_FLUDS& cbc_fluds)
{
  /*
  // opensn::log.Log() << "Created CBCD_FLUDS\n";
  device_buffer_ = crb::DeviceMemory<double>(cbc_fluds.GetGPULocalPsiDataSize());

  cell_dof_map_storage_ = Storage<size_t>(cbc_fluds.GetCellDOFMap().size());
  cell_dof_map_storage_.Copy(cbc_fluds.GetCellDOFMap().begin(), cbc_fluds.GetCellDOFMap().end());

  // cell_id_storage_ = Storage<uint64_t>(cbc_fluds.GetNumLocalCells());
  // cell_face_offset_storage_ = Storage<int>(cbc_fluds.GetNumLocalCells());

  // Auxiliary maps for dealing with cells with purely local dependencies
  face_neighbor_cell_node_map_storage_ =
    Storage<unsigned int>(cbc_fluds.num_total_faces_ * cbc_max_face_dofs);
  face_neighbor_cell_node_map_storage_.Copy(cbc_fluds.face_neighbor_cell_node_map_.begin(),
                                            cbc_fluds.face_neighbor_cell_node_map_.end());

  face_neighbor_local_ids_storage_ = Storage<uint64_t>(cbc_fluds.num_total_faces_);
  face_neighbor_local_ids_storage_.Copy(cbc_fluds.face_neighbor_local_ids_.begin(),
                                        cbc_fluds.face_neighbor_local_ids_.end());

  // Boundary
  boundary_psi_buffer_ = Storage<double>(cbc_fluds.incoming_boundary_psi_buffer_size_);
  boundary_psi_map_storage_ = Storage<int>(cbc_fluds.num_total_faces_);
  boundary_psi_map_storage_.Copy(cbc_fluds.boundary_psi_map_.begin(),
                                 cbc_fluds.boundary_psi_map_.end());

  cell_to_local_face_offset_storage_ =
    Storage<int>(cbc_fluds.cell_to_local_face_offset_map_.size());
  cell_to_local_face_offset_storage_.Copy(cbc_fluds.cell_to_local_face_offset_map_.begin(),
                                          cbc_fluds.cell_to_local_face_offset_map_.end());

  incoming_face_category_map_storage_ = Storage<int>(cbc_fluds.num_total_faces_);
  incoming_face_category_map_storage_.Copy(cbc_fluds.incoming_face_category_map_.begin(),
                                           cbc_fluds.incoming_face_category_map_.end());

  outgoing_face_category_map_storage_ = Storage<int>(cbc_fluds.num_total_faces_);
  outgoing_face_category_map_storage_.Copy(cbc_fluds.outgoing_face_category_map_.begin(),
                                           cbc_fluds.outgoing_face_category_map_.end());

  // 11/4: Containers for performing transport sweeps with streams
  cells_for_stream_1_storage_ = Storage<uint64_t>(cbc_fluds.GetNumLocalCells());

  cells_for_stream_2_storage_ = Storage<uint64_t>(cbc_fluds.GetNumLocalCells());
  cell_face_offset_for_stream_2_storage_ = Storage<int>(cbc_fluds.GetNumLocalCells());

  upwind_psi_buffer_storage_ = Storage<double>(cbc_fluds.non_local_upwind_psi_buffer_size_);
  upwind_psi_offsets_storage_ = Storage<int>(cbc_fluds.num_total_faces_);

  downwind_psi_buffer_storage_ =
    Storage<double>(cbc_fluds.non_local_and_reflecting_psi_buffer_size_);
  downwind_psi_offsets_storage_ = Storage<int>(cbc_fluds.num_total_faces_);
  */
  // ---------------------------------------------------------------------------
  cell_local_ids_storage_ = Storage<uint64_t>(cbc_fluds.GetNumLocalCells());

  cell_to_face_offset_map_storage_ =
    Storage<uint64_t>(cbc_fluds.cell_to_face_offset_map_gpu_.size());
  cell_to_face_offset_map_storage_.Copy(cbc_fluds.cell_to_face_offset_map_gpu_.begin(),
                                        cbc_fluds.cell_to_face_offset_map_gpu_.end());

  cell_face_node_angle_group_offsets_map_storage_ =
    Storage<uint64_t>(cbc_fluds.cell_face_node_angle_group_offsets_map_gpu_.size());
  cell_face_node_angle_group_offsets_map_storage_.Copy(
    cbc_fluds.cell_face_node_angle_group_offsets_map_gpu_.begin(),
    cbc_fluds.cell_face_node_angle_group_offsets_map_gpu_.end());

  cell_psi_data_buffer_storage_ = Storage<double>(cbc_fluds.GetGPULocalPsiDataSize() +
                                                  cbc_fluds.nonlocal_and_boundary_psi_buffer_size_);
  // ---------------------------------------------------------------------------
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

  std::vector<int> incoming_face_category_map(num_total_faces, 0);
  std::vector<int> outgoing_face_category_map(num_total_faces, 0);

  std::vector<int> boundary_psi_map(num_total_faces, -1);
  size_t incoming_boundary_psi_buffer_size = 0;
  size_t face_offset_stride = 0;

  // Size for all incoming psi
  size_t cell_face_psi_buffer_size = 0;

  // Size for device storage for incoming non-local upwind psi
  size_t non_local_upwind_psi_buffer_size = 0;

  // Size for device storage for outgoing non-local and reflecting boundary psi
  size_t non_local_and_reflecting_psi_buffer_size = 0;

  for (const auto& cell : grid->local_cells)
  {
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size =
        num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups();
      const FaceNodalMapping& face_nodal_mapping =
        GetCommonData().GetFaceNodalMapping(cell.local_id, f);
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face) and
        (angle_set.GetBoundaries().at(cell.faces[f].neighbor_id)->IsReflecting());

      const size_t current_face_offset = face_offset_stride + f;

      if ((face_orientations[f] == FaceOrientation::INCOMING))
      {
        if (is_local_face)
        {
          face_neighbor_local_ids[current_face_offset] =
            face.GetNeighborLocalID(sdm_.GetGrid().get());
          const size_t neighbor_node_map_offset = current_face_offset * cbc_max_face_dofs;
          for (size_t fj = 0; fj < num_face_nodes; ++fj)
            face_neighbor_cell_node_map[neighbor_node_map_offset + fj] =
              face_nodal_mapping.cell_node_mapping_[fj];

          incoming_face_category_map[current_face_offset] = -1; // Local face
        }
        else if (not is_boundary_face)
        {
          incoming_face_category_map[current_face_offset] = 0; // Non-local face
          non_local_upwind_psi_buffer_size += face_data_size;
        }
        else
        {
          boundary_psi_map[current_face_offset] = incoming_boundary_psi_buffer_size;
          incoming_boundary_psi_buffer_size +=
            (num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups());
          incoming_face_category_map[current_face_offset] = -2; // Boundary face
        }
      }
      else if (face_orientations[f] == FaceOrientation::OUTGOING)
      {
        if (((not is_local_face) and (not is_boundary_face)) or (is_reflecting_boundary_face))
          non_local_and_reflecting_psi_buffer_size += face_data_size;
        else
          outgoing_face_category_map[current_face_offset] = -1; // Local face
      }

      cell_face_psi_buffer_size += face_data_size;
    }
    face_offset_stride += cell.faces.size();
  }

  num_total_faces_ = num_total_faces;
  incoming_boundary_psi_buffer_size_ = incoming_boundary_psi_buffer_size;
  face_neighbor_local_ids_ = face_neighbor_local_ids;
  face_neighbor_cell_node_map_ = face_neighbor_cell_node_map;
  cell_to_local_face_offset_map_ = cell_to_local_face_offset_map;
  boundary_psi_map_ = boundary_psi_map;
  incoming_face_category_map_ = incoming_face_category_map;
  outgoing_face_category_map_ = outgoing_face_category_map;
  non_local_upwind_psi_buffer_size_ = non_local_upwind_psi_buffer_size;
  non_local_and_reflecting_psi_buffer_size_ = non_local_and_reflecting_psi_buffer_size;
  cell_face_psi_buffer_size_ = cell_face_psi_buffer_size;
}

std::vector<double>
CBC_FLUDS::GetBoundaryPsiData(SweepChunk& sweep_chunk, AngleSet& angle_set)
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

      if (face_orientations[f] != FaceOrientation::INCOMING or is_local_face or
          not is_boundary_face)
        continue;

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
        {
          const auto direction_num = angle_indices[as_ss_idx];
          const double* psi_in = cbc_angle_set.PsiBoundary(face.neighbor_id,
                                                           direction_num,
                                                           cell.local_id,
                                                           f,
                                                           fj,
                                                           sweep_chunk.GetGroupSetGroupIndex(),
                                                           sweep_chunk.IsSurfaceSourceActive());

          if (psi_in)
            std::copy(psi_in,
                      psi_in + angle_set.GetNumGroups(),
                      &boundary_psi_buffer[boundary_buffer_offset +
                                           (fj * angle_set.GetNumAngles() + as_ss_idx) *
                                             angle_set.GetNumGroups()]);
        }
      }
      boundary_buffer_offset +=
        (num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups());
    }
  }
  return boundary_psi_buffer;
}

void
CBC_FLUDS::SetBoundaryPsiData(SweepChunk& sweep_chunk, AngleSet& angle_set)
{
  const auto& boundary_psi_buffer = GetBoundaryPsiData(sweep_chunk, angle_set);
  incoming_boundary_psi_buffer_ = boundary_psi_buffer;
  reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
    ->boundary_psi_buffer_.Copy(incoming_boundary_psi_buffer_.begin(),
                                incoming_boundary_psi_buffer_.end());
}

// -----------------------------------------------------------------------------

// /*
void
CBC_FLUDS::Prepare_CBCD_FLUDS_V2(AngleSet& angle_set)
{
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();

  size_t cell_face_node_angle_group_offsets_map_size = 0;
  std::vector<std::uint64_t> cell_to_face_offset_map(grid->local_cells.size(), 0);

  bool has_nonlocal_faces = false;
  bool has_reflecting_boundary_faces = false;

  for (const auto& cell : grid->local_cells)
  {
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    const auto& cell_transport_view = cell_transport_views_[cell.local_id];

    cell_to_face_offset_map[cell.local_id] = cell_face_node_angle_group_offsets_map_size;

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_groups_and_angles_;

      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        is_boundary_face &&
        cbc_angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting();

      if (!is_local_face && !is_boundary_face)
        has_nonlocal_faces = true;
      if (is_reflecting_boundary_face)
        has_reflecting_boundary_faces = true;

      cell_face_node_angle_group_offsets_map_size += face_data_size;
    }
  }

  std::vector<std::uint64_t> cell_face_node_angle_group_offsets_map(
    cell_face_node_angle_group_offsets_map_size, 0);

  // dense counter for second section
  std::uint64_t next_nonlocal_and_boundary = 0;

  for (const auto& cell : grid->local_cells)
  {
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    const auto& cell_transport_view = cell_transport_views_[cell.local_id];

    size_t face_offset_stride = 0;

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_groups_and_angles_;
      const FaceNodalMapping& face_nodal_mapping =
        GetCommonData().GetFaceNodalMapping(cell.local_id, f);

      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        is_boundary_face &&
        angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting();

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < num_angles_; ++as_ss_idx)
        {
          for (size_t gsg = 0; gsg < num_groups_; ++gsg)
          {
            const std::uint64_t map_idx =
              cell_to_face_offset_map[cell.local_id] +
              face_offset_stride +
              (fj * num_groups_and_angles_) +
              (as_ss_idx * num_groups_) + gsg;

            std::uint64_t map_val = 0;

            if (face_orientations[f] == FaceOrientation::INCOMING)
            {
              if (is_local_face)
              {
                const std::uint64_t nbr_lid =
                  cell_transport_view.FaceNeighbor(f)->local_id;
                const std::uint64_t nbr_base = cell_dof_map_[nbr_lid];
                const std::uint64_t adj_cell_node =
                  face_nodal_mapping.cell_node_mapping_[fj];
                const std::uint64_t addr_offset =
                  adj_cell_node * num_groups_and_angles_ +
                  as_ss_idx * num_groups_ + gsg;

                map_val = nbr_base + addr_offset;

                // incoming, local
                cell_face_node_angle_group_offsets_map[map_idx] =
                  (map_val & 0x0FFFFFFFFFFFFFFFULL) | (0x1ULL << 62) | (0x1ULL << 60);
              }
              else if (!is_boundary_face)
              {
                // incoming, nonlocal
                const std::uint64_t base =
                  gpu_local_psi_data_size_ + next_nonlocal_and_boundary++;
                map_val = base;

                cell_face_node_angle_group_offsets_map[map_idx] =
                  (map_val & 0x0FFFFFFFFFFFFFFFULL) |
                  (0x1ULL << 62) | (0x2ULL << 60);
              }
              else
              {
                // incoming, boundary (non‑reflecting or reflecting)
                const std::uint64_t base =
                  gpu_local_psi_data_size_ + next_nonlocal_and_boundary++;
                map_val = base;

                cell_face_node_angle_group_offsets_map[map_idx] =
                  (map_val & 0x0FFFFFFFFFFFFFFFULL) |
                  (0x1ULL << 62) | (0x3ULL << 60);
              }
            }
            else if (face_orientations[f] == FaceOrientation::OUTGOING)
            {
              if (is_local_face)
              {
                const std::uint64_t cur_base = cell_dof_map_[cell.local_id];
                const std::uint64_t j = cell_mapping.MapFaceNode(f, fj);
                const std::uint64_t addr_offset =
                  j * num_groups_and_angles_ +
                  as_ss_idx * num_groups_ + gsg;

                map_val = cur_base + addr_offset;

                // outgoing, local
                cell_face_node_angle_group_offsets_map[map_idx] =
                  (map_val & 0x0FFFFFFFFFFFFFFFULL) |
                  (0x2ULL << 62) | (0x1ULL << 60);
              }
              else if (!is_boundary_face)
              {
                // outgoing, nonlocal
                const std::uint64_t base =
                  gpu_local_psi_data_size_ + next_nonlocal_and_boundary++;
                map_val = base;

                cell_face_node_angle_group_offsets_map[map_idx] =
                  (map_val & 0x0FFFFFFFFFFFFFFFULL) |
                  (0x2ULL << 62) | (0x2ULL << 60);
              }
              else if (is_reflecting_boundary_face)
              {
                // outgoing, reflecting boundary
                const std::uint64_t base =
                  gpu_local_psi_data_size_ + next_nonlocal_and_boundary++;
                map_val = base;

                cell_face_node_angle_group_offsets_map[map_idx] =
                  (map_val & 0x0FFFFFFFFFFFFFFFULL) |
                  (0x2ULL << 62) | (0x3ULL << 60);
              }
              else
              {
                // outgoing, non‑reflecting boundary:
                // keep encoded index (for outflow tally), but
                // we don’t need to reuse psi later.
                const std::uint64_t base =
                  gpu_local_psi_data_size_ + next_nonlocal_and_boundary++;
                map_val = base;

                cell_face_node_angle_group_offsets_map[map_idx] =
                  (map_val & 0x0FFFFFFFFFFFFFFFULL) |
                  (0x2ULL << 62) | (0x0ULL << 60);
              }
            }
          } // gsg
        }   // as_ss_idx
      }     // fj

      face_offset_stride += face_data_size;
    } // faces
  }   // cells

  has_nonlocal_faces_ = has_nonlocal_faces;
  has_reflecting_boundary_faces_ = has_reflecting_boundary_faces;

  nonlocal_and_boundary_psi_buffer_size_ = next_nonlocal_and_boundary;
  cell_to_face_offset_map_gpu_ = std::move(cell_to_face_offset_map);
  cell_face_node_angle_group_offsets_map_gpu_ =
    std::move(cell_face_node_angle_group_offsets_map);
}
// */

// /*
void
CBC_FLUDS::GetAndSetBoundaryPsiData(SweepChunk& sweep_chunk, AngleSet& angle_set)
{
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();
  const auto& angle_indices = cbc_angle_set.GetAngleIndices();

  for (const auto& cell : grid->local_cells)
  {
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    size_t face_offset_stride = 0;

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_groups_and_angles_;

      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;

      if (face_orientations[f] != FaceOrientation::INCOMING ||
          is_local_face || !is_boundary_face)
      {
        face_offset_stride += face_data_size;
        continue;
      }

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
        {
          const auto direction_num = angle_indices[as_ss_idx];
          const double* psi = cbc_angle_set.PsiBoundary(face.neighbor_id,
                                                        direction_num,
                                                        cell.local_id,
                                                        f,
                                                        fj,
                                                        sweep_chunk.GetGroupSetGroupIndex(),
                                                        sweep_chunk.IsSurfaceSourceActive());

          const std::uint64_t map_idx =
            cell_to_face_offset_map_gpu_[cell.local_id] +
            face_offset_stride +
            (fj * num_groups_and_angles_) +
            (as_ss_idx * num_groups_);

          const std::uint64_t encoded =
            cell_face_node_angle_group_offsets_map_gpu_[map_idx];
          const std::uint64_t true_idx = encoded & 0x0FFFFFFFFFFFFFFFULL;

          double* buffer =
            &reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
               ->cell_psi_data_buffer_storage_
               .GetHostVector()[true_idx];

          if (psi)
            std::copy(psi, psi + num_groups_, buffer);
        }
      }

      face_offset_stride += face_data_size;
    }
  }

  // copy only the dense tail
  crb::copy(
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetDeviceMemory(),
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetHostVector(),
    nonlocal_and_boundary_psi_buffer_size_,
    gpu_local_psi_data_size_,
    gpu_local_psi_data_size_);
}
// */

void
CBC_FLUDS::GetAndSetBoundaryPsiDataAsync(SweepChunk& sweep_chunk, AngleSet& angle_set)
{
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();
  const auto& angle_indices = cbc_angle_set.GetAngleIndices();

  for (const auto& cell : grid->local_cells)
  {
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    size_t face_offset_stride = 0;

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_groups_and_angles_;

      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;

      if (face_orientations[f] != FaceOrientation::INCOMING ||
          is_local_face || !is_boundary_face)
      {
        face_offset_stride += face_data_size;
        continue;
      }

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
        {
          const auto direction_num = angle_indices[as_ss_idx];
          const double* psi = cbc_angle_set.PsiBoundary(face.neighbor_id,
                                                        direction_num,
                                                        cell.local_id,
                                                        f,
                                                        fj,
                                                        sweep_chunk.GetGroupSetGroupIndex(),
                                                        sweep_chunk.IsSurfaceSourceActive());

          const std::uint64_t map_idx =
            cell_to_face_offset_map_gpu_[cell.local_id] +
            face_offset_stride +
            (fj * num_groups_and_angles_) +
            (as_ss_idx * num_groups_);

          const std::uint64_t encoded =
            cell_face_node_angle_group_offsets_map_gpu_[map_idx];
          const std::uint64_t true_idx = encoded & 0x0FFFFFFFFFFFFFFFULL;

          double* buffer =
            &reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
               ->cell_psi_data_buffer_storage_
               .GetHostVector()[true_idx];

          if (psi)
            std::copy(psi, psi + num_groups_, buffer);
        }
      }

      face_offset_stride += face_data_size;
    }
  }

  // copy only the dense tail
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cbc_angle_set.stream_ptr);

  cudaMemcpyAsync(
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetDeviceMemory().get() + 
      gpu_local_psi_data_size_,
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetHostVector().data() + 
      gpu_local_psi_data_size_,
    nonlocal_and_boundary_psi_buffer_size_ * sizeof(double),
    cudaMemcpyHostToDevice,
    stream);
}

// /*
void
CBC_FLUDS::GetNonlocalPsiData(SweepChunk& sweep_chunk,
                              AngleSet& angle_set,
                              std::vector<Task*>& tasks)
{
  if (!has_nonlocal_faces_) return;

  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();
  const auto& angle_indices = cbc_angle_set.GetAngleIndices();

  for (Task* task : tasks)
  {
    const auto& cell = grid->local_cells[task->reference_id];
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    size_t face_offset_stride = 0;

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_groups_and_angles_;
      const FaceNodalMapping& face_nodal_mapping =
        GetCommonData().GetFaceNodalMapping(cell.local_id, f);

      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = !face.has_neighbor;

      if (face_orientations[f] != FaceOrientation::INCOMING ||
          is_local_face || is_boundary_face)
      {
        face_offset_stride += face_data_size;
        continue;
      }

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < angle_indices.size(); ++as_ss_idx)
        {
          const std::uint64_t map_idx =
            cell_to_face_offset_map_gpu_[cell.local_id] +
            face_offset_stride +
            (fj * num_groups_and_angles_) +
            (as_ss_idx * num_groups_);

          const std::uint64_t encoded =
            cell_face_node_angle_group_offsets_map_gpu_[map_idx];
          const std::uint64_t true_idx = encoded & 0x0FFFFFFFFFFFFFFFULL;

          double* buffer =
            &reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
               ->cell_psi_data_buffer_storage_
               .GetHostVector()[true_idx];

          const double* psi =
            NLUpwindPsi(cell.global_id,
                        f,
                        face_nodal_mapping.face_node_mapping_[fj],
                        as_ss_idx);

          if (psi)
            std::copy(psi, psi + num_groups_, buffer);
        }
      }

      face_offset_stride += face_data_size;
    }
  }

  crb::copy(
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetDeviceMemory(),
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetHostVector(),
    nonlocal_and_boundary_psi_buffer_size_,
    gpu_local_psi_data_size_,
    gpu_local_psi_data_size_);
}
// */

void
CBC_FLUDS::GetNonlocalPsiDataAsync(SweepChunk& sweep_chunk,
                                   AngleSet& angle_set,
                                   std::vector<Task*>& tasks)
{
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();
  const auto& angle_indices = cbc_angle_set.GetAngleIndices();

  // Determine if there are non-local faces to process
  bool has_incoming_nonlocal_faces = false;

  for (Task* task : tasks)
  {
    const auto& cell = grid->local_cells[task->reference_id];
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_transport_view = cell_transport_views_[cell.local_id];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = !face.has_neighbor;

      if (face_orientations[f] == FaceOrientation::INCOMING &&
          !is_local_face && !is_boundary_face)
      {
        has_incoming_nonlocal_faces = true;
        break; // Exit inner loop early
      }
    }

    if (has_incoming_nonlocal_faces)
      break; // Exit outer loop early
  }

  if (!has_incoming_nonlocal_faces) return;

  for (Task* task : tasks)
  {
    const auto& cell = grid->local_cells[task->reference_id];
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    size_t face_offset_stride = 0;

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_groups_and_angles_;
      const FaceNodalMapping& face_nodal_mapping =
        GetCommonData().GetFaceNodalMapping(cell.local_id, f);

      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = !face.has_neighbor;

      if (face_orientations[f] != FaceOrientation::INCOMING ||
          is_local_face || is_boundary_face)
      {
        face_offset_stride += face_data_size;
        continue;
      }

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < angle_indices.size(); ++as_ss_idx)
        {
          const std::uint64_t map_idx =
            cell_to_face_offset_map_gpu_[cell.local_id] +
            face_offset_stride +
            (fj * num_groups_and_angles_) +
            (as_ss_idx * num_groups_);

          const std::uint64_t encoded =
            cell_face_node_angle_group_offsets_map_gpu_[map_idx];
          const std::uint64_t true_idx = encoded & 0x0FFFFFFFFFFFFFFFULL;

          double* buffer =
            &reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
               ->cell_psi_data_buffer_storage_
               .GetHostVector()[true_idx];

          const double* psi =
            NLUpwindPsi(cell.global_id,
                        f,
                        face_nodal_mapping.face_node_mapping_[fj],
                        as_ss_idx);

          if (psi)
            std::copy(psi, psi + num_groups_, buffer);
        }
      }

      face_offset_stride += face_data_size;
    }
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cbc_angle_set.stream_ptr);

  cudaMemcpyAsync(reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetDevicePtr() + gpu_local_psi_data_size_,
                  reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetHostVector().data() + gpu_local_psi_data_size_,
                  sizeof(double) * nonlocal_and_boundary_psi_buffer_size_,
                  cudaMemcpyHostToDevice,
                  stream);
}

// /*
void
CBC_FLUDS::SetNonlocalAndReflectingBoundaryPsiData(SweepChunk& sweep_chunk,
                                                   AngleSet& angle_set,
                                                   std::vector<Task*>& tasks)
{
  if (!has_nonlocal_faces_ && !has_reflecting_boundary_faces_) return;

  // bring back only the dense tail
  crb::copy(
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetHostVector(),
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetDeviceMemory(),
    nonlocal_and_boundary_psi_buffer_size_,
    GetGPULocalPsiDataSize(),
    GetGPULocalPsiDataSize());

  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();
  const auto& angle_indices = cbc_angle_set.GetAngleIndices();

  for (Task* task : tasks)
  {
    const auto& cell = grid->local_cells[task->reference_id];
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    const size_t cell_to_face_offset = cell_to_face_offset_map_gpu_[cell.local_id];
    size_t face_offset_stride = 0;

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_groups_and_angles_;
      const FaceNodalMapping& face_nodal_mapping =
        GetCommonData().GetFaceNodalMapping(cell.local_id, f);

      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = !face.has_neighbor;
      const bool is_reflecting_boundary_face =
        is_boundary_face &&
        angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting();

      if (face_orientations[f] != FaceOrientation::OUTGOING || is_local_face)
      {
        face_offset_stride += face_data_size;
        continue;
      }

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < num_angles_; ++as_ss_idx)
        {
          const std::uint64_t map_idx =
            cell_to_face_offset + face_offset_stride +
            (fj * num_groups_and_angles_) +
            (as_ss_idx * num_groups_);

          const std::uint64_t encoded =
            cell_face_node_angle_group_offsets_map_gpu_[map_idx];
          const std::uint64_t true_idx = encoded & 0x0FFFFFFFFFFFFFFFULL;

          if (!is_boundary_face)
          {
            const int locality = cell_transport_view.FaceLocality(f);
            auto& async_comm = *cbc_angle_set.GetCommunicator();
            std::vector<double>* psi_nonlocal_out =
              &async_comm.InitGetDownwindMessageData(locality,
                                                     face.neighbor_id,
                                                     face_nodal_mapping.associated_face_,
                                                     cbc_angle_set.GetID(),
                                                     face_data_size);

            double* psi = NLOutgoingPsi(psi_nonlocal_out, fj, as_ss_idx);

            if (psi)
            {
              auto& buf =
                reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
                  ->cell_psi_data_buffer_storage_.GetHostVector();
              std::copy(buf.data() + true_idx,
                        buf.data() + true_idx + num_groups_,
                        psi);
            }
          }
          else if (is_reflecting_boundary_face)
          {
            const auto direction_num = angle_indices[as_ss_idx];
            double* psi =
              cbc_angle_set.PsiReflected(face.neighbor_id,
                                         direction_num,
                                         cell.local_id,
                                         f,
                                         fj);

            if (psi)
            {
              auto& buf =
                reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
                  ->cell_psi_data_buffer_storage_.GetHostVector();
              std::copy(buf.data() + true_idx,
                        buf.data() + true_idx + num_groups_,
                        psi);
            }
          }
        }
      }

      face_offset_stride += face_data_size;
    }
  }
}
// */

void 
CBC_FLUDS::SetNonlocalAndReflectingBoundaryPsiDataAsync(SweepChunk& sweep_chunk,
                                                        AngleSet& angle_set,
                                                        std::vector<Task*>& tasks)
{
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();
  const auto& angle_indices = cbc_angle_set.GetAngleIndices();

  // Determine if there are outgoing non-local or reflecting boundary faces to process
  bool has_outgoing_nonlocal_faces = false;
  bool has_outgoing_reflecting_boundary_faces = false;

  for (Task* task : tasks)
  {
    const auto& cell = grid->local_cells[task->reference_id];
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_transport_view = cell_transport_views_[cell.local_id];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = !face.has_neighbor;
      const bool is_reflecting_boundary_face =
        is_boundary_face &&
        angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting();

      if (face_orientations[f] == FaceOrientation::OUTGOING && !is_local_face)
      {
        if (!is_boundary_face)
          has_outgoing_nonlocal_faces = true;
        else if (is_reflecting_boundary_face)
          has_outgoing_reflecting_boundary_faces = true;
      }
    }

    if (has_outgoing_nonlocal_faces or has_outgoing_reflecting_boundary_faces)
      break; // Exit outer loop early
  }

  if (!has_outgoing_nonlocal_faces && !has_outgoing_reflecting_boundary_faces)
    return;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cbc_angle_set.stream_ptr);

  cudaMemcpyAsync(reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetHostVector().data() + gpu_local_psi_data_size_,
                  reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.GetDevicePtr() + gpu_local_psi_data_size_,
                  sizeof(double) * nonlocal_and_boundary_psi_buffer_size_,
                  cudaMemcpyDeviceToHost,
                  stream);

  cudaStreamSynchronize(stream);

  for (Task* task : tasks)
  {
    const auto& cell = grid->local_cells[task->reference_id];
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    const size_t cell_to_face_offset = cell_to_face_offset_map_gpu_[cell.local_id];
    size_t face_offset_stride = 0;

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_groups_and_angles_;
      const FaceNodalMapping& face_nodal_mapping =
        GetCommonData().GetFaceNodalMapping(cell.local_id, f);

      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = !face.has_neighbor;
      const bool is_reflecting_boundary_face =
        is_boundary_face &&
        angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting();

      if (face_orientations[f] != FaceOrientation::OUTGOING || is_local_face)
      {
        face_offset_stride += face_data_size;
        continue;
      }

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < num_angles_; ++as_ss_idx)
        {
          const std::uint64_t map_idx =
            cell_to_face_offset + face_offset_stride +
            (fj * num_groups_and_angles_) +
            (as_ss_idx * num_groups_);

          const std::uint64_t encoded =
            cell_face_node_angle_group_offsets_map_gpu_[map_idx];
          const std::uint64_t true_idx = encoded & 0x0FFFFFFFFFFFFFFFULL;

          if (!is_boundary_face)
          {
            const int locality = cell_transport_view.FaceLocality(f);
            auto& async_comm = *cbc_angle_set.GetCommunicator();
            std::vector<double>* psi_nonlocal_out =
              &async_comm.InitGetDownwindMessageData(locality,
                                                     face.neighbor_id,
                                                     face_nodal_mapping.associated_face_,
                                                     cbc_angle_set.GetID(),
                                                     face_data_size);

            double* psi = NLOutgoingPsi(psi_nonlocal_out, fj, as_ss_idx);

            if (psi)
            {
              auto& buf =
                reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
                  ->cell_psi_data_buffer_storage_.GetHostVector();
              std::copy(buf.data() + true_idx,
                        buf.data() + true_idx + num_groups_,
                        psi);
            }
          }
          else if (is_reflecting_boundary_face)
          {
            const auto direction_num = angle_indices[as_ss_idx];
            double* psi =
              cbc_angle_set.PsiReflected(face.neighbor_id,
                                         direction_num,
                                         cell.local_id,
                                         f,
                                         fj);

            if (psi)
            {
              auto& buf =
                reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
                  ->cell_psi_data_buffer_storage_.GetHostVector();
              std::copy(buf.data() + true_idx,
                        buf.data() + true_idx + num_groups_,
                        psi);
            }
          }
        }
      }

      face_offset_stride += face_data_size;
    }
  }
}

void
CBC_FLUDS::Reset_CBCD_FLUDS_Device_Data()
{
  if (cbcd_fluds_ != nullptr)
  {
    std::vector<double> zero_data(
      gpu_local_psi_data_size_ + nonlocal_and_boundary_psi_buffer_size_, 0.0);
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->cell_psi_data_buffer_storage_.Copy(zero_data.begin(), zero_data.end());
  }
}

// -----------------------------------------------------------------------------

void
CBC_FLUDS::Create_CBCD_FLUDS(AngleSet& angle_set)
{ 
  if (cbcd_fluds_ == nullptr)
  {
    // For V1 of GPUSweep
    // Prepare_CBCD_FLUDS(angle_set);
    // This buffer is used to track changes in boundary psi data
    // If boundary data changes between sweeps, we need to
    // update the device boundary psi buffer
    // incoming_boundary_psi_buffer_.resize(incoming_boundary_psi_buffer_size_, 0.0);

    // V2 of GPUSweep
    Prepare_CBCD_FLUDS_V2(angle_set);
    CBCD_FLUDS* cbcd_fluds = new CBCD_FLUDS(*this);
    cbcd_fluds_ = cbcd_fluds;
  }
}

void
CBC_FLUDS::Destroy_CBCD_FLUDS()
{
  if (cbcd_fluds_ != nullptr)
  {
    CBCD_FLUDS* cbcd_fluds = reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_);
    delete cbcd_fluds;
    cbcd_fluds_ = nullptr;
  }
}

} // namespace opensn
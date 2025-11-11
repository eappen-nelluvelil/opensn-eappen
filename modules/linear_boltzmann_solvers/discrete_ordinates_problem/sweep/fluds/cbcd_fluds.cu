// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include <sys/types.h>

namespace opensn
{

CBCD_FLUDS::CBCD_FLUDS(CBC_FLUDS& cbc_fluds)
{
  // opensn::log.Log() << "Created CBCD_FLUDS\n";
  device_buffer_ = crb::DeviceMemory<double>(cbc_fluds.GetGPULocalPsiDataSize());

  cell_dof_map_storage_ = Storage<size_t>(cbc_fluds.GetCellDOFMap().size());
  cell_dof_map_storage_.Copy(cbc_fluds.GetCellDOFMap().begin(), cbc_fluds.GetCellDOFMap().end());

  // cell_id_storage_ = Storage<uint64_t>(cbc_fluds.GetNumLocalCells());
  // cell_face_offset_storage_ = Storage<int>(cbc_fluds.GetNumLocalCells());

  // Auxiliary maps for dealing with cells with purely local dependencies
  face_neighbor_cell_node_map_storage_ = Storage<unsigned int>(cbc_fluds.num_total_faces_ * cbc_max_face_dofs);
  face_neighbor_cell_node_map_storage_.Copy(cbc_fluds.face_neighbor_cell_node_map_.begin(),
                                            cbc_fluds.face_neighbor_cell_node_map_.end());

  face_neighbor_local_ids_storage_ = Storage<uint64_t>(cbc_fluds.num_total_faces_);
  face_neighbor_local_ids_storage_.Copy(cbc_fluds.face_neighbor_local_ids_.begin(),
                                        cbc_fluds.face_neighbor_local_ids_.end());

  // Boundary
  boundary_psi_buffer_ = Storage<double>(cbc_fluds.incoming_boundary_psi_buffer_size_);
  boundary_psi_map_storage_ = Storage<int>(cbc_fluds.num_total_faces_);
  boundary_psi_map_storage_.Copy(cbc_fluds.boundary_psi_map_.begin(), cbc_fluds.boundary_psi_map_.end());

  cell_to_local_face_offset_storage_ = Storage<int>(cbc_fluds.cell_to_local_face_offset_map_.size());
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

  downwind_psi_buffer_storage_ = Storage<double>(cbc_fluds.non_local_and_reflecting_psi_buffer_size_);
  downwind_psi_offsets_storage_ = Storage<int>(cbc_fluds.num_total_faces_);

  // Idea for passing in a single cell's angular fluxes to the incoming surface integral
  // device function
  // incoming_psi_buffer_ = crb::DeviceMemory<double>(cbc_fluds.cell_face_psi_buffer_size_);

  // // -----------------------------------------------------------------------------
  // cell_to_local_face_offset_map_gpu_storage_ =
  //   Storage<uint64_t>(cbc_fluds.cell_to_local_face_offset_map_gpu_size_);
  // cell_to_local_face_offset_map_gpu_storage_.Copy(
  //   cbc_fluds.cell_to_local_face_offset_map_gpu_.begin(),
  //   cbc_fluds.cell_to_local_face_offset_map_gpu_.end());
  // nonlocal_and_boundary_psi_buffer_ =
  //  Storage<double>(cbc_fluds.cell_to_local_face_offset_map_gpu_size_);
  // -----------------------------------------------------------------------------

  // opensn::log.Log() << "CBCD_FLUDS: " << cbc_fluds.cell_to_local_face_offset_map_gpu_.size()
  //                   << " entries in cell_to_local_face_offset_map_gpu_\n";
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
      const size_t face_data_size = num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups();
      const FaceNodalMapping& face_nodal_mapping =
        GetCommonData().GetFaceNodalMapping(cell.local_id, f);
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face) and (angle_set.GetBoundaries().at(cell.faces[f].neighbor_id)->IsReflecting());

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
  reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->boundary_psi_buffer_.Copy(incoming_boundary_psi_buffer_.begin(),
                                incoming_boundary_psi_buffer_.end());
}

// -----------------------------------------------------------------------------

void
CBC_FLUDS::Prepare_CBCD_FLUDS_V2(AngleSet& angle_set)
{
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();

  size_t cell_to_local_face_offset_map_size = 0;
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
      cell_to_local_face_offset_map_size += face_data_size;
    }
  }

  cell_to_local_face_offset_map_gpu_size_ = cell_to_local_face_offset_map_size;

  size_t face_offset_stride = 0;
  size_t current_face_offset = 0;
  cell_to_local_face_offset_map_gpu_.resize(cell_to_local_face_offset_map_size, 0);
 
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

      current_face_offset = face_offset_stride + f;

      if (face_orientations[f] == FaceOrientation::INCOMING)
      {
        if (is_local_face) // Incoming local face
        {
          const auto& face_neighbor = *cell_transport_view.FaceNeighbor(f);
          const size_t face_nbr_spatial_dof_0_index = cell_dof_map_[face_neighbor.local_id];
          const size_t face_nbr_data_start_index =
              face_nbr_spatial_dof_0_index * (angle_set.GetNumAngles() * angle_set.GetNumGroups());
          for (size_t fj = 0; fj < num_face_nodes; ++fj)
          {
            for (size_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
            {
              const size_t addr_offset = (face_nodal_mapping.cell_node_mapping_[fj] *
                                          angle_set.GetNumAngles() * angle_set.GetNumGroups()) +
                                         (as_ss_idx * angle_set.GetNumGroups());
              const size_t face_nbr_data_index = face_nbr_data_start_index + addr_offset;
              for (size_t gsg = 0; gsg < angle_set.GetNumGroups(); ++gsg)
              {
                const size_t idx = current_face_offset * face_data_size +
                                   (fj * angle_set.GetNumAngles() * angle_set.GetNumGroups()) +
                                   (as_ss_idx * angle_set.GetNumGroups()) + gsg;
                const std::uint64_t value = face_nbr_data_index + gsg;

                // 1. Set first bit of cell_to_local_face_offset_map[idx] to 1 to indicate incoming
                // face
                // 2. Set second bit of cell_to_local_face_offset_map[idx] to 1 to indicate local
                // face
                // 3. Set the remaining 62 bits of cell_to_local_face_offset_map[idx] to the
                // appropriate index in the device-side local cell angular flux buffer
                cell_to_local_face_offset_map_gpu_[idx] =
                  (1ULL << 63) | (1ULL << 62) | (value & ((1ULL << 62) - 1));
              }
            }
          }
        }
        else // Incoming non-local or incoming boundary face
        {
          for (size_t fj = 0; fj < num_face_nodes; ++fj)
          {
            for (size_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
            {
              for (size_t gsg = 0; gsg < angle_set.GetNumGroups(); ++gsg)
              {
                const size_t idx = current_face_offset * face_data_size +
                                   (fj * angle_set.GetNumAngles() * angle_set.GetNumGroups()) +
                                   (as_ss_idx * angle_set.GetNumGroups()) + gsg;


                // 1. Set first bit of cell_to_local_face_offset_map[idx] to 1 to indicate incoming face
                // 2. Set second bit of cell_to_local_face_offset_map[idx] to 0 to indicate non-local or boundary face
                // 3. Set the remaining 62 bits of cell_to_local_face_offset_map[idx] to the appropriate index in the device-side non-local/boundary upwind psi buffer  
                cell_to_local_face_offset_map_gpu_[idx] =
                  (1ULL << 63) | (idx & ((1ULL << 62) - 1));
              }
            }
          }
        }
      }
      else if (face_orientations[f] == FaceOrientation::OUTGOING)
      {
        if (is_local_face) // Outgoing local face
        {
          const size_t cur_cell_spatial_dof_0_index = cell_dof_map_[cell.local_id];
          const size_t cur_cell_data_start_index =
              cur_cell_spatial_dof_0_index * (angle_set.GetNumAngles() * angle_set.GetNumGroups());
          for (size_t fj = 0; fj < num_face_nodes; ++fj)
          {
            for (size_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
            {
              const size_t addr_offset = (face_nodal_mapping.cell_node_mapping_[fj] *
                                          angle_set.GetNumAngles() * angle_set.GetNumGroups()) +
                                         (as_ss_idx * angle_set.GetNumGroups());
              const size_t cur_cell_data_index = cur_cell_data_start_index + addr_offset;
              for (size_t gsg = 0; gsg < angle_set.GetNumGroups(); ++gsg)
              {
                const size_t idx = current_face_offset * face_data_size +
                                   (fj * angle_set.GetNumAngles() * angle_set.GetNumGroups()) +
                                   (as_ss_idx * angle_set.GetNumGroups()) + gsg;
                const std::uint64_t value = cur_cell_data_index + gsg;

                // 1. Set first bit of cell_to_local_face_offset_map[idx] to 0 to indicate outgoing face
                // 2. Set second bit of cell_to_local_face_offset_map[idx] to 1 to indicate local face
                // 3. Set the remaining 62 bits of cell_to_local_face_offset_map[idx] to the
                // appropriate index in the device-side local cell angular flux buffer

                cell_to_local_face_offset_map_gpu_[idx] =
                  (1ULL << 62) | (value & ((1ULL << 62) - 1));
              }
            }
          }
        }
        else // Outgoing non-local or boundary face
        {
          for (size_t fj = 0; fj < num_face_nodes; ++fj)
          {
            for (size_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
            {
              for (size_t gsg = 0; gsg < angle_set.GetNumGroups(); ++gsg)
              {
                const size_t idx = current_face_offset * face_data_size +
                                   (fj * angle_set.GetNumAngles() * angle_set.GetNumGroups()) +
                                   (as_ss_idx * angle_set.GetNumGroups()) + gsg;


                // 1. Set first bit of cell_to_local_face_offset_map[idx] to 0 to indicate outgoing face
                // 2. Set second bit of cell_to_local_face_offset_map[idx] to 0 to indicate non-local or boundary face
                // 3. Set the remaining 62 bits of cell_to_local_face_offset_map[idx] to the
                // appropriate index in the device-side non-local/boundary upwind psi buffer
                cell_to_local_face_offset_map_gpu_[idx] =
                  (idx & ((1ULL << 62) - 1));
              }
            }
          }
        }
      }
    }
    face_offset_stride += cell.faces.size();
  }
}

void
CBC_FLUDS::GetAndSetBoundaryPsiData(SweepChunk& sweep_chunk,
                                    AngleSet& angle_set)
{
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();
  const auto& angle_indices = cbc_angle_set.GetAngleIndices();

  auto& boundary_psi_buffer = reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->nonlocal_and_boundary_psi_buffer_.GetHostVector();

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
      {
        boundary_buffer_offset +=
          (num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups());
        continue;
      }

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
          {
            std::copy(psi_in,
                      psi_in + angle_set.GetNumGroups(),
                      &boundary_psi_buffer[boundary_buffer_offset +
                                           (fj * angle_set.GetNumAngles() + as_ss_idx) *
                                             angle_set.GetNumGroups()]);
            
            // NOTE: This is really inefficient, and ideally, I would just perform one H2D copy of
            // incoming boundary data after all data has been set
            // But for now, this will work
            crb::copy(reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
                        ->nonlocal_and_boundary_psi_buffer_.GetDeviceMemory(),
                      boundary_psi_buffer,
                      static_cast<size_t>(angle_set.GetNumGroups()),
                      static_cast<size_t>(boundary_buffer_offset +
                        (fj * angle_set.GetNumAngles() + as_ss_idx) * angle_set.GetNumGroups()),
                      static_cast<size_t>(boundary_buffer_offset +
                        (fj * angle_set.GetNumAngles() + as_ss_idx) * angle_set.GetNumGroups()));
          }
        }
      }
      boundary_buffer_offset +=
        (num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups());
    }
  }
}

void
CBC_FLUDS::GetNonlocalPsiData(SweepChunk& sweep_chunk,
                               AngleSet& angle_set,
                               std::vector<Task*> tasks)
{
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS());
  const auto& grid = cbc_spds.GetGrid();
  const auto& angle_indices = cbc_angle_set.GetAngleIndices();

  auto& nonlocal_psi_buffer =
    reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)->nonlocal_and_boundary_psi_buffer_.GetHostVector();

  size_t nonlocal_psi_buffer_offset = 0;
  for (size_t t = 0; t < tasks.size(); ++t)
  {
    const auto& task = tasks[t];
    const auto& cell = grid->local_cells[task->reference_id];
    const auto& face_orientations = cbc_spds.GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const FaceNodalMapping& face_nodal_mapping = GetCommonData().GetFaceNodalMapping(cell.local_id, f);

      if (face_orientations[f] != FaceOrientation::INCOMING or is_local_face or is_boundary_face)
      {
        nonlocal_psi_buffer_offset +=
          (num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups());
        continue;
      }

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
        {
          const double* psi_in = NLUpwindPsi(cell.global_id,
                                             f,
                                             face_nodal_mapping.face_node_mapping_[fj],
                                             as_ss_idx);

          if (psi_in)
          {
            std::copy(psi_in,
                      psi_in + angle_set.GetNumGroups(),
                      &nonlocal_psi_buffer[nonlocal_psi_buffer_offset +
                                           (fj * angle_set.GetNumAngles() + as_ss_idx) *
                                             angle_set.GetNumGroups()]);

            // NOTE: This is really inefficient, and ideally, I would just perform one H2D copy of
            // incoming nonlocal data after all data has been set
            // But for now, this will work
            // This approach also disallows for overlapping communication and computation
            // because of the synchronous copies - something to improve later
            crb::copy(reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_)
                        ->nonlocal_and_boundary_psi_buffer_.GetDeviceMemory(),
                      nonlocal_psi_buffer,
                      static_cast<size_t>(angle_set.GetNumGroups()),
                      static_cast<size_t>(nonlocal_psi_buffer_offset +
                        (fj * angle_set.GetNumAngles() + as_ss_idx) * angle_set.GetNumGroups()),
                      static_cast<size_t>(nonlocal_psi_buffer_offset +
                        (fj * angle_set.GetNumAngles() + as_ss_idx) * angle_set.GetNumGroups()));
          }
        }
      }
      nonlocal_psi_buffer_offset +=
        (num_face_nodes * angle_set.GetNumAngles() * angle_set.GetNumGroups());
    }
  }
}

// -----------------------------------------------------------------------------

void
CBC_FLUDS::Create_CBCD_FLUDS(AngleSet& angle_set)
{
  if (cbcd_fluds_ == nullptr)
  {
    Prepare_CBCD_FLUDS(angle_set);
    // Prepare_CBCD_FLUDS_V2(angle_set);
    CBCD_FLUDS* cbcd_fluds = new CBCD_FLUDS(*this);
    cbcd_fluds_ = cbcd_fluds;

    // This buffer is used to track changes in boundary psi data
    // If boundary data changes between sweeps, we need to
    // update the device boundary psi buffer
    incoming_boundary_psi_buffer_.resize(incoming_boundary_psi_buffer_size_, 0.0);
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
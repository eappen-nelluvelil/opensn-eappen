// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"
#include <cstddef>

namespace opensn
{

CBC_FLUDS::CBC_FLUDS(size_t num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     const UnknownManager& psi_uk_man,
                     const SpatialDiscretization& sdm,
                     size_t num_local_cells,
                     size_t max_cell_dof_count,
                     size_t min_num_pool_allocator_slots,
                     bool use_gpus)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(psi_uk_man),
    sdm_(sdm),
    num_local_cells_(num_local_cells),
    num_angles_in_gs_quadrature_(psi_uk_man_.GetNumberOfUnknowns()),
    num_quadrature_local_dofs_(sdm_.GetNumLocalDOFs(psi_uk_man_)),
    num_local_spatial_dofs_(num_quadrature_local_dofs_ / num_angles_in_gs_quadrature_ /
                            num_groups_),
    gpu_local_psi_data_size_(num_local_spatial_dofs_ * num_groups_and_angles_),
    use_gpus_(use_gpus),
    slot_size_(max_cell_dof_count * num_groups_and_angles_)
{
  if (use_gpus_)
  {
    local_psi_data_gpu_buffer_.resize(gpu_local_psi_data_size_);
  }
  else
  {
    cell_local_ID_to_psi_map_.resize(num_local_cells, nullptr);
    std::fill(cell_local_ID_to_psi_map_.begin(), cell_local_ID_to_psi_map_.end(), nullptr);
    local_psi_data_backing_buffer_.resize(min_num_pool_allocator_slots * slot_size_);
    local_psi_data_.add_block(local_psi_data_backing_buffer_.data(),
                            (min_num_pool_allocator_slots * slot_size_) * sizeof(double),
                            slot_size_ * sizeof(double));
  }
}

CBC_FLUDS::~CBC_FLUDS()
{
  if (use_gpus_)
    Destroy_CBCD_FLUDS();
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

void
CBC_FLUDS::Allocate(uint64_t cell_local_ID)
{
  assert(cell_local_ID_to_psi_map_[cell_local_ID] == nullptr);
  void* cell_block_ptr = local_psi_data_.malloc();
  cell_local_ID_to_psi_map_[cell_local_ID] = static_cast<double*>(cell_block_ptr);
}

void
CBC_FLUDS::Deallocate(uint64_t cell_local_ID)
{
  assert(cell_local_ID_to_psi_map_[cell_local_ID] != nullptr);
  local_psi_data_.free(cell_local_ID_to_psi_map_[cell_local_ID]);
  cell_local_ID_to_psi_map_[cell_local_ID] = nullptr;
}

double*
CBC_FLUDS::UpwindPsi(uint64_t cell_local_id, unsigned int adj_cell_node, size_t as_ss_idx)
{
  assert(cell_local_ID_to_psi_map_[cell_local_id] != nullptr);
  const size_t addr_offset = adj_cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return cell_local_ID_to_psi_map_[cell_local_id] + addr_offset;
}

double*
CBC_FLUDS::GPUUpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx)
{
  // Map to face neighbor cell's first spatial DOF index
  // (0 to (num_local_spatial_dofs_ - 1))
  const size_t face_nbr_spatial_dof_0_index =
    (sdm_.MapDOFLocal(face_neighbor, 0, psi_uk_man_, 0, 0) / num_angles_in_gs_quadrature_ /
     num_groups_);

  // Index to start of neighbor cell's data block in local_psi_data_
  const size_t face_nbr_data_start_index = face_nbr_spatial_dof_0_index * num_groups_and_angles_;
  const size_t addr_offset = adj_cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  const size_t face_nbr_data_index = face_nbr_data_start_index + addr_offset;

  assert((face_nbr_data_index >= 0) and (face_nbr_data_index < local_psi_data_gpu_buffer_.size()));

  return &local_psi_data_gpu_buffer_[face_nbr_data_index];
}

double*
CBC_FLUDS::OutgoingPsi(uint64_t cell_local_ID, unsigned int cell_node, size_t as_ss_idx)
{
  assert(cell_local_ID_to_psi_map_[cell_local_ID] != nullptr);
  const size_t addr_offset = cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return cell_local_ID_to_psi_map_[cell_local_ID] + addr_offset;
}

double*
CBC_FLUDS::GPUOutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx)
{
  // Map to current cell's first spatial DOF index
  // (0 to (num_local_spatial_dofs_ - 1))
  const size_t cur_cell_spatial_dof_0_index =
    (sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0) / num_angles_in_gs_quadrature_ / num_groups_);

  // Index to start of current cell's data block in local_psi_data_
  const size_t cur_cell_data_start_index = cur_cell_spatial_dof_0_index * num_groups_and_angles_;
  const size_t addr_offset = cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  const size_t cur_cell_data_index = cur_cell_data_start_index + addr_offset;

  assert((cur_cell_data_index >= 0) and (cur_cell_data_index < local_psi_data_gpu_buffer_.size()));

  return &local_psi_data_gpu_buffer_[cur_cell_data_index];
}

double*
CBC_FLUDS::NLUpwindPsi(uint64_t cell_global_id,
                       unsigned int face_id,
                       unsigned int face_node_mapped,
                       size_t as_ss_idx)
{
  // std::vector<double>& psi = deplocs_outgoing_messages_.at({cell_global_id, face_id});
  std::vector<double>& psi = deplocs_outgoing_messages_[{cell_global_id, face_id}];
  const size_t dof_map =
    face_node_mapped * num_groups_and_angles_ + //  Offset to start of data for face_node_mapped
    as_ss_idx * num_groups_;                    // Offset to start of data for angle_set_index
  assert((dof_map >= 0) and (dof_map < psi.size()));
  return &psi[dof_map];
}

double*
CBC_FLUDS::NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing,
                         size_t face_node,
                         size_t as_ss_idx)
{
  assert(psi_nonlocal_outgoing != nullptr);
  const size_t addr_offset = face_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return &(*psi_nonlocal_outgoing)[addr_offset];
}

void
CBC_FLUDS::BuildDeviceCellDOFMap()
{
  const auto& grid = sdm_.GetGrid();
  cell_dof_map_.resize(grid->local_cells.size());

  for (const auto& cell : grid->local_cells)
  {
    const size_t dof_map_idx = cell.local_id;
    const size_t spatial_dof_0_idx = 
      (sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0) / 
        num_angles_in_gs_quadrature_ / num_groups_);
    cell_dof_map_[dof_map_idx] = spatial_dof_0_idx * num_groups_and_angles_;
  }
}

#ifndef __OPENSN_USE_CUDA__
void
CBC_FLUDS::Create_CBCD_FLUDS(size_t num_total_faces,
                             size_t incoming_boundary_psi_buffer_size,
                             const std::vector<int>& cell_to_local_face_offset_map,
                             const std::vector<int>& boundary_psi_map)
{
}

void
CBC_FLUDS::SetBoundaryPsiData(const std::vector<double>& boundary_psi)
{
}

void
CBC_FLUDS::Destroy_CBCD_FLUDS()
{
}
#endif // __OPENSN_USE_CUDA__

} // namespace opensn
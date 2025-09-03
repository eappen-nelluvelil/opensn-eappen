// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"
#include "caliper/cali.h"
#include <memory_resource>

namespace opensn
{

CBC_FLUDS::CBC_FLUDS(size_t num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     const UnknownManager& psi_uk_man,
                     const SpatialDiscretization& sdm,
                     const size_t peak_number_alive_cells,
                     const size_t max_cell_dof_count)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(psi_uk_man),
    sdm_(sdm),
    max_cell_dof_count_(max_cell_dof_count),
    peak_number_alive_cells_(static_cast<size_t>((sdm.GetNumLocalDOFs(psi_uk_man) / psi_uk_man.GetNumberOfUnknowns() / num_groups)/max_cell_dof_count_)),
    // peak_number_alive_cells_(31), // for testing
    // peak_number_alive_cells_(47),   // hard-coded magic value that works for transport_2d_1_poly.py
    // peak_number_alive_cells_(17),       // hard-coded magic value that works for transport_1d_1.py
    // peak_number_alive_cells_(32),     // hard-coded magic value that works for hdpe_balance.py
    memory_buffer_(peak_number_alive_cells_ * max_cell_dof_count * num_angles * num_groups *
                   sizeof(double)),
    memory_resource_(
      memory_buffer_.data(), memory_buffer_.size(), std::pmr::null_memory_resource()),
    pool_options_(peak_number_alive_cells_,
                  max_cell_dof_count * num_angles * num_groups * sizeof(double)),
    // pool_resource_(pool_options_, &memory_resource_)
    pool_resource_(&memory_resource_)
{
  CALI_CXX_MARK_SCOPE("CBC_FLUDS::CBC_FLUDS");

  opensn::log.Log() << "CBC_FLUDS::CBC_FLUDS: Max cell DOF count = " << max_cell_dof_count << ", Peak number of alive cells = "
                    << peak_number_alive_cells_ << " (NOTE: Right now, this is a hard-coded magic number for debugging purposes - I'm trying to figure out why this number doesn't match the peak alive cell count that the CBC_FLUDS reports)";

  num_angles_in_gs_quadrature_ = psi_uk_man_.GetNumberOfUnknowns();
  num_quadrature_local_dofs_ = sdm_.GetNumLocalDOFs(psi_uk_man_);
  num_local_spatial_dofs_ = num_quadrature_local_dofs_ / num_angles_in_gs_quadrature_ / num_groups_;
  local_psi_data_size_ = num_local_spatial_dofs_ * num_groups_and_angles_;

  local_psi_data_.resize(local_psi_data_size_);

  opensn::log.Log() << "CBC_FLUDS::CBC_FLUDS: local psi data size = " << local_psi_data_.size();
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

const double*
CBC_FLUDS::GetLocalUpwindPsi(const Cell& face_neighbor) const
{
  // Map to face neighbor cell's first spatial DOF index
  // (0 to (num_local_spatial_dofs_ - 1))
  const size_t face_nbr_spatial_dof_0_index =
    (sdm_.MapDOFLocal(face_neighbor, 0, psi_uk_man_, 0, 0) / num_angles_in_gs_quadrature_ /
     num_groups_);

  // Index to start of neighbor cell's data block in local_psi_data_
  const size_t face_nbr_data_start_index = face_nbr_spatial_dof_0_index * num_groups_and_angles_;

  if ((face_nbr_data_start_index < 0) or (face_nbr_data_start_index >= local_psi_data_.size()))
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetLocalUpwindPsi: Invalid index " << face_nbr_data_start_index
               << " (max allowed = " << local_psi_data_.size() << ")";
    throw std::runtime_error(err_stream.str());
  }

  return &local_psi_data_[face_nbr_data_start_index];
}

double*
CBC_FLUDS::GetLocalDownwindPsi(const Cell& cell)
{
  // Map to current cell's first spatial DOF index
  // (0 to (num_local_spatial_dofs_ - 1))
  const size_t cur_cell_spatial_dof_0_index =
    (sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0) / num_angles_in_gs_quadrature_ / num_groups_);

  // Index to start of current cell's data block in local_psi_data_
  const size_t cur_cell_data_start_index = cur_cell_spatial_dof_0_index * num_groups_and_angles_;

  if ((cur_cell_data_start_index < 0) or (cur_cell_data_start_index >= local_psi_data_.size()))
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetLocalDownwindPsi: Invalid index " << cur_cell_data_start_index
               << " (max allowed = " << local_psi_data_.size() << ")";
    throw std::runtime_error(err_stream.str());
  }

  return &local_psi_data_[cur_cell_data_start_index];
}

const std::vector<double>&
CBC_FLUDS::GetNonLocalUpwindData(uint64_t cell_global_id, unsigned int face_id) const
{
  return deplocs_outgoing_messages_.at({cell_global_id, face_id});
}

const double*
CBC_FLUDS::GetNonLocalUpwindPsi(const std::vector<double>& psi_data,
                                unsigned int face_node_mapped,
                                unsigned int angle_set_index)
{
  const size_t dof_map =
    face_node_mapped * num_groups_and_angles_ + //  Offset to start of data for face_node_mapped
    angle_set_index * num_groups_;              // Offset to start of data for angle_set_index

  if ((dof_map < 0) or (dof_map > psi_data.size()))
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetNonLocalUpwindPsi: Invalid index " << dof_map
               << " (max allowed = " << psi_data.size() << ")";
    throw std::runtime_error(err_stream.str());
  }

  return &psi_data[dof_map];
}

double*
CBC_FLUDS::Allocate(const uint64_t cell_local_id)
{
  CALI_CXX_MARK_SCOPE("CBC_FLUDS::Allocate");

  if (cell_to_chunk_map_.count(cell_local_id))
  {
    // std::ostringstream err_stream;
    // err_stream << "CBC_FLUDS::Allocate: Cell with local ID " << cell_local_id
    //            << " already has an allocated chunk.";
    // throw std::runtime_error(err_stream.str());
    return cell_to_chunk_map_[cell_local_id];
  }

  double* chunk = static_cast<double*>(pool_resource_.allocate(num_groups_and_angles_ *
                                                              max_cell_dof_count_ * sizeof(double),
                                                              alignof(double)));
  cell_to_chunk_map_[cell_local_id] = chunk;

  ++num_allocations_;
  ++num_current_allocations_;
  num_peak_allocations_ = std::max(num_peak_allocations_, num_current_allocations_);

  return chunk;
}

void
CBC_FLUDS::Deallocate(const uint64_t cell_local_id)
{
  CALI_CXX_MARK_SCOPE("CBC_FLUDS::Deallocate");

  auto it = cell_to_chunk_map_.find(cell_local_id);
  if (it == cell_to_chunk_map_.end())
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::Deallocate: Cell with local ID " << cell_local_id
               << " does not have an allocated chunk.";
    throw std::runtime_error(err_stream.str());
  }

  double* chunk = it->second;
  pool_resource_.deallocate(chunk, 
                            num_groups_and_angles_ * max_cell_dof_count_ * sizeof(double),
                            alignof(double));
  cell_to_chunk_map_.erase(it);

  ++num_deallocations_;
  --num_current_allocations_;
}

const double*
CBC_FLUDS::GetChunk(const uint64_t cell_local_id) const
{
  auto it = cell_to_chunk_map_.find(cell_local_id);
  if (it == cell_to_chunk_map_.end())
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetChunk: Cell with local ID " << cell_local_id
               << " does not have an allocated chunk.";
    throw std::runtime_error(err_stream.str());
  }

  return it->second;
}

double*
CBC_FLUDS::GetChunk(const uint64_t cell_local_id)
{
  auto it = cell_to_chunk_map_.find(cell_local_id);
  if (it == cell_to_chunk_map_.end())
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetChunk: Cell with local ID " << cell_local_id
               << " does not have an allocated chunk.";
    throw std::runtime_error(err_stream.str());
  }

  return it->second;
}

} // namespace opensn

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

// size_t CBC_FLUDS::true_buffer_size_in_bytes_ = 0;

CBC_FLUDS::CBC_FLUDS(size_t num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     const UnknownManager& psi_uk_man,
                     const SpatialDiscretization& sdm,
                     const size_t num_local_cells,
                     const size_t peak_number_alive_cells,
                     const size_t max_cell_dof_count)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(psi_uk_man),
    sdm_(sdm),
    num_blocks_(peak_number_alive_cells),
    block_size_(max_cell_dof_count * num_groups_and_angles_),
    backing_buffer_(num_blocks_ * block_size_)
{
  storage_.add_block(backing_buffer_.data(), // Pointer to starting address for buffer
                    (num_blocks_ * block_size_) * sizeof(double), // Total size of buffer in bytes
                    block_size_ * sizeof(double));  // Size of each block in buffer in bytes

  cell_local_ID_to_ptr_map_.resize(num_local_cells, nullptr);
  
  opensn::log.Log() << "CBC_FLUDS: Allocated for " << num_blocks_ << "blocks , block size = "
                    << block_size_ << " doubles,"
                    << " buffer size: " << backing_buffer_.size() << " doubles"
                    << ", # of local cells = " << num_local_cells;
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
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

  return &psi_data[dof_map];
}

void
CBC_FLUDS::Allocate(const uint64_t cell_local_ID)
{
  if (cell_local_ID_to_ptr_map_[cell_local_ID] != nullptr)
    return;

  void* cell_block_ptr = storage_.malloc();
  cell_local_ID_to_ptr_map_[cell_local_ID] = static_cast<double*>(cell_block_ptr);

  ++num_allocations_;
  ++num_current_allocations_;
  num_peak_allocations_ = std::max(num_peak_allocations_, num_current_allocations_);
}

void
CBC_FLUDS::Deallocate(const uint64_t cell_local_ID)
{
  if (cell_local_ID_to_ptr_map_[cell_local_ID] == nullptr)
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::Deallocate: Cell with local ID " << cell_local_ID
               << " does not have an allocated chunk.";
    throw std::runtime_error(err_stream.str());
  }

  storage_.free(cell_local_ID_to_ptr_map_[cell_local_ID]);
  cell_local_ID_to_ptr_map_[cell_local_ID] = nullptr;

  ++num_deallocations_;
  --num_current_allocations_;
}

double* 
CBC_FLUDS::GetCellBlock(const uint64_t cell_local_ID)
{
  if (cell_local_ID_to_ptr_map_[cell_local_ID] == nullptr)
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetChunk: Cell with local ID " << cell_local_ID
               << " does not have an allocated chunk.";
    throw std::runtime_error(err_stream.str());
  }

  return cell_local_ID_to_ptr_map_[cell_local_ID];
}

const double* 
CBC_FLUDS::GetCellBlock(const uint64_t cell_local_ID) const
{
  if (cell_local_ID_to_ptr_map_[cell_local_ID] == nullptr)
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetChunk: Cell with local ID " << cell_local_ID
               << " does not have an allocated chunk.";
    throw std::runtime_error(err_stream.str());
  }

  return cell_local_ID_to_ptr_map_[cell_local_ID];
}

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/math/unknown_manager/unknown_manager.h"

namespace opensn
{

CBC_FLUDS::CBC_FLUDS(size_t num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     const SpatialDiscretization& sdm,
                     size_t max_wavefront_size,
                     size_t max_num_cell_dofs)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    sdm_(sdm),
    memory_pool_(),
    psi_allocator_(&memory_pool_) // Point the allocator to the memory pool
{
  // ---------------------------------------------------------------------------
  // Phase 2: UPR-specific code modifications
  // ---------------------------------------------------------------------------
  log.Log() << "[UPR] CBC_FLUDS: Max number of spatial DOFs per cell: " << max_num_cell_dofs;

  single_cell_block_size_ = max_num_cell_dofs * this->num_angles_ * this->num_groups_;

  // Allocate memory
  if (max_wavefront_size > 0 and single_cell_block_size_ > 0)
  {
    void* initial_chunk = memory_pool_.allocate(max_wavefront_size * single_cell_block_size_);
    memory_pool_.deallocate(initial_chunk, max_wavefront_size * single_cell_block_size_);
    log.Log() << "[UPR] CBC_FLUDS: Memory pool initialized for maximum wavefront size of "
              << max_wavefront_size << " cells.";
  }
  // ---------------------------------------------------------------------------
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
  // Stride to jump from one face node's data block to the next within `psi_data`.
  // Each face node block contains data for all angles in this AngleSet and all groups.
  const size_t num_psi_per_face_node_for_set = this->num_angles_ * this->num_groups_;

  // Stride to jump from one angle's data block to the next (for the same face node) within
  // `psi_data`. Each angle block contains data for all groups.
  const size_t num_groups_stride = this->num_groups_;

  // Calculate the 1D offset into psi_data.
  const size_t dof_map =
    face_node_mapped *
      num_psi_per_face_node_for_set +    // Offset to the start of data for this face_node_mapped
    angle_set_index * num_groups_stride; // Further offset to the start of data for this specific
                                         // angle_set_index (for group 0)

  if (dof_map + num_groups_stride > psi_data.size() && num_groups_stride > 0)
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetNonLocalUpwindPsi: Offset out of bounds. "
               << "Calculated_dof_map=" << dof_map << ", num_groups_stride=" << num_groups_stride
               << ", psi_data.size()=" << psi_data.size()
               << ", face_node_mapped=" << face_node_mapped
               << ", angle_set_index=" << angle_set_index
               << ", this->num_angles_=" << this->num_angles_
               << ", this->num_groups_=" << this->num_groups_;
    throw std::runtime_error(err_stream.str());
  }
  if (psi_data.empty() && (face_node_mapped > 0 || angle_set_index > 0))
  {
    throw std::runtime_error(
      "CBC_FLUDS::GetNonLocalUpwindPsi: Accessing non-empty psi_data with non-zero indices.");
  }

  return &psi_data[dof_map];
}

double*
CBC_FLUDS::AllocateForCell(uint64_t cell_local_id)
{
  if (single_cell_block_size_ == 0)
    return nullptr;

  // Polymorphic allocator handles getting a block from the pool
  double* block_ptr = psi_allocator_.allocate(single_cell_block_size_);
  cell_memory_map_[cell_local_id] = block_ptr;
  return block_ptr;
}

void
CBC_FLUDS::DeallocateForCell(uint64_t cell_local_id)
{
  if (single_cell_block_size_ == 0)
    return;

  auto it = cell_memory_map_.find(cell_local_id);
  if (it == cell_memory_map_.end())
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::DeallocateForCell: Attempted to deallocate memory for cell "
               << cell_local_id << " that has no allocated block.";
    throw std::runtime_error(err_stream.str());
  }

  // Return the block to the pool via the allocator
  psi_allocator_.deallocate(it->second, single_cell_block_size_);
  cell_memory_map_.erase(it);
}

const double*
CBC_FLUDS::GetPsiForCell(uint64_t cell_local_id) const
{
  auto it = cell_memory_map_.find(cell_local_id);
  if (it == cell_memory_map_.end())
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetPsiForCell: Attempted to get psi for cell " << cell_local_id
               << " that has no allocated block.";
    throw std::runtime_error(err_stream.str());
  }

  return it->second;
}

} // namespace opensn

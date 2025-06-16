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
                     const UnknownManager& psi_uk_man,
                     const SpatialDiscretization& sdm,
                     size_t max_wavefront_size)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(psi_uk_man),
    sdm_(sdm),
    memory_pool_(),
    psi_allocator_(&memory_pool_) // Point the allocator to the memory pool
{
  // --- Calculate required size for the optimized `local_psi_data_` ---

  // 1. Get number of purely spatial DOFs on this MPI rank.
  //    This is achieved by using a temporary UnknownManager representing a single
  //    scalar unknown per spatial node.
  const UnknownManager temp_unitary_uk_man({std::make_pair(UnknownType::SCALAR, 0)},
                                           UnknownStorageType::NODAL);
  const size_t num_local_spatial_dofs = sdm_.GetNumLocalDOFs(temp_unitary_uk_man);

  // 2. Calculate the required size for the `local_psi_data_` vector.
  //    Layout: spatial DOF major -> angle in set major -> group major
  //    @note `this->num_angles_` = Number of angles specific to this AngleSet
  //    @note `this->num_groups_` = Number of groups specific to this AngleSet
  size_t local_psi_data_size = num_local_spatial_dofs * this->num_angles_ * this->num_groups_;

  // 3. Allocate the local_psi_data_ vector.
  local_psi_data_.assign(local_psi_data_size, 0.0);

  // --- Verification logging ---
  // Calculate what the old size would have been for comparison,
  // when `local_psi_data_` was sized to account for the number of angles
  // in the groupset's qaudrature

  // Number of angles in the parent LBSGroupset's full quadrature
  const size_t num_angles_in_gs_quadrature = psi_uk_man_.GetNumberOfUnknowns();

  // Calculate estimated old size in number of doubles
  size_t size_before_doubles_calc =
    num_local_spatial_dofs * num_angles_in_gs_quadrature * this->num_groups_;

  // For direct comparison, get size using the LBSGroupset's UnknownManager
  size_t size_before_doubles_direct = sdm.GetNumLocalDOFs(psi_uk_man);

  if (local_psi_data_.size() != local_psi_data_size) // Sanity check allocation
  {
    log.Log0Warning() << "CBC_FLUDS Warning: Allocated local_psi_data_ size ("
                      << local_psi_data_.size() << ") does not match calculated size ("
                      << local_psi_data_size << ").";
  }

  const double mb_divisor = 1024.0 * 1024.0; // For conversion to MBs
  auto size_before_mb = static_cast<double>(size_before_doubles_calc * sizeof(double)) / mb_divisor;
  auto size_before_direct_mb =
    static_cast<double>(size_before_doubles_direct * sizeof(double)) / mb_divisor;

  auto size_after_mb = static_cast<double>(local_psi_data_size * sizeof(double)) / mb_divisor;

  log.Log() << "CBC_FLUDS Size Comparison for AngleSet (Angles in Set: " << this->num_angles_
            << ", Angles in Groupset Quadrature: " << num_angles_in_gs_quadrature
            << ", Groups: " << this->num_groups_ << "):";
  log.Log() << "  Original estimated size: " << size_before_doubles_calc << " doubles ("
            << std::fixed << std::setprecision(3) << size_before_mb << " MB)";
  log.Log() << "  Original direct size (for verification): " << size_before_doubles_direct
            << " doubles (" << std::fixed << std::setprecision(3) << size_before_direct_mb
            << " MB)";
  log.Log() << "  Optimized size (current):  " << local_psi_data_size << " doubles (" << std::fixed
            << std::setprecision(3) << size_after_mb << " MB)";

  // Avoid division by zero if original size was effectively 0
  if (size_before_mb > 1e-6)
  {
    auto reduction = (size_before_mb - size_after_mb) / size_before_mb * 100.0;
    log.Log() << "  Memory Reduction for local_psi_data_: " << std::fixed << std::setprecision(1)
              << reduction << "%";
  }

  // ---------------------------------------------------------------------------
  // Phase 2: UPR-specific code modifications
  // ---------------------------------------------------------------------------
  single_cell_block_size_ = num_local_spatial_dofs * this->num_angles_ * this->num_groups_;

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
  if (single_cell_block_size_ == 0) return nullptr;

  // Polymorphic allocator handles getting a block from the pool
  double* block_ptr = psi_allocator_.allocate(single_cell_block_size_);
  cell_memory_map_[cell_local_id] = block_ptr;
  return block_ptr;
}

void
CBC_FLUDS::DeallocateForCell(uint64_t cell_local_id)
{
  if (single_cell_block_size_ == 0) return;

  auto it = cell_memory_map_.find(cell_local_id);
  if (it == cell_memory_map_.end())
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::DeallocateForCell: Attempted to deallocate memory for cell "
               << cell_local_id
               << " that has no allocated block.";
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
    err_stream << "CBC_FLUDS::GetPsiForCell: Attempted to get psi for cell "
               << cell_local_id
               << " that has no allocated block.";   
    throw std::runtime_error(err_stream.str());
  }

  return it->second;
}

} // namespace opensn

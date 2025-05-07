// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/fluds/cbc_fluds.h"
#include "cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include <stdexcept>

#include "framework/logging/log.h"

namespace opensn
{

CBC_FLUDS::CBC_FLUDS(size_t num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     const UnknownManager& psi_uk_man,
                     const SpatialDiscretization& sdm)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(psi_uk_man),
    sdm_(sdm),
    num_angles_in_set_(num_angles)
{
  // Need to be sized to hold angles in an angleset for a groupset, instead of all angles in the
  // groupset i.e., num angles in angle set * num groups in groupset for that angle set For the time
  // being, use the full vector Each groupset is associated with a quadrature set
  // size_t num_ang_unknowns = sdm.GetNumLocalDOFs(psi_uk_man);
  // local_psi_data_.assign(num_ang_unknowns, 0.0);

  uint64_t total_nodes_across_local_cells = sdm_.GetNumLocalNodes(); // Uses virtual call

  size_t compact_dofs_per_node = num_angles_in_set_ * num_groups;
  size_t total_compact_psi_dofs = total_nodes_across_local_cells * compact_dofs_per_node;
  if (total_compact_psi_dofs > 0) {
      local_psi_data_.assign(total_compact_psi_dofs, 0.0);
  }
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

// NEW METHOD to avoid having to access psi_new_local_[groupset.id]
// const double*
// CBC_FLUDS::GetLocalUpwindPsi(const Cell& face_neighbor,
//                              const unsigned int adj_cell_node_offset) const
// {
//   // Starting index for upwind cell's angular flux data
//   const auto dof_map = sdm_.MapDOFLocal(face_neighbor, 0, psi_uk_man_, 0, 0);

//   const auto local_face_upwind_psi = &local_psi_data_[dof_map];
//   return &local_face_upwind_psi[adj_cell_node_offset];
// }

// NEW METHOD to set angular value data for a downwind face of the cell
// double*
// CBC_FLUDS::GetLocalDownwindPsi(const Cell& cell)
// {
//   const auto dof_map = sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0);
//   return &local_psi_data_[dof_map];
// }

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
  const size_t dof_map = face_node_mapped * num_groups_and_angles_ + angle_set_index * num_groups_;
  return &psi_data[dof_map];
}

/*
size_t
CBC_FLUDS::MapDOFCompactLocal(const Cell& cell,
                              unsigned int node_in_cell,
                              unsigned int angle_idx_as,
                              unsigned int group_idx_gs) const
{
  int64_t absolute_global_node_idx =
    sdm_.MapDOFLocal(cell, node_in_cell, UnknownManager::GetUnitaryUnknownManager(), 0, 0);

  if (absolute_global_node_idx < 0)
  {
    throw std::runtime_error("Failed to map compact DOF: Invalid absolute_global_node_idx.");
  }

  size_t stride_angle_within_node = num_groups_;
  size_t stride_node = num_angles_in_set_ * stride_angle_within_node;

  size_t address = static_cast<size_t>(absolute_global_node_idx) * stride_node + // Cast to size_t
                   angle_idx_as * stride_angle_within_node + group_idx_gs;

  if (!local_psi_data_.empty() && address >= local_psi_data_.size()) {
    // Log all input parameters and intermediate values here
    log.Log() << "MapDOFCompactLocal OOB: addr=" + std::to_string(address) +
                          " size=" + std::to_string(local_psi_data_.size()) +
                          " cell_id=" + std::to_string(cell.local_id) + // or global_id
                          " node_in_cell=" + std::to_string(node_in_cell) +
                          " angle_idx_as=" + std::to_string(angle_idx_as) +
                          " group_idx_gs=" + std::to_string(group_idx_gs) +
                          " abs_loc_node_idx=" + std::to_string(absolute_global_node_idx);
    // throw std::out_of_range(err_msg);
  }

  return address;
}
*/

// *** Enhance MapDOFCompactLocal ***
size_t
CBC_FLUDS::MapDOFCompactLocal(const Cell& cell,
                              unsigned int node_in_cell,
                              unsigned int angle_idx_as,
                              unsigned int group_idx_gs) const
{
  // Log entry and inputs
  log.Log0Verbose1() << "FLUDS_MAP_DEBUG: MapDOFCompactLocal called for Cell " << cell.global_id
                     << " Node " << node_in_cell << " AngleAS " << angle_idx_as
                     << " GroupGS " << group_idx_gs;

  int64_t absolute_local_node_idx_64 =
    sdm_.MapDOFLocal(cell, node_in_cell, UnknownManager::GetUnitaryUnknownManager(), 0, 0);

  // Log intermediate calculation
  log.Log0Verbose1() << "FLUDS_MAP_DEBUG:   AbsoluteLocalNodeIdx=" << absolute_local_node_idx_64;


  if (absolute_local_node_idx_64 < 0)
  {
    throw std::runtime_error("Failed to map compact DOF: Invalid absolute_local_node_idx.");
  }
  size_t absolute_local_node_idx = static_cast<size_t>(absolute_local_node_idx_64);

  size_t stride_angle_within_node = num_groups_; // num_groups_in_groupset
  size_t stride_node = num_angles_in_set_ * stride_angle_within_node;

  // Log strides
  log.Log0Verbose1() << "FLUDS_MAP_DEBUG:   NumGroups=" << num_groups_
                     << " NumAnglesAS=" << num_angles_in_set_
                     << " StrideAngleWithinNode=" << stride_angle_within_node
                     << " StrideNode=" << stride_node;

  size_t address = absolute_local_node_idx * stride_node +
                   angle_idx_as * stride_angle_within_node +
                   group_idx_gs;

  // Log final address and vector size
  log.Log0Verbose1() << "FLUDS_MAP_DEBUG:   Calculated Address=" << address
                     << " (Vector Size=" << local_psi_data_.size() << ")";


  // BOUNDS CHECK (already present, good)
  if (!local_psi_data_.empty() && address >= local_psi_data_.size())
  {
     std::string err_msg = "MapDOFCompactLocal OOB: addr=" + std::to_string(address) +
                          " size=" + std::to_string(local_psi_data_.size()) +
                          " cell_gid=" + std::to_string(cell.global_id) +
                          " node_in_cell=" + std::to_string(node_in_cell) +
                          " angle_idx_as=" + std::to_string(angle_idx_as) +
                          " group_idx_gs=" + std::to_string(group_idx_gs) +
                          " abs_loc_node_idx=" + std::to_string(absolute_local_node_idx);
     throw std::out_of_range(err_msg);
  }
  return address;
}

// const double*
// CBC_FLUDS::GetLocalUpwindPsi_Compact(const Cell& upwind_cell,
//                                      unsigned int upwind_node_in_cell,
//                                      unsigned int angle_idx_as) const
// {
//   if (local_psi_data_.empty())
//     return nullptr;
//   size_t index =
//     MapDOFCompactLocal(upwind_cell, upwind_node_in_cell, angle_idx_as, 0 /* group 0 */);
//   return &local_psi_data_[index];
// }

// *** Enhance GetLocalUpwindPsi_Compact ***
const double*
CBC_FLUDS::GetLocalUpwindPsi_Compact(const Cell& upwind_cell,
                                     unsigned int upwind_node_in_cell,
                                     unsigned int angle_idx_as) const
{
  log.Log0Verbose1() << "FLUDS_GETUP_DEBUG: GetLocalUpwindPsi_Compact called for UpwindCell " << upwind_cell.global_id
                     << " UpwindNode " << upwind_node_in_cell << " AngleAS " << angle_idx_as;

  if (local_psi_data_.empty())
  {
    log.Log0Verbose1() << "FLUDS_GETUP_DEBUG:   local_psi_data_ is empty, returning nullptr.";
    return nullptr;
  }
  size_t index =
    MapDOFCompactLocal(upwind_cell, upwind_node_in_cell, angle_idx_as, 0 /* group 0 */);

  log.Log0Verbose1() << "FLUDS_GETUP_DEBUG:   Reading from index " << index << ". Value[0]="
                     << std::scientific << std::setprecision(6) << local_psi_data_[index]; // Log value before returning ptr

  return &local_psi_data_[index]; // Pointer to group 0 data
}

// double* CBC_FLUDS::GetLocalDownwindPsi_Compact(const Cell& current_cell,
//                                                unsigned int current_node_in_cell,
//                                                unsigned int angle_idx_as)
// {
//   if (local_psi_data_.empty())
//     return nullptr;
//   size_t index =
//     MapDOFCompactLocal(current_cell, current_node_in_cell, angle_idx_as, 0 /* group 0 */);
//   return &local_psi_data_[index];
// }

// *** Enhance GetLocalDownwindPsi_Compact ***
double*
CBC_FLUDS::GetLocalDownwindPsi_Compact(const Cell& current_cell,
                                      unsigned int current_node_in_cell,
                                      unsigned int angle_idx_as)
{
  log.Log0Verbose1() << "FLUDS_GETDN_DEBUG: GetLocalDownwindPsi_Compact called for CurrentCell " << current_cell.global_id
  << " CurrentNode " << current_node_in_cell << " AngleAS " << angle_idx_as;

  if (local_psi_data_.empty())
  {
    log.Log0Verbose1() << "FLUDS_GETDN_DEBUG:   local_psi_data_ is empty, returning nullptr.";
    return nullptr;
  }
  size_t index =
  MapDOFCompactLocal(current_cell, current_node_in_cell, angle_idx_as, 0 /* group 0 */);

  log.Log0Verbose1() << "FLUDS_GETDN_DEBUG:   Returning pointer to index " << index;

  return &local_psi_data_[index]; // Pointer to group 0 data
}



} // namespace opensn

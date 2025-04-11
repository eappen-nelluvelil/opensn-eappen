// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>

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
    // Initialize optimal stride members to 0
    optimal_num_angles_(num_angles),
    optimal_num_groups_(num_groups),
    optimal_angle_stride_(0),
    optimal_node_stride_(0)
{
  /* OLD:
  // Need to be sized to hold angles in a angleset in a groupset, instead of all angles in all anglesets in the groupset
  // i.e., num angles in angle set * num groups in groupset for that angle set
  // For the time being, use the full vector

  // Each groupset is associated with a quadrature set 
  size_t num_ang_unknowns = sdm.GetNumLocalDOFs(psi_uk_man); 
  local_psi_data_.assign(num_ang_unknowns, 0.0);
  */

  // Calculate optimal strides
  optimal_angle_stride_ = optimal_num_groups_;
  optimal_node_stride_ = optimal_num_angles_ * optimal_angle_stride_;

  // Calculate optimal vector size
  // Get total number of local nodes (sum over all local cells)
  size_t num_local_nodes = sdm.GetNumLocalNodes();
  
  // Calculate size needed for these nodes with optimal components per node
  size_t optimal_size = num_local_nodes * optimal_node_stride_; // N_nodes * N_angles * N_groups

  // --- Resize the vector ---
  try
  {
    local_psi_data_.resize(optimal_size);
    std::fill(local_psi_data_.begin(), local_psi_data_.end(), 0.0);
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error(std::string("CBC_FLUDS constructor: failed to resize local_psi_data_: ") + e.what());
  }
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

// OLD METHOD: deprecate at some point
// const std::vector<double>&
// CBC_FLUDS::GetLocalUpwindDataBlock() const
// {
//   return local_psi_data_;
// }

// OLD METHOD: deprecate at some point
// const double*
// CBC_FLUDS::GetLocalCellUpwindPsi(const std::vector<double>& psi_data_block, const Cell& cell)
// {
//   const auto dof_map = sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0);
//   return &psi_data_block[dof_map];
// }

// NEW METHOD to avoid having to access psi_new_local_[groupset.id]
const double*
CBC_FLUDS::GetLocalUpwindPsi(const Cell& face_neighbor, 
                              const unsigned int adj_cell_node_offset)
                              const 
{
  // Starting index for upwind cell's angular flux data 
  const auto dof_map = sdm_.MapDOFLocal(face_neighbor, 0, psi_uk_man_, 0, 0);
  
  const auto local_face_upwind_psi = &local_psi_data_[dof_map];
  return &local_face_upwind_psi[adj_cell_node_offset];
}

// NEW METHOD to set angular value data for a downwind face of the cell
double*
CBC_FLUDS::GetLocalDownwindPsi(const Cell& cell)                
{
  const auto dof_map = sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0);
  return &local_psi_data_[dof_map];
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
  const size_t dof_map = face_node_mapped * num_groups_and_angles_ + angle_set_index * num_groups_;
  return &psi_data[dof_map];
}

size_t CBC_FLUDS::MapDOFOptimalLocal(const Cell& cell,
                                     size_t node_idx, 
                                     size_t angle_set_idx,
                                     size_t group_set_idx) const
{
  // 1. Get the base "flat" node index for this cell's node 
  // This index ranges from 0 to (total number of local nodes - 1)
  int64_t flat_node_index = sdm_.MapDOFLocal(cell, node_idx);

  if (flat_node_index < 0)
  {
    throw std::runtime_error("MapDOFOptimalLocal: Call to sdm_.MapDOFLocal failed");
  }

  // 2. Calculate the index using optimal strides (assuming nodal layout)
  // Index = BaseDOFForNode + OffsetWithinNode
  // BaseDOFForNode = angle_set_idx * DOFSPerAngle + group_set_idx
  size_t index = static_cast<size_t>(flat_node_index) * optimal_node_stride_ +
                 angle_set_idx * optimal_angle_stride_ + 
                 group_set_idx;

  // 3. Bounds checking
  if (index >= local_psi_data_.size())
  {
    throw std::out_of_range("MapDOFOptimalLocal: calculated index " + std::to_string(index) + 
                            " is out of bounds for vector size " + std::to_string(local_psi_data_.size()));
  }

  return index;
}

} // namespace opensn

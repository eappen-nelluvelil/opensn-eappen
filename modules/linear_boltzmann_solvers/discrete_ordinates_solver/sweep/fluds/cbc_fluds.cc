// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/math/unknown_manager/unknown_manager.h" // For UNITARY_UNKNOWN_MANAGER

namespace opensn
{

/*
CBC_FLUDS::CBC_FLUDS(size_t num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     const UnknownManager& psi_uk_man,
                     const SpatialDiscretization& sdm)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(psi_uk_man),
    sdm_(sdm)
{
  // Need to be sized to hold angles in an angleset for a groupset, instead of all angles in the
  // groupset i.e., num angles in angle set * num groups in groupset for that angle set For the time
  // being, use the full vector Each groupset is associated with a quadrature set
  size_t num_ang_unknowns = sdm.GetNumLocalDOFs(psi_uk_man);
  local_psi_data_.assign(num_ang_unknowns, 0.0);
}
*/

/*
CBC_FLUDS::CBC_FLUDS(
  size_t num_groups_in_angle_set, // Renamed for clarity: groups in LBSGroupset
  size_t num_angles_in_angle_set, // Renamed for clarity: angles in this specific AngleSet
  const CBC_FLUDSCommonData& common_data,
  const UnknownManager& lbs_groupset_psi_uk_man, // Renamed for clarity
  const SpatialDiscretization& sdm)
  : FLUDS(num_groups_in_angle_set, num_angles_in_angle_set, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(lbs_groupset_psi_uk_man), // This member stores the LBSGroupset's psi_uk_man
    sdm_(sdm)
{
  // Create a temporary UnknownManager representing ONE scalar DOF per node
  const UnknownManager temp_unitary_uk_man({std::make_pair(UnknownType::SCALAR, 0)},
                                           UnknownStorageType::NODAL);
  // Get number of local spatial DOFs (nodes for PWLD) using the temporary manager
  const size_t num_local_spatial_dofs =
    sdm_.GetNumLocalDOFs(temp_unitary_uk_man); // Use local temp manager
  const size_t num_angles = this->num_angles_;
  const size_t num_groups = this->num_groups_;
  const size_t required_size = num_local_spatial_dofs * num_angles * num_groups;
  local_psi_data_.assign(required_size, 0.0);
}
*/

CBC_FLUDS::CBC_FLUDS(
  size_t num_groups_in_angle_set, // = LBSGroupset groups
  size_t num_angles_in_angle_set, // = Angles in this specific AngleSet
  const CBC_FLUDSCommonData& common_data,
  const UnknownManager& lbs_groupset_psi_uk_man, // Original LBSGroupset's psi_uk_man
  const SpatialDiscretization& sdm)
  : FLUDS(num_groups_in_angle_set, num_angles_in_angle_set, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(lbs_groupset_psi_uk_man), // Not needed to store member
    sdm_(sdm)
{
  // --- Calculate Sizes ---

  // 1. Spatial DOFs (remains the same)
  const UnknownManager temp_unitary_uk_man({std::make_pair(UnknownType::SCALAR, 0)},
                                           UnknownStorageType::NODAL);
  const size_t num_local_spatial_dofs = sdm_.GetNumLocalDOFs(temp_unitary_uk_man);

  // 2. Groups (remains the same)
  const size_t num_groups = this->num_groups_; // = num_groups_in_angle_set

  // 3. Angles BEFORE refactoring (from the original groupset uk_man)
  const size_t N_TOTAL_gs_angles = lbs_groupset_psi_uk_man.GetNumberOfUnknowns();

  // 4. Angles AFTER refactoring (from the specific AngleSet)
  const size_t N_angles_in_set = this->num_angles_; // = num_angles_in_angle_set

  // 5. Calculate sizes in number of doubles
  size_t size_before_doubles = 0;
  if (N_TOTAL_gs_angles > 0 && num_groups > 0)
  { // Avoid multiplying by zero
    size_before_doubles = num_local_spatial_dofs * N_TOTAL_gs_angles * num_groups;
  }

  size_t size_after_doubles = 0;
  if (N_angles_in_set > 0 && num_groups > 0)
  { // Avoid multiplying by zero
    size_after_doubles = num_local_spatial_dofs * N_angles_in_set * num_groups;
  }

  // --- Allocate the optimized vector ---
  local_psi_data_.assign(size_after_doubles, 0.0);

  // --- Print/Log Comparison ---
  // Check if the actual allocated size matches the calculation
  if (local_psi_data_.size() != size_after_doubles)
  {
    log.Log0Warning() << "CBC_FLUDS Warning: Allocated local_psi_data_ size ("
                      << local_psi_data_.size() << ") does not match calculated size ("
                      << size_after_doubles << ").";
  }

  // Convert sizes to Megabytes (MB) for better readability
  const double mb_divisor = 1024.0 * 1024.0;
  double size_before_mb = static_cast<double>(size_before_doubles * sizeof(double)) / mb_divisor;
  double size_after_mb = static_cast<double>(size_after_doubles * sizeof(double)) / mb_divisor;

  // Use OpenSn Logger (adjust log level as needed, e.g., Log0Verbose1)
  log.Log() << "CBC_FLUDS Size Comparison for AngleSet (Angles: " << N_angles_in_set << " of "
            << N_TOTAL_gs_angles << ", Groups: " << num_groups << "):";
  log.Log() << "  Size BEFORE refactor: " << size_before_doubles << " doubles (" << std::fixed
            << std::setprecision(3) << size_before_mb << " MB)";
  log.Log() << "  Size AFTER refactor:  " << size_after_doubles << " doubles (" << std::fixed
            << std::setprecision(3) << size_after_mb << " MB)";
  if (size_before_mb > 1e-6)
  { // Avoid division by zero if original size was 0
    double reduction = (size_before_mb - size_after_mb) / size_before_mb * 100.0;
    log.Log() << "  Memory Reduction: " << std::fixed << std::setprecision(1) << reduction << "%";
  }

} // End Constructor

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

const double*
CBC_FLUDS::GetLocalUpwindPsi(const Cell& face_neighbor) const
{
  const size_t num_angles = this->num_angles_;
  const size_t num_groups = this->num_groups_;
  const size_t node_stride_compact = num_angles * num_groups;

  const UnknownManager temp_unitary_uk_man({std::make_pair(UnknownType::SCALAR, 0)},
                                           UnknownStorageType::NODAL);
  // Get the base spatial DOF index for the neighbor cell's first node using the temporary manager
  const int64_t node0_global_map =
    sdm_.MapDOFLocal(face_neighbor, 0, temp_unitary_uk_man, 0, 0); // Use local temp manager

  const int64_t offset = node0_global_map * node_stride_compact;

  if (offset < 0 || static_cast<size_t>(offset) >= local_psi_data_.size())
  {
    throw std::runtime_error("CBC_FLUDS::GetLocalUpwindPsi: Offset out of bounds.");
  }
  return &local_psi_data_[offset]; // Returns pointer to start of neighbor cell's data block
}

// NEW METHOD to set angular value data for a downwind face of the cell
// double*
// CBC_FLUDS::GetLocalDownwindPsi(const Cell& cell)
// {
//   const auto dof_map = sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0);
//   return &local_psi_data_[dof_map];
// }

double*
CBC_FLUDS::GetLocalDownwindPsi(const Cell& cell)
{
  const size_t num_angles = this->num_angles_;
  const size_t num_groups = this->num_groups_;
  const size_t node_stride_compact = num_angles * num_groups;

  const UnknownManager temp_unitary_uk_man({std::make_pair(UnknownType::SCALAR, 0)},
                                           UnknownStorageType::NODAL);

  // Get the base spatial DOF index for the cell's first node using the temporary manager
  const int64_t node0_global_map =
    sdm_.MapDOFLocal(cell, 0, temp_unitary_uk_man, 0, 0); // Use local temp manager

  const int64_t offset = node0_global_map * node_stride_compact;

  if (offset < 0 || static_cast<size_t>(offset) >= local_psi_data_.size())
  {
    throw std::runtime_error("CBC_FLUDS::GetLocalDownwindPsi: Offset out of bounds.");
  }
  return &local_psi_data_[offset]; // Returns pointer to start of cell's data block
}

const std::vector<double>&
CBC_FLUDS::GetNonLocalUpwindData(uint64_t cell_global_id, unsigned int face_id) const
{
  return deplocs_outgoing_messages_.at({cell_global_id, face_id});
}

// const double*
// CBC_FLUDS::GetNonLocalUpwindPsi(const std::vector<double>& psi_data,
//                                 unsigned int face_node_mapped,
//                                 unsigned int angle_set_index)
// {
//   const size_t dof_map = face_node_mapped * num_groups_and_angles_ + angle_set_index *
//   num_groups_; return &psi_data[dof_map];
// }

/*
const double*
CBC_FLUDS::GetNonLocalUpwindPsi(
  const std::vector<double>&
    psi_data_aggregated,             // Vector with data for ALL angles in the set for this face
  unsigned int face_node_mapped_idx, // Node index on face (0 to N_face_nodes-1)
  unsigned int local_angle_idx_in_set) const // Local angle index (0 to N_angles_in_set-1)
{
  const size_t num_groups = this->num_groups_;
  // Number of angles THIS AngleSet handles (and thus packed into psi_data_aggregated)
  const size_t num_angles_in_set = this->num_angles_;

  // Calculate strides WITHIN the aggregated vector for this face
  const size_t angle_stride_agg = num_groups; // Groups are contiguous per angle
  // Stride needed to get from Node K to Node K+1 for the same angle/group
  const size_t node_stride_agg = num_angles_in_set * num_groups;

  // Calculate the final 1D offset
  const size_t offset =
    face_node_mapped_idx * node_stride_agg + // Offset to the start of this node's data
    local_angle_idx_in_set *
      angle_stride_agg; // Offset to the start of this angle's data (for group 0)

  // Bounds check - crucial! Check if accessing groups would go out of bounds
  if (offset + num_groups > psi_data_aggregated.size())
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetNonLocalUpwindPsi: Offset calculation error. "
               << "Offset=" << offset << ", num_groups=" << num_groups
               << ", AggregatedVectorSize=" << psi_data_aggregated.size()
               << ", RequestedNodeIdx=" << face_node_mapped_idx
               << ", RequestedLocalAngleIdx=" << local_angle_idx_in_set
               << ", NumAnglesInSet=" << num_angles_in_set;
    throw std::runtime_error(err_stream.str());
  }

  return &psi_data_aggregated[offset]; // Return pointer to start of group data

  // ---- USE THE BELOW BLOCK; ABOVE BLOCK IS JUNK
  // const size_t num_groups = this->num_groups_;
  // // The input vector psi_data_aggregated contains data for ONE angle,
  // // packed as [node0_g0, node0_g1, ..., node1_g0, node1_g1, ...]
  // const size_t node_stride_in_packet = num_groups;
  // const size_t offset = face_node_mapped_idx * node_stride_in_packet;

  // return &psi_data_aggregated[offset]; // Pointer to group 0 for the requested node
}
*/

const double*
CBC_FLUDS::GetNonLocalUpwindPsi(
  const std::vector<double>&
    psi_data_aggregated,             // Vector with data for ONE angle, packed node-major
  unsigned int face_node_mapped_idx, // Node index on face (0 to N_face_nodes-1)
  unsigned int local_angle_idx_in_set)
  const // Local angle index (0 to N_angles_in_set-1) - Used for context/debug now
{
  // Number of energy groups (this remains the same)
  const size_t num_groups = this->num_groups_;

  // --- Calculate offset based on the ACTUAL structure of psi_data_aggregated ---
  // The received packet (psi_data_aggregated) contains data for ONLY ONE angle.
  // The structure is:
  // [node0_g0, node0_g1, ..., node0_g(N-1),   <-- Node 0 block (size = num_groups)
  //  node1_g0, node1_g1, ..., node1_g(N-1),   <-- Node 1 block (size = num_groups)
  //  ...
  //  nodeM_g0, ..., nodeM_g(N-1) ]             <-- Node M block (size = num_groups)
  // Where N=num_groups, M=num_face_nodes-1

  // The stride needed to jump from the start of one node's data block
  // to the start of the next node's data block IN THIS PACKET is simply num_groups.
  const size_t node_stride_in_packet = num_groups;

  // Calculate the final 1D offset to the start of the data for the requested node
  const size_t offset = face_node_mapped_idx * node_stride_in_packet;

  // --- Bounds check - crucial! ---
  // Check if accessing the data for the requested node (all groups) goes out of bounds.
  if (psi_data_aggregated.empty() || (offset + num_groups > psi_data_aggregated.size()))
  {
    // Provide a detailed error message
    const size_t num_angles_in_set = this->num_angles_; // Get for error message context
    size_t expected_size = 0;
    // Try to determine expected size based on other members if possible (e.g., face mapping size)
    // This part is tricky without access to num_face_nodes directly here.
    // We know the size SHOULD be num_face_nodes * num_groups, but we don't have num_face_nodes.
    // So, we report what we have.

    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetNonLocalUpwindPsi: Offset calculation error or empty vector. "
               << "Calculated Offset=" << offset << ", NumGroups=" << num_groups
               << ", Actual AggregatedVectorSize="
               << psi_data_aggregated.size()
               // << ", Expected AggregatedVectorSize (num_face_nodes*num_groups)= ?"
               << ", RequestedNodeIdxOnFace=" << face_node_mapped_idx
               << ", RequestedLocalAngleIdxInSet=" << local_angle_idx_in_set
               << ", NumAnglesInOwningSet=" << num_angles_in_set;
    throw std::runtime_error(err_stream.str());
  }

  // Return pointer to the start of the energy group data for the requested node
  return &psi_data_aggregated[offset];
}

} // namespace opensn

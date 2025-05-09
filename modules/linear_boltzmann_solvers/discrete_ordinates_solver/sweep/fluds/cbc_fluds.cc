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

// local_psi_data_ is now sized optimally for the angles in this specific
// AngleSet, rather than all angles in the LBSGroupset's quadrature.
CBC_FLUDS::CBC_FLUDS(
  size_t num_groups_in_angle_set, // Number of groups in this AngleSet's LBSGroupset
  size_t num_angles_in_angle_set, // Number of angles in THIS specific AngleSet
  const CBC_FLUDSCommonData& common_data,
  const UnknownManager&
    lbs_groupset_psi_uk_man, // LBSGroupset's psi_uk_man (used for logging comparison)
  const SpatialDiscretization& sdm)
  : FLUDS(num_groups_in_angle_set, num_angles_in_angle_set, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(lbs_groupset_psi_uk_man),
    sdm_(sdm)
{
  // --- Calculate Required Size for the optimized local_psi_data_ ---

  // 1. Get number of purely spatial DOFs
  //    This uses a temporary UnknownManager representing 1 scalar unknown per node.
  const UnknownManager temp_unitary_uk_man({std::make_pair(UnknownType::SCALAR, 0)},
                                           UnknownStorageType::NODAL);
  const size_t num_local_spatial_dofs = sdm_.GetNumLocalDOFs(temp_unitary_uk_man);

  // 2. Get number of angles specific to this AngleSet instance (from base class member).
  const size_t N_angles_in_set = this->num_angles_; // = num_angles_in_angle_set

  // 3. Get number of groups for this AngleSet instance (from base class member).
  const size_t num_groups = this->num_groups_; // = num_groups_in_angle_set

  // 4. Calculate the required size for the compact local_psi_data_ vector.
  //    Layout: SpatialDOF_Major -> AngleInSet_Major -> Group_Major
  size_t size_after_doubles = num_local_spatial_dofs * N_angles_in_set * num_groups;

  // 5. Allocate the local_psi_data_ vector.
  local_psi_data_.assign(size_after_doubles, 0.0);

  // --- Verification logging ---
  // Calculate what the old size would have been for comparison.

  // Angles in groupset's quadrature
  const size_t N_TOTAL_gs_angles = lbs_groupset_psi_uk_man.GetNumberOfUnknowns();

  // Calculate sizes in number of doubles
  size_t size_before_doubles_calc = num_local_spatial_dofs * N_TOTAL_gs_angles * num_groups;
  size_t size_before_doubles_direct = sdm.GetNumLocalDOFs(lbs_groupset_psi_uk_man);

  if (local_psi_data_.size() != size_after_doubles) // Sanity check allocation
  {
    log.Log0Warning() << "CBC_FLUDS Warning: Allocated local_psi_data_ size ("
                      << local_psi_data_.size() << ") does not match calculated size ("
                      << size_after_doubles << ").";
  }

  const double mb_divisor = 1024.0 * 1024.0; // For conversion to MBs
  double size_before_mb =
    static_cast<double>(size_before_doubles_calc * sizeof(double)) / mb_divisor;
  double size_before_direct_mb =
    static_cast<double>(size_before_doubles_direct * sizeof(double)) / mb_divisor;

  double size_after_mb = static_cast<double>(size_after_doubles * sizeof(double)) / mb_divisor;

  log.Log() << "CBC_FLUDS Size Comparison for AngleSet (Angles in Set: " << N_angles_in_set
            << ", Angles in Groupset Quadrature: " << N_TOTAL_gs_angles
            << ", Groups: " << num_groups << "):";
  log.Log() << "  Original estimated size: " << size_before_doubles_calc << " doubles ("
            << std::fixed << std::setprecision(3) << size_before_mb << " MB)";
  log.Log() << "  Original direct size (for verification): " << size_before_doubles_direct
            << " doubles (" << std::fixed << std::setprecision(3) << size_before_direct_mb
            << " MB)";
  log.Log() << "  Optimized size (current):  " << size_after_doubles << " doubles (" << std::fixed
            << std::setprecision(3) << size_after_mb << " MB)";

  if (size_before_mb > 1e-6)
  { // Avoid division by zero if original size was effectively 0
    double reduction = (size_before_mb - size_after_mb) / size_before_mb * 100.0;
    log.Log() << "  Memory Reduction for local_psi_data_: " << std::fixed << std::setprecision(1)
              << reduction << "%";
  }
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

// Returns a base pointer to the start of the upwind neighbor cell's data block
// within the compact local_psi_data_.
// The caller (CbcSweepChunk) calculates the relative offset for the specific
// node within that cell and the specific angle.
const double*
CBC_FLUDS::GetLocalUpwindPsi(const Cell& face_neighbor) const
{
  const size_t num_angles = this->num_angles_; // Angles in THIS AngleSet
  const size_t num_groups = this->num_groups_;

  // Stride in compact local_psi_data_ to get from one spatial DOF's full angle/group block to the
  // next.
  const size_t node_stride_compact = num_angles * num_groups;

  const UnknownManager temp_unitary_uk_man({std::make_pair(UnknownType::SCALAR, 0)},
                                           UnknownStorageType::NODAL);

  // Get the unique index (0 to num_local_spatial_dofs-1) for the first node of the neighbor cell.
  const int64_t node0_spatial_map =
    sdm_.MapDOFLocal(face_neighbor, 0, temp_unitary_uk_man, 0, 0); // Use local temp manager

  // Offset to the start of the neighbor cell's data block in the compact local_psi_data_.
  const int64_t offset = node0_spatial_map * node_stride_compact;

  if (offset < 0 || static_cast<size_t>(offset) >= local_psi_data_.size())
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetLocalUpwindPsi: Offset out of bounds. "
               << "NeighborCell global_id = " << face_neighbor.global_id
               << ", Calculated Offset = " << offset
               << ", CompactVectorSize = " << local_psi_data_.size()
               << ", node0_spatial_map = " << node0_spatial_map
               << ", node_stride_compact = " << node_stride_compact;
    throw std::runtime_error(err_stream.str());
  }
  return &local_psi_data_[offset]; // Returns pointer to start of neighbor cell's data block
}

// Returns a base pointer to the start of the current cell's data block
// within the compact local_psi_data_ for writing.
// The caller (CbcSweepChunk) calculates the relative offset for the specific
// node and angle.
double*
CBC_FLUDS::GetLocalDownwindPsi(const Cell& cell)
{
  const size_t num_angles = this->num_angles_; // Angles in THIS AngleSet
  const size_t num_groups = this->num_groups_;

  // Stride in compact local_psi_data_ to get from one spatial DOF's full angle/group block to the
  // next.
  const size_t node_stride_compact = num_angles * num_groups;

  const UnknownManager temp_unitary_uk_man({std::make_pair(UnknownType::SCALAR, 0)},
                                           UnknownStorageType::NODAL);

  // Get the unique index (0 to num_local_spatial_dofs-1) for the first node of the current cell.
  const int64_t node0_spatial_map =
    sdm_.MapDOFLocal(cell, 0, temp_unitary_uk_man, 0, 0); // Use local temp manager

  // Offset to the start of the current cell's data block in the compact local_psi_data_.
  const int64_t offset = node0_global_map * node_stride_compact;

  if (offset < 0 || static_cast<size_t>(offset) >= local_psi_data_.size())
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetLocalDownwindPsi: Offset out of bounds. "
               << "CurrentCell global_id = " << cell.global_id << ", Calculated Offset =" << offset
               << ", CompactVectorSize = " << local_psi_data_.size()
               << ", node0_spatial_map = " << node0_spatial_map
               << ", node_stride_compact = " << node_stride_compact;
    throw std::runtime_error(err_stream.str());
  }
  return &local_psi_data_[offset]; // Returns pointer to start of cell's data block
}

const std::vector<double>&
CBC_FLUDS::GetNonLocalUpwindData(uint64_t cell_global_id, unsigned int face_id) const
{
  // This map stores vectors received from MPI.
  // Each vector is a multi-AngleSet-angle packet.
  return deplocs_outgoing_messages_.at({cell_global_id, face_id});
}

const double*
CBC_FLUDS::GetNonLocalUpwindPsi(
  const std::vector<double>& psi_data, // The multi-AngleSet-angle packet for a face
  unsigned int face_node_mapped,       // Index of the node on the face (0 to N_face_nodes-1)
  unsigned int
    angle_set_index) // Local index of the angle within this AngleSet (0 to N_angles_in_set-1)
{
  // Total number of psi values for ONE face node, across ALL angles in THIS AngleSet.
  // This is the stride needed to jump from one face node's full data block to the next within
  // psi_data.
  const size_t num_psi_per_face_node_for_set = this->num_angles_ * this->num_groups_;

  // Stride between angles for the SAME face node within psi_data.
  const size_t num_groups_stride = this->num_groups_;

  // Calculate the 1D offset into psi_data.
  const size_t dof_map =
    face_node_mapped *
      num_psi_per_face_node_for_set +    // Offset to the start of data for this face_node_mapped
    angle_set_index * num_groups_stride; // Further offset to the start of data for this specific
                                         // angle_set_index (for group 0)

  // Bounds check (commented out for time being)
  // if (dof_map + num_groups_stride > psi_data.size() && num_groups_stride > 0)
  // { // Avoid issues if num_groups_stride is 0
  //   // (More detailed error message)
  //   std::ostringstream err_stream;
  //   err_stream << "CBC_FLUDS::GetNonLocalUpwindPsi: Offset out of bounds. "
  //              << "Calculated_dof_map=" << dof_map << ", num_groups_stride=" << num_groups_stride
  //              << ", psi_data.size()=" << psi_data.size()
  //              << ", face_node_mapped=" << face_node_mapped
  //              << ", angle_set_index=" << angle_set_index
  //              << ", this->num_angles_=" << this->num_angles_
  //              << ", this->num_groups_=" << this->num_groups_;
  //   throw std::runtime_error(err_stream.str());
  // }
  // if (psi_data.empty() && (face_node_mapped > 0 || angle_set_index > 0))
  // {
  //   throw std::runtime_error(
  //     "CBC_FLUDS::GetNonLocalUpwindPsi: Accessing non-empty psi_data with non-zero indices.");
  // }
  // if (psi_data.empty())
  //   return nullptr;

  return &psi_data[dof_map];
}

} // namespace opensn

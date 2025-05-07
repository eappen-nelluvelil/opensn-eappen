// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"

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
  // Sizing of local_psi_data_ is based on the LBSGroupset's psi_uk_man,
  // which covers ALL angles in the LBSGroupset's quadrature.
  size_t num_ang_unknowns_for_allocation = sdm_.GetNumLocalDOFs(psi_uk_man_);
  local_psi_data_.assign(num_ang_unknowns_for_allocation, 0.0);

  // --- Verification Code Start ---

  // 1. Get the grid
  const auto& grid = common_data_.GetSPDS().GetGrid(); // Correct way to access the grid

  // 2. Determine N_TOTAL_gs_angles and N_TOTAL_gs_groups from psi_uk_man_
  //    (the LBSGroupset's UnknownManager for psi)
  //    This is what local_psi_data_ is actually dimensioned for.
  const size_t N_TOTAL_gs_angles = psi_uk_man_.GetNumberOfUnknowns();
  const size_t N_TOTAL_gs_groups =
    (N_TOTAL_gs_angles > 0) ? psi_uk_man_.GetUnknown(0).GetNumComponents() : 0;

  if (N_TOTAL_gs_angles == 0 || N_TOTAL_gs_groups == 0)
  {
    log.Log() << "CBC_FLUDS Warning: No angles or groups in psi_uk_man_ for verification. Skipping "
                 "layout check.";
  }
  else
  {
    // Write unique values
    for (const auto& cell : grid->local_cells) // Use the correctly accessed grid
    {
      const auto& cell_mapping = sdm_.GetCellMapping(cell);
      const size_t num_nodes_in_cell = cell_mapping.GetNumNodes();

      for (size_t n_idx = 0; n_idx < num_nodes_in_cell; ++n_idx) // Node index *within the cell*
      {
        for (size_t ang_idx = 0; ang_idx < N_TOTAL_gs_angles;
             ++ang_idx) // Iterate over ALL angles psi_uk_man_ defines
        {
          for (size_t grp_idx = 0; grp_idx < N_TOTAL_gs_groups;
               ++grp_idx) // Iterate over ALL groups psi_uk_man_ defines
          {
            int64_t G_map = sdm_.MapDOFLocal(cell, n_idx, psi_uk_man_, ang_idx, grp_idx);

            if (G_map < 0 || static_cast<size_t>(G_map) >= local_psi_data_.size())
            {
              log.Log() << "CBC_FLUDS Error (Write): Calculated G_map=" << G_map
                        << " is out of bounds for local_psi_data_ size=" << local_psi_data_.size()
                        << " for Cell=" << cell.local_id << ", Node=" << n_idx
                        << ", Angle=" << ang_idx << ", Group=" << grp_idx;
              // Decide on error handling: throw, exit, or continue with a flag
              throw std::runtime_error("G_map out of bounds during verification write.");
            }

            double unique_value =
              static_cast<double>(cell.local_id) * 1.0e9 + static_cast<double>(n_idx) * 1.0e7 +
              static_cast<double>(ang_idx) * 1.0e4 + static_cast<double>(grp_idx);
            local_psi_data_[G_map] = unique_value;
          }
        }
      }
    }

    // --- Stride Verification Code Start ---
    // Assumes the unique values have already been written using Method 1

    bool strides_correct = true;
    log.Log() << "\n--- Stride Verification ---";

    for (const auto& cell : grid->local_cells)
    {
      const auto& cell_mapping = sdm_.GetCellMapping(cell);
      const size_t num_nodes_in_cell = cell_mapping.GetNumNodes();

      for (size_t n_idx = 0; n_idx < num_nodes_in_cell; ++n_idx)
      {
        for (size_t ang_idx = 0; ang_idx < N_TOTAL_gs_angles; ++ang_idx)
        {
          // *** Check Group Stride (Contiguity) ***
          if (N_TOTAL_gs_groups > 1)
          {
            // Get map for group 0 and group 1
            int64_t G_map_g0 = sdm_.MapDOFLocal(cell, n_idx, psi_uk_man_, ang_idx, 0);
            int64_t G_map_g1 = sdm_.MapDOFLocal(cell, n_idx, psi_uk_man_, ang_idx, 1);

            // EXPECTATION: MapDOFLocal for group 1 should be exactly 1 greater than for group 0
            if (G_map_g1 != G_map_g0 + 1)
            {
              log.Log() << "Stride FAIL (Group): Cell=" << cell.local_id << " Node=" << n_idx
                        << " Angle=" << ang_idx << " G0_map=" << G_map_g0 << " G1_map=" << G_map_g1
                        << " Diff=" << (G_map_g1 - G_map_g0) << " Expected Diff=1";
              strides_correct = false;
            }

            // Optional: Also check the value tagged for g1 is at G_map_g0 + 1
            // Logical indices for next element: (cell, n_idx, ang_idx, grp_idx=1) since we start
            // the check from g=0
            double expected_val_g1 =
              static_cast<double>(cell.local_id) * 1.0e9 + static_cast<double>(n_idx) * 1.0e7 +
              static_cast<double>(ang_idx) * 1.0e4 + static_cast<double>(1); // Group index is 1

            if (std::abs(local_psi_data_[G_map_g0 + 1] - expected_val_g1) > 1e-9)
            {
              // Using std::cerr for consistency, replace with log.Log() if preferred
              log.Log() << "Value Check FAIL (Group): Cell=" << cell.local_id << " Node=" << n_idx
                        << " Angle=" << ang_idx << " Expected G1 Value at Map=" << G_map_g0 + 1
                        << ". Expected=" << expected_val_g1
                        << ", Got=" << local_psi_data_[G_map_g0 + 1];
              strides_correct = false; // Treat value mismatch same as stride mismatch
            }
          }

          // *** Check Angle Stride ***
          if (N_TOTAL_gs_angles > 1 &&
              ang_idx < (N_TOTAL_gs_angles - 1)) // Avoid checking beyond last angle
          {
            // Check stride between angle 'ang_idx' and 'ang_idx + 1' for group 0
            int64_t G_map_a0_g0 = sdm_.MapDOFLocal(cell, n_idx, psi_uk_man_, ang_idx, 0);
            int64_t G_map_a1_g0 = sdm_.MapDOFLocal(cell, n_idx, psi_uk_man_, ang_idx + 1, 0);

            // EXPECTATION: MapDOFLocal for angle ang_idx+1 should be N_TOTAL_gs_groups greater
            int64_t expected_stride_angle = static_cast<int64_t>(N_TOTAL_gs_groups);
            if (G_map_a1_g0 != G_map_a0_g0 + expected_stride_angle)
            {
              log.Log() << "Stride FAIL (Angle): Cell=" << cell.local_id << " Node=" << n_idx
                        << " Angle=" << ang_idx << "->" << ang_idx + 1 << " G0_map0=" << G_map_a0_g0
                        << " G0_map1=" << G_map_a1_g0 << " Diff=" << (G_map_a1_g0 - G_map_a0_g0)
                        << " Expected Diff=" << expected_stride_angle;
              strides_correct = false;
            }
            // Optional: Also check the value tagged for ang_idx+1, g=0 is at G_map_a0_g0 +
            // expected_stride_angle Logical indices for next element: (cell, n_idx, ang_idx+1,
            // grp_idx=0)
            double expected_val_a1g0 =
              static_cast<double>(cell.local_id) * 1.0e9 + static_cast<double>(n_idx) * 1.0e7 +
              static_cast<double>(ang_idx + 1) * 1.0e4 + // Angle index is ang_idx + 1
              static_cast<double>(0);                    // Group index is 0

            if (std::abs(local_psi_data_[G_map_a0_g0 + expected_stride_angle] - expected_val_a1g0) >
                1e-9)
            {
              log.Log() << "Value Check FAIL (Angle): Cell=" << cell.local_id << " Node=" << n_idx
                        << " Expected Angle " << ang_idx + 1 << " Group 0 Value"
                        << " at Map=" << G_map_a0_g0 + expected_stride_angle
                        << ". Expected=" << expected_val_a1g0
                        << ", Got=" << local_psi_data_[G_map_a0_g0 + expected_stride_angle];
              strides_correct = false;
            }
          }

          // *** Check Node Stride (within a cell) ***
          if (num_nodes_in_cell > 1 &&
              n_idx < (num_nodes_in_cell - 1)) // Avoid checking beyond last node
          {
            // Check stride between node n_idx and n_idx+1 for angle 0, group 0
            int64_t G_map_n0_a0_g0 = sdm_.MapDOFLocal(cell, n_idx, psi_uk_man_, 0, 0);
            int64_t G_map_n1_a0_g0 = sdm_.MapDOFLocal(cell, n_idx + 1, psi_uk_man_, 0, 0);

            // EXPECTATION: MapDOFLocal for node n_idx+1 should be (N_TOTAL_gs_angles *
            // N_TOTAL_gs_groups) greater
            int64_t expected_stride_node =
              static_cast<int64_t>(N_TOTAL_gs_angles * N_TOTAL_gs_groups);
            if (G_map_n1_a0_g0 != G_map_n0_a0_g0 + expected_stride_node)
            {
              log.Log() << "Stride FAIL (Node): Cell=" << cell.local_id << " Node=" << n_idx << "->"
                        << n_idx + 1 << " A0G0_map0=" << G_map_n0_a0_g0
                        << " A0G0_map1=" << G_map_n1_a0_g0
                        << " Diff=" << (G_map_n1_a0_g0 - G_map_n0_a0_g0)
                        << " Expected Diff=" << expected_stride_node;
              strides_correct = false;
            }
            // Optional: Check value tagged for n_idx+1, a=0, g=0 is at G_map_n0_a0_g0 +
            // expected_stride_node Logical indices for next element: (cell, n_idx+1, ang_idx=0,
            // grp_idx=0)
            double expected_val_n1a0g0 =
              static_cast<double>(cell.local_id) * 1.0e9 +
              static_cast<double>(n_idx + 1) * 1.0e7 + // Node index is n_idx + 1
              static_cast<double>(0) * 1.0e4 +         // Angle index is 0
              static_cast<double>(0);                  // Group index is 0

            if (std::abs(local_psi_data_[G_map_n0_a0_g0 + expected_stride_node] -
                         expected_val_n1a0g0) > 1e-9)
            {
              log.Log() << "Value Check FAIL (Node): Cell=" << cell.local_id << " Expected Node "
                        << n_idx + 1 << " Angle 0 Group 0 Value"
                        << " at Map=" << G_map_n0_a0_g0 + expected_stride_node
                        << ". Expected=" << expected_val_n1a0g0
                        << ", Got=" << local_psi_data_[G_map_n0_a0_g0 + expected_stride_node];
              strides_correct = false;
            }
          }
        } // End Angle Loop
      } // End Node Loop
    } // End Cell Loop

    if (strides_correct)
    {
      log.Log() << "Stride verification PASSED!";
    }
    else
    {
      log.Log() << "Stride verification FAILED!";
      // throw std::runtime_error("CBC_FLUDS stride verification failed.");
    }
    // --- Stride Verification Code End ---

    // Read and verify values
    bool layout_is_correct = true;
    for (const auto& cell : grid->local_cells) // Use the correctly accessed grid
    {
      const auto& cell_mapping = sdm_.GetCellMapping(cell);
      const size_t num_nodes_in_cell = cell_mapping.GetNumNodes();

      for (size_t n_idx = 0; n_idx < num_nodes_in_cell; ++n_idx)
      {
        for (size_t ang_idx = 0; ang_idx < N_TOTAL_gs_angles; ++ang_idx) // Iterate over ALL angles
        {
          for (size_t grp_idx = 0; grp_idx < N_TOTAL_gs_groups;
               ++grp_idx) // Iterate over ALL groups
          {
            int64_t G_map = sdm_.MapDOFLocal(cell, n_idx, psi_uk_man_, ang_idx, grp_idx);

            if (G_map < 0 || static_cast<size_t>(G_map) >= local_psi_data_.size())
            {
              log.Log() << "CBC_FLUDS Error (Read): Calculated G_map=" << G_map
                        << " is out of bounds for local_psi_data_ size=" << local_psi_data_.size()
                        << " for Cell=" << cell.local_id << ", Node=" << n_idx
                        << ", Angle=" << ang_idx << ", Group=" << grp_idx;
              layout_is_correct = false;
              // Decide on error handling
              throw std::runtime_error("G_map out of bounds during verification read.");
            }

            double expected_value =
              static_cast<double>(cell.local_id) * 1.0e9 + static_cast<double>(n_idx) * 1.0e7 +
              static_cast<double>(ang_idx) * 1.0e4 + static_cast<double>(grp_idx);

            if (std::abs(local_psi_data_[G_map] - expected_value) > 1e-9)
            {
              log.Log() << "CBC_FLUDS Verification FAILED for:"
                        << " Cell=" << cell.local_id << " Node_in_cell=" << n_idx
                        << " Angle=" << ang_idx << " Group=" << grp_idx << " G_map=" << G_map
                        << ". Expected=" << expected_value << ", Got=" << local_psi_data_[G_map];
              layout_is_correct = false;
            }
          }
        }
      }
    }

    if (layout_is_correct)
    {
      log.Log() << "CBC_FLUDS: Memory layout verification PASSED!";
    }
    else
    {
      log.Log() << "CBC_FLUDS: Memory layout verification FAILED overall.";
      // Potentially throw an error here if a failed verification should halt execution
      // throw std::runtime_error("CBC_FLUDS memory layout verification failed.");
    }
  } // end else (if N_TOTAL_gs_angles > 0 ...)
  // --- Verification Code End ---

  local_psi_data_.assign(num_ang_unknowns_for_allocation, 0.0);
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

// NEW METHOD to avoid having to access psi_new_local_[groupset.id]
const double*
CBC_FLUDS::GetLocalUpwindPsi(const Cell& face_neighbor,
                             const unsigned int adj_cell_node_offset) const
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

} // namespace opensn

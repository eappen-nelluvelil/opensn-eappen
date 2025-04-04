// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"

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
    sdm_(sdm)
{
  // Need to be sized to hold angles in an angleset for a groupset, instead of all angles in the
  // groupset i.e., num angles in angle set * num groups in groupset for that angle set For the time
  // being, use the full vector Each groupset is associated with a quadrature set
  size_t num_ang_unknowns = sdm.GetNumLocalDOFs(psi_uk_man);
  local_psi_data_.assign(num_ang_unknowns, 0.0);
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

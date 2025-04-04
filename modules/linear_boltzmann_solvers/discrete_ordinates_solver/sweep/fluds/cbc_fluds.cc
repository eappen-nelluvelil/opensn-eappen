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
                     std::vector<double>& local_psi_data,
                     const UnknownManager& psi_uk_man,
                     const SpatialDiscretization& sdm)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    local_psi_data_(local_psi_data),
    psi_uk_man_(psi_uk_man),
    sdm_(sdm)
{
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

const std::vector<double>&
CBC_FLUDS::GetLocalUpwindDataBlock() const
{
  return local_psi_data_;
}

const double*
CBC_FLUDS::GetLocalCellUpwindPsi(const std::vector<double>& psi_data_block, const Cell& cell)
{
  const auto dof_map = sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0);
  return &psi_data_block[dof_map];
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

void CBC_FLUDS::UpdateCurrentCellPsiSubset(const double* psi_vector_start, size_t subset_size)
{
  // Resize the internal storage if necessary
  if (current_cell_psi_subset_.size() != subset_size)
  {
    current_cell_psi_subset_.resize(subset_size);
  }

  // Copy the data from the main psi vector into the local subset storage
  if (subset_size > 0 && psi_vector_start != nullptr)
  {
    // Using std::copy
    std::copy(psi_vector_start,
              psi_vector_start + subset_size,
              current_cell_psi_subset_.begin());
  }
  else if (subset_size == 0)
  {
     current_cell_psi_subset_.clear(); // Handle zero size case if needed
  }
}

const std::vector<double>& CBC_FLUDS::GetCurrentCellPsiSubset() const
{
  return current_cell_psi_subset_;
}

std::vector<double>& CBC_FLUDS::GetCurrentCellPsiSubset()
{
  return current_cell_psi_subset_;
}

} // namespace opensn

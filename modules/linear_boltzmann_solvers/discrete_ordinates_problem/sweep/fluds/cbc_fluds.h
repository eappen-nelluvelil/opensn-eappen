// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "framework/math/unknown_manager/unknown_manager.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace opensn
{

class UnknownManager;
class SpatialDiscretization;
class Cell;

/// Host CBC angular-flux storage.
class CBC_FLUDS : public FLUDS
{
public:
  /// Incoming nonlocal payload and unlocked cell.
  struct IncomingNonlocalPsi
  {
    /// Incoming face payload storage.
    std::vector<double>& psi;

    /// Local cell with one newly satisfied CBC dependency.
    std::uint32_t cell_local_id = 0;
  };

  /// Construct host CBC flux data structures.
  CBC_FLUDS(unsigned int num_groups,
            size_t num_angles,
            const CBC_FLUDSCommonData& common_data,
            const UnknownManager& psi_uk_man,
            const SpatialDiscretization& sdm);

  [[nodiscard]] const CBC_FLUDSCommonData& GetCommonData() const;

  /// Return local upwind angular-flux group data.
  [[nodiscard]] double*
  UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx);

  /// Return local outgoing angular-flux group data.
  [[nodiscard]] double* OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx);

  /// Return nonlocal upwind angular-flux group data by incoming face slot.
  [[nodiscard]] double*
  NLUpwindPsiBySlot(size_t incoming_face_slot, unsigned int face_node_mapped, size_t as_ss_idx);

  /// Return nonlocal outgoing angular-flux group data.
  [[nodiscard]] double*
  NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx);

  /// Clear local and received nonlocal angular-flux storage.
  void ClearLocalAndReceivePsi() override;

  /// Prepare storage for an incoming payload and return the local task it unlocks.
  IncomingNonlocalPsi PrepareIncomingNonlocalPsiBySlot(size_t incoming_face_slot,
                                                       size_t data_size);

  void ClearSendPsi() override {}

  void AllocateInternalLocalPsi() override {}

  void AllocateOutgoingPsi() override {}

  void AllocateDelayedLocalPsi() override {}

  void AllocatePrelocIOutgoingPsi() override {}

  void AllocateDelayedPrelocIOutgoingPsi() override {}

protected:
  /// Common CBC FLUDS metadata.
  const CBC_FLUDSCommonData& common_data_;

  /// Unknown manager for angular fluxes.
  const UnknownManager& psi_uk_man_;

  /// Spatial discretization used for local DOF lookup.
  const SpatialDiscretization& sdm_;

  /// Number of angles in the groupset quadrature.
  size_t num_angles_in_gs_quadrature_;

  /// Number of quadrature-local DOFs.
  size_t num_quadrature_local_dofs_;

  /// Number of local spatial DOFs.
  size_t num_local_spatial_dofs_;

  /// Number of entries in local angular-flux storage.
  size_t local_psi_data_size_;

  /// Spatial DOF -> angle-set subset -> group angular-flux storage.
  std::vector<double> local_psi_data_;

  /// Incoming nonlocal angular-flux payload storage indexed by face slot.
  std::vector<std::vector<double>> incoming_nonlocal_psi_;

  /// Readiness flags for incoming nonlocal angular-flux payload slots.
  std::vector<unsigned char> incoming_nonlocal_psi_ready_;

  /// Precomputed start index into local angular-flux storage for each local cell.
  std::vector<size_t> cell_psi_start_;
};

} // namespace opensn

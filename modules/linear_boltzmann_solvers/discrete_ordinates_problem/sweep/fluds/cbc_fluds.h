// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "framework/math/unknown_manager/unknown_manager.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include <cstddef>
#include <vector>

namespace opensn
{

class UnknownManager;
class SpatialDiscretization;
class Cell;

/**
 * Flux data structures for the host cell-by-cell sweep algorithm.
 * This class manages local and nonlocal angular-flux storage during a CBC sweep.
 */
class CBC_FLUDS : public FLUDS
{
public:
  /// Construct host CBC flux data structures.
  CBC_FLUDS(unsigned int num_groups,
            size_t num_angles,
            const CBC_FLUDSCommonData& common_data,
            const UnknownManager& psi_uk_man,
            const SpatialDiscretization& sdm);

  /// Return immutable common CBC FLUDS metadata.
  [[nodiscard]] const CBC_FLUDSCommonData& GetCommonData() const;

  /**
   * Return local upwind angular-flux group data.
   * The returned pointer addresses the first group for the requested node and angle-set subset.
   */
  [[nodiscard]] double*
  UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx);

  /**
   * Return local outgoing angular-flux group data.
   * The returned pointer addresses storage for writing the just-solved node and angle-set subset.
   */
  [[nodiscard]] double* OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx);

  /**
   * Return nonlocal upwind angular-flux group data.
   * The returned pointer is null until the remote payload has been received.
   */
  [[nodiscard]] double* NLUpwindPsi(std::uint64_t cell_global_id,
                                    unsigned int face_id,
                                    unsigned int face_node_mapped,
                                    size_t as_ss_idx);

  /// Return nonlocal upwind angular-flux group data by incoming face slot.
  [[nodiscard]] double*
  NLUpwindPsiBySlot(size_t incoming_face_slot, unsigned int face_node_mapped, size_t as_ss_idx);

  /**
   * Return nonlocal outgoing angular-flux group data.
   * The returned pointer addresses a caller-owned face payload buffer.
   */
  [[nodiscard]] double*
  NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx);

  /// Clear local and received nonlocal angular-flux storage.
  void ClearLocalAndReceivePsi() override;

  /// Prepare storage for an incoming nonlocal face payload.
  std::vector<double>& PrepareIncomingNonlocalPsi(std::uint64_t cell_global_id,
                                                  unsigned int face_id,
                                                  size_t data_size) override;

  /// Clear outgoing angular-flux storage.
  void ClearSendPsi() override {}

  /// Allocate internal local angular-flux storage.
  void AllocateInternalLocalPsi() override {}

  /// Allocate outgoing angular-flux storage.
  void AllocateOutgoingPsi() override {}

  /// Allocate delayed local angular-flux storage.
  void AllocateDelayedLocalPsi() override {}

  /// Allocate pre-location outgoing angular-flux storage.
  void AllocatePrelocIOutgoingPsi() override {}

  /// Allocate delayed pre-location outgoing angular-flux storage.
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

  /**
   * Local angular-flux storage.
   * spatial DOF major -> angle in angleset major -> group in groupset major
   */
  std::vector<double> local_psi_data_;

  /// Incoming nonlocal angular-flux payload storage indexed by face slot.
  std::vector<std::vector<double>> incoming_nonlocal_psi_;

  /// Readiness flags for incoming nonlocal angular-flux payload slots.
  std::vector<unsigned char> incoming_nonlocal_psi_ready_;

  /// Precomputed start index into local angular-flux storage for each local cell.
  std::vector<size_t> cell_psi_start_;
};

} // namespace opensn

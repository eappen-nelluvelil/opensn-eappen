// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "framework/math/unknown_manager/unknown_manager.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include <cstddef>
#include <map>
#include <functional>

namespace opensn
{

class UnknownManager;
class SpatialDiscretization;
class Cell;

/**
 * Flux data structures (FLUDS) specific to the cell-by-cell (CBC) sweep algorithm
 *
 * This class manages the storage and access of angular flux data during a CBC sweep
 *
 * It provides methods to access:
 * - Upwind angular flux data from local neighbor cells
 * - Storage locations for downwind angular flux data for the current cell
 * - Upwind angular flux data received from remote MPI ranks
 * - Delayed angular flux data for cycle-breaking (local FAS edges and delayed locations)
 */
class CBC_FLUDS : public FLUDS
{
public:
  CBC_FLUDS(unsigned int num_groups,
            size_t num_angles,
            const CBC_FLUDSCommonData& common_data,
            const UnknownManager& psi_uk_man,
            const SpatialDiscretization& sdm);

  virtual const FLUDSCommonData& GetCommonData() const;

  /**
   * Given a local upwind neighbor cell, a node index on this cell, and an
   * angleset subset index, this function returns a pointer to
   * the start of the group data for the specified node and angle.
   */
  double* UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx);

  /**
   * Given a local cell, a node index on this cell, and an angleset subset index,
   * this function returns a pointer to the start of the group data for the specified
   * node and angle for writing its just solved angular fluxes.
   */
  double* OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx);

  /**
   * Given a remote upwind cell's global ID, a face ID on this cell,
   * a node index on this face, and an angleset subset index,
   * this function returns a pointer to the start of the group data for the specified
   * face node and angle.
   */
  double* NLUpwindPsi(uint64_t cell_global_id,
                      unsigned int face_id,
                      unsigned int face_node_mapped,
                      size_t as_ss_idx);

  /**
   * Given a pointer to a vector holding the non-local outgoing psi data for a face,
   * a node index on this face, and an angleset subset index,
   * this function returns a pointer to the start of the group data for the specified
   * face node and angle.
   */
  double*
  NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx);

  /**
   * Returns a pointer to delayed local incoming psi (old values) for a given
   * cell/face/node/angle. Used when reading from a local FAS edge.
   */
  double* DelayedLocalUpwindPsi(uint64_t cell_global_id,
                                unsigned int face_id,
                                unsigned int face_node_mapped,
                                size_t as_ss_idx);

  /**
   * Returns a pointer to delayed local outgoing psi (new values) for a given
   * cell/face/node/angle. Used when writing to a local FAS edge.
   */
  double* DelayedLocalOutgoingPsi(uint64_t cell_global_id,
                                  unsigned int face_id,
                                  unsigned int face_node_mapped,
                                  size_t as_ss_idx);

  /**
   * Returns a pointer to delayed non-local incoming psi (old values) for a given
   * cell/face/node/angle. Used when reading from a delayed location dependency.
   */
  double* DelayedNLUpwindPsi(uint64_t cell_global_id,
                             unsigned int face_id,
                             unsigned int face_node_mapped,
                             size_t as_ss_idx);

  /**
   * Stores delayed non-local psi data received from MPI into the flat array.
   * Used by ReceiveDelayedData() in the async communicator.
   */
  void StoreDelayedNonlocalData(uint64_t cell_global_id,
                                unsigned int face_id,
                                const double* data,
                                size_t data_size);

  void ClearLocalAndReceivePsi() override { deplocs_outgoing_messages_.clear(); }
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocateDelayedLocalPsi() override;
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override;

  void SetDelayedLocalPsiOldToNew() override;
  void SetDelayedLocalPsiNewToOld() override;

  void SetDelayedOutgoingPsiOldToNew() override;
  void SetDelayedOutgoingPsiNewToOld() override;

  // cell_global_id, face_id
  using CellFaceKey = std::pair<uint64_t, unsigned int>;

  std::map<CellFaceKey, std::vector<double>>& GetDeplocsOutgoingMessages()
  {
    return deplocs_outgoing_messages_;
  }

protected:
  const CBC_FLUDSCommonData& common_data_;
  const UnknownManager& psi_uk_man_;
  const SpatialDiscretization& sdm_;
  size_t num_angles_in_gs_quadrature_;
  size_t num_quadrature_local_dofs_;
  size_t num_local_spatial_dofs_;
  size_t local_psi_data_size_;

  /**
   * Layout for storage for local angular fluxes:
   * spatial DOF major -> angle in angleset major -> group in groupset major
   */
  std::vector<double> local_psi_data_;

  std::vector<std::vector<double>> boundryI_incoming_psi_;

  std::map<CellFaceKey, std::vector<double>> deplocs_outgoing_messages_;

  // --- Delayed local psi (local FAS edges) ---
  // Flat arrays backing the base class span views
  std::vector<double> delayed_local_psi_data_;
  std::vector<double> delayed_local_psi_old_data_;

  /// Lookup from (cell_global_id, face_id) to offset in delayed_local_psi_data_
  struct DelayedFaceInfo
  {
    size_t offset;
    size_t num_face_nodes;
  };
  std::map<CellFaceKey, DelayedFaceInfo> delayed_local_face_lookup_;

  // --- Delayed nonlocal psi (delayed location dependencies) ---
  // Per-dependency flat arrays backing the base class span views
  std::vector<std::vector<double>> delayed_prelocI_psi_data_;
  std::vector<std::vector<double>> delayed_prelocI_psi_old_data_;

  /// Lookup from (cell_global_id, face_id) to (dep_idx, offset) in delayed_prelocI_psi_data_
  struct DelayedNLFaceInfo
  {
    size_t dep_idx;
    size_t offset;
    size_t num_face_nodes;
  };
  std::map<CellFaceKey, DelayedNLFaceInfo> delayed_nonlocal_face_lookup_;

  /// Helper to update span views from vector-of-vectors
  static void UpdateRange(std::vector<std::vector<double>>& data,
                          std::vector<std::span<double>>& spans);
};

} // namespace opensn

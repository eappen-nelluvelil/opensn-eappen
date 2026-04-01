// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "framework/math/unknown_manager/unknown_manager.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include <cstddef>
#include <unordered_map>
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

  double* UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx);
    double* OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx);

    double* DelayedLocalUpwindPsi(const Cell& face_neighbor,
                                  unsigned int adj_cell_node,
                                  size_t as_ss_idx);
    double* DelayedLocalOutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx);

    double* NLUpwindPsi(uint64_t cell_global_id,
                        unsigned int face_id,
                        unsigned int face_node_mapped,
                        size_t as_ss_idx);

    double* DelayedNLUpwindPsi(const CBC_FLUDSCommonData::DelayedNonlocalFaceInfo& info,
                               unsigned int face_node_mapped,
                               size_t as_ss_idx);

    double*
    NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx);

    void StoreIncomingFaceData(uint64_t cell_global_id,
                               unsigned int face_id,
                               std::vector<double>&& psi_data);

    void StoreDelayedIncomingFaceData(uint64_t cell_global_id,
                                      unsigned int face_id,
                                      const double* psi_data,
                                      size_t data_size);

    void ClearLocalAndReceivePsi() override { deplocs_outgoing_messages_.clear(); }
    void ClearSendPsi() override {}
    void AllocateInternalLocalPsi() override {}
    void AllocateOutgoingPsi() override {}

    void AllocateDelayedLocalPsi() override;
    void AllocatePrelocIOutgoingPsi() override {}
    void AllocateDelayedPrelocIOutgoingPsi() override;

    void SetDelayedOutgoingPsiOldToNew() override;
    void SetDelayedOutgoingPsiNewToOld() override;
    void SetDelayedLocalPsiOldToNew() override;
    void SetDelayedLocalPsiNewToOld() override;

  protected:
    const CBC_FLUDSCommonData& common_data_;
    const UnknownManager& psi_uk_man_;
    const SpatialDiscretization& sdm_;
    size_t num_angles_in_gs_quadrature_;
    size_t num_quadrature_local_dofs_;
    size_t num_local_spatial_dofs_;
    size_t local_psi_data_size_;

    std::vector<double> local_psi_data_;
    std::vector<double> delayed_local_psi_data_;
    std::vector<double> delayed_local_psi_old_data_;
    std::vector<std::vector<double>> delayed_preloc_outgoing_psi_;
    std::vector<std::vector<double>> delayed_preloc_outgoing_psi_old_;

    std::vector<std::vector<double>> boundryI_incoming_psi_;
    std::vector<size_t> cell_psi_start_;
};

} // namespace opensn

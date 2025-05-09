// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/fluds/fluds.h"
#include <map>
#include <functional>

namespace opensn
{

class UnknownManager;
class SpatialDiscretization;
class Cell;

class CBC_FLUDS : public FLUDS
{
public:
  CBC_FLUDS(size_t num_groups, // Number of groups in this AngleSet's LBSGroupset
            size_t num_angles, // Number of angles in THIS specific AngleSet
            const CBC_FLUDSCommonData& common_data,
            const UnknownManager&
              psi_uk_man, // LBSGroupset's psi_uk_man (used for context/logging in constructor)
            const SpatialDiscretization& sdm);

  const FLUDSCommonData& GetCommonData() const;

  // --- Methods for local angular flux data ---

  // GetLocalUpwindPsi returns a base pointer to the start of an upwind neighbor cell's
  // data block within the compact local_psi_data_.
  // The caller (CbcSweepChunk) adds a relative offset for the specific node/angle.
  const double* GetLocalUpwindPsi(const Cell& face_neighbor) const;

  // GetLocalDownwindPsi returns a base pointer to the start of the current cell's
  // data block within the compact local_psi_data_ for writing.
  // The caller (CbcSweepChunk) adds a relative offset for the specific node/angle.
  double* GetLocalDownwindPsi(const Cell& cell);

  // --- Methods for non-local (remote) angular flux data ---

  const std::vector<double>& GetNonLocalUpwindData(uint64_t cell_global_id,
                                                   unsigned int face_id) const;
  // GetNonLocalUpwindPsi interprets a received multi-angle packet
  // (for all angles in this AngleSet, for all nodes on a face)
  // and returns a pointer to the data for a specific angle and face node.
  const double*
  GetNonLocalUpwindPsi(const std::vector<double>& psi_data, // Received multi-AngleSet-angle packet
                       unsigned int face_node_mapped,       // Node index on the face
                       unsigned int angle_set_index); // Local index of angle within this AngleSet

  void ClearLocalAndReceivePsi() override { deplocs_outgoing_messages_.clear(); }
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi(size_t num_grps, size_t num_angles) override {}
  void AllocateOutgoingPsi(size_t num_grps, size_t num_angles, size_t num_loc_sucs) override {}

  void AllocateDelayedLocalPsi(size_t num_grps, size_t num_angles) override {}
  void AllocatePrelocIOutgoingPsi(size_t num_grps, size_t num_angles, size_t num_loc_deps) override
  {
  }
  void AllocateDelayedPrelocIOutgoingPsi(size_t num_grps,
                                         size_t num_angles,
                                         size_t num_loc_deps) override
  {
  }

  std::vector<double>& DelayedLocalPsi() override { return delayed_local_psi_; }
  std::vector<double>& DelayedLocalPsiOld() override { return delayed_local_psi_old_; }

  std::vector<std::vector<double>>& DeplocIOutgoingPsi() override { return deplocI_outgoing_psi_; }

  std::vector<std::vector<double>>& PrelocIOutgoingPsi() override { return prelocI_outgoing_psi_; }

  std::vector<std::vector<double>>& DelayedPrelocIOutgoingPsi() override
  {
    return delayed_prelocI_outgoing_psi_;
  }
  std::vector<std::vector<double>>& DelayedPrelocIOutgoingPsiOld() override
  {
    return delayed_prelocI_outgoing_psi_old_;
  }

  // For managing messages received from remote processors
  using CellFaceKey = std::pair<uint64_t, unsigned int>;

  std::map<CellFaceKey, std::vector<double>>& GetDeplocsOutgoingMessages()
  {
    return deplocs_outgoing_messages_;
  }

  // Accessors for the primary local angular flux data buffer
  // For reading
  const std::vector<double>& GetLocalPsiData() const { return local_psi_data_; }

  // For writing (direct indexing, if needed)
  std::vector<double>& GetLocalPsiData() { return local_psi_data_; }

private:
  const CBC_FLUDSCommonData& common_data_;
  std::vector<double> local_psi_data_; // Now optimally sized for angles in THIS AngleSet
  const UnknownManager& psi_uk_man_;   // LBSGroupset's psi_uk_man (stored, but not used for
                                       // local_psi_data_ sizing/indexing)
  const SpatialDiscretization& sdm_;

  std::vector<double> delayed_local_psi_;
  std::vector<double> delayed_local_psi_old_;
  std::vector<std::vector<double>> deplocI_outgoing_psi_;
  std::vector<std::vector<double>> prelocI_outgoing_psi_;
  std::vector<std::vector<double>> boundryI_incoming_psi_;

  std::vector<std::vector<double>> delayed_prelocI_outgoing_psi_;
  std::vector<std::vector<double>> delayed_prelocI_outgoing_psi_old_;

  std::map<CellFaceKey, std::vector<double>> deplocs_outgoing_messages_;
};

} // namespace opensn

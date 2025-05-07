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
  CBC_FLUDS(size_t num_groups,
            size_t num_angles,
            const CBC_FLUDSCommonData& common_data,
            const UnknownManager& psi_uk_man,
            const SpatialDiscretization& sdm);

  const FLUDSCommonData& GetCommonData() const;

  // OLD METHOD: deprecate at some point
  // const std::vector<double>& GetLocalUpwindDataBlock() const;

  // OLD METHOD: deprecate at some point
  // const double* GetLocalCellUpwindPsi(const std::vector<double>& psi_data_block, const Cell& cell);

  // const double* GetLocalUpwindPsi(const Cell& face_neighbor,
                                  // const unsigned int adj_cell_node_offset) const;

  // double* GetLocalDownwindPsi(const Cell& cell);

  const std::vector<double>& GetNonLocalUpwindData(uint64_t cell_global_id,
                                                   unsigned int face_id) const;

  const double* GetNonLocalUpwindPsi(const std::vector<double>& psi_data,
                                     unsigned int face_node_mapped,
                                     unsigned int angle_set_index);

  // --- NEW:
  // Private mapping for compact local_psi_data_
  size_t MapDOFCompactLocal(const Cell& cell,
    unsigned int node_in_cell,
    unsigned int angle_idx_ss, // 0 to num_angles_in_set_ - 1
    unsigned int group_idx_gs // 0 to num_groups_ - 1
  ) const;

  // --- NEW:
  const double* GetLocalUpwindPsi_Compact(const Cell& upwind_cell,
                                         unsigned int upwind_node_in_cell,
                                         unsigned int angle_idx_ss) const;  // Angle index within current angle set

  // --- NEW:
  double* GetLocalDownwindPsi_Compact(const Cell& current_cell,
                                      unsigned int current_node_in_cell,
                                      unsigned int angle_idx_ss); // Angle index within current angle set

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

  // cell_global_id, face_id
  using CellFaceKey = std::pair<uint64_t, unsigned int>;

  std::map<CellFaceKey, std::vector<double>>& GetDeplocsOutgoingMessages()
  {
    return deplocs_outgoing_messages_;
  }

  // Const getter (for reading from the subset)
  const std::vector<double>& GetLocalPsiData() const { return local_psi_data_; }

  // Non-const getter (for writing into the subset)
  std::vector<double>& GetLocalPsiData() { return local_psi_data_; }

private:
  const CBC_FLUDSCommonData& common_data_;
  std::vector<double> local_psi_data_;
  const UnknownManager& psi_uk_man_;
  const SpatialDiscretization& sdm_;

  // --- NEW:
  const size_t num_angles_in_set_;

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

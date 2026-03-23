// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include <cstddef>
#include <unordered_map>
#include <functional>

namespace opensn
{

class UnknownManager;
class SpatialDiscretization;
class Cell;

class CBC_FLUDS : public FLUDS
{
public:
  CBC_FLUDS(unsigned int num_groups,
            size_t num_angles,
            const CBC_FLUDSCommonData& common_data,
            const UnknownManager& psi_uk_man,
            const SpatialDiscretization& sdm);

  const FLUDSCommonData& GetCommonData() const;

  double* UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx);

  double* OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx);

  double* NLUpwindPsi(uint64_t cell_global_id,
                      unsigned int face_id,
                      unsigned int face_node_mapped,
                      size_t as_ss_idx);

  double*
  NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx);

  void ClearLocalAndReceivePsi() override { deplocs_outgoing_messages_.clear(); }
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocateDelayedLocalPsi() override {}
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override {}

  // cell_global_id, face_id
  using CellFaceKey = std::pair<uint64_t, unsigned int>;

  struct CellFaceKeyHash
  {
    size_t operator()(const CellFaceKey& k) const
    {
      return std::hash<uint64_t>()(k.first) ^ (std::hash<unsigned int>()(k.second) << 32);
    }
  };

  std::unordered_map<CellFaceKey, std::vector<double>, CellFaceKeyHash>&
  GetDeplocsOutgoingMessages()
  {
    return deplocs_outgoing_messages_;
  }

private:
  const CBC_FLUDSCommonData& common_data_;
  const UnknownManager& psi_uk_man_;
  const SpatialDiscretization& sdm_;
  size_t num_angles_in_gs_quadrature_;
  size_t num_local_spatial_dofs_;
  size_t local_psi_data_size_;

  std::vector<double> local_psi_data_;

  /// Pre-computed mapping: cell_local_id -> start index in local_psi_data_.
  std::vector<size_t> cell_psi_data_start_;

  std::unordered_map<CellFaceKey, std::vector<double>, CellFaceKeyHash> deplocs_outgoing_messages_;
};

} // namespace opensn

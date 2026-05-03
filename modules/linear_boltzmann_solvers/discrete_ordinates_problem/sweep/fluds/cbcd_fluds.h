// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "framework/mesh/cell/cell.h"
#include "caribou/main.hpp"
#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace crb = caribou;

namespace opensn
{

class CBCD_AngleSet;
class UnknownManager;
class SpatialDiscretization;
class Cell;
class CBCDSweepChunk;

/// CBCD FLUDS.
class CBCD_FLUDS : public FLUDS
{
public:
  CBCD_FLUDS(size_t num_groups,
             size_t num_angles,
             size_t num_local_cells,
             const CBCD_FLUDSCommonData& common_data,
             const UnknownManager& psi_uk_man,
             const SpatialDiscretization& sdm,
             bool save_angular_flux);

  ~CBCD_FLUDS();

  const CBCD_FLUDSCommonData& GetCommonData() const { return common_data_; }

  crb::Stream& GetStream() { return stream_; }

  void AllocateLocalAndSavedPsi();

  inline std::size_t GetStrideSize() const { return num_groups_and_angles_; }

  crb::MappedHostVector<std::uint32_t>& GetLocalCellIDs() { return local_cell_ids_; }

  double* GetSavedAngularFluxDevicePointer() { return device_saved_psi_.get(); }

  void CopySavedPsiFromDevice();

  void CopySavedPsiToDestinationPsi(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  CBCD_FLUDSPointerSet& GetDevicePointerSet() { return pointer_set_; }

  void CopyIncomingBoundaryPsiToDevice(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  void CopyIncomingNonlocalPsiToDevice(CBCD_AngleSet* angle_set,
                                       const std::vector<std::uint32_t>& cell_local_ids);

  void CopyOutgoingPsiBackToHost(CBCDSweepChunk& sweep_chunk,
                                 CBCD_AngleSet* angle_set,
                                 const std::vector<std::uint32_t>& cell_local_ids);

  double* NLUpwindPsi(uint64_t cell_global_id,
                      unsigned int face_id,
                      unsigned int face_node_mapped,
                      size_t as_ss_idx);

  std::unordered_map<CellFaceKey, std::vector<double>>& GetDeplocsOutgoingMessages()
  {
    return deplocs_outgoing_messages_;
  }

  double*
  NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx);

  void ClearLocalAndReceivePsi() override { deplocs_outgoing_messages_.clear(); }
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocateDelayedLocalPsi() override {}
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override {}

private:
  const CBCD_FLUDSCommonData& common_data_;
  const UnknownManager& psi_uk_man_;
  const SpatialDiscretization& sdm_;
  size_t num_angles_in_gs_quadrature_;
  size_t num_quadrature_local_dofs_;
  size_t num_local_spatial_dofs_;
  size_t local_psi_data_size_;
  std::vector<BoundaryNodeInfo> incoming_boundary_node_map_;
  std::map<std::uint64_t, std::vector<BoundaryNodeInfo>> cell_to_outgoing_boundary_nodes_;
  std::map<std::uint64_t, std::vector<NonlocalNodeInfo>> cell_to_incoming_nonlocal_nodes_;
  std::map<std::uint64_t, std::vector<NonlocalNodeInfo>> cell_to_outgoing_nonlocal_nodes_;
  crb::MappedHostVector<double> incoming_boundary_psi_;
  crb::MappedHostVector<double> outgoing_boundary_psi_;
  crb::MappedHostVector<double> incoming_nonlocal_psi_;
  crb::MappedHostVector<double> outgoing_nonlocal_psi_;
  crb::Stream stream_ = crb::Stream::get_null_stream();
  crb::MappedHostVector<std::uint32_t> local_cell_ids_;
  bool save_angular_flux_;
  crb::DeviceMemory<double> local_psi_;
  crb::DeviceMemory<double> device_saved_psi_;
  crb::HostVector<double> host_saved_psi_;
  CBCD_FLUDSPointerSet pointer_set_;
  std::unordered_map<CellFaceKey, std::vector<double>> deplocs_outgoing_messages_;

  void CreatePointerSet();

  std::vector<std::vector<double>> boundaryI_incoming_psi_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "caribou/main.hpp"
#include <array>
#include <cstddef>
#include <span>

namespace crb = caribou;

namespace opensn
{

class CBC_SPDS;
class CBCD_AngleSet;
class CBCD_AsynchronousCommunicator;
class CBCDSweepChunk;
class UnknownManager;
class SpatialDiscretization;
class SweepBoundary;
class MeshContinuum;

/// Device CBC FLUDS.
class CBCD_FLUDS : public FLUDS
{
public:
  CBCD_FLUDS(std::size_t num_groups,
             std::size_t num_angles,
             std::size_t num_local_cells,
             const CBCD_FLUDSCommonData& common_data,
             const UnknownManager& psi_uk_man,
             const SpatialDiscretization& sdm,
             bool save_angular_flux);

  ~CBCD_FLUDS();

  const CBCD_FLUDSCommonData& GetCommonData() const { return common_data_; }

  crb::Stream& GetStream() { return stream_; }

  std::size_t GetLocalPsiBytes() const noexcept { return local_psi_data_size_ * sizeof(double); }

  void AllocateLocalAndSavedPsi();

  void InitializeReflectingBoundaryNodes(
    const std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries);

  inline std::size_t GetStrideSize() const { return num_groups_and_angles_; }

  crb::MappedHostVector<std::uint32_t>& GetLocalCellIDs(const std::size_t buffer_index)
  {
    return local_cell_ids_[buffer_index];
  }

  double* GetSavedAngularFluxDevicePointer() { return device_saved_psi_.get(); }

  void CopySavedPsiFromDevice();

  void CopySavedPsiToDestinationPsi(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  CBCD_FLUDSPointerSet& GetDevicePointerSet() { return pointer_set_; }

  void CopyIncomingBoundaryPsiToDevice(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  /// Copy a completed batch's outgoing psi to host-visible destinations.
  void CopyOutgoingPsiBackToHost(CBCDSweepChunk& sweep_chunk,
                                 CBCD_AsynchronousCommunicator& async_comm,
                                 std::size_t producer_id,
                                 std::size_t angle_set_id,
                                 const std::vector<std::uint32_t>& angle_indices,
                                 std::span<const std::uint32_t> cell_local_ids);

  /// Scatter one received nonlocal face and return its local cell ID.
  std::uint32_t ScatterReceivedFaceData(std::uint32_t source_slot,
                                        std::uint32_t source_face_index,
                                        const double* psi_data);

  void ClearLocalAndReceivePsi() override;
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocatePrelocIOutgoingPsi() override {}

  std::span<const ReflectingBoundaryFacePlan>
  GetReflectingOutgoingBoundaryFaces(const std::uint64_t cell_local_id) const
  {
    const auto begin = reflecting_outgoing_boundary_face_offsets_[cell_local_id];
    const auto end = reflecting_outgoing_boundary_face_offsets_[cell_local_id + 1];
    return {reflecting_boundary_face_plans_.data() + begin, end - begin};
  }

private:
  const CBCD_FLUDSCommonData& common_data_;
  const CBC_SPDS& cbc_spds_;
  const UnknownManager& psi_uk_man_;
  const SpatialDiscretization& sdm_;
  std::size_t num_local_spatial_dofs_;
  std::size_t local_psi_data_size_;
  std::size_t saved_psi_data_size_;
  const MeshContinuum* grid_ptr_ = nullptr;
  /// Mapped host vectors for boundary and non-local angular fluxes.
  crb::MappedHostVector<double> incoming_boundary_psi_;
  crb::MappedHostVector<double> outgoing_boundary_psi_;
  crb::MappedHostVector<double> incoming_nonlocal_psi_;
  crb::MappedHostVector<double> outgoing_nonlocal_psi_;
  /// Associated angleset's stream.
  crb::Stream stream_;
  std::array<crb::MappedHostVector<std::uint32_t>, 3> local_cell_ids_;
  bool save_angular_flux_;
  /// Device storage for local angular fluxes.
  crb::DeviceMemory<double> local_psi_;
  crb::DeviceMemory<double> device_saved_psi_;
  crb::HostVector<double> host_saved_psi_;
  /// Pointer set used by the CBCD sweep kernel.
  CBCD_FLUDSPointerSet pointer_set_;
  std::vector<std::uint32_t> reflecting_outgoing_boundary_face_offsets_;
  std::vector<ReflectingBoundaryFacePlan> reflecting_boundary_face_plans_;
  std::vector<OutgoingNodeMemcpy> outgoing_node_memcpy_plan_;

  void CreatePointerSet();
};

} // namespace opensn

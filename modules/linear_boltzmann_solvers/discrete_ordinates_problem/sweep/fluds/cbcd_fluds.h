// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/storage.h"
#include "caribou/main.hpp"
#include <cstddef>
#include <vector>

namespace crb = caribou;

namespace opensn
{

class CBCD_AngleSet;
class UnknownManager;
class SpatialDiscretization;
class Cell;
class MeshContinuum;
class CBCDSweepChunk;

/// CBC FLUDS for device.
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

  /// Get constant reference to CBCD_FLUDS common data.
  const CBCD_FLUDSCommonData& GetCommonData() const { return common_data_; }

  /// Get reference to stream.
  crb::Stream& GetStream() { return stream_; }

  /// Allocate buffers asynchronously on the associated stream.
  void AllocateLocalAndSavedPsi();

  /// Get the stride size for each face node's angular flux data.
  inline std::size_t GetStrideSize() const { return num_groups_and_angles_; }

  /// Get vector of local cells to be swept.
  crb::MappedHostVector<std::uint64_t>& GetLocalCellIDs() { return local_cell_ids_; }

  /// Get saved angular flux device pointer.
  double* GetSavedAngularFluxDevicePointer() { return device_saved_psi_.get(); }

  /// Copy saved psi from device to host.
  void CopySavedPsiFromDevice();

  /// Copy saved psi from host to destination psi host buffer.
  void CopySavedPsiToDestinationPsi(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  /// Gets pointer set to device angular flux data.
  CBCD_FLUDSPointerSet& GetDevicePointerSet() { return pointer_set_; }

  /// Copies incoming boundary psi from host to device.
  void CopyIncomingBoundaryPsiToDevice(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  /// Scatter received face data directly into incoming_nonlocal_psi_.
  /// Returns the local cell ID (avoids caller needing a second global→local lookup).
  uint64_t ScatterReceivedFaceData(uint64_t cell_global_id,
                                   unsigned int face_id,
                                   const std::vector<double>& psi_data);

  /// Copy outgoing psi on host after D2H copy is done.
  void CopyOutgoingPsiBackToHost(CBCDSweepChunk& sweep_chunk,
                                 CBCD_AngleSet* angle_set,
                                 const std::vector<std::uint64_t>& cell_local_ids);

  void ClearLocalAndReceivePsi() override {}
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}
  void AllocateDelayedLocalPsi() override {}
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override {}

  /// Get the outgoing boundary node map (indexed by cell_local_id).
  const std::vector<std::vector<BoundaryNodeInfo>>& GetOutgoingBoundaryNodeMap() const
  {
    return common_data_.GetOutgoingBoundaryNodeMap();
  }

  size_t GetNumOutgoingFaces() const { return num_outgoing_faces_; }
  size_t GetNumIncomingFaces() const { return num_incoming_faces_; }

private:
  /// Reference to the common data.
  const CBCD_FLUDSCommonData& common_data_;
  /// Unknown manager for psi.
  const UnknownManager& psi_uk_man_;
  /// Spatial discretization reference.
  const SpatialDiscretization& sdm_;
  /// Size computation helpers (pulled up from CBC_FLUDS).
  size_t num_angles_in_gs_quadrature_;
  size_t num_quadrature_local_dofs_;
  size_t num_local_spatial_dofs_;
  size_t local_psi_data_size_;
  /// Cached grid pointer — avoids shared_ptr copy on the hot path.
  const MeshContinuum* grid_ptr_;
  /// Mapped host vectors for boundary and non-local angular fluxes.
  crb::MappedHostVector<double> incoming_boundary_psi_;
  crb::MappedHostVector<double> outgoing_boundary_psi_;
  crb::MappedHostVector<double> incoming_nonlocal_psi_;
  crb::MappedHostVector<double> outgoing_nonlocal_psi_;
  /// Associated angleset's stream.
  crb::Stream stream_;
  crb::MappedHostVector<std::uint64_t> local_cell_ids_;
  bool save_angular_flux_;
  /// Device storage for local angular fluxes.
  crb::DeviceMemory<double> local_psi_;
  /// Host and device buffers for saved angular fluxes.
  crb::DeviceMemory<double> device_saved_psi_;
  crb::HostVector<double> host_saved_psi_;
  /// Pointer set to device angular flux data
  CBCD_FLUDSPointerSet pointer_set_;

  /// Pre-computed face-grouped outgoing nonlocal nodes (indexed by cell_local_id).
  struct FaceOutgoingInfo
  {
    unsigned int face_id;
    std::vector<const NonlocalNodeInfo*> nodes;
    size_t face_data_size;        ///< num_face_nodes * num_groups_and_angles_
    int locality;                 ///< Destination MPI rank (neighbor partition ID).
    uint64_t neighbor_global_id;  ///< Global ID of the neighbor cell across this face.
    unsigned int associated_face; ///< Face index on the neighbor cell.
  };
  std::vector<std::vector<FaceOutgoingInfo>> cell_to_face_grouped_outgoing_;

  /// Pre-computed face-grouped incoming nonlocal nodes (indexed by cell_local_id).
  struct FaceIncomingInfo
  {
    unsigned int face_id;
    std::vector<const NonlocalNodeInfo*> nodes;
  };
  std::vector<std::vector<FaceIncomingInfo>> cell_to_face_grouped_incoming_;

  size_t num_outgoing_faces_ = 0;
  size_t num_incoming_faces_ = 0;

  /// Fast global→local cell ID lookup for incoming nonlocal cells only.
  /// Uses unordered_map (O(1) amortized) instead of std::map (O(log n)) on the hot path.
  std::unordered_map<uint64_t, uint64_t> incoming_global_to_local_;

  /// Creates device pointer set to the local, boundary, and non-local angular flux buffers.
  void CreatePointerSet();
};

} // namespace opensn

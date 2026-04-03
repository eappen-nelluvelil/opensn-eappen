// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/storage.h"
#include "framework/data_types/byte_array.h"
#include "caribou/main.hpp"
#include <cstddef>
#include <limits>
#include <span>

namespace crb = caribou;

namespace opensn
{

class CBCD_AngleSet;
class CBCD_AsynchronousCommunicator;
class UnknownManager;
class SpatialDiscretization;
class CBCDSweepChunk;

/// CBC FLUDS for device.
class CBCD_FLUDS : public FLUDS
{
public:
  static constexpr std::uint32_t INVALID_SLOT_OFFSET = std::numeric_limits<std::uint32_t>::max();

  CBCD_FLUDS(size_t num_groups,
             size_t num_angles,
             size_t num_local_cells,
             const CBCD_FLUDSCommonData& common_data,
             const UnknownManager& psi_uk_man,
             const SpatialDiscretization& sdm,
             bool save_angular_flux);

  ~CBCD_FLUDS();

  /// Get reference to the common data.
  const CBCD_FLUDSCommonData& GetCommonData() const { return common_data_; }

  /// Get reference to stream.
  crb::Stream& GetStream() { return stream_; }

  /// Allocate buffers asynchronously on the associated stream.
  void AllocateLocalAndSavedPsi();

  /// Resolve outgoing queue indices once the aggregated communicator exists.
  void InitializeQueueIndices(const CBCD_AsynchronousCommunicator& async_comm);

  /// Get the stride size for each face node's angular flux data.
  inline std::size_t GetStrideSize() const { return num_groups_and_angles_; }

  /// Get vector of local cells to be swept.
  crb::MappedHostVector<std::uint32_t>& GetLocalCellIDs() { return local_cell_ids_; }

  void AllocateSlots(const std::vector<std::uint32_t>& cell_local_ids);
  void DeallocateSlots(const std::vector<std::uint32_t>& cell_local_ids);

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

  /// Copies incoming non-local psi from host to device.
  void CopyIncomingNonlocalPsiToDevice(CBCD_AngleSet* angle_set,
                                       const std::vector<std::uint32_t>& cell_local_ids);

  /// Copy outgoing psi on host after D2H copy is done.
  void CopyOutgoingPsiBackToHost(CBCDSweepChunk& sweep_chunk,
                                 CBCD_AngleSet* angle_set,
                                 const std::vector<std::uint32_t>& cell_local_ids);

  /// Scatter one received nonlocal face payload directly into the mapped incoming buffer.
  std::uint64_t ScatterReceivedFaceData(std::uint64_t cell_global_id,
                                        unsigned int face_id,
                                        const double* psi_data);

  void ClearLocalAndReceivePsi() override;
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocateDelayedLocalPsi() override {}
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override {}

private:
  /// Reference to the common data.
  const CBCD_FLUDSCommonData& common_data_;
  const UnknownManager& psi_uk_man_;
  const SpatialDiscretization& sdm_;
  size_t num_angles_in_gs_quadrature_;
  size_t num_quadrature_local_dofs_;
  size_t num_local_spatial_dofs_;
  size_t num_local_psi_slots_;
  size_t local_psi_slot_stride_;
  size_t local_psi_data_size_;
  size_t saved_psi_data_size_;
  std::vector<BoundaryNodeInfo> incoming_boundary_node_map_;
  /// Mapped host vectors for boundary and non-local angular fluxes.
  crb::MappedHostVector<double> incoming_boundary_psi_;
  crb::MappedHostVector<double> outgoing_boundary_psi_;
  crb::MappedHostVector<double> incoming_nonlocal_psi_;
  crb::MappedHostVector<double> outgoing_nonlocal_psi_;
  /// Associated angleset's stream.
  crb::Stream stream_;
  crb::MappedHostVector<std::uint32_t> local_cell_ids_;
  crb::MappedHostVector<std::uint32_t> local_slot_offsets_;
  bool save_angular_flux_;
  /// Device storage for local angular fluxes.
  crb::DeviceMemory<double> local_psi_;
  /// Host and device buffers for saved angular fluxes.
  crb::DeviceMemory<double> device_saved_psi_;
  crb::HostVector<double> host_saved_psi_;
  /// Pointer set to device angular flux data
  CBCD_FLUDSPointerSet pointer_set_;
  std::vector<std::uint32_t> free_slot_stack_;
  /// Ordered outgoing destination metadata.
  struct OutgoingDestination
  {
    int locality = 0;
    int queue_index = -1;
  };
  std::vector<OutgoingDestination> outgoing_destinations_;
  /// Per-destination face counts for the current pack pass.
  std::vector<size_t> scratch_dest_face_counts_;
  /// Per-destination touched flags for the current pack pass.
  std::vector<std::uint8_t> scratch_dest_touched_;
  /// Destinations touched during the current pack pass.
  std::vector<std::uint32_t> active_dest_indices_;
  /// Reusable destination buffers for outgoing wire-format sections.
  std::vector<ByteArray> dest_buffers_;
  /// Flat byte-level memcpy descriptors referenced by outgoing faces.
  std::vector<OutgoingNodeMemcpy> outgoing_node_memcpy_plan_;
  /// Packed payload size, in doubles, for each grouped outgoing nonlocal face.
  std::vector<std::size_t> outgoing_face_payload_sizes_;

  /// Creates device pointer set to the local, boundary, and non-local angular flux buffers.
  void CreatePointerSet();
};

} // namespace opensn

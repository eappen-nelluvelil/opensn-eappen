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
#include <map>
#include <span>

namespace crb = caribou;

namespace opensn
{

class CBCD_AngleSet;
class CBCD_AsynchronousCommunicator;
class UnknownManager;
class SpatialDiscretization;
class CBCDSweepChunk;
class SweepBoundary;
class MeshContinuum;

/**
 * Device-side flux data structures for the cell-by-cell (CBCD) sweep algorithm.
 *
 * Manages GPU-resident and mapped-host angular-flux buffers for local, nonlocal,
 * and boundary face data during a CBCD sweep. The local angular-flux buffer is
 * allocated on the device with \f$s^*\f$ slots determined by the CBC_SPDS static
 * slot assignment, analogous to the host-side CBC_FLUDS but using device memory
 * (caribou::DeviceMemory) and mapped host vectors (caribou::MappedHostVector) for
 * zero-copy CPU-GPU data transfer.
 *
 * ## Data flow
 *
 * 1. **Incoming boundary/nonlocal psi:** Copied from host to device before
 *    kernel launch via CopyIncomingBoundaryPsiToDevice.
 * 2. **Local psi:** Computed and consumed entirely on-device using slot-based
 *    addressing through CBCD_FLUDSPointerSet.
 * 3. **Outgoing nonlocal psi:** Copied from device to host after kernel
 *    completion via CopyOutgoingPsiBackToHost, then packed into wire-format
 *    ByteArray sections and enqueued to the aggregated communicator.
 * 4. **Saved psi:** If angular-flux storage is enabled, the device kernel writes
 *    to device_saved_psi_, which is later copied to the host destination vector.
 *
 * ## Outgoing buffer management
 *
 * Each outgoing destination locality has a reusable ByteArray (dest_buffers_)
 * with pre-reserved capacity (dest_buffer_capacities_). The swap-and-re-reserve
 * pattern in CopyOutgoingPsiBackToHost preserves allocation across batches,
 * avoiding geometric-growth reallocations.
 */
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

  /// Get reference to the common data.
  const CBCD_FLUDSCommonData& GetCommonData() const { return common_data_; }

  /// Get reference to stream.
  crb::Stream& GetStream() { return stream_; }

  /// Bytes in the local psi backing buffer for this FLUDS instance.
  size_t GetLocalPsiBytes() const noexcept { return local_psi_data_size_ * sizeof(double); }

  /// Allocate buffers asynchronously on the associated stream.
  void AllocateLocalAndSavedPsi();

  /// Resolve outgoing queue indices once the aggregated communicator exists.
  void InitializeQueueIndices(const CBCD_AsynchronousCommunicator& async_comm);

  /// Build reflecting-boundary copy plans for this angle set.
  void InitializeReflectingBoundaryNodes(
    const std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries);

  /// Get the stride size for each face node's angular flux data.
  inline std::size_t GetStrideSize() const { return num_groups_and_angles_; }

  /// Get vector of local cells to be swept.
  crb::MappedHostVector<std::uint32_t>& GetLocalCellIDs() { return local_cell_ids_; }

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

  /// Copy outgoing psi on host after D2H copy is done.
  void CopyOutgoingPsiBackToHost(CBCDSweepChunk& sweep_chunk,
                                 CBCD_AngleSet* angle_set,
                                 const std::vector<std::uint32_t>& cell_local_ids);

  /// Scatter one received nonlocal face payload directly into the mapped incoming buffer.
  std::uint32_t ScatterReceivedFaceData(std::uint64_t cell_global_id,
                                        unsigned int face_id,
                                        const double* psi_data);

  void ClearLocalAndReceivePsi() override;
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocateDelayedLocalPsi() override {}
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override {}

  std::span<const ReflectingBoundaryFacePlan>
  GetReflectingOutgoingBoundaryFaces(const std::uint64_t cell_local_id) const
  {
    const auto begin = reflecting_outgoing_boundary_face_offsets_[cell_local_id];
    const auto end = reflecting_outgoing_boundary_face_offsets_[cell_local_id + 1];
    return {reflecting_boundary_face_plans_.data() + begin, end - begin};
  }

private:
  /// Reference to the common data.
  const CBCD_FLUDSCommonData& common_data_;
  const UnknownManager& psi_uk_man_;
  const SpatialDiscretization& sdm_;
  size_t num_local_spatial_dofs_;
  size_t local_psi_data_size_;
  size_t saved_psi_data_size_;
  const MeshContinuum* grid_ptr_ = nullptr;
  /// Mapped host vectors for boundary and non-local angular fluxes.
  crb::MappedHostVector<double> incoming_boundary_psi_;
  crb::MappedHostVector<double> outgoing_boundary_psi_;
  crb::MappedHostVector<double> incoming_nonlocal_psi_;
  crb::MappedHostVector<double> outgoing_nonlocal_psi_;
  /// Associated angleset's stream.
  crb::Stream stream_;
  crb::MappedHostVector<std::uint32_t> local_cell_ids_;
  bool save_angular_flux_;
  /// Device storage for local angular fluxes.
  crb::DeviceMemory<double> local_psi_;
  /// Host and device buffers for saved angular fluxes.
  crb::DeviceMemory<double> device_saved_psi_;
  crb::HostVector<double> host_saved_psi_;
  /// Pointer set to device angular flux data
  CBCD_FLUDSPointerSet pointer_set_;
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
  /// Exact per-destination section capacities in bytes.
  std::vector<std::size_t> dest_buffer_capacities_;
  /// Cell-to-reflecting-face offset table.
  std::vector<std::uint32_t> reflecting_outgoing_boundary_face_offsets_;
  /// Flat reflecting-boundary face plans.
  std::vector<ReflectingBoundaryFacePlan> reflecting_boundary_face_plans_;
  /// Flat byte-level memcpy descriptors referenced by outgoing faces.
  std::vector<OutgoingNodeMemcpy> outgoing_node_memcpy_plan_;

  /// Creates device pointer set to the local, boundary, and non-local angular flux buffers.
  void CreatePointerSet();
};

} // namespace opensn

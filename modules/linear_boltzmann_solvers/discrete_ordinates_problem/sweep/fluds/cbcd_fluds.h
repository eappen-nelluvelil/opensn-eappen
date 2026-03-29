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
#include <memory>
#include <vector>

namespace crb = caribou;

namespace opensn
{

class CBCD_AngleSet;
class CBCD_AggregatedCommunicator;
class SweepBoundary;
class UnknownManager;
class SpatialDiscretization;
class MeshContinuum;
class CBCDSweepChunk;

/**
 * Per-angle-set CBCD FLUDS storage and host/device transfer helper.
 *
 * Each instance owns the zero-copy host buffers and device buffers needed by
 * one `CBCD_AngleSet`. Shared, angle-set-independent topology is stored in
 * `CBCD_FLUDSCommonData`; this class adds the mutable per-angle-set state used
 * for GPU launches, MPI packing, boundary exchange, and optional saved angular
 * flux storage.
 */
class CBCD_FLUDS : public FLUDS
{
public:
  /// Construct per-angle-set CBCD FLUDS storage.
  ///
  /// \param num_groups Number of energy groups.
  /// \param num_angles Number of angles in the angle set.
  /// \param num_local_cells Number of local cells.
  /// \param common_data Shared topology data.
  /// \param psi_uk_man Angular-flux unknown manager.
  /// \param sdm Spatial discretization.
  /// \param save_angular_flux Whether saved angular flux storage is needed.
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

  /// Allocate device-side local and saved-psi buffers.
  void AllocateLocalAndSavedPsi();

  /// Resolve aggregated-communicator queue indices for outgoing destinations.
  ///
  /// \param agg_comm Aggregated communicator.
  void InitializeQueueIndices(const CBCD_AggregatedCommunicator& agg_comm);

  /// Precompute reflecting outgoing-boundary nodes for this angle set.
  ///
  /// \param boundaries Sweep-boundary map.
  void InitializeReflectingBoundaryNodes(
    const std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries);

  /// Return the per-node angular-flux stride.
  std::size_t GetStrideSize() const { return num_groups_and_angles_; }

  /// Return the mapped local-cell work list for the next kernel launch.
  crb::MappedHostVector<std::uint32_t>& GetLocalCellIDs() { return local_cell_ids_; }

  /// Return the saved-angular-flux device pointer.
  double* GetSavedAngularFluxDevicePointer() { return device_saved_psi_.get(); }

  /// Copy saved angular flux back into the destination psi storage.
  ///
  /// \param sweep_chunk Owning sweep chunk.
  /// \param angle_set Owning angle set.
  void CopySavedPsiToDestinationPsi(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  /// Return the device pointer bundle used by the GPU kernel.
  CBCD_FLUDSPointerSet& GetDevicePointerSet() { return pointer_set_; }

  /// Copy incoming boundary values into the mapped boundary buffer.
  ///
  /// \param sweep_chunk Owning sweep chunk.
  /// \param angle_set Owning angle set.
  void CopyIncomingBoundaryPsiToDevice(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  /// Scatter one received nonlocal face payload into the incoming buffer.
  ///
  /// \param cell_global_id Receiving cell global identifier.
  /// \param face_id Receiving face index.
  /// \param psi_data Received psi payload.
  /// \return Local cell identifier for dependency updates.
  uint64_t ScatterReceivedFaceData(uint64_t cell_global_id,
                                   unsigned int face_id,
                                   const double* psi_data);

  /// Pack completed outgoing psi and enqueue it for communication.
  ///
  /// The method also copies reflecting-boundary values directly into the
  /// associated sweep boundary storage.
  ///
  /// \param angle_set Owning angle set.
  /// \param cell_local_ids Completed local-cell identifiers.
  void CopyOutgoingPsiBackToHost(CBCD_AngleSet* angle_set,
                                 const std::vector<std::uint64_t>& cell_local_ids);

  void ClearLocalAndReceivePsi() override {}
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}
  void AllocateDelayedLocalPsi() override {}
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override {}

  /// Return per-cell reflecting outgoing-boundary nodes for this angle set.
  const std::vector<std::vector<BoundaryNodeInfo>>& GetReflectingOutgoingBoundaryNodeMap() const
  {
    return reflecting_outgoing_boundary_nodes_;
  }

  size_t GetNumOutgoingFaces() const { return common_data_.GetNumOutgoingNonlocalFaces(); }
  size_t GetNumIncomingFaces() const { return common_data_.GetNumIncomingNonlocalFaces(); }

private:
  /// Shared topology data.
  const CBCD_FLUDSCommonData& common_data_;
  /// Unknown-manager view for destination psi layout.
  const UnknownManager& psi_uk_man_;
  /// Total local-psi allocation size in doubles.
  size_t local_psi_data_size_;
  /// Mesh pointer used for host-side destination copyback.
  const MeshContinuum* grid_ptr_;

  /// Incoming boundary psi buffer visible to host and device.
  crb::MappedHostVector<double> incoming_boundary_psi_;
  /// Outgoing boundary psi buffer visible to host and device.
  crb::MappedHostVector<double> outgoing_boundary_psi_;
  /// Incoming nonlocal psi buffer visible to host and device.
  crb::MappedHostVector<double> incoming_nonlocal_psi_;
  /// Outgoing nonlocal psi buffer visible to host and device.
  crb::MappedHostVector<double> outgoing_nonlocal_psi_;

  /// CUDA/HIP stream associated with this angle set.
  crb::Stream stream_;
  /// Local-cell work list for the next GPU launch.
  crb::MappedHostVector<std::uint32_t> local_cell_ids_;
  /// Saved-angular-flux enable flag.
  bool save_angular_flux_;

  /// Device-resident local psi storage.
  crb::DeviceMemory<double> local_psi_;
  /// Device-resident saved angular flux.
  crb::DeviceMemory<double> device_saved_psi_;
  /// Host-side saved angular flux buffer.
  crb::HostVector<double> host_saved_psi_;

  /// Device pointer bundle passed to the sweep kernel.
  CBCD_FLUDSPointerSet pointer_set_;

  /// Outgoing destination metadata for one locality.
  struct OutgoingDestination
  {
    /// Destination partition ID.
    int locality;
    /// Resolved aggregated-communicator queue index.
    int queue_index = -1;
  };
  /// Ordered outgoing destination table.
  std::vector<OutgoingDestination> outgoing_destinations_;

  /// Per-destination face counts for the current pack pass.
  std::vector<size_t> scratch_dest_face_counts_;
  /// Per-destination psi payload sizes in bytes for the current pack pass.
  std::vector<size_t> scratch_dest_psi_bytes_;
  /// Per-destination write offsets for the current pack pass.
  std::vector<size_t> scratch_dest_offsets_;

  /// Reusable destination buffers for outgoing wire-format sections.
  std::vector<ByteArray> dest_buffers_;

  /// Per-cell reflecting outgoing-boundary nodes.
  std::vector<std::vector<BoundaryNodeInfo>> reflecting_outgoing_boundary_nodes_;

  /// Populate the device pointer bundle after allocation.
  void CreatePointerSet();
};

} // namespace opensn

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

/**
 * Per-angle-set FLUDS for cell-by-cell device (CBCD) sweep.
 *
 * Owns the MappedHostVector buffers that the GPU kernel reads/writes through
 * zero-copy mapped memory: local psi, incoming/outgoing boundary psi, and
 * incoming/outgoing non-local psi.  Also owns the device allocation for local
 * psi and (optionally) saved angular flux.
 *
 * Host-side methods handle:
 *  - Scattering received MPI face data into the incoming non-local buffer
 *    (ScatterReceivedFaceData).
 *  - Packing outgoing face data into wire-format ByteArrays for the
 *    aggregated communicator (CopyOutgoingPsiBackToHost).
 *  - Copying boundary psi to/from the sweep boundary conditions.
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

  const CBCD_FLUDSCommonData& GetCommonData() const { return common_data_; }
  crb::Stream& GetStream() { return stream_; }

  /// Allocate device-side local psi and (optionally) saved psi buffers.
  void AllocateLocalAndSavedPsi();

  /// Stride (num_groups * num_angles) for each face node's psi data.
  std::size_t GetStrideSize() const { return num_groups_and_angles_; }

  /// Mapped host vector of cell IDs to sweep (written by the caller before Sweep).
  crb::MappedHostVector<std::uint32_t>& GetLocalCellIDs() { return local_cell_ids_; }

  double* GetSavedAngularFluxDevicePointer() { return device_saved_psi_.get(); }

  void CopySavedPsiFromDevice();
  void CopySavedPsiToDestinationPsi(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  CBCD_FLUDSPointerSet& GetDevicePointerSet() { return pointer_set_; }

  void CopyIncomingBoundaryPsiToDevice(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set);

  /// Scatter received wire-format face data into incoming_nonlocal_psi_.
  /// Returns the cell_local_id so the caller can update task dependencies.
  uint64_t ScatterReceivedFaceData(uint64_t cell_global_id,
                                   unsigned int face_id,
                                   const double* psi_data);

  /// Pack outgoing psi into wire-format ByteArrays and enqueue for MPI send.
  /// Also copies outgoing boundary (reflecting) psi to the sweep boundary.
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

  const std::vector<std::vector<BoundaryNodeInfo>>& GetOutgoingBoundaryNodeMap() const
  {
    return common_data_.GetOutgoingBoundaryNodeMap();
  }

  size_t GetNumOutgoingFaces() const { return num_outgoing_faces_; }
  size_t GetNumIncomingFaces() const { return num_incoming_faces_; }

private:
  const CBCD_FLUDSCommonData& common_data_;
  const UnknownManager& psi_uk_man_;
  size_t local_psi_data_size_; ///< Total DOFs * groups_and_angles (device alloc size).
  const MeshContinuum* grid_ptr_;

  // Zero-copy mapped host vectors visible to both CPU and GPU.
  crb::MappedHostVector<double> incoming_boundary_psi_;
  crb::MappedHostVector<double> outgoing_boundary_psi_;
  crb::MappedHostVector<double> incoming_nonlocal_psi_;
  crb::MappedHostVector<double> outgoing_nonlocal_psi_;

  crb::Stream stream_;
  crb::MappedHostVector<std::uint32_t> local_cell_ids_;
  bool save_angular_flux_;

  crb::DeviceMemory<double> local_psi_;
  crb::DeviceMemory<double> device_saved_psi_;
  crb::HostVector<double> host_saved_psi_;

  CBCD_FLUDSPointerSet pointer_set_;

  /// Face-grouped outgoing non-local node metadata (indexed by cell_local_id).
  struct FaceOutgoingInfo
  {
    unsigned int face_id;
    std::vector<const NonlocalNodeInfo*> nodes;
    size_t face_data_size;        ///< num_face_nodes * stride_size
    int locality;                 ///< Destination MPI rank.
    uint64_t neighbor_global_id;  ///< Neighbor cell global ID (packed into wire format).
    unsigned int associated_face; ///< Face index on the neighbor cell.
  };
  std::vector<std::vector<FaceOutgoingInfo>> cell_to_face_grouped_outgoing_;

  /// Face-grouped incoming non-local node metadata (indexed by cell_local_id).
  struct FaceIncomingInfo
  {
    unsigned int face_id;
    std::vector<const NonlocalNodeInfo*> nodes;
  };
  std::vector<std::vector<FaceIncomingInfo>> cell_to_face_grouped_incoming_;

  size_t num_outgoing_faces_ = 0;
  size_t num_incoming_faces_ = 0;

  /// O(1) global→local lookup for cells receiving non-local face data.
  std::unordered_map<uint64_t, uint64_t> incoming_global_to_local_;

  /// Pre-resolved per-destination info for batched outgoing enqueue.
  struct OutgoingDestination
  {
    int locality;
    int queue_index = -1; ///< Resolved lazily (agg_comm not available at construction).
  };
  std::vector<OutgoingDestination> outgoing_destinations_;
  std::unordered_map<int, size_t> locality_to_dest_index_;

  /// Scratch buffers for CopyOutgoingPsiBackToHost (sized once, reused across calls).
  std::vector<size_t> scratch_dest_face_counts_;
  std::vector<size_t> scratch_dest_psi_bytes_;
  std::vector<size_t> scratch_dest_offsets_;

  /// Reusable dest buffer vector (avoids outer vector heap allocation per call).
  std::vector<ByteArray> dest_buffers_;

  void CreatePointerSet();
};

} // namespace opensn

// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include <span>
#include <unordered_map>
#include <cstdint>
#include <vector>

namespace opensn
{

class SpatialDiscretization;

/**
 * Shared face-level indexing metadata for device CBCD FLUDS instances.
 *
 * Precomputes and stores the mapping between local cell faces and their
 * boundary, incoming nonlocal, and outgoing nonlocal counterparts. Also
 * builds and uploads to the device a flattened cell-face-node index map
 * used by the GPU sweep kernel to resolve angular-flux buffer addresses
 * via CBCD_NodeIndex encoding. Multiple CBCD_FLUDS instances (one per
 * angle set) reference the same CBCD_FLUDSCommonData.
 */
class CBCD_FLUDSCommonData : public FLUDSCommonData
{
public:
  CBCD_FLUDSCommonData(const SPDS& spds,
                       const std::vector<CellFaceNodalMapping>& grid_nodal_mappings,
                       const SpatialDiscretization& sdm);

  ~CBCD_FLUDSCommonData() override;

  /// Get number of incoming boundary face nodes.
  std::size_t GetNumIncomingBoundaryNodes() const { return num_incoming_boundary_nodes_; }

  /// Get number of outgoing boundary face nodes.
  std::size_t GetNumOutgoingBoundaryNodes() const { return num_outgoing_boundary_nodes_; }

  /// Get number of incoming non-local face nodes.
  std::size_t GetNumIncomingNonlocalNodes() const { return num_incoming_nonlocal_nodes_; }

  /// Get number of outgoing non-local face nodes.
  std::size_t GetNumOutgoingNonlocalNodes() const { return num_outgoing_nonlocal_nodes_; }

  /// Get number of incoming non-local faces.
  std::size_t GetNumIncomingNonlocalFaces() const { return num_incoming_nonlocal_faces_; }

  /// Get number of outgoing non-local faces.
  std::size_t GetNumOutgoingNonlocalFaces() const { return num_outgoing_nonlocal_faces_; }

  /// Get incoming boundary node map.
  const std::vector<BoundaryNodeInfo>& GetIncomingBoundaryNodeMap() const
  {
    return incoming_boundary_node_map_;
  }

  /// Return outgoing-boundary nodes for one cell.
  std::span<const BoundaryNodeInfo> GetOutgoingBoundaryNodes(std::uint64_t cell_local_id) const
  {
    const auto begin = cell_to_outgoing_boundary_node_offsets_[cell_local_id];
    const auto end = cell_to_outgoing_boundary_node_offsets_[cell_local_id + 1];
    return {outgoing_boundary_nodes_.data() + begin, end - begin};
  }

  /// Return grouped outgoing nonlocal faces for one cell.
  std::span<const GroupedOutgoingNonlocalFace>
  GetOutgoingNonlocalFaces(std::uint64_t cell_local_id) const
  {
    const auto begin = cell_to_outgoing_nonlocal_face_offsets_[cell_local_id];
    const auto end = cell_to_outgoing_nonlocal_face_offsets_[cell_local_id + 1];
    return {outgoing_nonlocal_faces_.data() + begin, end - begin};
  }

  /// Return grouped incoming nonlocal faces for one cell.
  std::span<const GroupedIncomingNonlocalFace>
  GetIncomingNonlocalFaces(std::uint64_t cell_local_id) const
  {
    const auto begin = cell_to_incoming_nonlocal_face_offsets_[cell_local_id];
    const auto end = cell_to_incoming_nonlocal_face_offsets_[cell_local_id + 1];
    return {incoming_nonlocal_faces_.data() + begin, end - begin};
  }

  /// Return the number of local cells represented in the grouped-face tables.
  std::size_t GetNumLocalCells() const { return cell_to_incoming_nonlocal_face_offsets_.size() - 1; }

  /// Return the ordered outgoing-locality table.
  const std::vector<int>& GetOutgoingLocalities() const { return outgoing_localities_; }

  /// Resolve one grouped incoming nonlocal face from wire identifiers.
  const GroupedIncomingNonlocalFace&
  FindIncomingNonlocalFace(std::uint64_t cell_global_id, unsigned int face_id) const;

  /// Return the outgoing-node-copy descriptors for one grouped outgoing face.
  std::span<const OutgoingNodeCopy>
  GetOutgoingNodeCopies(const GroupedOutgoingNonlocalFace& face) const
  {
    return {outgoing_nonlocal_face_node_copies_.data() + face.node_copy_offset,
            face.num_node_copies};
  }

  /// Get pointer to cell-face-node map on device.
  const std::uint64_t* GetDeviceIndex() const { return device_cell_face_node_map_; }

private:
  /// Number of incoming boundary face nodes.
  size_t num_incoming_boundary_nodes_;
  /// Number of outgoing boundary face nodes.
  size_t num_outgoing_boundary_nodes_;
  /// Number of incoming non-local faces.
  size_t num_incoming_nonlocal_faces_;
  /// Number of incoming non-local face nodes.
  size_t num_incoming_nonlocal_nodes_;
  /// Number of outgoing non-local faces.
  size_t num_outgoing_nonlocal_faces_;
  /// Number of outgoing non-local face nodes.
  size_t num_outgoing_nonlocal_nodes_;
  /// Device pointer to cell-face-node map for angular flux buffer access.
  std::uint64_t* device_cell_face_node_map_;
  /// Map from incoming face boundary node to indexing metadata.
  std::vector<BoundaryNodeInfo> incoming_boundary_node_map_;
  /// Cell-to-outgoing-boundary-node offset table.
  std::vector<std::uint32_t> cell_to_outgoing_boundary_node_offsets_;
  /// Flat outgoing-boundary node list.
  std::vector<BoundaryNodeInfo> outgoing_boundary_nodes_;
  /// Cell-to-incoming-face offset table.
  std::vector<std::uint32_t> cell_to_incoming_nonlocal_face_offsets_;
  /// Cell-to-outgoing-face offset table.
  std::vector<std::uint32_t> cell_to_outgoing_nonlocal_face_offsets_;
  /// Flat grouped incoming nonlocal faces.
  std::vector<GroupedIncomingNonlocalFace> incoming_nonlocal_faces_;
  /// Flat grouped outgoing nonlocal faces.
  std::vector<GroupedOutgoingNonlocalFace> outgoing_nonlocal_faces_;
  /// Flat outgoing-node-copy metadata referenced by grouped outgoing faces.
  std::vector<OutgoingNodeCopy> outgoing_nonlocal_face_node_copies_;
  /// Incoming wire-format face key to grouped-face descriptor lookup.
  std::unordered_map<IncomingFaceKey, std::uint32_t, IncomingFaceKeyHash> incoming_face_map_;
  /// Ordered table of distinct outgoing localities.
  std::vector<int> outgoing_localities_;

  /**
   * Compute cell-face-node map for device angular flux buffer access, and
   * create auxiliary indexing maps for boundary and non-local nodes for host access.
   */
  void CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm);
  /// Deallocate device memory for cell-face-node map.
  void DeallocateDeviceMemory();
};

} // namespace opensn

// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace opensn
{

class SpatialDiscretization;

/**
 * Shared CBCD FLUDS topology and indexing data.
 *
 * The object is built once per groupset and shared by every `CBCD_FLUDS`
 * instance for that sweep ordering. It owns the device-resident face-node
 * index map consumed by the GPU kernel and the host-side grouped-face metadata
 * consumed by the host communication path.
 */
class CBCD_FLUDSCommonData : public CBC_FLUDSCommonData
{
public:
  /// Incoming nonlocal face grouped by face ID.
  struct GroupedIncomingNonlocalFace
  {
    /// Face-node metadata in face-node order.
    std::vector<NonlocalNodeInfo> nodes;
  };

  /// Outgoing node-copy descriptor for one face node.
  struct OutgoingNodeCopy
  {
    /// Source offset in the outgoing nonlocal psi buffer.
    std::uint32_t storage_index = 0;
    /// Destination face-node index inside the packed face payload.
    std::uint16_t face_node = 0;
  };

  /// Outgoing nonlocal face grouped by face ID and destination.
  struct GroupedOutgoingNonlocalFace
  {
    /// Fixed wire-format prefix `[neighbor_global_id][associated_face]`.
    std::array<std::byte, sizeof(std::uint64_t) + sizeof(unsigned int)> entry_header_prefix{};
    /// Destination slot in the outgoing locality table.
    std::uint32_t dest_slot = 0;
    /// Number of nodes on this face.
    std::uint16_t num_face_nodes = 0;
    /// Face-node copy descriptors in face-node order.
    std::vector<OutgoingNodeCopy> node_copies;
  };

  /// Build shared CBCD topology for one sweep ordering.
  ///
  /// \param spds Sweep ordering.
  /// \param grid_nodal_mappings Face-node mappings for local cells.
  /// \param sdm Spatial discretization.
  CBCD_FLUDSCommonData(const SPDS& spds,
                       const std::vector<CellFaceNodalMapping>& grid_nodal_mappings,
                       const SpatialDiscretization& sdm);

  ~CBCD_FLUDSCommonData() override;

  std::size_t GetNumIncomingBoundaryNodes() const { return num_incoming_boundary_nodes_; }
  std::size_t GetNumOutgoingBoundaryNodes() const { return num_outgoing_boundary_nodes_; }
  std::size_t GetNumIncomingNonlocalNodes() const { return num_incoming_nonlocal_nodes_; }
  std::size_t GetNumOutgoingNonlocalNodes() const { return num_outgoing_nonlocal_nodes_; }
  std::size_t GetNumIncomingNonlocalFaces() const { return num_incoming_nonlocal_faces_; }
  std::size_t GetNumOutgoingNonlocalFaces() const { return num_outgoing_nonlocal_faces_; }

  /// Return the flat incoming-boundary node list.
  const std::vector<BoundaryNodeInfo>& GetIncomingBoundaryNodeMap() const
  {
    return incoming_boundary_node_map_;
  }

  /// Return per-cell outgoing-boundary nodes.
  const std::vector<std::vector<BoundaryNodeInfo>>& GetOutgoingBoundaryNodeMap() const
  {
    return cell_to_outgoing_boundary_nodes_;
  }

  /// Return per-cell grouped outgoing nonlocal faces.
  const std::vector<std::vector<GroupedOutgoingNonlocalFace>>& GetOutgoingNonlocalFaces() const
  {
    return cell_to_outgoing_nonlocal_faces_;
  }

  /// Return per-cell grouped incoming nonlocal faces.
  const std::vector<std::vector<GroupedIncomingNonlocalFace>>& GetIncomingNonlocalFaces() const
  {
    return cell_to_incoming_nonlocal_faces_;
  }

  /// Return the ordered outgoing locality table.
  const std::vector<int>& GetOutgoingLocalities() const { return outgoing_localities_; }

  /// Resolve one grouped incoming nonlocal face.
  ///
  /// \param cell_local_id Local cell identifier.
  /// \param face_id Face index on that cell.
  /// \return Metadata for the requested grouped face.
  const GroupedIncomingNonlocalFace* FindIncomingNonlocalFace(std::uint64_t cell_local_id,
                                                              unsigned int face_id) const;

  /// Map a receiving cell global ID to its local ID.
  ///
  /// \param cell_global_id Global cell identifier.
  /// \return Local cell identifier.
  std::uint64_t MapIncomingGlobalToLocal(std::uint64_t cell_global_id) const
  {
    const auto it = incoming_global_to_local_.find(cell_global_id);
    assert(it != incoming_global_to_local_.end());
    return it->second;
  }

  /// Return the device pointer to the packed face-node index table.
  const std::uint64_t* GetDeviceIndex() const { return device_cell_face_node_map_; }

private:
  /// Number of incoming boundary nodes.
  size_t num_incoming_boundary_nodes_;
  /// Number of outgoing boundary nodes.
  size_t num_outgoing_boundary_nodes_;
  /// Number of incoming nonlocal nodes.
  size_t num_incoming_nonlocal_nodes_;
  /// Number of outgoing nonlocal nodes.
  size_t num_outgoing_nonlocal_nodes_;
  /// Number of grouped incoming nonlocal faces.
  size_t num_incoming_nonlocal_faces_;
  /// Number of grouped outgoing nonlocal faces.
  size_t num_outgoing_nonlocal_faces_;

  /// Device-resident packed face-node index table.
  std::uint64_t* device_cell_face_node_map_;

  /// Flat incoming-boundary node list.
  std::vector<BoundaryNodeInfo> incoming_boundary_node_map_;

  /// Per-cell outgoing-boundary nodes.
  std::vector<std::vector<BoundaryNodeInfo>> cell_to_outgoing_boundary_nodes_;
  /// Per-cell grouped incoming nonlocal faces.
  std::vector<std::vector<GroupedIncomingNonlocalFace>> cell_to_incoming_nonlocal_faces_;
  /// Per-cell grouped outgoing nonlocal faces.
  std::vector<std::vector<GroupedOutgoingNonlocalFace>> cell_to_outgoing_nonlocal_faces_;
  /// Face-ID to grouped-face index lookup for incoming nonlocal faces.
  std::vector<std::vector<int>> incoming_nonlocal_face_lookup_;
  /// Receiving-cell global-to-local map for incoming nonlocal traffic.
  std::unordered_map<std::uint64_t, std::uint64_t> incoming_global_to_local_;
  /// Ordered table of distinct outgoing localities.
  std::vector<int> outgoing_localities_;

  /// Build the device table and grouped host-side metadata.
  void CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm);

  /// Release the packed device table.
  void DeallocateDeviceMemory();
};

} // namespace opensn

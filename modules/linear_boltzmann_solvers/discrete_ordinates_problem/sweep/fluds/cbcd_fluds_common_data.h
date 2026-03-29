// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace opensn
{

class SpatialDiscretization;

/**
 * Shared (angle-set-independent) topology data for CBCD FLUDS.
 *
 * Built once per groupset from the mesh and SPDS, then referenced by every
 * CBCD_FLUDS instance.  Holds:
 *
 *  - A device-resident bit-packed cell-face-node index map used by the GPU
 *    kernel to resolve angular flux buffer pointers (see CBCD_NodeIndex).
 *
 *  - Host-side auxiliary maps that classify every face node as incoming/outgoing
 *    and boundary/local/non-local, with compact metadata structs
 *    (BoundaryNodeInfo 24 B, NonlocalNodeInfo 24 B) used by CBCD_FLUDS for
 *    host-side scatter/gather during MPI communication.
 */
class CBCD_FLUDSCommonData : public CBC_FLUDSCommonData
{
public:
  struct GroupedIncomingNonlocalFace
  {
    std::vector<NonlocalNodeInfo> nodes;
  };

  struct GroupedOutgoingNonlocalFace
  {
    std::vector<NonlocalNodeInfo> nodes;
    std::uint64_t neighbor_global_id = 0;
    int locality = -1;
    std::uint32_t dest_slot = 0;
    unsigned int associated_face = 0;
    std::uint16_t num_face_nodes = 0;
  };

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

  /// Flat list of all incoming boundary face nodes (for host-side psi copy at init).
  const std::vector<BoundaryNodeInfo>& GetIncomingBoundaryNodeMap() const
  {
    return incoming_boundary_node_map_;
  }

  /// Per-cell outgoing boundary nodes (indexed by cell_local_id).
  const std::vector<std::vector<BoundaryNodeInfo>>& GetOutgoingBoundaryNodeMap() const
  {
    return cell_to_outgoing_boundary_nodes_;
  }

  /// Per-cell outgoing non-local faces (indexed by cell_local_id).
  const std::vector<std::vector<GroupedOutgoingNonlocalFace>>& GetOutgoingNonlocalFaces() const
  {
    return cell_to_outgoing_nonlocal_faces_;
  }

  /// Per-cell incoming non-local faces (indexed by cell_local_id).
  const std::vector<std::vector<GroupedIncomingNonlocalFace>>& GetIncomingNonlocalFaces() const
  {
    return cell_to_incoming_nonlocal_faces_;
  }

  const std::vector<int>& GetOutgoingLocalities() const { return outgoing_localities_; }

  /// Resolve one incoming non-local face by cell-local-id and face-id.
  const GroupedIncomingNonlocalFace* FindIncomingNonlocalFace(std::uint64_t cell_local_id,
                                                              unsigned int face_id) const;

  /// O(1) global-to-local lookup for cells that receive non-local face data.
  std::uint64_t MapIncomingGlobalToLocal(std::uint64_t cell_global_id) const
  {
    const auto it = incoming_global_to_local_.find(cell_global_id);
    assert(it != incoming_global_to_local_.end());
    return it->second;
  }

  /// Device pointer to the bit-packed cell-face-node index map.
  const std::uint64_t* GetDeviceIndex() const { return device_cell_face_node_map_; }

private:
  size_t num_incoming_boundary_nodes_;
  size_t num_outgoing_boundary_nodes_;
  size_t num_incoming_nonlocal_nodes_;
  size_t num_outgoing_nonlocal_nodes_;
  size_t num_incoming_nonlocal_faces_;
  size_t num_outgoing_nonlocal_faces_;

  /// Device-resident array: [cell_offset, num_face_nodes] pairs followed by
  /// packed CBCD_NodeIndex values for every face node of every local cell.
  std::uint64_t* device_cell_face_node_map_;

  /// Flat list of incoming boundary face nodes (traversed once per sweep init).
  std::vector<BoundaryNodeInfo> incoming_boundary_node_map_;

  /// Per-cell auxiliary maps (indexed by cell_local_id) for the four non-local
  /// and boundary categories.  Built once during construction.
  std::vector<std::vector<BoundaryNodeInfo>> cell_to_outgoing_boundary_nodes_;
  std::vector<std::vector<GroupedIncomingNonlocalFace>> cell_to_incoming_nonlocal_faces_;
  std::vector<std::vector<GroupedOutgoingNonlocalFace>> cell_to_outgoing_nonlocal_faces_;
  std::vector<std::vector<int>> incoming_nonlocal_face_lookup_;
  std::unordered_map<std::uint64_t, std::uint64_t> incoming_global_to_local_;
  std::vector<int> outgoing_localities_;

  /// Build the device index map and populate all auxiliary host-side maps.
  void CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm);

  /// Free device memory for the cell-face-node map.
  void DeallocateDeviceMemory();
};

} // namespace opensn

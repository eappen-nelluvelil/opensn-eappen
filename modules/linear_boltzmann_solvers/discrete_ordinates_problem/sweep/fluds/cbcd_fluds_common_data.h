// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include <cassert>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace opensn
{

class SpatialDiscretization;

/**
 * Shared CBCD FLUDS topology and indexing data.
 *
 * The object is built once per sweep ordering and shared by all `CBCD_FLUDS`
 * instances in the groupset. It owns the packed face-node table consumed by
 * the GPU kernel and the flat host metadata consumed by the CBCD host
 * communication path.
 */
class CBCD_FLUDSCommonData : public CBC_FLUDSCommonData
{
public:
  /// Construct shared CBCD topology for one sweep ordering.
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
  std::size_t GetNumIncomingNonlocalFaces() const { return incoming_nonlocal_faces_.size(); }
  std::size_t GetNumOutgoingNonlocalFaces() const { return outgoing_nonlocal_faces_.size(); }

  /// Return the flat incoming-boundary node table.
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
  ///
  /// \param cell_local_id Local cell identifier.
  /// \return Span of grouped outgoing faces for the cell.
  std::span<const GroupedOutgoingNonlocalFace>
  GetOutgoingNonlocalFaces(std::uint64_t cell_local_id) const
  {
    const auto begin = cell_to_outgoing_nonlocal_face_offsets_[cell_local_id];
    const auto end = cell_to_outgoing_nonlocal_face_offsets_[cell_local_id + 1];
    return {outgoing_nonlocal_faces_.data() + begin, end - begin};
  }

  /// Return grouped incoming nonlocal faces for one cell.
  ///
  /// \param cell_local_id Local cell identifier.
  /// \return Span of grouped incoming faces for the cell.
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
  ///
  /// \param cell_global_id Receiving cell global identifier.
  /// \param face_id Receiving face index.
  /// \return Grouped-face metadata.
  const GroupedIncomingNonlocalFace&
  FindIncomingNonlocalFace(std::uint64_t cell_global_id, unsigned int face_id) const;

  /// Return the outgoing-node-copy descriptors for one grouped outgoing face.
  ///
  /// \param face Grouped outgoing face descriptor.
  /// \return Span of outgoing-node-copy descriptors.
  std::span<const OutgoingNodeCopy>
  GetOutgoingNodeCopies(const GroupedOutgoingNonlocalFace& face) const
  {
    return {outgoing_nonlocal_face_node_copies_.data() + face.node_copy_offset, face.num_node_copies};
  }

  /// Return the device pointer to the packed face-node table.
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
  /// Device-resident packed face-node index table.
  std::uint64_t* device_cell_face_node_map_;

  /// Flat incoming-boundary node list.
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

  /// Build the device table and flat host metadata.
  void CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm);

  /// Release the packed device table.
  void DeallocateDeviceMemory();
};

} // namespace opensn

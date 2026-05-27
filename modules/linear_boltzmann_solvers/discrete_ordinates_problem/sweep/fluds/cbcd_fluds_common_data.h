// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include <cstdint>
#include <span>
#include <vector>

namespace opensn
{

class SpatialDiscretization;

/// Shared CBCD FLUDS metadata.
class CBCD_FLUDSCommonData : public FLUDSCommonData
{
public:
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

  const std::vector<IncomingBoundaryFacePlan>& GetIncomingBoundaryFaces() const
  {
    return incoming_boundary_face_plans_;
  }

  std::size_t GetNumIncomingFacesFromSource(const std::size_t source_slot) const
  {
    return source_to_incoming_face_offsets_[source_slot + 1] -
           source_to_incoming_face_offsets_[source_slot];
  }

  std::span<const BoundaryNodeInfo> GetOutgoingBoundaryNodes(std::uint64_t cell_local_id) const
  {
    const auto begin = cell_to_outgoing_boundary_node_offsets_[cell_local_id];
    const auto end = cell_to_outgoing_boundary_node_offsets_[cell_local_id + 1];
    return {outgoing_boundary_nodes_.data() + begin, end - begin};
  }

  std::span<const GroupedOutgoingNonlocalFace>
  GetOutgoingNonlocalFaces(std::uint64_t cell_local_id) const
  {
    const auto begin = cell_to_outgoing_nonlocal_face_offsets_[cell_local_id];
    const auto end = cell_to_outgoing_nonlocal_face_offsets_[cell_local_id + 1];
    return {outgoing_nonlocal_faces_.data() + begin, end - begin};
  }

  std::span<const GroupedIncomingNonlocalFace>
  GetIncomingNonlocalFaces(std::uint64_t cell_local_id) const
  {
    const auto begin = cell_to_incoming_nonlocal_face_offsets_[cell_local_id];
    const auto end = cell_to_incoming_nonlocal_face_offsets_[cell_local_id + 1];
    return {incoming_nonlocal_faces_.data() + begin, end - begin};
  }

  std::size_t GetNumLocalCells() const
  {
    return cell_to_outgoing_nonlocal_face_offsets_.size() - 1;
  }

  const std::vector<int>& GetOutgoingLocalities() const { return outgoing_localities_; }

  const std::vector<int>& GetIncomingSourcePartitions() const
  {
    return incoming_source_partitions_;
  }

  const GroupedIncomingNonlocalFace& GetIncomingNonlocalFace(std::uint32_t source_slot,
                                                             std::uint32_t source_face_index) const;

  std::span<const OutgoingNodeCopy>
  GetOutgoingNodeCopies(const GroupedOutgoingNonlocalFace& face) const
  {
    return {outgoing_nonlocal_face_node_copies_.data() + face.node_copy_offset,
            face.num_node_copies};
  }

  const std::uint64_t* GetDeviceIndex() const { return device_cell_face_node_map_; }

private:
  size_t num_incoming_boundary_nodes_;
  size_t num_outgoing_boundary_nodes_;
  size_t num_incoming_nonlocal_faces_;
  size_t num_incoming_nonlocal_nodes_;
  size_t num_outgoing_nonlocal_faces_;
  size_t num_outgoing_nonlocal_nodes_;
  std::uint64_t* device_cell_face_node_map_;
  std::vector<IncomingBoundaryFacePlan> incoming_boundary_face_plans_;
  std::vector<std::uint32_t> cell_to_outgoing_boundary_node_offsets_;
  std::vector<BoundaryNodeInfo> outgoing_boundary_nodes_;
  std::vector<std::uint32_t> cell_to_incoming_nonlocal_face_offsets_;
  std::vector<std::uint32_t> cell_to_outgoing_nonlocal_face_offsets_;
  std::vector<GroupedIncomingNonlocalFace> incoming_nonlocal_faces_;
  std::vector<GroupedOutgoingNonlocalFace> outgoing_nonlocal_faces_;
  std::vector<OutgoingNodeCopy> outgoing_nonlocal_face_node_copies_;
  std::vector<int> outgoing_localities_;
  std::vector<int> incoming_source_partitions_;
  std::vector<std::uint32_t> source_to_incoming_face_offsets_;
  std::vector<std::uint32_t> incoming_face_indices_by_source_;

  void CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm);
  void DeallocateDeviceMemory();
};

} // namespace opensn

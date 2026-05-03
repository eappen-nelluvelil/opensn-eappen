// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include <cstdint>
#include <map>

namespace opensn
{

class SpatialDiscretization;

/// CBCD FLUDS common data.
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

  const std::vector<BoundaryNodeInfo>& GetIncomingBoundaryNodeMap() const
  {
    return incoming_boundary_node_map_;
  }

  const std::map<std::uint64_t, std::vector<BoundaryNodeInfo>>& GetOutgoingBoundaryNodeMap() const
  {
    return cell_to_outgoing_boundary_nodes_;
  }

  const std::map<std::uint64_t, std::vector<NonlocalNodeInfo>>& GetIncomingNonlocalNodeMap() const
  {
    return cell_to_incoming_nonlocal_nodes_;
  }

  const std::map<std::uint64_t, std::vector<NonlocalNodeInfo>>& GetOutgoingNonlocalNodeMap() const
  {
    return cell_to_outgoing_nonlocal_nodes_;
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
  std::vector<BoundaryNodeInfo> incoming_boundary_node_map_;
  std::map<std::uint64_t, std::vector<BoundaryNodeInfo>> cell_to_outgoing_boundary_nodes_;
  std::map<std::uint64_t, std::vector<NonlocalNodeInfo>> cell_to_incoming_nonlocal_nodes_;
  std::map<std::uint64_t, std::vector<NonlocalNodeInfo>> cell_to_outgoing_nonlocal_nodes_;

  /// Compute device and host angular-flux indexing maps.
  void CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm);
  void DeallocateDeviceMemory();
};

} // namespace opensn

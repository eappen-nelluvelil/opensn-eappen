// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "framework/utils/error.h"

namespace opensn
{

class SpatialDiscretization;

CBCD_FLUDSCommonData::CBCD_FLUDSCommonData(
  const SPDS& spds,
  const std::vector<CellFaceNodalMapping>& grid_nodal_mappings,
  const SpatialDiscretization& sdm)
  : FLUDSCommonData(spds, grid_nodal_mappings),
    num_incoming_boundary_nodes_(0),
    num_outgoing_boundary_nodes_(0),
    num_incoming_nonlocal_faces_(0),
    num_incoming_nonlocal_nodes_(0),
    num_outgoing_nonlocal_faces_(0),
    num_outgoing_nonlocal_nodes_(0),
    max_local_outgoing_node_count_(0),
    device_cell_face_node_map_(nullptr),
    device_local_cell_node_offsets_(nullptr),
    device_local_compact_node_indices_(nullptr),
    incoming_boundary_node_map_()
{
  CopyFlattenedNodeIndexToDevice(sdm);
}

CBCD_FLUDSCommonData::~CBCD_FLUDSCommonData()
{
  DeallocateDeviceMemory();
}

#ifndef __OPENSN_WITH_GPU__
void
CBCD_FLUDSCommonData::CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm)
{
}

void
CBCD_FLUDSCommonData::DeallocateDeviceMemory()
{
}
#endif

const GroupedIncomingNonlocalFace&
CBCD_FLUDSCommonData::FindIncomingNonlocalFace(std::uint64_t cell_global_id,
                                               unsigned int face_id) const
{
  const auto it = incoming_face_map_.find(IncomingFaceKey{cell_global_id, face_id});
  OpenSnLogicalErrorIf(it == incoming_face_map_.end(),
                       "Unknown incoming nonlocal face requested in CBCD_FLUDSCommonData");
  return incoming_nonlocal_faces_[it->second];
}

} // namespace opensn

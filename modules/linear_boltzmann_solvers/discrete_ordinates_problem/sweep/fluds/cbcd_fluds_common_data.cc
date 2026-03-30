// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"

namespace opensn
{

CBCD_FLUDSCommonData::CBCD_FLUDSCommonData(
  const SPDS& spds,
  const std::vector<CellFaceNodalMapping>& grid_nodal_mappings,
  const SpatialDiscretization& sdm)
  : CBC_FLUDSCommonData(spds, grid_nodal_mappings),
    num_incoming_boundary_nodes_(0),
    num_outgoing_boundary_nodes_(0),
    num_incoming_nonlocal_nodes_(0),
    num_outgoing_nonlocal_nodes_(0),
    num_incoming_nonlocal_faces_(0),
    num_outgoing_nonlocal_faces_(0),
    device_cell_face_node_map_(nullptr),
    incoming_boundary_node_map_()
{
  const size_t num_local_cells = spds_.GetGrid()->local_cells.size();
  cell_to_outgoing_boundary_node_offsets_.resize(num_local_cells + 1, 0);
  cell_to_incoming_nonlocal_face_offsets_.resize(num_local_cells + 1, 0);
  cell_to_outgoing_nonlocal_face_offsets_.resize(num_local_cells + 1, 0);

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
  (void)sdm;
}

void
CBCD_FLUDSCommonData::DeallocateDeviceMemory()
{
}
#endif

std::pair<std::uint64_t, const CBCD_FLUDSCommonData::GroupedIncomingNonlocalFace*>
CBCD_FLUDSCommonData::FindIncomingNonlocalFace(std::uint64_t cell_global_id,
                                               unsigned int face_id) const
{
  const auto it = incoming_face_map_.find({cell_global_id, face_id});
  assert(it != incoming_face_map_.end());
  const auto& face_ref = it->second;
  return {face_ref.cell_local_id, &incoming_nonlocal_faces_[face_ref.grouped_face_index]};
}

} // namespace opensn

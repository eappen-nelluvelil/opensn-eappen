// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "framework/utils/error.h"
#include <cassert>
#include <algorithm>

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
    device_cell_face_node_map_(nullptr)
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
CBCD_FLUDSCommonData::FindIncomingNonlocalFace(const std::uint32_t source_slot,
                                               const std::uint64_t cell_global_id,
                                               const unsigned int face_id) const
{
  const auto source_faces = GetIncomingFaceLookupsBySource(source_slot);
  const auto face_it =
    std::lower_bound(source_faces.begin(),
                     source_faces.end(),
                     std::pair<std::uint64_t, unsigned int>{cell_global_id, face_id},
                     [](const IncomingFaceLookup& lookup,
                        const std::pair<std::uint64_t, unsigned int>& key)
                     {
                       return std::pair<std::uint64_t, unsigned int>{lookup.cell_global_id,
                                                                     lookup.face_id} < key;
                     });
  assert(face_it != source_faces.end() and face_it->cell_global_id == cell_global_id and
         face_it->face_id == face_id);
  return incoming_nonlocal_faces_[face_it->face_index];
}

} // namespace opensn

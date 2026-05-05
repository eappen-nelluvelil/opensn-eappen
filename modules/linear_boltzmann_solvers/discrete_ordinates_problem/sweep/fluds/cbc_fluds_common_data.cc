// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/cell/cell.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mpi/mpi_utils.h"
#include "framework/utils/error.h"
#include <boost/unordered/unordered_flat_map.hpp>
#include <algorithm>
#include <cassert>
#include <limits>
#include <map>

namespace opensn
{

namespace
{

static_assert(sizeof(size_t) <= sizeof(std::uint64_t),
              "CBC face slot exchange assumes size_t fits in uint64_t.");

constexpr std::size_t FACE_SLOT_RECORD_SIZE = 5;

struct RemoteFaceSlot
{
  size_t slot = CBC_FLUDSCommonData::INVALID_FACE_SLOT;
  size_t num_face_nodes = 0;
  bool delayed = false;
};

} // namespace

CBC_FLUDSCommonData::CBC_FLUDSCommonData(
  const SPDS& spds, const std::vector<CellFaceNodalMapping>& grid_nodal_mappings)
  : FLUDSCommonData(spds, grid_nodal_mappings),
    num_incoming_nonlocal_faces_(0),
    num_outgoing_nonlocal_faces_(0),
    num_delayed_local_face_nodes_(0)
{
  const auto& grid = *spds.GetGrid();
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(spds);
  const auto& face_orientations = spds.GetCellFaceOrientations();
  const auto& delayed_location_dependencies = spds.GetDelayedLocationDependencies();
  local_face_slot_offsets_.resize(grid.local_cells.size(), 0);

  std::size_t num_local_faces = 0;
  std::size_t num_incoming_nonlocal_faces = 0;
  for (const auto& cell : grid.local_cells)
  {
    assert(cell.local_id < local_face_slot_offsets_.size());
    local_face_slot_offsets_[cell.local_id] = num_local_faces;
    num_local_faces += cell.faces.size();

    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      if ((not face.has_neighbor) or (face.IsNeighborLocal(&grid)))
        continue;

      const auto orientation = face_orientations[cell.local_id][f];
      if (orientation == FaceOrientation::INCOMING)
        ++num_incoming_nonlocal_faces;
      else if (orientation == FaceOrientation::OUTGOING)
        ++num_outgoing_nonlocal_faces_;
    }
  }

  incoming_nonlocal_face_slots_by_local_face_.assign(num_local_faces, INVALID_FACE_SLOT);
  delayed_nonlocal_face_slots_by_local_face_.assign(num_local_faces, INVALID_FACE_SLOT);
  incoming_nonlocal_face_local_cells_.reserve(num_incoming_nonlocal_faces);
  outgoing_nonlocal_face_slots_by_local_face_.assign(num_local_faces, INVALID_FACE_SLOT);
  outgoing_nonlocal_face_peer_indices_by_local_face_.assign(num_local_faces, INVALID_PEER_INDEX);
  outgoing_nonlocal_face_locations_by_local_face_.assign(num_local_faces, -1);
  outgoing_nonlocal_face_node_counts_by_local_face_.assign(num_local_faces, 0);
  delayed_local_face_info_by_local_face_.assign(num_local_faces, {});
  delayed_nonlocal_face_info_by_local_face_.assign(num_local_faces, {});
  delayed_prelocI_face_node_counts_.assign(delayed_location_dependencies.size(), 0);
  delayed_local_incoming_by_local_face_.assign(num_local_faces, 0);
  delayed_local_outgoing_by_local_face_.assign(num_local_faces, 0);
  delayed_nonlocal_incoming_by_local_face_.assign(num_local_faces, 0);
  delayed_nonlocal_outgoing_by_local_face_.assign(num_local_faces, 0);

  boost::unordered_flat_map<int, std::size_t> outgoing_peer_index_by_location;
  const auto& location_successors = spds.GetLocationSuccessors();
  outgoing_peer_index_by_location.reserve(location_successors.size());
  for (std::size_t i = 0; i < location_successors.size(); ++i)
    outgoing_peer_index_by_location.emplace(location_successors[i], i);

  std::map<int, std::vector<std::uint64_t>> incoming_slot_records_by_upstream_location;
  for (const auto& cell : grid.local_cells)
  {
    const auto local_face_slot_offset = local_face_slot_offsets_[cell.local_id];
    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = face_orientations[cell.local_id][f];
      const auto face_storage_index = local_face_slot_offset + f;
      const auto num_face_nodes =
        GetFaceNodalMapping(cell.local_id, static_cast<unsigned int>(f)).face_node_mapping_.size();

      if (face.has_neighbor and face.IsNeighborLocal(&grid))
      {
        const auto& adj_cell = grid.cells[face.neighbor_id];
        const bool delayed_incoming =
          orientation == FaceOrientation::INCOMING and
          cbc_spds.IsDelayedLocalDependency(adj_cell.local_id, cell.local_id);
        const bool delayed_outgoing =
          orientation == FaceOrientation::OUTGOING and
          cbc_spds.IsDelayedLocalDependency(cell.local_id, adj_cell.local_id);

        if (delayed_incoming)
        {
          delayed_local_incoming_by_local_face_[face_storage_index] = 1;
          delayed_local_face_info_by_local_face_[face_storage_index] =
            DelayedLocalFaceInfo{num_delayed_local_face_nodes_, num_face_nodes};
          num_delayed_local_face_nodes_ += num_face_nodes;
        }
        if (delayed_outgoing)
          delayed_local_outgoing_by_local_face_[face_storage_index] = 1;

        continue;
      }

      if ((not face.has_neighbor) or (face.IsNeighborLocal(&grid)))
        continue;

      if (orientation == FaceOrientation::INCOMING)
      {
        const auto neighbor_location = face.GetNeighborPartitionID(&grid);
        auto& records = incoming_slot_records_by_upstream_location[neighbor_location];
        const auto delayed_it = std::find(delayed_location_dependencies.begin(),
                                          delayed_location_dependencies.end(),
                                          neighbor_location);
        if (delayed_it != delayed_location_dependencies.end())
        {
          const auto prelocI =
            static_cast<size_t>(std::distance(delayed_location_dependencies.begin(), delayed_it));
          const auto slot = delayed_nonlocal_face_info_by_slot_.size();
          const DelayedNonlocalFaceInfo info{
            prelocI, delayed_prelocI_face_node_counts_[prelocI], num_face_nodes};
          delayed_nonlocal_face_slots_by_local_face_[face_storage_index] = slot;
          delayed_nonlocal_face_info_by_local_face_[face_storage_index] = info;
          delayed_nonlocal_face_info_by_slot_.push_back(info);
          delayed_nonlocal_incoming_by_local_face_[face_storage_index] = 1;
          delayed_prelocI_face_node_counts_[prelocI] += num_face_nodes;
          records.push_back(cell.global_id);
          records.push_back(static_cast<std::uint64_t>(f));
          records.push_back(static_cast<std::uint64_t>(slot));
          records.push_back(1);
          records.push_back(static_cast<std::uint64_t>(num_face_nodes));
          continue;
        }

        const auto slot = num_incoming_nonlocal_faces_;
        incoming_nonlocal_face_slots_by_local_face_[face_storage_index] = slot;
        incoming_nonlocal_face_local_cells_.push_back(cell.local_id);
        records.push_back(cell.global_id);
        records.push_back(static_cast<std::uint64_t>(f));
        records.push_back(static_cast<std::uint64_t>(slot));
        records.push_back(0);
        records.push_back(static_cast<std::uint64_t>(num_face_nodes));
        ++num_incoming_nonlocal_faces_;
      }
    }
  }

  const auto downstream_slot_records = MapAllToAll(incoming_slot_records_by_upstream_location);
  boost::unordered_flat_map<CellFaceKey, RemoteFaceSlot, std::hash<CellFaceKey>>
    downstream_slot_by_face;
  downstream_slot_by_face.reserve(num_outgoing_nonlocal_faces_);
  for (const auto& location_records : downstream_slot_records)
  {
    const auto& records = location_records.second;
    OpenSnLogicalErrorIf(records.size() % FACE_SLOT_RECORD_SIZE != 0,
                         "CBC non-local face slot exchange returned a malformed record set.");

    for (std::size_t i = 0; i < records.size(); i += FACE_SLOT_RECORD_SIZE)
    {
      const auto face_id = records[i + 1];
      OpenSnLogicalErrorIf(face_id > std::numeric_limits<unsigned int>::max(),
                           "CBC non-local face slot record has invalid face ID.");

      const auto slot_record = records[i + 2];
      if constexpr (sizeof(size_t) < sizeof(std::uint64_t))
        OpenSnLogicalErrorIf(slot_record > std::numeric_limits<size_t>::max(),
                             "CBC non-local face slot record has invalid slot.");
      const auto delayed_record = records[i + 3];
      OpenSnLogicalErrorIf(delayed_record > 1,
                           "CBC non-local face slot record has invalid delayed flag.");
      const auto num_face_nodes_record = records[i + 4];
      if constexpr (sizeof(size_t) < sizeof(std::uint64_t))
        OpenSnLogicalErrorIf(num_face_nodes_record > std::numeric_limits<size_t>::max(),
                             "CBC non-local face slot record has invalid face-node count.");
      OpenSnLogicalErrorIf(num_face_nodes_record == 0,
                           "CBC non-local face slot record has zero face-node count.");

      const CellFaceKey key{records[i], static_cast<unsigned int>(face_id)};
      const RemoteFaceSlot slot{static_cast<size_t>(slot_record),
                                static_cast<size_t>(num_face_nodes_record),
                                delayed_record != 0};
      OpenSnLogicalErrorIf(not downstream_slot_by_face.try_emplace(key, slot).second,
                           "CBC non-local face slot exchange returned duplicate records.");
    }
  }

  for (const auto& cell : grid.local_cells)
  {
    const auto local_face_slot_offset = local_face_slot_offsets_[cell.local_id];
    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      if ((not face.has_neighbor) or (face.IsNeighborLocal(&grid)))
        continue;

      if (face_orientations[cell.local_id][f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face_nodal_mapping =
        GetFaceNodalMapping(cell.local_id, static_cast<unsigned int>(f));
      OpenSnLogicalErrorIf(face_nodal_mapping.associated_face_ < 0,
                           "CBC non-local outgoing face is missing an associated face.");

      const CellFaceKey key{face.neighbor_id,
                            static_cast<unsigned int>(face_nodal_mapping.associated_face_)};
      const auto slot_it = downstream_slot_by_face.find(key);
      OpenSnLogicalErrorIf(slot_it == downstream_slot_by_face.end(),
                           "CBC non-local face slot exchange did not resolve an outgoing face.");

      const auto face_storage_index = local_face_slot_offset + f;
      const auto neighbor_location = face.GetNeighborPartitionID(&grid);
      const auto local_num_face_nodes = face_nodal_mapping.face_node_mapping_.size();
      OpenSnLogicalErrorIf(local_num_face_nodes != slot_it->second.num_face_nodes,
                           "CBC non-local face slot exchange found inconsistent face-node counts.");
      outgoing_nonlocal_face_slots_by_local_face_[face_storage_index] = slot_it->second.slot;
      delayed_nonlocal_outgoing_by_local_face_[face_storage_index] =
        slot_it->second.delayed ? 1 : 0;
      outgoing_nonlocal_face_locations_by_local_face_[face_storage_index] = neighbor_location;
      outgoing_nonlocal_face_node_counts_by_local_face_[face_storage_index] =
        slot_it->second.num_face_nodes;

      if (not slot_it->second.delayed)
      {
        const auto peer_it = outgoing_peer_index_by_location.find(neighbor_location);
        OpenSnLogicalErrorIf(peer_it == outgoing_peer_index_by_location.end(),
                             "CBC outgoing non-local face is missing an SPDS successor.");

        outgoing_nonlocal_face_peer_indices_by_local_face_[face_storage_index] = peer_it->second;
      }
    }
  }
}

size_t
CBC_FLUDSCommonData::GetDelayedPrelocIFaceNodeCount(size_t prelocI) const
{
  assert(prelocI < delayed_prelocI_face_node_counts_.size());
  return delayed_prelocI_face_node_counts_[prelocI];
}

size_t
CBC_FLUDSCommonData::GetDelayedNonlocalFaceNodeCount(size_t delayed_face_slot) const
{
  assert(delayed_face_slot < delayed_nonlocal_face_info_by_slot_.size());
  return delayed_nonlocal_face_info_by_slot_[delayed_face_slot].num_face_nodes;
}

bool
CBC_FLUDSCommonData::IsDelayedLocalIncomingFace(std::uint32_t cell_local_id,
                                                unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < delayed_local_incoming_by_local_face_.size());
  return delayed_local_incoming_by_local_face_[slot_offset] != 0;
}

bool
CBC_FLUDSCommonData::IsDelayedLocalOutgoingFace(std::uint32_t cell_local_id,
                                                unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < delayed_local_outgoing_by_local_face_.size());
  return delayed_local_outgoing_by_local_face_[slot_offset] != 0;
}

bool
CBC_FLUDSCommonData::IsDelayedNonlocalIncomingFace(std::uint32_t cell_local_id,
                                                   unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < delayed_nonlocal_incoming_by_local_face_.size());
  return delayed_nonlocal_incoming_by_local_face_[slot_offset] != 0;
}

bool
CBC_FLUDSCommonData::IsDelayedNonlocalOutgoingFace(std::uint32_t cell_local_id,
                                                   unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < delayed_nonlocal_outgoing_by_local_face_.size());
  return delayed_nonlocal_outgoing_by_local_face_[slot_offset] != 0;
}

const CBC_FLUDSCommonData::DelayedLocalFaceInfo&
CBC_FLUDSCommonData::GetDelayedLocalFaceInfo(std::uint32_t cell_local_id,
                                             unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < delayed_local_face_info_by_local_face_.size());
  return delayed_local_face_info_by_local_face_[slot_offset];
}

const CBC_FLUDSCommonData::DelayedNonlocalFaceInfo&
CBC_FLUDSCommonData::GetDelayedNonlocalFaceInfoByLocalFace(std::uint32_t cell_local_id,
                                                           unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < delayed_nonlocal_face_info_by_local_face_.size());
  return delayed_nonlocal_face_info_by_local_face_[slot_offset];
}

const CBC_FLUDSCommonData::DelayedNonlocalFaceInfo&
CBC_FLUDSCommonData::GetDelayedNonlocalFaceInfoBySlot(size_t delayed_face_slot) const
{
  assert(delayed_face_slot < delayed_nonlocal_face_info_by_slot_.size());
  return delayed_nonlocal_face_info_by_slot_[delayed_face_slot];
}

size_t
CBC_FLUDSCommonData::GetIncomingNonlocalFaceSlotByLocalFace(std::uint32_t cell_local_id,
                                                            unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < incoming_nonlocal_face_slots_by_local_face_.size());
  return incoming_nonlocal_face_slots_by_local_face_[slot_offset];
}

size_t
CBC_FLUDSCommonData::GetOutgoingNonlocalFaceSlotByLocalFace(std::uint32_t cell_local_id,
                                                            unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < outgoing_nonlocal_face_slots_by_local_face_.size());
  return outgoing_nonlocal_face_slots_by_local_face_[slot_offset];
}

size_t
CBC_FLUDSCommonData::GetOutgoingNonlocalFacePeerIndexByLocalFace(std::uint32_t cell_local_id,
                                                                 unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < outgoing_nonlocal_face_peer_indices_by_local_face_.size());
  return outgoing_nonlocal_face_peer_indices_by_local_face_[slot_offset];
}

int
CBC_FLUDSCommonData::GetOutgoingNonlocalFaceLocationByLocalFace(std::uint32_t cell_local_id,
                                                                unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < outgoing_nonlocal_face_locations_by_local_face_.size());
  return outgoing_nonlocal_face_locations_by_local_face_[slot_offset];
}

size_t
CBC_FLUDSCommonData::GetOutgoingNonlocalFaceNodeCountByLocalFace(std::uint32_t cell_local_id,
                                                                 unsigned int face_id) const
{
  assert(cell_local_id < local_face_slot_offsets_.size());
  const auto slot_offset = local_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < outgoing_nonlocal_face_node_counts_by_local_face_.size());
  return outgoing_nonlocal_face_node_counts_by_local_face_[slot_offset];
}

std::uint32_t
CBC_FLUDSCommonData::GetIncomingNonlocalFaceLocalCell(size_t incoming_face_slot) const
{
  assert(incoming_face_slot < incoming_nonlocal_face_local_cells_.size());
  return incoming_nonlocal_face_local_cells_[incoming_face_slot];
}

} // namespace opensn

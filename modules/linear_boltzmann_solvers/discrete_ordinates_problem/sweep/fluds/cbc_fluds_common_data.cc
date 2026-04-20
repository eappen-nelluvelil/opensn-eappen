// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/cell/cell.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include <cassert>
#include <set>
#include <sstream>
#include <unordered_set>

namespace opensn
{

namespace
{

std::uint64_t
PackEdge(const std::uint32_t upwind_local_id, const std::uint32_t downwind_local_id) noexcept
{
  return (static_cast<std::uint64_t>(upwind_local_id) << 32) |
         static_cast<std::uint64_t>(downwind_local_id);
}

} // namespace

CBC_FLUDSCommonData::CBC_FLUDSCommonData(
  const SPDS& spds, const std::vector<CellFaceNodalMapping>& grid_nodal_mappings)
  : FLUDSCommonData(spds, grid_nodal_mappings),
    num_incoming_nonlocal_faces_(0),
    num_incoming_nonlocal_face_nodes_(0),
    num_outgoing_nonlocal_faces_(0),
    num_local_faces_(0),
    max_local_face_node_count_(0),
    num_local_face_slots_(dynamic_cast<const CBC_SPDS&>(spds).GetMaxNumLocalPsiSlots())
{
  // Pre-compute non-local face counts for hash map capacity reservation
  const auto& grid = *spds.GetGrid();
  const auto& cbc_spds = dynamic_cast<const CBC_SPDS&>(spds);
  const auto& face_orientations = spds.GetCellFaceOrientations();
  const auto& delayed_loc_deps = spds.GetDelayedLocationDependencies();

  std::unordered_set<std::uint64_t> delayed_local_edges;
  delayed_local_edges.reserve(spds.GetLocalSweepFAS().size());
  for (const auto& [u, v] : spds.GetLocalSweepFAS())
    delayed_local_edges.insert(PackEdge(u, v));

  has_delayed_local_dependencies_ = not delayed_local_edges.empty();

  outgoing_nonlocal_face_counts_.assign(spds.GetLocationSuccessors().size(), 0);
  outgoing_nonlocal_face_node_counts_.assign(spds.GetLocationSuccessors().size(), 0);
  delayed_preloc_face_node_count_.assign(delayed_loc_deps.size(), 0);
  cell_face_offsets_.resize(grid.local_cells.size() + 1, 0);
  delayed_local_incoming_faces_.resize(grid.local_cells.size());
  delayed_local_outgoing_faces_.resize(grid.local_cells.size());
  delayed_nonlocal_incoming_faces_.resize(grid.local_cells.size());
  delayed_nonlocal_face_info_by_cell_.resize(grid.local_cells.size());
  size_t total_num_faces = 0;

  for (const auto& cell : grid.local_cells)
  {
    cell_face_offsets_[cell.local_id] = static_cast<std::uint32_t>(total_num_faces);
    total_num_faces += cell.faces.size();
  }
  cell_face_offsets_.back() = static_cast<std::uint32_t>(total_num_faces);
  std::set<std::tuple<int, std::uint64_t, unsigned int>> outgoing_nonlocal_face_keys;
  local_face_slot_ids_.assign(total_num_faces, CBC_SPDS::INVALID_LOCAL_FACE_TASK_ID);
  incoming_face_kinds_.assign(total_num_faces, IncomingFaceKind::NONE);
  outgoing_face_kinds_.assign(total_num_faces, OutgoingFaceKind::NONE);
  incoming_nonlocal_face_info_.resize(total_num_faces);
  outgoing_nonlocal_face_info_.resize(total_num_faces);
  delayed_local_face_info_by_storage_index_.assign(total_num_faces, {});

  for (const auto& cell : grid.local_cells)
  {
    const size_t face_offset = cell_face_offsets_[cell.local_id];
    delayed_local_incoming_faces_[cell.local_id].assign(cell.faces.size(), 0);
    delayed_local_outgoing_faces_[cell.local_id].assign(cell.faces.size(), 0);
    delayed_nonlocal_incoming_faces_[cell.local_id].assign(cell.faces.size(), 0);
    delayed_nonlocal_face_info_by_cell_[cell.local_id].resize(cell.faces.size());
    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = face_orientations[cell.local_id][f];
      const size_t face_storage_index = face_offset + f;
      const auto num_face_nodes =
        static_cast<std::uint32_t>(grid_nodal_mappings[cell.local_id][f].face_node_mapping_.size());

      if ((not face.has_neighbor) or (face.IsNeighborLocal(&grid)))
      {
        if (face.has_neighbor)
        {
          max_local_face_node_count_ = std::max(max_local_face_node_count_, face.vertex_ids.size());
          const auto& adj_cell = grid.cells[face.neighbor_id];
          const bool is_delayed_local_incoming =
            (orientation == FaceOrientation::INCOMING) and
            delayed_local_edges.contains(PackEdge(adj_cell.local_id, cell.local_id));
          const bool is_delayed_local_outgoing =
            (orientation == FaceOrientation::OUTGOING) and
            delayed_local_edges.contains(PackEdge(cell.local_id, adj_cell.local_id));

          if (orientation == FaceOrientation::OUTGOING)
          {
            if (is_delayed_local_outgoing)
            {
              delayed_local_outgoing_faces_[cell.local_id][f] = 1;
              outgoing_face_kinds_[face_storage_index] = OutgoingFaceKind::DELAYED_LOCAL;
            }
            else
            {
              const auto task_id =
                cbc_spds.GetOutgoingLocalFaceTaskID(cell.local_id, static_cast<unsigned int>(f));
              assert(task_id != CBC_SPDS::INVALID_LOCAL_FACE_TASK_ID);
              local_face_slot_ids_[face_storage_index] = cbc_spds.GetLocalFaceSlotIDs()[task_id];
              outgoing_face_kinds_[face_storage_index] = OutgoingFaceKind::NORMAL_LOCAL;
              ++num_local_faces_;
            }
          }
          else if (orientation == FaceOrientation::INCOMING)
          {
            if (is_delayed_local_incoming)
            {
              delayed_local_incoming_faces_[cell.local_id][f] = 1;
              incoming_face_kinds_[face_storage_index] = IncomingFaceKind::DELAYED_LOCAL;
              delayed_local_face_info_by_storage_index_[face_storage_index] =
                DelayedLocalFaceInfo{static_cast<std::uint32_t>(delayed_local_face_node_count_),
                                     static_cast<std::uint16_t>(num_face_nodes)};
              delayed_local_face_node_count_ += num_face_nodes;
            }
            else
            {
              const auto task_id =
                cbc_spds.GetIncomingLocalFaceTaskID(cell.local_id, static_cast<unsigned int>(f));
              assert(task_id != CBC_SPDS::INVALID_LOCAL_FACE_TASK_ID);
              local_face_slot_ids_[face_storage_index] = cbc_spds.GetLocalFaceSlotIDs()[task_id];
              incoming_face_kinds_[face_storage_index] = IncomingFaceKind::NORMAL_LOCAL;
            }
          }
        }
        continue;
      }

      if (orientation == FaceOrientation::INCOMING)
      {
        const int neighbor_loc = face.GetNeighborPartitionID(&grid);
        const auto delayed_it =
          std::find(delayed_loc_deps.begin(), delayed_loc_deps.end(), neighbor_loc);
        if (delayed_it != delayed_loc_deps.end())
        {
          const auto prelocI =
            static_cast<std::uint32_t>(std::distance(delayed_loc_deps.begin(), delayed_it));
          const DelayedNonlocalFaceInfo info{
            prelocI,
            static_cast<std::uint32_t>(delayed_preloc_face_node_count_[prelocI]),
            static_cast<std::uint16_t>(num_face_nodes)};
          delayed_nonlocal_incoming_faces_[cell.local_id][f] = 1;
          incoming_face_kinds_[face_storage_index] = IncomingFaceKind::DELAYED_NONLOCAL;
          delayed_nonlocal_face_info_by_cell_[cell.local_id][f] = info;
          delayed_nonlocal_face_info_by_key_.emplace(
            CellFaceKey{cell.global_id, static_cast<unsigned int>(f)}, info);
          delayed_preloc_face_node_count_[prelocI] += num_face_nodes;
        }
        else
        {
          incoming_face_kinds_[face_storage_index] = IncomingFaceKind::NORMAL_NONLOCAL;
          ++num_incoming_nonlocal_faces_;
          IncomingNonlocalFaceInfo info{
            static_cast<std::uint32_t>(cell.local_id),
            static_cast<std::uint32_t>(num_incoming_nonlocal_face_nodes_),
            num_face_nodes};
          incoming_nonlocal_face_info_[face_storage_index] = info;
          incoming_nonlocal_face_info_by_key_.emplace(
            CellFaceKey{cell.global_id, static_cast<unsigned int>(f)}, face_storage_index);
          num_incoming_nonlocal_face_nodes_ += num_face_nodes;
        }
      }
      else if (orientation == FaceOrientation::OUTGOING)
      {
        outgoing_face_kinds_[face_storage_index] = OutgoingFaceKind::NORMAL_NONLOCAL;
        ++num_outgoing_nonlocal_faces_;
        const int locality = face.GetNeighborPartitionID(&grid);
        const auto associated_face =
          static_cast<unsigned int>(grid_nodal_mappings[cell.local_id][f].associated_face_);
        const auto [_, inserted] =
          outgoing_nonlocal_face_keys.emplace(locality, face.neighbor_id, associated_face);
        if (not inserted)
        {
          std::ostringstream out;
          out << "CBC_FLUDSCommonData: duplicate outgoing nonlocal face key detected."
              << " cell_local_id=" << cell.local_id << " cell_global_id=" << cell.global_id
              << " face_id=" << f << " neighbor_loc=" << locality
              << " neighbor_cell_global_id=" << face.neighbor_id
              << " associated_face=" << associated_face;
          throw std::logic_error(out.str());
        }
        const auto deplocI = static_cast<std::size_t>(spds.MapLocJToDeplocI(locality));
        ++outgoing_nonlocal_face_counts_[deplocI];
        outgoing_nonlocal_face_node_counts_[deplocI] +=
          grid_nodal_mappings[cell.local_id][f].face_node_mapping_.size();
        outgoing_nonlocal_face_info_[face_storage_index] = OutgoingNonlocalFaceInfo{
          locality,
          face.neighbor_id,
          associated_face,
          static_cast<std::uint32_t>(
            grid_nodal_mappings[cell.local_id][f].face_node_mapping_.size())};
      }
      else if (orientation == FaceOrientation::INCOMING)
      {
        incoming_face_kinds_[face_storage_index] = IncomingFaceKind::NONE;
      }
    }
  }
}

bool
CBC_FLUDSCommonData::IsDelayedLocalIncomingFace(const std::uint32_t cell_local_id,
                                                const unsigned int face_id) const noexcept
{
  return delayed_local_incoming_faces_[cell_local_id][face_id] != 0;
}

bool
CBC_FLUDSCommonData::IsDelayedLocalOutgoingFace(const std::uint32_t cell_local_id,
                                                const unsigned int face_id) const noexcept
{
  return delayed_local_outgoing_faces_[cell_local_id][face_id] != 0;
}

bool
CBC_FLUDSCommonData::IsDelayedNonlocalIncomingFace(const std::uint32_t cell_local_id,
                                                   const unsigned int face_id) const noexcept
{
  return delayed_nonlocal_incoming_faces_[cell_local_id][face_id] != 0;
}

const CBC_FLUDSCommonData::IncomingNonlocalFaceInfo&
CBC_FLUDSCommonData::GetIncomingNonlocalFaceInfo(const std::uint32_t cell_local_id,
                                                 const unsigned int face_id) const noexcept
{
  return incoming_nonlocal_face_info_[cell_face_offsets_[cell_local_id] + face_id];
}

const CBC_FLUDSCommonData::IncomingNonlocalFaceInfo&
CBC_FLUDSCommonData::GetIncomingNonlocalFaceInfoByKey(const std::uint64_t cell_global_id,
                                                      const unsigned int face_id) const noexcept
{
  return incoming_nonlocal_face_info_[GetIncomingNonlocalFaceStorageIndexByKey(cell_global_id,
                                                                               face_id)];
}

const CBC_FLUDSCommonData::IncomingNonlocalFaceInfo&
CBC_FLUDSCommonData::GetIncomingNonlocalFaceInfoByStorageIndex(
  const std::size_t storage_index) const noexcept
{
  return incoming_nonlocal_face_info_[storage_index];
}

std::size_t
CBC_FLUDSCommonData::GetIncomingNonlocalFaceStorageIndexByKey(
  const std::uint64_t cell_global_id, const unsigned int face_id) const noexcept
{
  const auto it = incoming_nonlocal_face_info_by_key_.find({cell_global_id, face_id});
  assert(it != incoming_nonlocal_face_info_by_key_.end());
  return it->second;
}

const CBC_FLUDSCommonData::OutgoingNonlocalFaceInfo&
CBC_FLUDSCommonData::GetOutgoingNonlocalFaceInfo(const std::uint32_t cell_local_id,
                                                 const unsigned int face_id) const noexcept
{
  return outgoing_nonlocal_face_info_[cell_face_offsets_[cell_local_id] + face_id];
}

const CBC_FLUDSCommonData::DelayedLocalFaceInfo&
CBC_FLUDSCommonData::GetDelayedLocalFaceInfo(const std::uint32_t cell_local_id,
                                             const unsigned int face_id) const noexcept
{
  const auto& info =
    delayed_local_face_info_by_storage_index_[cell_face_offsets_[cell_local_id] + face_id];
  assert(info.num_face_nodes != 0);
  return info;
}

const CBC_FLUDSCommonData::DelayedNonlocalFaceInfo&
CBC_FLUDSCommonData::GetDelayedNonlocalFaceInfo(const std::uint32_t cell_local_id,
                                                const unsigned int face_id) const noexcept
{
  return delayed_nonlocal_face_info_by_cell_[cell_local_id][face_id];
}

bool
CBC_FLUDSCommonData::TryGetDelayedNonlocalFaceInfo(const std::uint64_t cell_global_id,
                                                   const unsigned int face_id,
                                                   DelayedNonlocalFaceInfo& info) const noexcept
{
  const auto it = delayed_nonlocal_face_info_by_key_.find({cell_global_id, face_id});
  if (it == delayed_nonlocal_face_info_by_key_.end())
    return false;
  info = it->second;
  return true;
}

} // namespace opensn

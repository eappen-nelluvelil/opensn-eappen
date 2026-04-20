// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"

#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "caribou/main.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace crb = caribou;

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

std::uint64_t
CBCD_FLUDSCommonData::PackCellFaceKey(const std::uint64_t cell_global_id,
                                      const unsigned int face_id) noexcept
{
  return (cell_global_id << 32) | static_cast<std::uint64_t>(face_id);
}

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

void
CBCD_FLUDSCommonData::CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm)
{
  const MeshContinuum& grid = *(spds_.GetGrid());
  const auto& cbc_spds = static_cast<const CBC_SPDS&>(spds_);
  const auto& face_orientations = spds_.GetCellFaceOrientations();
  const auto& local_face_slot_ids = cbc_spds.GetLocalFaceSlotIDs();
  const auto& local_face_slot_node_offsets = cbc_spds.GetLocalFaceSlotNodeOffsets();
  const auto& delayed_loc_deps = spds_.GetDelayedLocationDependencies();

  std::unordered_set<std::uint64_t> delayed_local_edges;
  delayed_local_edges.reserve(spds_.GetLocalSweepFAS().size());
  for (const auto& [upwind, downwind] : spds_.GetLocalSweepFAS())
    delayed_local_edges.insert(PackEdge(upwind, downwind));

  cell_face_offsets_.assign(grid.local_cells.size() + 1, 0);
  std::uint64_t total_face_nodes = 0;
  std::size_t total_num_faces = 0;
  for (const auto& cell : grid.local_cells)
  {
    cell_face_offsets_[cell.local_id] = static_cast<std::uint32_t>(total_num_faces);
    total_num_faces += cell.faces.size();
    for (std::uint32_t f = 0; f < cell.faces.size(); ++f)
      total_face_nodes += sdm.GetCellMapping(cell).GetNumFaceNodes(f);
  }
  cell_face_offsets_.back() = static_cast<std::uint32_t>(total_num_faces);

  delayed_local_faces_.assign(total_num_faces, {});
  delayed_incoming_nonlocal_faces_.assign(total_num_faces, {});
  delayed_incoming_nonlocal_node_counts_.assign(delayed_loc_deps.size(), 0);
  delayed_incoming_nonlocal_node_offsets_.assign(delayed_loc_deps.size() + 1, 0);
  delayed_incoming_face_lookup_.reserve(total_num_faces);

  for (const auto& cell : grid.local_cells)
  {
    const auto face_offset = cell_face_offsets_[cell.local_id];
    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = face_orientations[cell.local_id][f];
      if ((orientation != FaceOrientation::INCOMING) || !face.has_neighbor)
        continue;

      const auto num_face_nodes =
        static_cast<std::uint16_t>(sdm.GetCellMapping(cell).GetNumFaceNodes(f));
      const auto face_storage_index = face_offset + f;

      if (face.IsNeighborLocal(&grid))
      {
        const auto& upwind = grid.cells[face.neighbor_id];
        if (!delayed_local_edges.contains(PackEdge(upwind.local_id, cell.local_id)))
          continue;

        const DelayedLocalFaceInfo info{num_delayed_local_nodes_, num_face_nodes};
        delayed_local_faces_[face_storage_index] = info;
        const auto upwind_face_id =
          static_cast<unsigned int>(grid_nodal_mappings_[cell.local_id][f].associated_face_);
        const auto upwind_storage_index = cell_face_offsets_[upwind.local_id] + upwind_face_id;
        delayed_local_faces_[upwind_storage_index] = info;
        num_delayed_local_nodes_ += num_face_nodes;
      }
      else
      {
        const int source_partition = grid.cells[face.neighbor_id].partition_id;
        const auto delayed_it =
          std::find(delayed_loc_deps.begin(), delayed_loc_deps.end(), source_partition);
        if (delayed_it == delayed_loc_deps.end())
          continue;

        const auto source_slot =
          static_cast<std::uint32_t>(std::distance(delayed_loc_deps.begin(), delayed_it));
        const DelayedNonlocalFaceInfo info{static_cast<std::uint32_t>(cell.local_id),
                                           source_slot,
                                           delayed_incoming_nonlocal_node_counts_[source_slot],
                                           num_face_nodes};
        delayed_incoming_nonlocal_faces_[face_storage_index] = info;
        delayed_incoming_face_lookup_.emplace(
          PackCellFaceKey(cell.global_id, static_cast<unsigned int>(f)), info);
        ++num_delayed_incoming_nonlocal_faces_;
        delayed_incoming_nonlocal_node_counts_[source_slot] += num_face_nodes;
      }
    }
  }

  for (std::size_t source_slot = 0; source_slot < delayed_incoming_nonlocal_node_counts_.size();
       ++source_slot)
    delayed_incoming_nonlocal_node_offsets_[source_slot + 1] =
      delayed_incoming_nonlocal_node_offsets_[source_slot] +
      delayed_incoming_nonlocal_node_counts_[source_slot];

  for (auto& info : delayed_incoming_nonlocal_faces_)
    if (info.num_nodes != 0)
      info.base_storage_index += delayed_incoming_nonlocal_node_offsets_[info.source_slot];

  for (auto& [_, info] : delayed_incoming_face_lookup_)
    info.base_storage_index += delayed_incoming_nonlocal_node_offsets_[info.source_slot];

  const size_t offsets_size = 2 * grid.local_cells.size();
  const size_t total_size = offsets_size + total_face_nodes;
  std::vector<std::uint64_t> local_map(total_size);
  std::uint64_t* cell_offsets_ptr = local_map.data();
  std::uint64_t* indices_ptr = local_map.data() + offsets_size;
  std::uint64_t current_index_offset = offsets_size;
  std::uint64_t local_indices_filled = 0;

  cell_to_outgoing_boundary_node_offsets_.assign(grid.local_cells.size() + 1, 0);
  cell_to_incoming_nonlocal_face_offsets_.assign(grid.local_cells.size() + 1, 0);
  cell_to_outgoing_nonlocal_face_offsets_.assign(grid.local_cells.size() + 1, 0);

  std::unordered_map<int, std::uint32_t> locality_to_dest_slot;
  std::unordered_map<int, std::uint32_t> source_partition_to_slot;
  outgoing_localities_.reserve(grid.local_cells.size());
  incoming_source_partitions_.reserve(grid.local_cells.size());
  outgoing_boundary_nodes_.reserve(total_face_nodes);
  outgoing_nonlocal_face_node_copies_.reserve(total_face_nodes);

  struct SourceLookupBuild
  {
    std::uint32_t source_slot = 0;
    IncomingFaceLookup lookup;
  };
  std::vector<SourceLookupBuild> incoming_lookup_build;
  incoming_lookup_build.reserve(total_face_nodes);

  const auto update_cell_offsets = [this](const std::uint64_t cell_local_id)
  {
    cell_to_outgoing_boundary_node_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(outgoing_boundary_nodes_.size());
    cell_to_incoming_nonlocal_face_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(incoming_nonlocal_faces_.size());
    cell_to_outgoing_nonlocal_face_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(outgoing_nonlocal_faces_.size());
  };

  for (const auto& cell : grid.local_cells)
  {
    update_cell_offsets(cell.local_id);
    cell_offsets_ptr[2 * cell.local_id] = current_index_offset;
    std::uint64_t num_cell_nodes = 0;
    std::vector<int> incoming_face_to_grouped_index(cell.faces.size(), -1);
    std::vector<int> outgoing_face_to_grouped_index(cell.faces.size(), -1);

    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const CellFace& face = cell.faces[f];
      const auto orientation = face_orientations[cell.local_id][f];
      const auto& face_nodal_mapping = grid_nodal_mappings_[cell.local_id][f];
      const auto num_face_nodes = sdm.GetCellMapping(cell).GetNumFaceNodes(f);
      const bool is_outgoing_face = (orientation == FaceOrientation::OUTGOING);
      const bool is_incoming_face = (orientation == FaceOrientation::INCOMING);
      const bool is_local_face = face.IsNeighborLocal(&grid);
      const bool is_boundary_face = not face.has_neighbor;
      const auto face_storage_index = cell_face_offsets_[cell.local_id] + f;
      const auto delayed_local_info = delayed_local_faces_[face_storage_index];
      const auto delayed_nonlocal_info = delayed_incoming_nonlocal_faces_[face_storage_index];

      for (std::size_t fn = 0; fn < num_face_nodes; ++fn)
      {
        CBCD_NodeIndex node_index;

        if (is_incoming_face)
        {
          if (is_local_face)
          {
            if (delayed_local_info.num_nodes != 0)
            {
              const auto local_face_node =
                static_cast<std::uint64_t>(face_nodal_mapping.face_node_mapping_[fn]);
              node_index = CBCD_NodeIndex(
                static_cast<std::uint64_t>(delayed_local_info.base_storage_index) + local_face_node,
                false,
                true,
                true);
            }
            else
            {
              const auto task_id = cbc_spds.GetIncomingLocalFaceTaskID(
                static_cast<std::uint32_t>(cell.local_id), static_cast<unsigned int>(f));
              const auto slot_id = local_face_slot_ids[task_id];
              const auto local_face_node =
                static_cast<std::uint64_t>(face_nodal_mapping.face_node_mapping_[fn]);
              node_index = CBCD_NodeIndex(
                static_cast<std::uint64_t>(local_face_slot_node_offsets[slot_id]) + local_face_node,
                false,
                true,
                false);
            }
          }
          else if (!is_boundary_face)
          {
            if (delayed_nonlocal_info.num_nodes != 0)
            {
              node_index = CBCD_NodeIndex(
                static_cast<std::uint64_t>(delayed_nonlocal_info.base_storage_index + fn),
                false,
                false,
                true);
            }
            else
            {
              node_index = CBCD_NodeIndex(num_incoming_nonlocal_nodes_, false, false, false);
              int& grouped_face_index = incoming_face_to_grouped_index[f];
              if (grouped_face_index < 0)
              {
                grouped_face_index =
                  static_cast<int>(incoming_nonlocal_faces_.size() -
                                   cell_to_incoming_nonlocal_face_offsets_[cell.local_id]);
                auto& grouped_face = incoming_nonlocal_faces_.emplace_back();
                const int source_partition = grid.cells[face.neighbor_id].partition_id;
                auto [source_it, inserted] = source_partition_to_slot.try_emplace(
                  source_partition, static_cast<std::uint32_t>(incoming_source_partitions_.size()));
                if (inserted)
                  incoming_source_partitions_.push_back(source_partition);
                grouped_face.cell_local_id = static_cast<std::uint32_t>(cell.local_id);
                grouped_face.base_storage_index =
                  static_cast<std::uint32_t>(num_incoming_nonlocal_nodes_);
                grouped_face.source_slot = source_it->second;
                incoming_lookup_build.push_back(
                  {grouped_face.source_slot,
                   {cell.global_id,
                    static_cast<unsigned int>(f),
                    static_cast<std::uint32_t>(incoming_nonlocal_faces_.size() - 1)}});
                ++num_incoming_nonlocal_faces_;
              }

              auto& grouped_face =
                incoming_nonlocal_faces_[cell_to_incoming_nonlocal_face_offsets_[cell.local_id] +
                                         grouped_face_index];
              ++grouped_face.num_nodes;
              ++num_incoming_nonlocal_nodes_;
            }
          }
          else
          {
            node_index = CBCD_NodeIndex(num_incoming_boundary_nodes_, false);
            if (fn == 0)
            {
              incoming_boundary_face_plans_.push_back(
                {face.neighbor_id,
                 static_cast<std::uint32_t>(cell.local_id),
                 static_cast<unsigned int>(f),
                 0,
                 static_cast<std::uint32_t>(num_incoming_boundary_nodes_),
                 static_cast<std::uint16_t>(num_face_nodes)});
            }
            ++num_incoming_boundary_nodes_;
          }
        }
        else if (is_outgoing_face)
        {
          if (is_local_face)
          {
            if (delayed_local_info.num_nodes != 0)
            {
              node_index = CBCD_NodeIndex(
                static_cast<std::uint64_t>(delayed_local_info.base_storage_index + fn),
                true,
                true,
                true);
            }
            else
            {
              const auto task_id = cbc_spds.GetOutgoingLocalFaceTaskID(
                static_cast<std::uint32_t>(cell.local_id), static_cast<unsigned int>(f));
              const auto slot_id = local_face_slot_ids[task_id];
              node_index =
                CBCD_NodeIndex(static_cast<std::uint64_t>(local_face_slot_node_offsets[slot_id]) +
                                 static_cast<std::uint64_t>(fn),
                               true,
                               true,
                               false);
            }
          }
          else if (!is_boundary_face)
          {
            node_index = CBCD_NodeIndex(num_outgoing_nonlocal_nodes_, true, false, false);
            int& grouped_face_index = outgoing_face_to_grouped_index[f];
            if (grouped_face_index < 0)
            {
              const int locality = grid.cells[face.neighbor_id].partition_id;
              auto dest_slot_it = locality_to_dest_slot.find(locality);
              std::uint32_t dest_slot = 0;
              if (dest_slot_it == locality_to_dest_slot.end())
              {
                dest_slot = static_cast<std::uint32_t>(outgoing_localities_.size());
                locality_to_dest_slot.emplace(locality, dest_slot);
                outgoing_localities_.push_back(locality);
              }
              else
                dest_slot = dest_slot_it->second;

              grouped_face_index =
                static_cast<int>(outgoing_nonlocal_faces_.size() -
                                 cell_to_outgoing_nonlocal_face_offsets_[cell.local_id]);
              auto& grouped_face = outgoing_nonlocal_faces_.emplace_back();
              grouped_face.cell_global_id = face.neighbor_id;
              grouped_face.dest_slot = dest_slot;
              grouped_face.face_id = static_cast<unsigned int>(face_nodal_mapping.associated_face_);
              grouped_face.num_face_nodes = static_cast<std::uint16_t>(num_face_nodes);
              grouped_face.node_copy_offset =
                static_cast<std::uint32_t>(outgoing_nonlocal_face_node_copies_.size());
              ++num_outgoing_nonlocal_faces_;
            }

            auto& grouped_face =
              outgoing_nonlocal_faces_[cell_to_outgoing_nonlocal_face_offsets_[cell.local_id] +
                                       grouped_face_index];
            outgoing_nonlocal_face_node_copies_.push_back(
              {static_cast<std::uint32_t>(num_outgoing_nonlocal_nodes_),
               static_cast<std::uint16_t>(face_nodal_mapping.face_node_mapping_[fn])});
            ++grouped_face.num_node_copies;
            ++num_outgoing_nonlocal_nodes_;
          }
          else
          {
            node_index = CBCD_NodeIndex(num_outgoing_boundary_nodes_, true);
            outgoing_boundary_nodes_.push_back(
              {face.neighbor_id,
               static_cast<std::uint32_t>(cell.local_id),
               static_cast<unsigned int>(f),
               static_cast<std::uint32_t>(num_outgoing_boundary_nodes_),
               static_cast<std::uint16_t>(fn)});
            ++num_outgoing_boundary_nodes_;
          }
        }
        else
        {
          node_index = CBCD_NodeIndex();
        }

        indices_ptr[local_indices_filled++] = node_index.GetCoreValue();
      }

      num_cell_nodes += num_face_nodes;
    }

    update_cell_offsets(cell.local_id + 1);
    cell_offsets_ptr[2 * cell.local_id + 1] = num_cell_nodes;
    current_index_offset += num_cell_nodes;
  }

  source_to_incoming_face_offsets_.assign(incoming_source_partitions_.size() + 1, 0);
  for (const auto& build : incoming_lookup_build)
    ++source_to_incoming_face_offsets_[build.source_slot + 1];
  for (std::size_t i = 0; i < incoming_source_partitions_.size(); ++i)
    source_to_incoming_face_offsets_[i + 1] += source_to_incoming_face_offsets_[i];

  incoming_face_lookups_by_source_.resize(incoming_lookup_build.size());
  auto source_write_offsets = source_to_incoming_face_offsets_;
  for (const auto& build : incoming_lookup_build)
    incoming_face_lookups_by_source_[source_write_offsets[build.source_slot]++] = build.lookup;

  for (std::size_t source_slot = 0; source_slot < incoming_source_partitions_.size(); ++source_slot)
  {
    const auto begin = source_to_incoming_face_offsets_[source_slot];
    const auto end = source_to_incoming_face_offsets_[source_slot + 1];
    std::sort(incoming_face_lookups_by_source_.begin() + begin,
              incoming_face_lookups_by_source_.begin() + end,
              [](const IncomingFaceLookup& lhs, const IncomingFaceLookup& rhs)
              {
                return std::pair<std::uint64_t, unsigned int>{lhs.cell_global_id, lhs.face_id} <
                       std::pair<std::uint64_t, unsigned int>{rhs.cell_global_id, rhs.face_id};
              });
  }

  if (local_map.empty())
    return;

  crb::HostVector<std::uint64_t> host_mem(local_map.begin(), local_map.end());
  crb::DeviceMemory<std::uint64_t> device_mem(local_map.size());
  crb::copy(device_mem, host_mem, host_mem.size());
  device_cell_face_node_map_ = device_mem.release();
}

const GroupedIncomingNonlocalFace&
CBCD_FLUDSCommonData::FindIncomingNonlocalFace(const std::uint32_t source_slot,
                                               const std::uint64_t cell_global_id,
                                               const unsigned int face_id) const
{
  const auto begin = source_to_incoming_face_offsets_[source_slot];
  const auto end = source_to_incoming_face_offsets_[source_slot + 1];
  const auto it = std::lower_bound(
    incoming_face_lookups_by_source_.begin() + begin,
    incoming_face_lookups_by_source_.begin() + end,
    std::pair<std::uint64_t, unsigned int>{cell_global_id, face_id},
    [](const IncomingFaceLookup& lhs, const std::pair<std::uint64_t, unsigned int>& rhs)
    { return std::pair<std::uint64_t, unsigned int>{lhs.cell_global_id, lhs.face_id} < rhs; });
  assert(it != incoming_face_lookups_by_source_.begin() + end);
  assert(it->cell_global_id == cell_global_id);
  assert(it->face_id == face_id);
  return incoming_nonlocal_faces_[it->face_index];
}

const CBCD_FLUDSCommonData::DelayedLocalFaceInfo&
CBCD_FLUDSCommonData::GetDelayedLocalFace(const std::uint32_t cell_local_id,
                                          const unsigned int face_id) const
{
  const auto& info = delayed_local_faces_[cell_face_offsets_[cell_local_id] + face_id];
  assert(info.num_nodes != 0);
  return info;
}

const CBCD_FLUDSCommonData::DelayedNonlocalFaceInfo&
CBCD_FLUDSCommonData::GetDelayedIncomingNonlocalFace(const std::uint32_t cell_local_id,
                                                     const unsigned int face_id) const
{
  const auto& info = delayed_incoming_nonlocal_faces_[cell_face_offsets_[cell_local_id] + face_id];
  assert(info.num_nodes != 0);
  return info;
}

bool
CBCD_FLUDSCommonData::TryFindDelayedIncomingNonlocalFace(const std::uint64_t cell_global_id,
                                                         const unsigned int face_id,
                                                         DelayedNonlocalFaceInfo& info) const
{
  const auto it = delayed_incoming_face_lookup_.find(PackCellFaceKey(cell_global_id, face_id));
  if (it == delayed_incoming_face_lookup_.end())
    return false;
  info = it->second;
  return true;
}

void
CBCD_FLUDSCommonData::DeallocateDeviceMemory()
{
  if (device_cell_face_node_map_ != nullptr)
  {
    crb::DeviceMemory<std::uint64_t> device_cell_face_node_map(device_cell_face_node_map_);
    device_cell_face_node_map.reset();
    device_cell_face_node_map_ = nullptr;
  }
}

} // namespace opensn

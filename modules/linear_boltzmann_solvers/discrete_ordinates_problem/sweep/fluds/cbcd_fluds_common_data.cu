// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caribou/main.hpp"
#include <cstring>

namespace crb = caribou;

namespace opensn
{

void
CBCD_FLUDSCommonData::CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm)
{
  const MeshContinuum& grid = *(spds_.GetGrid());
  const size_t num_local_cells = grid.local_cells.size();
  const auto& face_orientations = spds_.GetCellFaceOrientations();
  std::uint64_t total_face_nodes = 0;
  for (const auto& cell : grid.local_cells)
    for (std::uint32_t f = 0; f < cell.faces.size(); ++f)
      total_face_nodes += sdm.GetCellMapping(cell).GetNumFaceNodes(f);
  std::vector<size_t> cell_spatial_dof_offsets(num_local_cells);
  size_t current_dof_offset = 0;
  for (const auto& cell : grid.local_cells)
  {
    cell_spatial_dof_offsets[cell.local_id] = current_dof_offset;
    current_dof_offset += sdm.GetCellMapping(cell).GetNumNodes();
  }
  const size_t offsets_size = 2 * num_local_cells;
  const size_t total_size = offsets_size + total_face_nodes;
  std::vector<std::uint64_t> local_map(total_size);
  std::uint64_t* cell_offsets_ptr = local_map.data();
  std::uint64_t* indices_ptr = local_map.data() + offsets_size;
  std::uint64_t current_index_offset = offsets_size;
  std::uint64_t local_indices_filled = 0;
  std::unordered_map<int, std::uint32_t> locality_to_dest_slot;
  incoming_face_map_.reserve(num_local_cells);
  outgoing_localities_.reserve(num_local_cells);
  outgoing_boundary_nodes_.reserve(total_face_nodes);
  incoming_nonlocal_face_nodes_.reserve(total_face_nodes);
  outgoing_nonlocal_face_node_copies_.reserve(total_face_nodes);

  for (const auto& cell : grid.local_cells)
  {
    cell_to_outgoing_boundary_node_offsets_[cell.local_id] =
      static_cast<std::uint32_t>(outgoing_boundary_nodes_.size());
    cell_to_incoming_nonlocal_face_offsets_[cell.local_id] =
      static_cast<std::uint32_t>(incoming_nonlocal_faces_.size());
    cell_to_outgoing_nonlocal_face_offsets_[cell.local_id] =
      static_cast<std::uint32_t>(outgoing_nonlocal_faces_.size());

    cell_offsets_ptr[2 * cell.local_id] = current_index_offset;
    std::uint64_t num_cell_nodes = 0;
    std::vector<int> incoming_face_to_grouped_index(cell.faces.size(), -1);
    std::vector<int> outgoing_face_to_grouped_index(cell.faces.size(), -1);
    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const CellFace& face = cell.faces[f];
      const FaceOrientation& orientation = face_orientations[cell.local_id][f];
      const FaceNodalMapping& face_nodal_mapping = grid_nodal_mappings_[cell.local_id][f];
      const size_t num_face_nodes = sdm.GetCellMapping(cell).GetNumFaceNodes(f);
      const bool is_outgoing_face = (orientation == FaceOrientation::OUTGOING);
      const bool is_incoming_face = (orientation == FaceOrientation::INCOMING);
      const bool is_local_face = face.IsNeighborLocal(&grid);
      const bool is_boundary_face = not face.has_neighbor;
      for (size_t fn = 0; fn < num_face_nodes; ++fn)
      {
        CBCD_NodeIndex node_index;

        if (is_incoming_face)
        {
          if (is_local_face)
          {
            std::uint32_t nbr_local_idx = face.GetNeighborLocalID(&grid);
            std::uint32_t adj_cell_node = face_nodal_mapping.cell_node_mapping_[fn];
            const std::uint64_t index = cell_spatial_dof_offsets[nbr_local_idx] + adj_cell_node;
            node_index = CBCD_NodeIndex(index, is_outgoing_face, is_local_face);
          }
          else if (not is_boundary_face)
          {
            node_index =
              CBCD_NodeIndex(num_incoming_nonlocal_nodes_, is_outgoing_face, is_local_face);
            int& grouped_face_index = incoming_face_to_grouped_index[f];
            if (grouped_face_index < 0)
            {
              grouped_face_index = static_cast<int>(incoming_nonlocal_faces_.size() -
                                                    cell_to_incoming_nonlocal_face_offsets_[cell.local_id]);
              auto& grouped_face = incoming_nonlocal_faces_.emplace_back();
              grouped_face.node_offset = static_cast<std::uint32_t>(incoming_nonlocal_face_nodes_.size());
              incoming_face_map_.emplace(IncomingFaceKey{cell.global_id, static_cast<unsigned int>(f)},
                                         IncomingFaceRef{static_cast<std::uint32_t>(cell.local_id),
                                                         static_cast<std::uint32_t>(
                                                           cell_to_incoming_nonlocal_face_offsets_[cell.local_id] +
                                                           grouped_face_index)});
              ++num_incoming_nonlocal_faces_;
            }

            auto& grouped_face =
              incoming_nonlocal_faces_[cell_to_incoming_nonlocal_face_offsets_[cell.local_id] +
                                      grouped_face_index];
            incoming_nonlocal_face_nodes_.emplace_back(NonlocalNodeInfo{cell.global_id,
                                                                        static_cast<unsigned int>(f),
                                                                        static_cast<std::uint32_t>(
                                                                          num_incoming_nonlocal_nodes_),
                                                                        static_cast<std::uint16_t>(fn),
                                                                        face_nodal_mapping.face_node_mapping_[fn]});
            ++grouped_face.num_nodes;
            ++num_incoming_nonlocal_nodes_;
          }
          else
          {
            node_index = CBCD_NodeIndex(num_incoming_boundary_nodes_, is_outgoing_face);
            incoming_boundary_node_map_.emplace_back(
              BoundaryNodeInfo{face.neighbor_id,
                               static_cast<std::uint32_t>(cell.local_id),
                               static_cast<unsigned int>(f),
                               static_cast<std::uint32_t>(num_incoming_boundary_nodes_),
                               static_cast<std::uint16_t>(fn)});
            ++num_incoming_boundary_nodes_;
          }
        }
        else if (is_outgoing_face)
        {
          if (is_local_face)
          {
            const int cell_node = sdm.GetCellMapping(cell).MapFaceNode(f, fn);
            const std::uint64_t index = cell_spatial_dof_offsets[cell.local_id] + cell_node;
            node_index = CBCD_NodeIndex(index, is_outgoing_face, is_local_face);
          }
          else if (not is_boundary_face)
          {
            node_index =
              CBCD_NodeIndex(num_outgoing_nonlocal_nodes_, is_outgoing_face, is_local_face);
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

              grouped_face_index = static_cast<int>(outgoing_nonlocal_faces_.size() -
                                                    cell_to_outgoing_nonlocal_face_offsets_[cell.local_id]);
              auto& grouped_face = outgoing_nonlocal_faces_.emplace_back();
              grouped_face.pack_plan_index =
                static_cast<std::uint32_t>(outgoing_nonlocal_faces_.size() - 1);
              grouped_face.dest_slot = dest_slot;
              grouped_face.num_face_nodes = static_cast<std::uint16_t>(num_face_nodes);
              grouped_face.node_copy_offset =
                static_cast<std::uint32_t>(outgoing_nonlocal_face_node_copies_.size());
              std::memcpy(grouped_face.entry_header_prefix.data(),
                          &face.neighbor_id,
                          sizeof(std::uint64_t));
              const auto associated_face =
                static_cast<unsigned int>(face_nodal_mapping.associated_face_);
              std::memcpy(grouped_face.entry_header_prefix.data() + sizeof(std::uint64_t),
                          &associated_face,
                          sizeof(unsigned int));
              ++num_outgoing_nonlocal_faces_;
            }

            auto& grouped_face =
              outgoing_nonlocal_faces_[cell_to_outgoing_nonlocal_face_offsets_[cell.local_id] +
                                       grouped_face_index];
            outgoing_nonlocal_face_node_copies_.push_back(
              {static_cast<std::uint32_t>(num_outgoing_nonlocal_nodes_),
               static_cast<std::uint16_t>(fn)});
            ++grouped_face.num_node_copies;
            ++num_outgoing_nonlocal_nodes_;
          }
          else
          {
            node_index = CBCD_NodeIndex(num_outgoing_boundary_nodes_, is_outgoing_face);
            outgoing_boundary_nodes_.emplace_back(
              BoundaryNodeInfo{face.neighbor_id,
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
    cell_to_outgoing_boundary_node_offsets_[cell.local_id + 1] =
      static_cast<std::uint32_t>(outgoing_boundary_nodes_.size());
    cell_to_incoming_nonlocal_face_offsets_[cell.local_id + 1] =
      static_cast<std::uint32_t>(incoming_nonlocal_faces_.size());
    cell_to_outgoing_nonlocal_face_offsets_[cell.local_id + 1] =
      static_cast<std::uint32_t>(outgoing_nonlocal_faces_.size());
    cell_offsets_ptr[2 * cell.local_id + 1] = num_cell_nodes;
    current_index_offset += num_cell_nodes;
  }
  if (local_map.empty())
    return;
  crb::HostVector<std::uint64_t> host_mem(local_map.begin(), local_map.end());
  crb::DeviceMemory<std::uint64_t> device_mem(local_map.size());
  crb::copy(device_mem, host_mem, host_mem.size());
  device_cell_face_node_map_ = device_mem.release();
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

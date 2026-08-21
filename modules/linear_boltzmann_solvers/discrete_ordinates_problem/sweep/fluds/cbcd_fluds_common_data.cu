// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caribou/main.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace crb = caribou;

namespace opensn
{

namespace
{

/// Pack a directed local edge `(producer_cell_local_id, producer_face_id)` into one 64-bit key.
constexpr std::uint64_t
PackProducerFaceKey(std::uint32_t producer_cell_local_id, unsigned int producer_face_id) noexcept
{
  return (static_cast<std::uint64_t>(producer_cell_local_id) << 32) |
         static_cast<std::uint64_t>(producer_face_id);
}

} // namespace

void
CBCD_FLUDSCommonData::CopyFlattenedNodeIndexToDevice(const SpatialDiscretization& sdm)
{
  const MeshContinuum& grid = *(spds_.GetGrid());
  const auto& cbc_spds = static_cast<const CBC_SPDS&>(spds_);
  const size_t num_local_cells = grid.local_cells.size();
  const auto& face_orientations = spds_.GetCellFaceOrientations();
  const auto& local_face_slot_ids = cbc_spds.GetLocalFaceSlotIDs();
  const auto& local_face_slot_node_offsets = cbc_spds.GetLocalFaceSlotNodeOffsets();

  // Cycle-aware metadata.  When the CBC SPDS has been built with `allow_cycles = true` these
  // lists hold the upstream/downstream ranks whose dependency was removed by the global
  // feedback-arc set; otherwise both lists are empty and the delayed routes below are never
  // taken.
  const auto& delayed_location_dependencies = spds_.GetDelayedLocationDependencies();
  const auto& delayed_location_successors = spds_.GetDelayedLocationSuccessors();
  const auto is_delayed_source_partition = [&](int partition) noexcept
  {
    return std::find(delayed_location_dependencies.begin(),
                     delayed_location_dependencies.end(),
                     partition) != delayed_location_dependencies.end();
  };
  const auto is_delayed_dest_partition = [&](int partition) noexcept
  {
    return std::find(delayed_location_successors.begin(),
                     delayed_location_successors.end(),
                     partition) != delayed_location_successors.end();
  };

  // Canonical (producer_cell, producer_face) -> delayed-local-bank node offset.  The same
  // directed edge appears once as a producer's outgoing face and once as a consumer's
  // incoming face; both sides must use the same offset so the kernel's outgoing write hits
  // the bank slot that the next sweep's downwind read consumes.
  std::unordered_map<std::uint64_t, std::uint32_t> delayed_local_offset_by_producer_face;
  std::uint64_t total_face_nodes = 0;
  for (const auto& cell : grid.local_cells)
    for (std::uint32_t f = 0; f < cell.faces.size(); ++f)
      total_face_nodes += sdm.GetCellMapping(cell).GetNumFaceNodes(f);

  const size_t offsets_size = 2 * num_local_cells;
  const size_t total_size = offsets_size + total_face_nodes;
  std::vector<std::uint64_t> local_map(total_size);
  std::uint64_t* cell_offsets_ptr = local_map.data();
  std::uint64_t* indices_ptr = local_map.data() + offsets_size;
  std::uint64_t current_index_offset = offsets_size;
  std::uint64_t local_indices_filled = 0;

  cell_to_outgoing_boundary_node_offsets_.assign(num_local_cells + 1, 0);
  cell_to_incoming_nonlocal_face_offsets_.assign(num_local_cells + 1, 0);
  cell_to_outgoing_nonlocal_face_offsets_.assign(num_local_cells + 1, 0);
  cell_to_delayed_incoming_nonlocal_face_offsets_.assign(num_local_cells + 1, 0);
  cell_to_delayed_outgoing_nonlocal_face_offsets_.assign(num_local_cells + 1, 0);

  std::unordered_map<int, std::uint32_t> locality_to_dest_slot;
  std::unordered_map<int, std::uint32_t> source_partition_to_slot;
  std::unordered_map<int, std::uint32_t> delayed_locality_to_dest_slot;
  std::unordered_map<int, std::uint32_t> delayed_source_partition_to_slot;
  outgoing_localities_.reserve(num_local_cells);
  incoming_source_partitions_.reserve(num_local_cells);
  outgoing_boundary_nodes_.reserve(total_face_nodes);
  outgoing_nonlocal_face_node_copies_.reserve(total_face_nodes);
  delayed_outgoing_nonlocal_face_node_copies_.reserve(total_face_nodes);
  struct OrderedIncomingFaceBuild
  {
    std::uint32_t source_slot = 0;
    std::uint64_t cell_global_id = 0;
    unsigned int face_id = 0;
    std::uint32_t face_index = 0;
  };
  struct OrderedOutgoingFaceBuild
  {
    std::uint32_t dest_slot = 0;
    std::uint64_t cell_global_id = 0;
    unsigned int face_id = 0;
    std::uint32_t face_index = 0;
  };
  std::vector<OrderedIncomingFaceBuild> incoming_face_order;
  std::vector<OrderedOutgoingFaceBuild> outgoing_face_order;
  std::vector<OrderedIncomingFaceBuild> delayed_incoming_face_order;
  std::vector<OrderedOutgoingFaceBuild> delayed_outgoing_face_order;
  incoming_face_order.reserve(total_face_nodes);
  outgoing_face_order.reserve(total_face_nodes);

  const auto update_cell_offsets = [this](const std::uint64_t cell_local_id)
  {
    cell_to_outgoing_boundary_node_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(outgoing_boundary_nodes_.size());
    cell_to_incoming_nonlocal_face_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(incoming_nonlocal_faces_.size());
    cell_to_outgoing_nonlocal_face_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(outgoing_nonlocal_faces_.size());
    cell_to_delayed_incoming_nonlocal_face_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(delayed_incoming_nonlocal_faces_.size());
    cell_to_delayed_outgoing_nonlocal_face_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(delayed_outgoing_nonlocal_faces_.size());
  };

  for (const auto& cell : grid.local_cells)
  {
    update_cell_offsets(cell.local_id);

    cell_offsets_ptr[2 * cell.local_id] = current_index_offset;
    std::uint64_t num_cell_nodes = 0;
    std::vector<int> incoming_face_to_grouped_index(cell.faces.size(), -1);
    std::vector<int> outgoing_face_to_grouped_index(cell.faces.size(), -1);
    std::vector<int> delayed_incoming_face_to_grouped_index(cell.faces.size(), -1);
    std::vector<int> delayed_outgoing_face_to_grouped_index(cell.faces.size(), -1);
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

      // Cycle-aware classification: identify faces whose dependency was removed by the
      // local or interpartition feedback-arc set and routed through the lagged banks.
      bool is_delayed_local_face = false;
      std::uint64_t delayed_local_producer_face_key = 0;
      if (is_local_face and (is_incoming_face or is_outgoing_face))
      {
        const auto adj_cell_local_id =
          static_cast<std::uint32_t>(grid.cells[face.neighbor_id].local_id);
        const auto producer_cell_local_id =
          is_outgoing_face ? static_cast<std::uint32_t>(cell.local_id) : adj_cell_local_id;
        const auto consumer_cell_local_id =
          is_outgoing_face ? adj_cell_local_id : static_cast<std::uint32_t>(cell.local_id);
        if (cbc_spds.IsDelayedLocalDependency(producer_cell_local_id, consumer_cell_local_id))
        {
          is_delayed_local_face = true;
          const auto producer_face_id =
            is_outgoing_face ? static_cast<unsigned int>(f)
                             : static_cast<unsigned int>(face_nodal_mapping.associated_face_);
          delayed_local_producer_face_key =
            PackProducerFaceKey(producer_cell_local_id, producer_face_id);
        }
      }
      bool is_delayed_nonlocal_face = false;
      if ((not is_local_face) and (not is_boundary_face))
      {
        const int neighbor_partition = grid.cells[face.neighbor_id].partition_id;
        if (is_incoming_face)
          is_delayed_nonlocal_face = is_delayed_source_partition(neighbor_partition);
        else if (is_outgoing_face)
          is_delayed_nonlocal_face = is_delayed_dest_partition(neighbor_partition);
      }

      for (size_t fn = 0; fn < num_face_nodes; ++fn)
      {
        CBCD_NodeIndex node_index;

        if (is_incoming_face)
        {
          if (is_local_face)
          {
            if (is_delayed_local_face)
            {
              // Resolve (producer_cell, producer_face) -> delayed-local bank offset.  The
              // producer-side outgoing iteration may run before or after this consumer-side
              // incoming iteration; whichever side encounters the directed face first assigns
              // the canonical offset, which the other side then reuses.
              auto [it, inserted] = delayed_local_offset_by_producer_face.try_emplace(
                delayed_local_producer_face_key,
                static_cast<std::uint32_t>(num_delayed_local_nodes_));
              if (inserted)
                num_delayed_local_nodes_ += num_face_nodes;
              const auto local_face_node =
                static_cast<std::uint64_t>(face_nodal_mapping.face_node_mapping_[fn]);
              node_index = CBCD_NodeIndex::DelayedLocal(
                static_cast<std::uint64_t>(it->second) + local_face_node, is_outgoing_face);
            }
            else
            {
              const auto task_id = cbc_spds.GetIncomingLocalFaceTaskID(
                static_cast<std::uint32_t>(cell.local_id), static_cast<unsigned int>(f));
              if (task_id == CBC_SPDS::INVALID_LOCAL_FACE_TASK_ID or
                  cbc_spds.GetLocalFaceNodeCount(task_id) != num_face_nodes)
                throw std::logic_error("CBCD FLUDS: incoming local-face extent is inconsistent.");
              const auto slot_id = local_face_slot_ids[task_id];
              if (static_cast<std::size_t>(slot_id) + 1 >= local_face_slot_node_offsets.size())
                throw std::logic_error("CBCD FLUDS: incoming local face has an invalid slot.");
              const auto local_face_node =
                static_cast<std::uint64_t>(face_nodal_mapping.face_node_mapping_[fn]);
              node_index = CBCD_NodeIndex::Local(
                static_cast<std::uint64_t>(local_face_slot_node_offsets[slot_id]) + local_face_node,
                is_outgoing_face);
            }
          }
          else if (is_delayed_nonlocal_face)
          {
            // Delayed-incoming nonlocal faces are grouped in parallel with the normal
            // tables so the receiver-side scatter can iterate by (source_slot, source_face)
            // independently of the normal route.
            node_index = CBCD_NodeIndex::DelayedNonlocal(num_delayed_incoming_nonlocal_nodes_,
                                                         is_outgoing_face);
            int& grouped_face_index = delayed_incoming_face_to_grouped_index[f];
            if (grouped_face_index < 0)
            {
              grouped_face_index =
                static_cast<int>(delayed_incoming_nonlocal_faces_.size() -
                                 cell_to_delayed_incoming_nonlocal_face_offsets_[cell.local_id]);
              auto& grouped_face = delayed_incoming_nonlocal_faces_.emplace_back();
              const int source_partition = grid.cells[face.neighbor_id].partition_id;
              auto [source_it, inserted] = delayed_source_partition_to_slot.try_emplace(
                source_partition,
                static_cast<std::uint32_t>(delayed_incoming_source_partitions_.size()));
              if (inserted)
                delayed_incoming_source_partitions_.push_back(source_partition);
              grouped_face.cell_local_id = static_cast<std::uint32_t>(cell.local_id);
              grouped_face.base_storage_index =
                static_cast<std::uint32_t>(num_delayed_incoming_nonlocal_nodes_);
              grouped_face.source_slot = source_it->second;
              delayed_incoming_face_order.push_back(
                {grouped_face.source_slot,
                 cell.global_id,
                 static_cast<unsigned int>(f),
                 static_cast<std::uint32_t>(delayed_incoming_nonlocal_faces_.size() - 1)});
            }

            auto& grouped_face = delayed_incoming_nonlocal_faces_
              [cell_to_delayed_incoming_nonlocal_face_offsets_[cell.local_id] + grouped_face_index];
            ++grouped_face.num_nodes;
            ++num_delayed_incoming_nonlocal_nodes_;
          }
          else if (not is_boundary_face)
          {
            node_index =
              CBCD_NodeIndex(num_incoming_nonlocal_nodes_, is_outgoing_face, is_local_face);
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
              incoming_face_order.push_back(
                {grouped_face.source_slot,
                 cell.global_id,
                 static_cast<unsigned int>(f),
                 static_cast<std::uint32_t>(incoming_nonlocal_faces_.size() - 1)});
              ++num_incoming_nonlocal_faces_;
            }

            auto& grouped_face =
              incoming_nonlocal_faces_[cell_to_incoming_nonlocal_face_offsets_[cell.local_id] +
                                       grouped_face_index];
            ++grouped_face.num_nodes;
            ++num_incoming_nonlocal_nodes_;
          }
          else
          {
            node_index = CBCD_NodeIndex(num_incoming_boundary_nodes_, is_outgoing_face);
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
            if (is_delayed_local_face)
            {
              auto [it, inserted] = delayed_local_offset_by_producer_face.try_emplace(
                delayed_local_producer_face_key,
                static_cast<std::uint32_t>(num_delayed_local_nodes_));
              if (inserted)
                num_delayed_local_nodes_ += num_face_nodes;
              node_index = CBCD_NodeIndex::DelayedLocal(static_cast<std::uint64_t>(it->second) +
                                                          static_cast<std::uint64_t>(fn),
                                                        is_outgoing_face);
            }
            else
            {
              const auto task_id = cbc_spds.GetOutgoingLocalFaceTaskID(
                static_cast<std::uint32_t>(cell.local_id), static_cast<unsigned int>(f));
              if (task_id == CBC_SPDS::INVALID_LOCAL_FACE_TASK_ID or
                  cbc_spds.GetLocalFaceNodeCount(task_id) != num_face_nodes)
                throw std::logic_error("CBCD FLUDS: outgoing local-face extent is inconsistent.");
              const auto slot_id = local_face_slot_ids[task_id];
              if (static_cast<std::size_t>(slot_id) + 1 >= local_face_slot_node_offsets.size())
                throw std::logic_error("CBCD FLUDS: outgoing local face has an invalid slot.");
              node_index = CBCD_NodeIndex::Local(
                static_cast<std::uint64_t>(local_face_slot_node_offsets[slot_id]) +
                  static_cast<std::uint64_t>(fn),
                is_outgoing_face);
            }
          }
          else if (is_delayed_nonlocal_face)
          {
            // Delayed-outgoing nonlocal faces build a grouped table in parallel with the
            // normal outgoing table.  Each face owns a slice of the delayed outgoing
            // node-copy plan; the lagged outgoing bank is sized exactly to the total
            // delayed outgoing nodes; remote face indices use their own deterministic
            // sequence (assigned after the deferred sort below).
            node_index = CBCD_NodeIndex::DelayedNonlocal(num_delayed_outgoing_nonlocal_nodes_,
                                                         is_outgoing_face);
            int& grouped_face_index = delayed_outgoing_face_to_grouped_index[f];
            if (grouped_face_index < 0)
            {
              const int locality = grid.cells[face.neighbor_id].partition_id;
              auto dest_slot_it = delayed_locality_to_dest_slot.find(locality);
              std::uint32_t dest_slot = 0;
              if (dest_slot_it == delayed_locality_to_dest_slot.end())
              {
                dest_slot = static_cast<std::uint32_t>(delayed_outgoing_localities_.size());
                delayed_locality_to_dest_slot.emplace(locality, dest_slot);
                delayed_outgoing_localities_.push_back(locality);
              }
              else
                dest_slot = dest_slot_it->second;

              const auto dest_cell_global_id = face.neighbor_id;
              const auto dest_face_id =
                static_cast<unsigned int>(face_nodal_mapping.associated_face_);
              grouped_face_index =
                static_cast<int>(delayed_outgoing_nonlocal_faces_.size() -
                                 cell_to_delayed_outgoing_nonlocal_face_offsets_[cell.local_id]);
              auto& grouped_face = delayed_outgoing_nonlocal_faces_.emplace_back();
              grouped_face.dest_slot = dest_slot;
              grouped_face.num_face_nodes = static_cast<std::uint16_t>(num_face_nodes);
              grouped_face.node_copy_offset =
                static_cast<std::uint32_t>(delayed_outgoing_nonlocal_face_node_copies_.size());
              delayed_outgoing_face_order.push_back(
                {dest_slot,
                 dest_cell_global_id,
                 dest_face_id,
                 static_cast<std::uint32_t>(delayed_outgoing_nonlocal_faces_.size() - 1)});
            }

            auto& grouped_face = delayed_outgoing_nonlocal_faces_
              [cell_to_delayed_outgoing_nonlocal_face_offsets_[cell.local_id] + grouped_face_index];
            delayed_outgoing_nonlocal_face_node_copies_.push_back(
              {static_cast<std::uint32_t>(num_delayed_outgoing_nonlocal_nodes_),
               static_cast<std::uint16_t>(face_nodal_mapping.face_node_mapping_[fn])});
            ++grouped_face.num_node_copies;
            ++num_delayed_outgoing_nonlocal_nodes_;
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

              const auto dest_cell_global_id = face.neighbor_id;
              const auto dest_face_id =
                static_cast<unsigned int>(face_nodal_mapping.associated_face_);
              grouped_face_index =
                static_cast<int>(outgoing_nonlocal_faces_.size() -
                                 cell_to_outgoing_nonlocal_face_offsets_[cell.local_id]);
              auto& grouped_face = outgoing_nonlocal_faces_.emplace_back();
              grouped_face.dest_slot = dest_slot;
              grouped_face.num_face_nodes = static_cast<std::uint16_t>(num_face_nodes);
              grouped_face.node_copy_offset =
                static_cast<std::uint32_t>(outgoing_nonlocal_face_node_copies_.size());
              outgoing_face_order.push_back(
                {dest_slot,
                 dest_cell_global_id,
                 dest_face_id,
                 static_cast<std::uint32_t>(outgoing_nonlocal_faces_.size() - 1)});
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
    update_cell_offsets(cell.local_id + 1);
    cell_offsets_ptr[2 * cell.local_id + 1] = num_cell_nodes;
    current_index_offset += num_cell_nodes;
  }

  std::sort(incoming_face_order.begin(),
            incoming_face_order.end(),
            [](const OrderedIncomingFaceBuild& lhs, const OrderedIncomingFaceBuild& rhs)
            {
              return std::tuple(lhs.source_slot, lhs.cell_global_id, lhs.face_id) <
                     std::tuple(rhs.source_slot, rhs.cell_global_id, rhs.face_id);
            });

  source_to_incoming_face_offsets_.assign(incoming_source_partitions_.size() + 1, 0);
  for (const auto& build : incoming_face_order)
    ++source_to_incoming_face_offsets_[build.source_slot + 1];
  for (std::size_t i = 0; i < incoming_source_partitions_.size(); ++i)
    source_to_incoming_face_offsets_[i + 1] += source_to_incoming_face_offsets_[i];

  incoming_face_indices_by_source_.resize(incoming_face_order.size());
  auto source_write_offsets = source_to_incoming_face_offsets_;
  for (const auto& build : incoming_face_order)
    incoming_face_indices_by_source_[source_write_offsets[build.source_slot]++] = build.face_index;

  std::sort(outgoing_face_order.begin(),
            outgoing_face_order.end(),
            [](const OrderedOutgoingFaceBuild& lhs, const OrderedOutgoingFaceBuild& rhs)
            {
              return std::tuple(lhs.dest_slot, lhs.cell_global_id, lhs.face_id) <
                     std::tuple(rhs.dest_slot, rhs.cell_global_id, rhs.face_id);
            });

  std::uint32_t current_dest_slot = 0;
  std::uint32_t remote_face_index = 0;
  bool first_outgoing_face = true;
  for (const auto& build : outgoing_face_order)
  {
    if (first_outgoing_face or (build.dest_slot != current_dest_slot))
    {
      current_dest_slot = build.dest_slot;
      remote_face_index = 0;
      first_outgoing_face = false;
    }
    outgoing_nonlocal_faces_[build.face_index].remote_face_index = remote_face_index++;
  }

  // Delayed-incoming face ordering and source-slot-major lookup tables.  Producer and
  // consumer must agree on the (source_slot, source_face_index) -> grouped-face index
  // mapping for delayed payloads.  The remote-face-index space is independent from the
  // normal one.
  std::sort(delayed_incoming_face_order.begin(),
            delayed_incoming_face_order.end(),
            [](const OrderedIncomingFaceBuild& lhs, const OrderedIncomingFaceBuild& rhs)
            {
              return std::tuple(lhs.source_slot, lhs.cell_global_id, lhs.face_id) <
                     std::tuple(rhs.source_slot, rhs.cell_global_id, rhs.face_id);
            });

  delayed_source_to_incoming_face_offsets_.assign(delayed_incoming_source_partitions_.size() + 1,
                                                  0);
  for (const auto& build : delayed_incoming_face_order)
    ++delayed_source_to_incoming_face_offsets_[build.source_slot + 1];
  for (std::size_t i = 0; i < delayed_incoming_source_partitions_.size(); ++i)
    delayed_source_to_incoming_face_offsets_[i + 1] += delayed_source_to_incoming_face_offsets_[i];

  delayed_incoming_face_indices_by_source_.resize(delayed_incoming_face_order.size());
  auto delayed_source_write_offsets = delayed_source_to_incoming_face_offsets_;
  for (const auto& build : delayed_incoming_face_order)
    delayed_incoming_face_indices_by_source_[delayed_source_write_offsets[build.source_slot]++] =
      build.face_index;

  std::sort(delayed_outgoing_face_order.begin(),
            delayed_outgoing_face_order.end(),
            [](const OrderedOutgoingFaceBuild& lhs, const OrderedOutgoingFaceBuild& rhs)
            {
              return std::tuple(lhs.dest_slot, lhs.cell_global_id, lhs.face_id) <
                     std::tuple(rhs.dest_slot, rhs.cell_global_id, rhs.face_id);
            });

  current_dest_slot = 0;
  remote_face_index = 0;
  first_outgoing_face = true;
  for (const auto& build : delayed_outgoing_face_order)
  {
    if (first_outgoing_face or (build.dest_slot != current_dest_slot))
    {
      current_dest_slot = build.dest_slot;
      remote_face_index = 0;
      first_outgoing_face = false;
    }
    delayed_outgoing_nonlocal_faces_[build.face_index].remote_face_index = remote_face_index++;
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

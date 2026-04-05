// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/cell/cell.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace opensn
{

/**
 * Static local-face reuse plan for CBC/CBCD.
 *
 * The reusable objects are local outgoing faces, grouped by exact face-node
 * count. One slot stores all nodes of one local outgoing face. Incoming local
 * faces do not own storage; they reference the slot assigned to their unique
 * upwind local outgoing face.
 *
 * For two local outgoing faces e = (u -> v) and f = (w -> x) of the same face
 * category, e may safely reuse into f if and only if w is v itself or a strict
 * descendant of v in the local task DAG. The equality case is important: during
 * the sweep of cell v, all incident local face data are consumed before any
 * outgoing local face data of v are written, so the slot for e may be
 * transferred to any outgoing local face of v without overlap. This remains
 * valid for CBCD because ready-cell batches do not contain task-successor
 * relationships, and the intra-cell read-before-write ordering is preserved
 * within each sweep kernel.
 */
struct CBCLocalFaceSlotPlan
{
  static constexpr std::uint32_t INVALID_SLOT = std::numeric_limits<std::uint32_t>::max();

  std::vector<std::uint32_t> face_node_offsets;
  std::vector<std::uint32_t> incoming_local_face_node_slot_indices;
  std::vector<std::uint32_t> outgoing_local_face_node_slot_indices;
  std::size_t num_local_psi_face_node_slots = 0;
};

namespace detail
{

class CBCPlannerBitMatrix
{
public:
  void ResizeAndClear(const std::size_t n)
  {
    n_ = n;
    words_per_row_ = (n + 63) / 64;
    data_.assign(n * words_per_row_, 0ULL);
  }

  void SetBit(const std::size_t i, const std::size_t j)
  {
    Row(i)[j / 64] |= (1ULL << (j % 64));
  }

  void OrRows(const std::size_t dst, const std::size_t src)
  {
    auto* dst_row = Row(dst);
    const auto* src_row = Row(src);
    for (std::size_t w = 0; w < words_per_row_; ++w)
      dst_row[w] |= src_row[w];
  }

  std::size_t FindFirstSet(const std::size_t row, const std::size_t start_pos) const
  {
    const auto* row_data = Row(row);
    std::size_t word = start_pos / 64;
    if (word >= words_per_row_)
      return n_;

    std::uint64_t masked = row_data[word] & (~0ULL << (start_pos % 64));
    if (masked != 0ULL)
      return word * 64 + static_cast<std::size_t>(std::countr_zero(masked));

    for (++word; word < words_per_row_; ++word)
      if (row_data[word] != 0ULL)
        return word * 64 + static_cast<std::size_t>(std::countr_zero(row_data[word]));

    return n_;
  }

  std::size_t FindNextSet(const std::size_t row, const std::size_t pos) const
  {
    return FindFirstSet(row, pos + 1);
  }

private:
  std::uint64_t* Row(const std::size_t i) { return data_.data() + i * words_per_row_; }
  const std::uint64_t* Row(const std::size_t i) const
  {
    return data_.data() + i * words_per_row_;
  }

  std::size_t n_ = 0;
  std::size_t words_per_row_ = 0;
  std::vector<std::uint64_t> data_;
};

struct CBCLocalOutgoingFace
{
  std::uint32_t producer_cell_local_id = 0;
  std::uint32_t consumer_cell_local_id = 0;
  std::uint32_t producer_face_storage_index = 0;
  std::uint32_t producer_cell_rank = 0;
  std::uint32_t consumer_cell_rank = 0;
  std::uint16_t producer_face_id = 0;
  std::uint16_t consumer_face_id = 0;
  std::uint16_t num_face_nodes = 0;
  std::uint32_t slot_id = CBCLocalFaceSlotPlan::INVALID_SLOT;
};

inline std::vector<std::uint32_t>
BuildCellFaceOffsets(const MeshContinuum& grid)
{
  std::vector<std::uint32_t> cell_face_offsets(grid.local_cells.size() + 1, 0);
  std::uint32_t total_num_faces = 0;
  for (const auto& cell : grid.local_cells)
  {
    cell_face_offsets[cell.local_id] = total_num_faces;
    total_num_faces += static_cast<std::uint32_t>(cell.faces.size());
  }
  cell_face_offsets.back() = total_num_faces;
  return cell_face_offsets;
}

inline CBCPlannerBitMatrix
BuildCellDescendantMatrix(const SPDS& spds, const std::vector<std::uint32_t>& topo_rank)
{
  const auto& grid = *spds.GetGrid();
  const auto& face_orientations = spds.GetCellFaceOrientations();
  std::vector<std::vector<std::uint32_t>> local_successors(grid.local_cells.size());

  for (const auto& cell : grid.local_cells)
  {
    auto& successors = local_successors[cell.local_id];
    successors.reserve(cell.faces.size());
    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      if (face_orientations[cell.local_id][f] != FaceOrientation::OUTGOING or
          not face.has_neighbor or not face.IsNeighborLocal(&grid))
        continue;

      successors.push_back(face.GetNeighborLocalID(&grid));
    }
  }

  CBCPlannerBitMatrix descendants;
  descendants.ResizeAndClear(grid.local_cells.size());

  const auto& topo_order = spds.GetLocalSubgrid();
  for (std::size_t reverse_i = topo_order.size(); reverse_i-- > 0;)
  {
    const auto cell_local_id = topo_order[reverse_i];
    const auto cell_rank = topo_rank[cell_local_id];
    descendants.SetBit(cell_rank, cell_rank);
    for (const auto successor : local_successors[cell_local_id])
    {
      const auto successor_rank = topo_rank[successor];
      descendants.OrRows(cell_rank, successor_rank);
    }
  }

  return descendants;
}

class CBCLocalFaceCategoryAllocator
{
public:
  CBCLocalFaceCategoryAllocator(const CBCPlannerBitMatrix& descendants,
                                const std::size_t num_cells,
                                std::vector<CBCLocalOutgoingFace>& faces)
    : descendants_(descendants),
      num_cells_(num_cells),
      faces_(faces),
      mate_u_(faces.size(), INVALID_INDEX),
      mate_v_(faces.size(), INVALID_INDEX),
      dist_(faces.size(), -1),
      queue_(faces.size())
  {
    std::sort(faces_.begin(),
              faces_.end(),
              [](const auto& lhs, const auto& rhs)
              {
                if (lhs.producer_cell_rank != rhs.producer_cell_rank)
                  return lhs.producer_cell_rank < rhs.producer_cell_rank;
                return lhs.producer_face_storage_index < rhs.producer_face_storage_index;
              });

    producer_cell_offsets_.assign(num_cells_ + 1, 0);
    producer_ranks_.resize(faces_.size(), 0);
    consumer_ranks_.resize(faces_.size(), 0);
    for (const auto& face : faces_)
      ++producer_cell_offsets_[face.producer_cell_rank + 1];

    std::partial_sum(
      producer_cell_offsets_.begin(), producer_cell_offsets_.end(), producer_cell_offsets_.begin());

    for (std::size_t i = 0; i < faces_.size(); ++i)
    {
      producer_ranks_[i] = faces_[i].producer_cell_rank;
      consumer_ranks_[i] = faces_[i].consumer_cell_rank;
    }
  }

  std::size_t Solve()
  {
    std::size_t matching_size = GreedyInit();
    while (BFS())
      for (std::uint32_t u = 0; u < faces_.size(); ++u)
        if (mate_u_[u] == INVALID_INDEX and DFS(u))
          ++matching_size;

    AssignSlots();
    return faces_.empty() ? 0 : *std::max_element(slot_ids_.begin(), slot_ids_.end()) + 1;
  }

private:
  static constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

  std::size_t FindFirstNeighbor(const std::uint32_t u) const
  {
    std::size_t producer_rank =
      descendants_.FindFirstSet(consumer_ranks_[u], consumer_ranks_[u]);
    while (producer_rank < num_cells_ and
           producer_cell_offsets_[producer_rank] == producer_cell_offsets_[producer_rank + 1])
      producer_rank = descendants_.FindNextSet(consumer_ranks_[u], producer_rank);

    if (producer_rank >= num_cells_)
      return faces_.size();
    return producer_cell_offsets_[producer_rank];
  }

  std::size_t FindNextNeighbor(const std::uint32_t u, const std::size_t v) const
  {
    const auto producer_rank = producer_ranks_[v];
    const auto next_v = v + 1;
    if (next_v < producer_cell_offsets_[producer_rank + 1])
      return next_v;

    std::size_t next_producer_rank = descendants_.FindNextSet(consumer_ranks_[u], producer_rank);
    while (next_producer_rank < num_cells_ and
           producer_cell_offsets_[next_producer_rank] ==
             producer_cell_offsets_[next_producer_rank + 1])
      next_producer_rank = descendants_.FindNextSet(consumer_ranks_[u], next_producer_rank);

    if (next_producer_rank >= num_cells_)
      return faces_.size();
    return producer_cell_offsets_[next_producer_rank];
  }

  std::size_t GreedyInit()
  {
    std::size_t count = 0;
    for (std::uint32_t u = 0; u < faces_.size(); ++u)
    {
      if (mate_u_[u] != INVALID_INDEX)
        continue;

      for (std::size_t v = FindFirstNeighbor(u); v < faces_.size(); v = FindNextNeighbor(u, v))
      {
        if (mate_v_[v] != INVALID_INDEX)
          continue;

        mate_u_[u] = static_cast<std::uint32_t>(v);
        mate_v_[v] = u;
        ++count;
        break;
      }
    }
    return count;
  }

  bool BFS()
  {
    std::fill(dist_.begin(), dist_.end(), -1);
    std::size_t head = 0;
    std::size_t tail = 0;
    for (std::uint32_t u = 0; u < faces_.size(); ++u)
      if (mate_u_[u] == INVALID_INDEX)
      {
        dist_[u] = 0;
        queue_[tail++] = u;
      }

    dist_null_ = std::numeric_limits<int>::max();
    while (head < tail)
    {
      const auto u = queue_[head++];
      if (dist_[u] >= dist_null_)
        continue;

      for (std::size_t v = FindFirstNeighbor(u); v < faces_.size(); v = FindNextNeighbor(u, v))
      {
        const auto mate_of_v = mate_v_[v];
        if (mate_of_v == INVALID_INDEX)
        {
          if (dist_null_ == std::numeric_limits<int>::max())
            dist_null_ = dist_[u] + 1;
        }
        else if (dist_[mate_of_v] == -1)
        {
          dist_[mate_of_v] = dist_[u] + 1;
          queue_[tail++] = mate_of_v;
        }
      }
    }

    return dist_null_ != std::numeric_limits<int>::max();
  }

  bool DFS(const std::uint32_t u)
  {
    for (std::size_t v = FindFirstNeighbor(u); v < faces_.size(); v = FindNextNeighbor(u, v))
    {
      const auto mate_of_v = mate_v_[v];
      if (mate_of_v == INVALID_INDEX)
      {
        if (dist_null_ != dist_[u] + 1)
          continue;

        mate_v_[v] = u;
        mate_u_[u] = static_cast<std::uint32_t>(v);
        dist_[u] = -1;
        return true;
      }

      if (dist_[mate_of_v] != dist_[u] + 1 or not DFS(mate_of_v))
        continue;

      mate_v_[v] = u;
      mate_u_[u] = static_cast<std::uint32_t>(v);
      dist_[u] = -1;
      return true;
    }

    dist_[u] = -1;
    return false;
  }

  void AssignSlots()
  {
    slot_ids_.assign(faces_.size(), INVALID_INDEX);
    std::uint32_t next_slot_id = 0;
    for (std::uint32_t v = 0; v < faces_.size(); ++v)
    {
      if (mate_v_[v] != INVALID_INDEX)
        continue;

      std::uint32_t current = v;
      while (current != INVALID_INDEX)
      {
        slot_ids_[current] = next_slot_id;
        faces_[current].slot_id = next_slot_id;
        current = mate_u_[current];
      }
      ++next_slot_id;
    }
  }

  const CBCPlannerBitMatrix& descendants_;
  std::size_t num_cells_ = 0;
  std::vector<CBCLocalOutgoingFace>& faces_;
  std::vector<std::uint32_t> producer_cell_offsets_;
  std::vector<std::uint32_t> producer_ranks_;
  std::vector<std::uint32_t> consumer_ranks_;
  std::vector<std::uint32_t> mate_u_;
  std::vector<std::uint32_t> mate_v_;
  std::vector<std::uint32_t> queue_;
  std::vector<std::uint32_t> slot_ids_;
  std::vector<int> dist_;
  int dist_null_ = 0;
};

} // namespace detail

inline CBCLocalFaceSlotPlan
ComputeCBCLocalFaceSlotPlan(const SPDS& spds,
                            const std::vector<CellFaceNodalMapping>& grid_nodal_mappings,
                            const SpatialDiscretization& sdm)
{
  const auto& grid = *spds.GetGrid();
  const auto& face_orientations = spds.GetCellFaceOrientations();
  const auto num_local_cells = grid.local_cells.size();
  const auto cell_face_offsets = detail::BuildCellFaceOffsets(grid);
  const auto total_num_faces = static_cast<std::size_t>(cell_face_offsets.back());

  CBCLocalFaceSlotPlan plan;
  plan.face_node_offsets.resize(total_num_faces + 1, 0);

  std::size_t total_num_face_nodes = 0;
  for (const auto& cell : grid.local_cells)
    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto face_storage_index = static_cast<std::size_t>(cell_face_offsets[cell.local_id] + f);
      plan.face_node_offsets[face_storage_index] = static_cast<std::uint32_t>(total_num_face_nodes);
      total_num_face_nodes += sdm.GetCellMapping(cell).GetNumFaceNodes(f);
    }
  plan.face_node_offsets.back() = static_cast<std::uint32_t>(total_num_face_nodes);
  plan.incoming_local_face_node_slot_indices.assign(
    total_num_face_nodes, CBCLocalFaceSlotPlan::INVALID_SLOT);
  plan.outgoing_local_face_node_slot_indices.assign(
    total_num_face_nodes, CBCLocalFaceSlotPlan::INVALID_SLOT);

  std::vector<std::uint32_t> topo_rank(num_local_cells, 0);
  const auto& topo_order = spds.GetLocalSubgrid();
  for (std::size_t rank = 0; rank < topo_order.size(); ++rank)
    topo_rank[topo_order[rank]] = static_cast<std::uint32_t>(rank);

  const auto descendants = detail::BuildCellDescendantMatrix(spds, topo_rank);

  std::map<std::uint16_t, std::vector<detail::CBCLocalOutgoingFace>> faces_by_size;
  for (const auto& cell : grid.local_cells)
    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      if (face_orientations[cell.local_id][f] != FaceOrientation::OUTGOING or
          not face.has_neighbor or not face.IsNeighborLocal(&grid))
        continue;

      const auto num_face_nodes = static_cast<std::uint16_t>(sdm.GetCellMapping(cell).GetNumFaceNodes(f));
      const auto face_storage_index = cell_face_offsets[cell.local_id] + static_cast<std::uint32_t>(f);
      faces_by_size[num_face_nodes].push_back(
        {cell.local_id,
         static_cast<std::uint32_t>(face.GetNeighborLocalID(&grid)),
         face_storage_index,
         topo_rank[cell.local_id],
         topo_rank[face.GetNeighborLocalID(&grid)],
         static_cast<std::uint16_t>(f),
         static_cast<std::uint16_t>(grid_nodal_mappings[cell.local_id][f].associated_face_),
         num_face_nodes,
         CBCLocalFaceSlotPlan::INVALID_SLOT});
    }

  std::map<std::uint16_t, std::size_t> slots_per_size;
  for (auto& [num_face_nodes, faces] : faces_by_size)
  {
    detail::CBCLocalFaceCategoryAllocator allocator(descendants, num_local_cells, faces);
    slots_per_size[num_face_nodes] = allocator.Solve();
  }

  std::uint32_t next_base_offset = 0;
  std::map<std::uint16_t, std::uint32_t> base_offsets_by_size;
  for (const auto& [num_face_nodes, num_slots] : slots_per_size)
  {
    base_offsets_by_size[num_face_nodes] = next_base_offset;
    next_base_offset += static_cast<std::uint32_t>(num_face_nodes * num_slots);
  }
  plan.num_local_psi_face_node_slots = next_base_offset;

  for (const auto& [num_face_nodes, faces] : faces_by_size)
  {
    const auto base_offset = base_offsets_by_size[num_face_nodes];
    for (const auto& face : faces)
    {
      const auto producer_face_node_offset = plan.face_node_offsets[face.producer_face_storage_index];
      const auto consumer_face_storage_index =
        cell_face_offsets[face.consumer_cell_local_id] + face.consumer_face_id;
      const auto consumer_face_node_offset = plan.face_node_offsets[consumer_face_storage_index];
      const auto& face_nodal_mapping =
        grid_nodal_mappings[face.producer_cell_local_id][face.producer_face_id];

      assert(face_nodal_mapping.face_node_mapping_.size() == face.num_face_nodes);
      for (std::uint32_t fi = 0; fi < face.num_face_nodes; ++fi)
      {
        const auto physical_slot = base_offset + face.slot_id * face.num_face_nodes + fi;
        plan.outgoing_local_face_node_slot_indices[producer_face_node_offset + fi] = physical_slot;

        const auto consumer_face_node = static_cast<std::uint32_t>(face_nodal_mapping.face_node_mapping_[fi]);
        auto& incoming_slot =
          plan.incoming_local_face_node_slot_indices[consumer_face_node_offset + consumer_face_node];
        if (incoming_slot != CBCLocalFaceSlotPlan::INVALID_SLOT)
          throw std::logic_error("CBC local face-slot planner encountered duplicate incoming slot mapping.");
        incoming_slot = physical_slot;
      }
    }
  }

  for (const auto& cell : grid.local_cells)
    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      if (not face.has_neighbor or not face.IsNeighborLocal(&grid))
        continue;

      const auto face_storage_index = cell_face_offsets[cell.local_id] + static_cast<std::uint32_t>(f);
      const auto face_node_offset = plan.face_node_offsets[face_storage_index];
      const auto num_face_nodes = sdm.GetCellMapping(cell).GetNumFaceNodes(f);
      const auto* slot_indices =
        (face_orientations[cell.local_id][f] == FaceOrientation::OUTGOING)
          ? plan.outgoing_local_face_node_slot_indices.data() + face_node_offset
          : plan.incoming_local_face_node_slot_indices.data() + face_node_offset;

      for (std::size_t fn = 0; fn < num_face_nodes; ++fn)
        if (slot_indices[fn] == CBCLocalFaceSlotPlan::INVALID_SLOT)
          throw std::logic_error("CBC local face-slot planner produced an incomplete slot map.");
    }

  return plan;
}

} // namespace opensn

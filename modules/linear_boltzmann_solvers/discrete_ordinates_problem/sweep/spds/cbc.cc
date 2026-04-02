// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/max_cardinality_matching.hpp>
#include <boost/graph/topological_sort.hpp>

namespace opensn
{

namespace
{

constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

struct ChainCoverData
{
  std::uint32_t num_chains = 0;
  std::vector<std::uint32_t> chain_id;
  std::vector<std::uint32_t> pos_in_chain;
  std::vector<std::uint32_t> chain_offsets;
  std::vector<std::uint32_t> chain_vertices;
};

struct SlotCalcScratch
{
  std::vector<std::uint32_t> mate_u;
  std::vector<std::uint32_t> mate_v;
  std::vector<int> dist;
  std::vector<std::uint32_t> bfs_queue;
  std::vector<std::uint32_t> first_reachable;
  std::vector<std::uint32_t> reuse_start;
  std::vector<std::uint8_t> is_immediate_successor;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> reuse_edges;
  std::vector<int> component_ids;
  std::vector<std::uint32_t> local_vertex_ids;
  std::vector<std::size_t> component_vertex_counts;
  std::vector<std::size_t> component_edge_counts;
  std::vector<std::size_t> component_vertex_offsets;
  std::vector<std::size_t> component_edge_offsets;
  std::vector<std::size_t> component_edge_write_offsets;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> component_edges;
  std::vector<std::size_t> component_matching_mate;

  void ResizeMatchingState(const std::uint32_t num_tasks)
  {
    mate_u.assign(num_tasks, INVALID_INDEX);
    mate_v.assign(num_tasks, INVALID_INDEX);
    dist.assign(num_tasks, -1);
    bfs_queue.resize(num_tasks);
    is_immediate_successor.assign(num_tasks, 0);
  }
};

class CSRBipartiteMatcher
{
public:
  CSRBipartiteMatcher(const std::vector<std::uint32_t>& row_offsets,
                      const std::vector<std::uint32_t>& columns,
                      SlotCalcScratch& scratch)
    : row_offsets_(row_offsets), columns_(columns), scratch_(scratch)
  {
  }

  std::size_t Solve()
  {
    std::size_t matching_size = GreedyInitialize();
    while (BFS())
    {
      for (std::uint32_t u = 0; u + 1 < row_offsets_.size(); ++u)
      {
        if (scratch_.mate_u[u] == INVALID_INDEX and DFS(u))
          ++matching_size;
      }
    }
    return matching_size;
  }

private:
  std::size_t GreedyInitialize()
  {
    std::size_t matching_size = 0;

    for (std::uint32_t u = 0; u + 1 < row_offsets_.size(); ++u)
    {
      for (auto e = row_offsets_[u]; e < row_offsets_[u + 1]; ++e)
      {
        const auto v = columns_[e];
        if (scratch_.mate_v[v] != INVALID_INDEX)
          continue;

        scratch_.mate_u[u] = v;
        scratch_.mate_v[v] = u;
        ++matching_size;
        break;
      }
    }

    return matching_size;
  }

  bool BFS()
  {
    std::fill(scratch_.dist.begin(), scratch_.dist.end(), -1);

    std::size_t queue_head = 0;
    std::size_t queue_tail = 0;

    for (std::uint32_t u = 0; u + 1 < row_offsets_.size(); ++u)
    {
      if (scratch_.mate_u[u] == INVALID_INDEX)
      {
        scratch_.dist[u] = 0;
        scratch_.bfs_queue[queue_tail++] = u;
      }
    }

    dist_null_ = std::numeric_limits<int>::max();

    while (queue_head < queue_tail)
    {
      const auto u = scratch_.bfs_queue[queue_head++];
      if (scratch_.dist[u] >= dist_null_)
        continue;

      for (auto e = row_offsets_[u]; e < row_offsets_[u + 1]; ++e)
      {
        const auto v = columns_[e];
        const auto mate_of_v = scratch_.mate_v[v];

        if (mate_of_v == INVALID_INDEX)
          dist_null_ = std::min(dist_null_, scratch_.dist[u] + 1);
        else if (scratch_.dist[mate_of_v] == -1)
        {
          scratch_.dist[mate_of_v] = scratch_.dist[u] + 1;
          scratch_.bfs_queue[queue_tail++] = mate_of_v;
        }
      }
    }

    return dist_null_ != std::numeric_limits<int>::max();
  }

  bool DFS(const std::uint32_t u)
  {
    for (auto e = row_offsets_[u]; e < row_offsets_[u + 1]; ++e)
    {
      const auto v = columns_[e];
      const auto mate_of_v = scratch_.mate_v[v];

      if (mate_of_v == INVALID_INDEX)
      {
        if (dist_null_ == scratch_.dist[u] + 1)
        {
          scratch_.mate_u[u] = v;
          scratch_.mate_v[v] = u;
          scratch_.dist[u] = -1;
          return true;
        }
      }
      else if (scratch_.dist[mate_of_v] == scratch_.dist[u] + 1 and DFS(mate_of_v))
      {
        scratch_.mate_u[u] = v;
        scratch_.mate_v[v] = u;
        scratch_.dist[u] = -1;
        return true;
      }
    }

    scratch_.dist[u] = -1;
    return false;
  }

  const std::vector<std::uint32_t>& row_offsets_;
  const std::vector<std::uint32_t>& columns_;
  SlotCalcScratch& scratch_;
  int dist_null_ = std::numeric_limits<int>::max();
};

ChainCoverData
BuildMinimumPathCover(const std::vector<std::uint32_t>& successor_offsets,
                      const std::vector<std::uint32_t>& successors,
                      SlotCalcScratch& scratch)
{
  const auto num_tasks = static_cast<std::uint32_t>(successor_offsets.size() - 1);

  scratch.ResizeMatchingState(num_tasks);
  CSRBipartiteMatcher matcher(successor_offsets, successors, scratch);
  matcher.Solve();

  ChainCoverData chain_cover;
  chain_cover.chain_id.assign(num_tasks, INVALID_INDEX);
  chain_cover.pos_in_chain.assign(num_tasks, INVALID_INDEX);
  chain_cover.chain_offsets.push_back(0);
  chain_cover.chain_vertices.reserve(num_tasks);

  auto append_chain = [&](std::uint32_t start_vertex)
  {
    std::uint32_t current = start_vertex;
    std::uint32_t chain_pos = 0;

    while (current != INVALID_INDEX)
    {
      if (chain_cover.chain_id[current] != INVALID_INDEX)
        throw std::logic_error("CBC_SPDS: Invalid minimum path cover construction.");

      chain_cover.chain_id[current] = chain_cover.num_chains;
      chain_cover.pos_in_chain[current] = chain_pos++;
      chain_cover.chain_vertices.push_back(current);
      current = scratch.mate_u[current];
    }

    ++chain_cover.num_chains;
    chain_cover.chain_offsets.push_back(
      static_cast<std::uint32_t>(chain_cover.chain_vertices.size()));
  };

  for (std::uint32_t v = 0; v < num_tasks; ++v)
    if (scratch.mate_v[v] == INVALID_INDEX)
      append_chain(v);

  for (std::uint32_t v = 0; v < num_tasks; ++v)
    if (chain_cover.chain_id[v] == INVALID_INDEX)
      append_chain(v);

  return chain_cover;
}

void
BuildFirstReachable(const std::vector<std::uint32_t>& successor_offsets,
                    const std::vector<std::uint32_t>& successors,
                    const std::vector<std::uint32_t>& topo_order,
                    const ChainCoverData& chain_cover,
                    SlotCalcScratch& scratch)
{
  const auto num_tasks = static_cast<std::uint32_t>(successor_offsets.size() - 1);
  const auto num_chains = chain_cover.num_chains;

  scratch.first_reachable.assign(static_cast<std::size_t>(num_tasks) * num_chains, INVALID_INDEX);

  for (auto topo_it = topo_order.rbegin(); topo_it != topo_order.rend(); ++topo_it)
  {
    const auto u = *topo_it;
    auto* const row = scratch.first_reachable.data() + static_cast<std::size_t>(u) * num_chains;

    row[chain_cover.chain_id[u]] = chain_cover.pos_in_chain[u];

    for (auto e = successor_offsets[u]; e < successor_offsets[u + 1]; ++e)
    {
      const auto succ = successors[e];
      const auto* const succ_row =
        scratch.first_reachable.data() + static_cast<std::size_t>(succ) * num_chains;

      for (std::uint32_t chain = 0; chain < num_chains; ++chain)
      {
        const auto succ_pos = succ_row[chain];
        auto& first_pos = row[chain];
        if (succ_pos == INVALID_INDEX)
          continue;

        if (first_pos == INVALID_INDEX or succ_pos < first_pos)
          first_pos = succ_pos;
      }
    }
  }
}

void
BuildReuseStart(const std::vector<std::uint32_t>& successor_offsets,
                const std::vector<std::uint32_t>& successors,
                const std::uint32_t num_chains,
                SlotCalcScratch& scratch)
{
  const auto num_tasks = static_cast<std::uint32_t>(successor_offsets.size() - 1);
  scratch.reuse_start.assign(static_cast<std::size_t>(num_tasks) * num_chains, INVALID_INDEX);

  for (std::uint32_t u = 0; u < num_tasks; ++u)
  {
    const auto succ_begin = successor_offsets[u];
    const auto succ_end = successor_offsets[u + 1];
    if (succ_begin == succ_end)
      continue;

    auto* const reuse_row = scratch.reuse_start.data() + static_cast<std::size_t>(u) * num_chains;
    const auto first_succ = successors[succ_begin];
    const auto* const first_succ_row =
      scratch.first_reachable.data() + static_cast<std::size_t>(first_succ) * num_chains;
    std::copy_n(first_succ_row, num_chains, reuse_row);

    for (auto e = succ_begin + 1; e < succ_end; ++e)
    {
      const auto succ = successors[e];
      const auto* const succ_row =
        scratch.first_reachable.data() + static_cast<std::size_t>(succ) * num_chains;

      for (std::uint32_t chain = 0; chain < num_chains; ++chain)
      {
        auto& candidate_pos = reuse_row[chain];
        const auto succ_pos = succ_row[chain];
        if (candidate_pos == INVALID_INDEX or succ_pos == INVALID_INDEX)
          candidate_pos = INVALID_INDEX;
        else
          candidate_pos = std::max(candidate_pos, succ_pos);
      }
    }
  }
}

using ReuseGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

template <class F>
void
ForEachReuseNeighbor(const std::uint32_t u,
                     const std::vector<std::uint32_t>& successor_offsets,
                     const std::vector<std::uint32_t>& successors,
                     const ChainCoverData& chain_cover,
                     SlotCalcScratch& scratch,
                     F&& func)
{
  const auto succ_begin = successor_offsets[u];
  const auto succ_end = successor_offsets[u + 1];
  if (succ_begin == succ_end)
    return;

  for (auto e = succ_begin; e < succ_end; ++e)
    scratch.is_immediate_successor[successors[e]] = 1;

  const auto num_chains = chain_cover.num_chains;
  const auto* const reuse_row =
    scratch.reuse_start.data() + static_cast<std::size_t>(u) * num_chains;

  for (std::uint32_t chain = 0; chain < num_chains; ++chain)
  {
    const auto start_pos = reuse_row[chain];
    if (start_pos == INVALID_INDEX)
      continue;

    const auto chain_begin = chain_cover.chain_offsets[chain];
    const auto chain_end = chain_cover.chain_offsets[chain + 1];
    for (std::uint32_t pos = chain_begin + start_pos; pos < chain_end; ++pos)
    {
      const auto v = chain_cover.chain_vertices[pos];
      if (not scratch.is_immediate_successor[v])
        func(v);
    }
  }

  for (auto e = succ_begin; e < succ_end; ++e)
    scratch.is_immediate_successor[successors[e]] = 0;
}

void
BuildReuseEdges(const std::vector<std::uint32_t>& successor_offsets,
                const std::vector<std::uint32_t>& successors,
                const ChainCoverData& chain_cover,
                SlotCalcScratch& scratch)
{
  const auto num_tasks = static_cast<std::uint32_t>(successor_offsets.size() - 1);

  std::size_t reuse_edge_count = 0;
  for (std::uint32_t u = 0; u < num_tasks; ++u)
    ForEachReuseNeighbor(u,
                         successor_offsets,
                         successors,
                         chain_cover,
                         scratch,
                         [&](const std::uint32_t) { ++reuse_edge_count; });

  scratch.reuse_edges.clear();
  scratch.reuse_edges.reserve(reuse_edge_count);

  for (std::uint32_t u = 0; u < num_tasks; ++u)
    ForEachReuseNeighbor(u,
                         successor_offsets,
                         successors,
                         chain_cover,
                         scratch,
                         [&](const std::uint32_t v)
                         { scratch.reuse_edges.emplace_back(u, num_tasks + v); });
}

std::size_t
ComputeReuseMatchingSize(const std::uint32_t num_tasks, SlotCalcScratch& scratch)
{
  if (scratch.reuse_edges.empty())
    return 0;

  ReuseGraph reuse_graph(scratch.reuse_edges.begin(),
                         scratch.reuse_edges.end(),
                         static_cast<std::size_t>(num_tasks) * 2);

  using Vertex = boost::graph_traits<ReuseGraph>::vertex_descriptor;
  const auto null_vertex = boost::graph_traits<ReuseGraph>::null_vertex();
  scratch.component_ids.assign(num_vertices(reuse_graph), -1);
  const auto component_map = boost::make_iterator_property_map(
    scratch.component_ids.begin(), get(boost::vertex_index, reuse_graph));
  const int num_components = boost::connected_components(reuse_graph, component_map);

  if (num_components <= 1)
  {
    scratch.component_matching_mate.assign(num_vertices(reuse_graph), null_vertex);
    const auto mate_map = boost::make_iterator_property_map(scratch.component_matching_mate.begin(),
                                                            get(boost::vertex_index, reuse_graph));
    const bool is_maximum_matching =
      boost::checked_edmonds_maximum_cardinality_matching(reuse_graph, mate_map);
    if (not is_maximum_matching)
      throw std::logic_error(
        "CBC_SPDS: Boost Edmonds matching failed to produce a maximum reuse matching.");
    return boost::matching_size(reuse_graph, mate_map);
  }

  scratch.component_vertex_counts.assign(num_components, 0);
  scratch.component_edge_counts.assign(num_components, 0);
  for (const auto component_id : scratch.component_ids)
    ++scratch.component_vertex_counts[component_id];
  for (const auto& [u, _] : scratch.reuse_edges)
    ++scratch.component_edge_counts[scratch.component_ids[u]];

  scratch.component_vertex_offsets.resize(num_components + 1, 0);
  scratch.component_edge_offsets.resize(num_components + 1, 0);
  for (int component_id = 0; component_id < num_components; ++component_id)
  {
    scratch.component_vertex_offsets[component_id + 1] =
      scratch.component_vertex_offsets[component_id] +
      scratch.component_vertex_counts[component_id];
    scratch.component_edge_offsets[component_id + 1] =
      scratch.component_edge_offsets[component_id] + scratch.component_edge_counts[component_id];
  }

  scratch.local_vertex_ids.assign(num_vertices(reuse_graph), INVALID_INDEX);
  auto next_component_vertex = scratch.component_vertex_offsets;
  for (Vertex vertex = 0; vertex < num_vertices(reuse_graph); ++vertex)
  {
    const auto component_id = scratch.component_ids[vertex];
    scratch.local_vertex_ids[vertex] = static_cast<std::uint32_t>(
      next_component_vertex[component_id] - scratch.component_vertex_offsets[component_id]);
    ++next_component_vertex[component_id];
  }

  scratch.component_edges.resize(scratch.reuse_edges.size());
  scratch.component_edge_write_offsets = scratch.component_edge_offsets;
  for (const auto& [u, v] : scratch.reuse_edges)
  {
    const auto component_id = scratch.component_ids[u];
    scratch.component_edges[scratch.component_edge_write_offsets[component_id]++] = {
      scratch.local_vertex_ids[u], scratch.local_vertex_ids[v]};
  }

  std::size_t matching_size = 0;
  for (int component_id = 0; component_id < num_components; ++component_id)
  {
    const auto vertex_count = scratch.component_vertex_counts[component_id];
    const auto edge_begin =
      scratch.component_edges.begin() +
      static_cast<std::ptrdiff_t>(scratch.component_edge_offsets[component_id]);
    const auto edge_end =
      scratch.component_edges.begin() +
      static_cast<std::ptrdiff_t>(scratch.component_edge_offsets[component_id + 1]);
    ReuseGraph component_graph(edge_begin, edge_end, vertex_count);
    scratch.component_matching_mate.assign(num_vertices(component_graph), null_vertex);
    const auto mate_map = boost::make_iterator_property_map(
      scratch.component_matching_mate.begin(), get(boost::vertex_index, component_graph));
    const bool is_maximum_matching =
      boost::checked_edmonds_maximum_cardinality_matching(component_graph, mate_map);
    if (not is_maximum_matching)
      throw std::logic_error(
        "CBC_SPDS: Boost Edmonds matching failed to produce a maximum reuse matching.");
    matching_size += boost::matching_size(component_graph, mate_map);
  }

  return matching_size;
}

} // namespace

CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const std::shared_ptr<MeshContinuum>& grid,
                   const bool allow_cycles)
  : SPDS(omega, grid)
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

  const auto num_loc_cells = grid->local_cells.size();

  std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
  std::set<int> location_successors;
  std::set<int> location_dependencies;

  PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

  location_successors_.reserve(location_successors.size());
  location_dependencies_.reserve(location_dependencies.size());

  for (const auto loc : location_successors)
    location_successors_.push_back(loc);

  for (const auto loc : location_dependencies)
    location_dependencies_.push_back(loc);

  Graph local_dg(num_loc_cells);
  for (std::size_t c = 0; c < num_loc_cells; ++c)
    for (const auto& successor : cell_successors[c])
      boost::add_edge(c, successor.first, successor.second, local_dg);

  if (allow_cycles)
  {
    const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
    for (const auto& [u, v] : edges_to_remove)
      local_sweep_fas_.emplace_back(u, v);
  }

  spls_.clear();
  boost::topological_sort(local_dg, std::back_inserter(spls_));
  std::reverse(spls_.begin(), spls_.end());
  if (spls_.empty())
  {
    throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
                           "Cycles need to be allowed by the calling application.");
  }

  topo_order_.reserve(spls_.size());
  for (const auto v : spls_)
    topo_order_.push_back(static_cast<std::uint32_t>(v));

  std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
  CommunicateLocationDependencies(location_dependencies_, global_dependencies);

  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

  task_list_.assign(num_loc_cells, Task{});
  local_successor_offsets_.resize(num_loc_cells + 1, 0);

  std::size_t successor_count = 0;
  for (const auto& cell : grid_->local_cells)
  {
    unsigned int num_dependencies = 0;
    std::vector<std::uint32_t> predecessors;
    std::vector<std::uint32_t> successors;

    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = cell_face_orientations_[cell.local_id][f];

      if (orientation == INCOMING and face.has_neighbor)
      {
        ++num_dependencies;
        if (face.IsNeighborLocal(grid_.get()))
          predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
      }
      else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
        successors.push_back(grid_->cells[face.neighbor_id].local_id);
    }

    successor_count += successors.size();
    local_successor_offsets_[cell.local_id + 1] = static_cast<std::uint32_t>(successor_count);

    task_list_[cell.local_id] = Task{
      0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
  }

  local_successors_.resize(successor_count);
  for (std::uint32_t cell_id = 0; cell_id < task_list_.size(); ++cell_id)
  {
    std::copy(task_list_[cell_id].successors.begin(),
              task_list_[cell_id].successors.end(),
              local_successors_.begin() + local_successor_offsets_[cell_id]);
  }

  min_num_local_psi_slots_ = num_loc_cells;
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const noexcept
{
  return task_list_;
}

void
CBC_SPDS::ComputeMinNumLocalPsiSlots()
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMinNumLocalPsiSlots");

  const auto num_tasks = static_cast<std::uint32_t>(task_list_.size());
  if (num_tasks == 0)
  {
    min_num_local_psi_slots_ = 0;
    return;
  }

  thread_local SlotCalcScratch scratch;

  const auto chain_cover =
    BuildMinimumPathCover(local_successor_offsets_, local_successors_, scratch);
  BuildFirstReachable(
    local_successor_offsets_, local_successors_, topo_order_, chain_cover, scratch);
  BuildReuseStart(local_successor_offsets_, local_successors_, chain_cover.num_chains, scratch);
  BuildReuseEdges(local_successor_offsets_, local_successors_, chain_cover, scratch);

  const auto reuse_matching_size = ComputeReuseMatchingSize(num_tasks, scratch);
  min_num_local_psi_slots_ = static_cast<std::size_t>(num_tasks) - reuse_matching_size;
}

} // namespace opensn

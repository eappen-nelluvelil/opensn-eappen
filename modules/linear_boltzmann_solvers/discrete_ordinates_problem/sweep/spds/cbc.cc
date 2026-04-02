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
using MatchingGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
using MatchingVertex = boost::graph_traits<MatchingGraph>::vertex_descriptor;

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
  std::vector<std::uint32_t> reuse_row;
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
  std::vector<std::uint8_t> is_active_vertex;

  void ResizeMatchingState(const std::uint32_t num_tasks)
  {
    mate_u.assign(num_tasks, INVALID_INDEX);
    mate_v.assign(num_tasks, INVALID_INDEX);
    dist.assign(num_tasks, -1);
    bfs_queue.resize(num_tasks);
    is_immediate_successor.assign(num_tasks, 0);
  }
};

std::size_t
ComputeCheckedMatchingSize(const MatchingGraph& graph,
                           std::vector<MatchingVertex>& mate_storage,
                           const char* failure_message)
{
  const auto null_vertex = boost::graph_traits<MatchingGraph>::null_vertex();
  mate_storage.assign(num_vertices(graph), null_vertex);
  const auto mate_map =
    boost::make_iterator_property_map(mate_storage.begin(), get(boost::vertex_index, graph));
  const bool is_maximum_matching =
    boost::checked_edmonds_maximum_cardinality_matching(graph, mate_map);
  if (not is_maximum_matching)
    throw std::logic_error(failure_message);
  return boost::matching_size(graph, mate_map);
}

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

class ExactSlotCounter
{
public:
  ExactSlotCounter(const std::vector<std::uint32_t>& successor_offsets,
                   const std::vector<std::uint32_t>& successors,
                   const std::vector<std::uint32_t>& topo_order,
                   SlotCalcScratch& scratch)
    : successor_offsets_(successor_offsets),
      successors_(successors),
      topo_order_(topo_order),
      scratch_(scratch),
      num_tasks_(static_cast<std::uint32_t>(successor_offsets.size() - 1))
  {
  }

  std::size_t Solve()
  {
    if (num_tasks_ == 0)
      return 0;

    BuildMinimumPathCover();
    BuildFirstReachable();
    BuildReuseEdges();
    return static_cast<std::size_t>(num_tasks_) - ComputeReuseMatchingSize();
  }

private:
  void BuildMinimumPathCover()
  {
    scratch_.ResizeMatchingState(num_tasks_);
    CSRBipartiteMatcher(successor_offsets_, successors_, scratch_).Solve();

    chain_cover_.chain_id.assign(num_tasks_, INVALID_INDEX);
    chain_cover_.pos_in_chain.assign(num_tasks_, INVALID_INDEX);
    chain_cover_.chain_offsets.clear();
    chain_cover_.chain_offsets.reserve(num_tasks_ + 1);
    chain_cover_.chain_offsets.push_back(0);
    chain_cover_.chain_vertices.clear();
    chain_cover_.chain_vertices.reserve(num_tasks_);
    chain_cover_.num_chains = 0;

    const auto append_chain = [this](std::uint32_t start_vertex)
    {
      std::uint32_t current = start_vertex;
      std::uint32_t chain_pos = 0;

      while (current != INVALID_INDEX)
      {
        if (chain_cover_.chain_id[current] != INVALID_INDEX)
          throw std::logic_error("CBC_SPDS: Invalid minimum path cover construction.");

        chain_cover_.chain_id[current] = chain_cover_.num_chains;
        chain_cover_.pos_in_chain[current] = chain_pos++;
        chain_cover_.chain_vertices.push_back(current);
        current = scratch_.mate_u[current];
      }

      ++chain_cover_.num_chains;
      chain_cover_.chain_offsets.push_back(
        static_cast<std::uint32_t>(chain_cover_.chain_vertices.size()));
    };

    for (std::uint32_t v = 0; v < num_tasks_; ++v)
      if (scratch_.mate_v[v] == INVALID_INDEX)
        append_chain(v);

    for (std::uint32_t v = 0; v < num_tasks_; ++v)
      if (chain_cover_.chain_id[v] == INVALID_INDEX)
        append_chain(v);
  }

  void BuildFirstReachable()
  {
    const auto num_chains = chain_cover_.num_chains;
    scratch_.first_reachable.assign(static_cast<std::size_t>(num_tasks_) * num_chains,
                                    INVALID_INDEX);

    for (auto topo_it = topo_order_.rbegin(); topo_it != topo_order_.rend(); ++topo_it)
    {
      const auto u = *topo_it;
      auto* const row = scratch_.first_reachable.data() + static_cast<std::size_t>(u) * num_chains;

      row[chain_cover_.chain_id[u]] = chain_cover_.pos_in_chain[u];

      for (auto e = successor_offsets_[u]; e < successor_offsets_[u + 1]; ++e)
      {
        const auto succ = successors_[e];
        const auto* const succ_row =
          scratch_.first_reachable.data() + static_cast<std::size_t>(succ) * num_chains;

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

  template <class F>
  void ForEachReuseNeighbor(const std::uint32_t u, F&& func)
  {
    const auto num_chains = chain_cover_.num_chains;
    const auto succ_begin = successor_offsets_[u];
    const auto succ_end = successor_offsets_[u + 1];
    if (succ_begin == succ_end)
      return;

    scratch_.reuse_row.resize(num_chains);
    const auto first_succ = successors_[succ_begin];
    const auto* const first_succ_row =
      scratch_.first_reachable.data() + static_cast<std::size_t>(first_succ) * num_chains;
    std::copy_n(first_succ_row, num_chains, scratch_.reuse_row.begin());

    for (auto e = succ_begin + 1; e < succ_end; ++e)
    {
      const auto succ = successors_[e];
      const auto* const succ_row =
        scratch_.first_reachable.data() + static_cast<std::size_t>(succ) * num_chains;

      for (std::uint32_t chain = 0; chain < num_chains; ++chain)
      {
        auto& candidate_pos = scratch_.reuse_row[chain];
        const auto succ_pos = succ_row[chain];
        if (candidate_pos == INVALID_INDEX or succ_pos == INVALID_INDEX)
          candidate_pos = INVALID_INDEX;
        else
          candidate_pos = std::max(candidate_pos, succ_pos);
      }
    }

    for (auto e = succ_begin; e < succ_end; ++e)
      scratch_.is_immediate_successor[successors_[e]] = 1;

    for (std::uint32_t chain = 0; chain < num_chains; ++chain)
    {
      const auto start_pos = scratch_.reuse_row[chain];
      if (start_pos == INVALID_INDEX)
        continue;

      const auto chain_begin = chain_cover_.chain_offsets[chain];
      const auto chain_end = chain_cover_.chain_offsets[chain + 1];
      for (std::uint32_t pos = chain_begin + start_pos; pos < chain_end; ++pos)
      {
        const auto v = chain_cover_.chain_vertices[pos];
        if (not scratch_.is_immediate_successor[v])
          func(v);
      }
    }

    for (auto e = succ_begin; e < succ_end; ++e)
      scratch_.is_immediate_successor[successors_[e]] = 0;
  }

  void BuildReuseEdges()
  {
    std::size_t reuse_edge_count = 0;
    for (std::uint32_t u = 0; u < num_tasks_; ++u)
      ForEachReuseNeighbor(u, [&](const std::uint32_t) { ++reuse_edge_count; });

    scratch_.reuse_edges.clear();
    scratch_.reuse_edges.reserve(reuse_edge_count);

    for (std::uint32_t u = 0; u < num_tasks_; ++u)
      ForEachReuseNeighbor(
        u, [&](const std::uint32_t v) { scratch_.reuse_edges.emplace_back(u, num_tasks_ + v); });
  }

  std::size_t ComputeReuseMatchingSize()
  {
    if (scratch_.reuse_edges.empty())
      return 0;

    const auto num_graph_vertices = static_cast<std::size_t>(num_tasks_) * 2;
    scratch_.is_active_vertex.assign(num_graph_vertices, 0);
    for (const auto& [u, v] : scratch_.reuse_edges)
    {
      scratch_.is_active_vertex[u] = 1;
      scratch_.is_active_vertex[v] = 1;
    }

    scratch_.local_vertex_ids.assign(num_graph_vertices, INVALID_INDEX);
    std::uint32_t active_vertex_count = 0;
    for (std::size_t vertex = 0; vertex < num_graph_vertices; ++vertex)
      if (scratch_.is_active_vertex[vertex])
        scratch_.local_vertex_ids[vertex] = active_vertex_count++;

    scratch_.component_edges.resize(scratch_.reuse_edges.size());
    std::transform(scratch_.reuse_edges.begin(),
                   scratch_.reuse_edges.end(),
                   scratch_.component_edges.begin(),
                   [&](const auto& edge)
                   {
                     return std::pair<std::uint32_t, std::uint32_t>{
                       scratch_.local_vertex_ids[edge.first],
                       scratch_.local_vertex_ids[edge.second]};
                   });

    MatchingGraph reuse_graph(
      scratch_.component_edges.begin(), scratch_.component_edges.end(), active_vertex_count);

    scratch_.component_ids.assign(num_vertices(reuse_graph), -1);
    const auto component_map = boost::make_iterator_property_map(
      scratch_.component_ids.begin(), get(boost::vertex_index, reuse_graph));
    const int num_components = boost::connected_components(reuse_graph, component_map);

    if (num_components <= 1)
    {
      return ComputeCheckedMatchingSize(
        reuse_graph,
        scratch_.component_matching_mate,
        "CBC_SPDS: Boost Edmonds matching failed to produce a maximum reuse matching.");
    }

    scratch_.component_vertex_counts.assign(num_components, 0);
    scratch_.component_edge_counts.assign(num_components, 0);
    for (const auto component_id : scratch_.component_ids)
      ++scratch_.component_vertex_counts[component_id];
    for (const auto& [u, _] : scratch_.reuse_edges)
      ++scratch_.component_edge_counts[scratch_.component_ids[u]];

    scratch_.component_vertex_offsets.resize(num_components + 1, 0);
    scratch_.component_edge_offsets.resize(num_components + 1, 0);
    for (int component_id = 0; component_id < num_components; ++component_id)
    {
      scratch_.component_vertex_offsets[component_id + 1] =
        scratch_.component_vertex_offsets[component_id] +
        scratch_.component_vertex_counts[component_id];
      scratch_.component_edge_offsets[component_id + 1] =
        scratch_.component_edge_offsets[component_id] +
        scratch_.component_edge_counts[component_id];
    }

    auto next_component_vertex = scratch_.component_vertex_offsets;
    for (MatchingVertex vertex = 0; vertex < num_vertices(reuse_graph); ++vertex)
    {
      const auto component_id = scratch_.component_ids[vertex];
      scratch_.local_vertex_ids[vertex] = static_cast<std::uint32_t>(
        next_component_vertex[component_id] - scratch_.component_vertex_offsets[component_id]);
      ++next_component_vertex[component_id];
    }

    scratch_.component_edges.resize(scratch_.reuse_edges.size());
    scratch_.component_edge_write_offsets = scratch_.component_edge_offsets;
    for (const auto& [u, v] : scratch_.reuse_edges)
    {
      const auto component_id = scratch_.component_ids[u];
      scratch_.component_edges[scratch_.component_edge_write_offsets[component_id]++] = {
        scratch_.local_vertex_ids[u], scratch_.local_vertex_ids[v]};
    }

    std::size_t matching_size = 0;
    for (int component_id = 0; component_id < num_components; ++component_id)
    {
      const auto vertex_count = scratch_.component_vertex_counts[component_id];
      const auto edge_begin =
        scratch_.component_edges.begin() +
        static_cast<std::ptrdiff_t>(scratch_.component_edge_offsets[component_id]);
      const auto edge_end =
        scratch_.component_edges.begin() +
        static_cast<std::ptrdiff_t>(scratch_.component_edge_offsets[component_id + 1]);
      MatchingGraph component_graph(edge_begin, edge_end, vertex_count);
      matching_size += ComputeCheckedMatchingSize(
        component_graph,
        scratch_.component_matching_mate,
        "CBC_SPDS: Boost Edmonds matching failed to produce a maximum reuse matching.");
    }

    return matching_size;
  }

private:
  const std::vector<std::uint32_t>& successor_offsets_;
  const std::vector<std::uint32_t>& successors_;
  const std::vector<std::uint32_t>& topo_order_;
  SlotCalcScratch& scratch_;
  const std::uint32_t num_tasks_;
  ChainCoverData chain_cover_;
};

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
  min_num_local_psi_slots_ =
    ExactSlotCounter(local_successor_offsets_, local_successors_, topo_order_, scratch).Solve();
}

} // namespace opensn

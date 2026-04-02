// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <boost/graph/topological_sort.hpp>

namespace opensn
{

namespace
{

constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

class ExplicitBipartiteMatcher
{
public:
  explicit ExplicitBipartiteMatcher(const std::vector<std::vector<std::uint32_t>>& adjacency)
    : adjacency_(adjacency),
      mate_u_(adjacency.size(), INVALID_INDEX),
      mate_v_(adjacency.size(), INVALID_INDEX),
      dist_(adjacency.size(), -1)
  {
  }

  std::size_t Solve()
  {
    std::size_t matching_size = GreedyInitialize();
    while (BFS())
    {
      for (std::uint32_t u = 0; u < adjacency_.size(); ++u)
      {
        if (mate_u_[u] == INVALID_INDEX and DFS(u))
          ++matching_size;
      }
    }
    return matching_size;
  }

  const std::vector<std::uint32_t>& MateU() const noexcept { return mate_u_; }
  const std::vector<std::uint32_t>& MateV() const noexcept { return mate_v_; }

private:
  std::size_t GreedyInitialize()
  {
    std::size_t matching_size = 0;
    for (std::uint32_t u = 0; u < adjacency_.size(); ++u)
    {
      for (const auto v : adjacency_[u])
      {
        if (mate_v_[v] != INVALID_INDEX)
          continue;

        mate_u_[u] = v;
        mate_v_[v] = u;
        ++matching_size;
        break;
      }
    }
    return matching_size;
  }

  bool BFS()
  {
    std::fill(dist_.begin(), dist_.end(), -1);
    std::queue<std::uint32_t> work_queue;

    for (std::uint32_t u = 0; u < adjacency_.size(); ++u)
    {
      if (mate_u_[u] == INVALID_INDEX)
      {
        dist_[u] = 0;
        work_queue.push(u);
      }
    }

    dist_null_ = std::numeric_limits<int>::max();

    while (not work_queue.empty())
    {
      const auto u = work_queue.front();
      work_queue.pop();

      if (dist_[u] >= dist_null_)
        continue;

      for (const auto v : adjacency_[u])
      {
        const auto mate_of_v = mate_v_[v];
        if (mate_of_v == INVALID_INDEX)
        {
          dist_null_ = std::min(dist_null_, dist_[u] + 1);
          continue;
        }

        if (dist_[mate_of_v] == -1)
        {
          dist_[mate_of_v] = dist_[u] + 1;
          work_queue.push(mate_of_v);
        }
      }
    }

    return dist_null_ != std::numeric_limits<int>::max();
  }

  bool DFS(std::uint32_t u)
  {
    for (const auto v : adjacency_[u])
    {
      const auto mate_of_v = mate_v_[v];
      if (mate_of_v == INVALID_INDEX)
      {
        if (dist_null_ == dist_[u] + 1)
        {
          mate_u_[u] = v;
          mate_v_[v] = u;
          dist_[u] = -1;
          return true;
        }
      }
      else if (dist_[mate_of_v] == dist_[u] + 1 and DFS(mate_of_v))
      {
        mate_u_[u] = v;
        mate_v_[v] = u;
        dist_[u] = -1;
        return true;
      }
    }

    dist_[u] = -1;
    return false;
  }

  const std::vector<std::vector<std::uint32_t>>& adjacency_;
  std::vector<std::uint32_t> mate_u_;
  std::vector<std::uint32_t> mate_v_;
  std::vector<int> dist_;
  int dist_null_ = std::numeric_limits<int>::max();
};

struct ChainCoverData
{
  std::uint32_t num_chains = 0;
  std::vector<std::uint32_t> chain_id;
  std::vector<std::uint32_t> pos_in_chain;
  std::vector<std::uint32_t> chain_offsets;
  std::vector<std::uint32_t> chain_vertices;
};

ChainCoverData
BuildMinimumPathCover(const std::vector<Task>& task_list)
{
  const auto num_tasks = static_cast<std::uint32_t>(task_list.size());

  std::vector<std::vector<std::uint32_t>> adjacency(num_tasks);
  for (std::uint32_t u = 0; u < num_tasks; ++u)
    adjacency[u] = task_list[u].successors;

  ExplicitBipartiteMatcher matcher(adjacency);
  matcher.Solve();

  const auto& next_in_chain = matcher.MateU();
  const auto& prev_in_chain = matcher.MateV();

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
      current = next_in_chain[current];
    }

    ++chain_cover.num_chains;
    chain_cover.chain_offsets.push_back(
      static_cast<std::uint32_t>(chain_cover.chain_vertices.size()));
  };

  for (std::uint32_t v = 0; v < num_tasks; ++v)
    if (prev_in_chain[v] == INVALID_INDEX)
      append_chain(v);

  for (std::uint32_t v = 0; v < num_tasks; ++v)
    if (chain_cover.chain_id[v] == INVALID_INDEX)
      append_chain(v);

  return chain_cover;
}

std::vector<std::uint32_t>
BuildFirstReachable(const std::vector<Task>& task_list,
                    const std::vector<std::uint32_t>& topo_order,
                    const ChainCoverData& chain_cover)
{
  const auto num_tasks = static_cast<std::uint32_t>(task_list.size());
  const auto num_chains = chain_cover.num_chains;

  std::vector<std::uint32_t> first_reachable(static_cast<std::size_t>(num_tasks) * num_chains,
                                             INVALID_INDEX);

  for (auto topo_it = topo_order.rbegin(); topo_it != topo_order.rend(); ++topo_it)
  {
    const auto u = *topo_it;
    auto* const row = first_reachable.data() + static_cast<std::size_t>(u) * num_chains;

    row[chain_cover.chain_id[u]] = chain_cover.pos_in_chain[u];

    for (const auto succ : task_list[u].successors)
    {
      const auto* const succ_row =
        first_reachable.data() + static_cast<std::size_t>(succ) * num_chains;

      for (std::uint32_t c = 0; c < num_chains; ++c)
      {
        const auto succ_pos = succ_row[c];
        auto& first_pos = row[c];
        if (succ_pos == INVALID_INDEX)
          continue;

        if (first_pos == INVALID_INDEX or succ_pos < first_pos)
          first_pos = succ_pos;
      }
    }
  }

  return first_reachable;
}

std::vector<std::uint32_t>
BuildReuseStart(const std::vector<Task>& task_list,
                std::uint32_t num_chains,
                const std::vector<std::uint32_t>& first_reachable)
{
  const auto num_tasks = static_cast<std::uint32_t>(task_list.size());
  std::vector<std::uint32_t> reuse_start(static_cast<std::size_t>(num_tasks) * num_chains,
                                         INVALID_INDEX);

  for (std::uint32_t u = 0; u < num_tasks; ++u)
  {
    const auto& successors = task_list[u].successors;
    if (successors.empty())
      continue;

    auto* const reuse_row = reuse_start.data() + static_cast<std::size_t>(u) * num_chains;
    const auto* const first_succ_row =
      first_reachable.data() + static_cast<std::size_t>(successors.front()) * num_chains;
    std::copy_n(first_succ_row, num_chains, reuse_row);

    for (std::size_t succ_i = 1; succ_i < successors.size(); ++succ_i)
    {
      const auto* const succ_row =
        first_reachable.data() + static_cast<std::size_t>(successors[succ_i]) * num_chains;

      for (std::uint32_t c = 0; c < num_chains; ++c)
      {
        auto& candidate_pos = reuse_row[c];
        const auto succ_pos = succ_row[c];
        if (candidate_pos == INVALID_INDEX or succ_pos == INVALID_INDEX)
          candidate_pos = INVALID_INDEX;
        else
          candidate_pos = std::max(candidate_pos, succ_pos);
      }
    }
  }

  return reuse_start;
}

class ReuseGraphMatcher
{
public:
  ReuseGraphMatcher(const std::vector<Task>& task_list,
                    const ChainCoverData& chain_cover,
                    const std::vector<std::uint32_t>& reuse_start)
    : task_list_(task_list),
      chain_cover_(chain_cover),
      reuse_start_(reuse_start),
      mate_u_(task_list.size(), INVALID_INDEX),
      mate_v_(task_list.size(), INVALID_INDEX),
      dist_(task_list.size(), -1)
  {
  }

  std::size_t Solve()
  {
    std::size_t matching_size = GreedyInitialize();
    while (BFS())
    {
      for (std::uint32_t u = 0; u < task_list_.size(); ++u)
      {
        if (mate_u_[u] == INVALID_INDEX and DFS(u))
          ++matching_size;
      }
    }
    return matching_size;
  }

private:
  template <class F>
  void ForEachNeighbor(std::uint32_t u, F&& func) const
  {
    const auto num_chains = chain_cover_.num_chains;
    const auto* const reuse_row = reuse_start_.data() + static_cast<std::size_t>(u) * num_chains;

    for (std::uint32_t chain = 0; chain < num_chains; ++chain)
    {
      const auto start_pos = reuse_row[chain];
      if (start_pos == INVALID_INDEX)
        continue;

      const auto chain_begin = chain_cover_.chain_offsets[chain];
      const auto chain_end = chain_cover_.chain_offsets[chain + 1];
      for (auto pos = chain_begin + start_pos; pos < chain_end; ++pos)
      {
        const auto v = chain_cover_.chain_vertices[pos];
        if (std::find(task_list_[u].successors.begin(), task_list_[u].successors.end(), v) !=
            task_list_[u].successors.end())
          continue;

        if (not func(v))
          return;
      }
    }
  }

  std::size_t GreedyInitialize()
  {
    std::size_t matching_size = 0;
    for (std::uint32_t u = 0; u < task_list_.size(); ++u)
    {
      if (mate_u_[u] != INVALID_INDEX)
        continue;

      bool matched = false;
      ForEachNeighbor(u,
                      [&](std::uint32_t v)
                      {
                        if (mate_v_[v] != INVALID_INDEX)
                          return true;

                        mate_u_[u] = v;
                        mate_v_[v] = u;
                        matched = true;
                        return false;
                      });

      if (matched)
        ++matching_size;
    }

    return matching_size;
  }

  bool BFS()
  {
    std::fill(dist_.begin(), dist_.end(), -1);
    std::queue<std::uint32_t> work_queue;

    for (std::uint32_t u = 0; u < task_list_.size(); ++u)
    {
      if (mate_u_[u] == INVALID_INDEX)
      {
        dist_[u] = 0;
        work_queue.push(u);
      }
    }

    dist_null_ = std::numeric_limits<int>::max();

    while (not work_queue.empty())
    {
      const auto u = work_queue.front();
      work_queue.pop();

      if (dist_[u] >= dist_null_)
        continue;

      ForEachNeighbor(u,
                      [&](std::uint32_t v)
                      {
                        const auto mate_of_v = mate_v_[v];
                        if (mate_of_v == INVALID_INDEX)
                        {
                          dist_null_ = std::min(dist_null_, dist_[u] + 1);
                          return true;
                        }

                        if (dist_[mate_of_v] == -1)
                        {
                          dist_[mate_of_v] = dist_[u] + 1;
                          work_queue.push(mate_of_v);
                        }
                        return true;
                      });
    }

    return dist_null_ != std::numeric_limits<int>::max();
  }

  bool DFS(std::uint32_t u)
  {
    bool augmented = false;

    ForEachNeighbor(u,
                    [&](std::uint32_t v)
                    {
                      const auto mate_of_v = mate_v_[v];
                      if (mate_of_v == INVALID_INDEX)
                      {
                        if (dist_null_ == dist_[u] + 1)
                        {
                          mate_u_[u] = v;
                          mate_v_[v] = u;
                          dist_[u] = -1;
                          augmented = true;
                          return false;
                        }
                        return true;
                      }

                      if (dist_[mate_of_v] == dist_[u] + 1 and DFS(mate_of_v))
                      {
                        mate_u_[u] = v;
                        mate_v_[v] = u;
                        dist_[u] = -1;
                        augmented = true;
                        return false;
                      }

                      return true;
                    });

    if (not augmented)
      dist_[u] = -1;

    return augmented;
  }

  const std::vector<Task>& task_list_;
  const ChainCoverData& chain_cover_;
  const std::vector<std::uint32_t>& reuse_start_;
  std::vector<std::uint32_t> mate_u_;
  std::vector<std::uint32_t> mate_v_;
  std::vector<int> dist_;
  int dist_null_ = std::numeric_limits<int>::max();
};

} // namespace

CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const std::shared_ptr<MeshContinuum>& grid,
                   bool allow_cycles)
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

  std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
  CommunicateLocationDependencies(location_dependencies_, global_dependencies);

  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

  task_list_.assign(num_loc_cells, Task{});
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

    task_list_[cell.local_id] = Task{
      0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
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

  std::vector<std::uint32_t> topo_order;
  topo_order.reserve(spls_.size());
  for (const auto v : spls_)
    topo_order.push_back(static_cast<std::uint32_t>(v));

  const auto chain_cover = BuildMinimumPathCover(task_list_);
  const auto first_reachable = BuildFirstReachable(task_list_, topo_order, chain_cover);
  const auto reuse_start = BuildReuseStart(task_list_, chain_cover.num_chains, first_reachable);

  ReuseGraphMatcher matcher(task_list_, chain_cover, reuse_start);
  const auto reuse_matching_size = matcher.Solve();
  min_num_local_psi_slots_ = static_cast<std::size_t>(num_tasks) - reuse_matching_size;
}

} // namespace opensn

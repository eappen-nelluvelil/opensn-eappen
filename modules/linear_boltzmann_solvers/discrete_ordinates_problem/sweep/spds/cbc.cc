// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"
#include <algorithm>
#include <bit>
#include <boost/graph/topological_sort.hpp>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace opensn
{

namespace
{

constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

class BitMatrix
{
public:
  BitMatrix() : n_(0), words_per_row_(0) {}

  void ResizeAndClear(const std::size_t n)
  {
    n_ = n;
    words_per_row_ = (((n + 63) / 64) + 7) & ~7ULL;
    const auto required_words = n * words_per_row_;

    if (data_.size() < required_words)
      data_.resize(required_words, 0ULL);
    else
      std::fill_n(data_.begin(), required_words, 0ULL);
  }

  std::uint64_t* Row(const std::size_t i) { return data_.data() + i * words_per_row_; }

  const std::uint64_t* Row(const std::size_t i) const
  {
    return data_.data() + i * words_per_row_;
  }

  void SetBit(const std::size_t i, const std::size_t j)
  {
    Row(i)[j / 64] |= (1ULL << (j % 64));
  }

  void ClearBit(const std::size_t i, const std::size_t j)
  {
    Row(i)[j / 64] &= ~(1ULL << (j % 64));
  }

  bool TestBit(const std::size_t i, const std::size_t j) const
  {
    return (Row(i)[j / 64] & (1ULL << (j % 64))) != 0ULL;
  }

  void CopyRow(const std::size_t dst,
               const BitMatrix& src_mat,
               const std::size_t src_row,
               const std::size_t start_pos = 0)
  {
    const auto start_word = start_pos / 64;
    auto* __restrict__ dst_row = Row(dst) + start_word;
    const auto* __restrict__ src_row_data = src_mat.Row(src_row) + start_word;
    const auto words_to_copy = words_per_row_ - start_word;
    std::memcpy(dst_row, src_row_data, words_to_copy * sizeof(std::uint64_t));
  }

  void OrRows(const std::size_t dst,
              const BitMatrix& src_mat,
              const std::size_t src_row,
              const std::size_t start_pos = 0)
  {
    const auto start_word = start_pos / 64;
    auto* __restrict__ dst_row = Row(dst) + start_word;
    const auto* __restrict__ src_row_data = src_mat.Row(src_row) + start_word;
    const auto words_to_process = words_per_row_ - start_word;

    for (std::size_t w = 0; w < words_to_process; ++w)
      dst_row[w] |= src_row_data[w];
  }

  void AndRows(const std::size_t dst,
               const BitMatrix& src_mat,
               const std::size_t src_row,
               const std::size_t start_pos = 0)
  {
    const auto start_word = start_pos / 64;
    auto* __restrict__ dst_row = Row(dst) + start_word;
    const auto* __restrict__ src_row_data = src_mat.Row(src_row) + start_word;
    const auto words_to_process = words_per_row_ - start_word;

    for (std::size_t w = 0; w < words_to_process; ++w)
      dst_row[w] &= src_row_data[w];
  }

  std::size_t FindFirstSet(const std::size_t row, const std::size_t start_pos = 0) const
  {
    const auto* __restrict__ row_data = Row(row);
    auto word = start_pos / 64;

    if (word >= words_per_row_)
      return n_;

    const auto masked = row_data[word] & (~0ULL << (start_pos % 64));
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
  std::size_t n_;
  std::size_t words_per_row_;
  std::vector<std::uint64_t> data_;
};

struct ThreadLocalWorkspace
{
  BitMatrix reachability;
  BitMatrix reuse_targets;
  std::vector<std::uint32_t> mate_u;
  std::vector<std::uint32_t> mate_v;
  std::vector<int> dist;
  std::vector<std::uint32_t> queue;
  std::vector<std::uint32_t> topo_rank;
  std::vector<std::uint8_t> is_sink_rank;

  void Prepare(const std::size_t n)
  {
    reachability.ResizeAndClear(n);
    reuse_targets.ResizeAndClear(n);
    mate_u.assign(n, INVALID_INDEX);
    mate_v.assign(n, INVALID_INDEX);
    dist.assign(n, -1);

    if (queue.size() < n)
      queue.resize(n);
    if (topo_rank.size() < n)
      topo_rank.resize(n);
    if (is_sink_rank.size() < n)
      is_sink_rank.resize(n);
  }
};

struct CBCSlotPlan
{
  std::size_t num_dynamic_slots = 0;
  std::size_t num_static_slots = 0;
};

class DenseHopcroftKarp
{
public:
  DenseHopcroftKarp(const std::uint32_t num_tasks,
                    const std::vector<Task>& task_list,
                    const std::vector<std::uint32_t>& topo_order,
                    std::vector<std::uint32_t>& task_slot_ids,
                    ThreadLocalWorkspace& ws)
    : num_tasks_(num_tasks),
      task_list_(task_list),
      topo_order_(topo_order),
      task_slot_ids_(task_slot_ids),
      ws_(ws)
  {
    ws_.Prepare(num_tasks_);

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      ws_.topo_rank[topo_order_[i]] = i;
      ws_.is_sink_rank[i] = task_list_[topo_order_[i]].successors.empty() ? 1U : 0U;
    }
  }

  CBCSlotPlan Solve()
  {
    BuildReachabilityAndReuseTargets();

    const auto dynamic_matching_size =
      ComputeMaximumMatching(/*skip_sink_sources=*/false, /*skip_sink_targets=*/false);

    AssignStaticSlots();

    return CBCSlotPlan{static_cast<std::size_t>(num_tasks_) - dynamic_matching_size,
                       num_static_slots_};
  }

private:
  void BuildReachabilityAndReuseTargets()
  {
    for (std::uint32_t i = num_tasks_; i-- > 0;)
    {
      const auto u = topo_order_[i];
      const auto& successors = task_list_[u].successors;

      if (successors.empty())
      {
        ws_.reachability.SetBit(i, i);
        continue;
      }

      const auto first_succ_rank = ws_.topo_rank[successors.front()];
      ws_.reachability.CopyRow(i, ws_.reachability, first_succ_rank, first_succ_rank);
      ws_.reachability.SetBit(i, i);

      for (std::size_t j = 1; j < successors.size(); ++j)
      {
        const auto succ_rank = ws_.topo_rank[successors[j]];
        ws_.reachability.OrRows(i, ws_.reachability, succ_rank, succ_rank);
      }
    }

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      const auto u = topo_order_[i];
      const auto& successors = task_list_[u].successors;
      if (successors.empty())
        continue;

      auto max_succ_rank = ws_.topo_rank[successors.front()];
      for (std::size_t j = 1; j < successors.size(); ++j)
        max_succ_rank = std::max(max_succ_rank, ws_.topo_rank[successors[j]]);

      ws_.reuse_targets.CopyRow(i,
                                ws_.reachability,
                                ws_.topo_rank[successors.front()],
                                max_succ_rank);

      for (std::size_t j = 1; j < successors.size(); ++j)
        ws_.reuse_targets.AndRows(i, ws_.reachability, ws_.topo_rank[successors[j]], max_succ_rank);

      for (const auto succ : successors)
        ws_.reuse_targets.ClearBit(i, ws_.topo_rank[succ]);
    }
  }

  void ResetMatchingState()
  {
    std::fill_n(ws_.mate_u.begin(), num_tasks_, INVALID_INDEX);
    std::fill_n(ws_.mate_v.begin(), num_tasks_, INVALID_INDEX);
    std::fill_n(ws_.dist.begin(), num_tasks_, -1);
  }

  std::size_t ComputeMaximumMatching(const bool skip_sink_sources,
                                     const bool skip_sink_targets)
  {
    ResetMatchingState();

    auto matching_size = GreedyInit(skip_sink_sources, skip_sink_targets);
    while (BFS(skip_sink_sources, skip_sink_targets))
      for (std::uint32_t i = 0; i < num_tasks_; ++i)
        if (IsEligibleSource(i, skip_sink_sources) and ws_.mate_u[i] == INVALID_INDEX and
            DFS(i, skip_sink_targets))
          ++matching_size;

    return matching_size;
  }

  bool IsEligibleSource(const std::uint32_t u, const bool skip_sink_sources) const
  {
    return (not skip_sink_sources) or ws_.is_sink_rank[u] == 0U;
  }

  std::size_t FindFirstMatchableNeighbor(const std::uint32_t u,
                                         const bool skip_sink_targets) const
  {
    auto v = ws_.reuse_targets.FindFirstSet(u, u + 1);
    while (skip_sink_targets and v < num_tasks_ and ws_.is_sink_rank[v] != 0U)
      v = ws_.reuse_targets.FindNextSet(u, v);
    return v;
  }

  std::size_t FindNextMatchableNeighbor(const std::uint32_t u,
                                        const std::size_t v,
                                        const bool skip_sink_targets) const
  {
    auto next = ws_.reuse_targets.FindNextSet(u, v);
    while (skip_sink_targets and next < num_tasks_ and ws_.is_sink_rank[next] != 0U)
      next = ws_.reuse_targets.FindNextSet(u, next);
    return next;
  }

  std::size_t GreedyInit(const bool skip_sink_sources, const bool skip_sink_targets)
  {
    std::size_t count = 0;
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_u[i] != INVALID_INDEX or not IsEligibleSource(i, skip_sink_sources))
        continue;

      auto v = FindFirstMatchableNeighbor(i, skip_sink_targets);
      while (v < num_tasks_)
      {
        if (ws_.mate_v[v] == INVALID_INDEX)
        {
          ws_.mate_u[i] = static_cast<std::uint32_t>(v);
          ws_.mate_v[v] = i;
          ++count;
          break;
        }
        v = FindNextMatchableNeighbor(i, v, skip_sink_targets);
      }
    }
    return count;
  }

  bool BFS(const bool skip_sink_sources, const bool skip_sink_targets)
  {
    std::fill_n(ws_.dist.begin(), num_tasks_, -1);
    std::size_t head = 0;
    std::size_t tail = 0;

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
      if (ws_.mate_u[i] == INVALID_INDEX and IsEligibleSource(i, skip_sink_sources))
      {
        ws_.dist[i] = 0;
        ws_.queue[tail++] = i;
      }

    dist_null_ = std::numeric_limits<int>::max();

    while (head < tail)
    {
      const auto u = ws_.queue[head++];
      if (ws_.dist[u] >= dist_null_)
        continue;

      auto v = FindFirstMatchableNeighbor(u, skip_sink_targets);
      while (v < num_tasks_)
      {
        const auto mate_of_v = ws_.mate_v[v];
        if (mate_of_v == INVALID_INDEX)
        {
          if (dist_null_ == std::numeric_limits<int>::max())
            dist_null_ = ws_.dist[u] + 1;
        }
        else if (ws_.dist[mate_of_v] == -1)
        {
          ws_.dist[mate_of_v] = ws_.dist[u] + 1;
          ws_.queue[tail++] = mate_of_v;
        }
        v = FindNextMatchableNeighbor(u, v, skip_sink_targets);
      }
    }
    return dist_null_ != std::numeric_limits<int>::max();
  }

  bool DFS(const std::uint32_t u, const bool skip_sink_targets)
  {
    auto v = FindFirstMatchableNeighbor(u, skip_sink_targets);
    while (v < num_tasks_)
    {
      const auto mate_of_v = ws_.mate_v[v];
      if (mate_of_v == INVALID_INDEX)
      {
        if (dist_null_ == ws_.dist[u] + 1)
        {
          ws_.mate_v[v] = u;
          ws_.mate_u[u] = static_cast<std::uint32_t>(v);
          ws_.dist[u] = -1;
          return true;
        }
      }
      else if (ws_.dist[mate_of_v] == ws_.dist[u] + 1 and DFS(mate_of_v, skip_sink_targets))
      {
        ws_.mate_v[v] = u;
        ws_.mate_u[u] = static_cast<std::uint32_t>(v);
        ws_.dist[u] = -1;
        return true;
      }
      v = FindNextMatchableNeighbor(u, v, skip_sink_targets);
    }
    ws_.dist[u] = -1;
    return false;
  }

  void AssignStaticSlots()
  {
    ComputeMaximumMatching(/*skip_sink_sources=*/true, /*skip_sink_targets=*/true);

    task_slot_ids_.assign(num_tasks_, INVALID_INDEX);
    std::uint32_t next_slot_id = 0;
    std::vector<std::vector<std::uint32_t>> non_sink_chains;
    std::vector<std::uint32_t> chain_slot_ids;
    non_sink_chains.reserve(num_tasks_);
    chain_slot_ids.reserve(num_tasks_);

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.is_sink_rank[i] != 0U or ws_.mate_v[i] != INVALID_INDEX)
        continue;

      std::vector<std::uint32_t> chain;
      auto current = i;
      while (current != INVALID_INDEX)
      {
        task_slot_ids_[topo_order_[current]] = next_slot_id;
        chain.push_back(current);
        current = ws_.mate_u[current];
      }

      non_sink_chains.push_back(std::move(chain));
      chain_slot_ids.push_back(next_slot_id);
      ++next_slot_id;
    }

    std::uint32_t shared_sink_slot_id = INVALID_INDEX;
    for (std::uint32_t sink_rank = 0; sink_rank < num_tasks_; ++sink_rank)
    {
      if (ws_.is_sink_rank[sink_rank] == 0U)
        continue;

      auto assigned_slot_id = INVALID_INDEX;
      for (std::size_t chain_idx = 0; chain_idx < non_sink_chains.size(); ++chain_idx)
      {
        bool chain_is_compatible = true;
        for (const auto task_rank : non_sink_chains[chain_idx])
          if (not ws_.reuse_targets.TestBit(task_rank, sink_rank))
          {
            chain_is_compatible = false;
            break;
          }

        if (chain_is_compatible)
        {
          assigned_slot_id = chain_slot_ids[chain_idx];
          break;
        }
      }

      if (assigned_slot_id == INVALID_INDEX)
      {
        if (shared_sink_slot_id == INVALID_INDEX)
          shared_sink_slot_id = next_slot_id++;
        assigned_slot_id = shared_sink_slot_id;
      }

      task_slot_ids_[topo_order_[sink_rank]] = assigned_slot_id;
    }

    num_static_slots_ = next_slot_id;
  }

  std::uint32_t num_tasks_;
  const std::vector<Task>& task_list_;
  const std::vector<std::uint32_t>& topo_order_;
  std::vector<std::uint32_t>& task_slot_ids_;
  ThreadLocalWorkspace& ws_;
  int dist_null_ = 0;
  std::size_t num_static_slots_ = 0;
};

} // namespace

void
CBC_SPDS::BuildTaskGraph()
{
  constexpr auto incoming = FaceOrientation::INCOMING;
  constexpr auto outgoing = FaceOrientation::OUTGOING;

  const auto num_loc_cells = grid_->local_cells.size();
  task_list_.assign(num_loc_cells, Task{});

  for (const auto& cell : grid_->local_cells)
  {
    unsigned int num_dependencies = 0;
    std::vector<std::uint32_t> predecessors;
    std::vector<std::uint32_t> successors;

    predecessors.reserve(cell.faces.size());
    successors.reserve(cell.faces.size());

    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = cell_face_orientations_[cell.local_id][f];

      if (orientation == incoming and face.has_neighbor)
      {
        ++num_dependencies;
        if (face.IsNeighborLocal(grid_.get()))
          predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
      }
      else if (orientation == outgoing and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
        successors.push_back(grid_->cells[face.neighbor_id].local_id);
    }

    task_list_[cell.local_id] = Task{
      0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
  }
}

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
  for (const auto loc : location_successors)
    location_successors_.push_back(loc);

  location_dependencies_.reserve(location_dependencies.size());
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

  BuildTaskGraph();

  max_num_local_psi_slots_ = num_loc_cells;
  num_static_local_psi_slots_ = num_loc_cells;
  task_slot_ids_.resize(num_loc_cells);
  std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), 0);
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const noexcept
{
  return task_list_;
}

void
CBC_SPDS::ComputeMaxNumLocalPsiSlots()
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMaxNumLocalPsiSlots");

  const auto num_tasks = static_cast<std::uint32_t>(task_list_.size());
  if (num_tasks == 0)
  {
    max_num_local_psi_slots_ = 0;
    num_static_local_psi_slots_ = 0;
    task_slot_ids_.clear();
    return;
  }

  thread_local ThreadLocalWorkspace workspace;

  DenseHopcroftKarp allocator(num_tasks, task_list_, topo_order_, task_slot_ids_, workspace);
  const auto slot_plan = allocator.Solve();
  max_num_local_psi_slots_ = slot_plan.num_dynamic_slots;
  num_static_local_psi_slots_ = slot_plan.num_static_slots;
}

} // namespace opensn

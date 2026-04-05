// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace opensn
{
namespace detail
{

/// Sentinel used throughout the CBC cell-level slot planner to mark unset matching / slot entries.
inline constexpr std::uint32_t CBC_SLOT_PLANNER_INVALID_INDEX =
  std::numeric_limits<std::uint32_t>::max();

/**
 * Packed bit matrix used by the CBC cell-level slot planner.
 *
 * Row stride is rounded up to a multiple of eight 64-bit words (one AVX-512 vector) to let
 * bitwise row-ops vectorize cleanly. Rows store task-rank-indexed bit sets: reachability from
 * task i, or the set of ranks v > i at which i's physical slot is safe to reuse.
 */
class CBCSlotPlannerBitMatrix
{
public:
  CBCSlotPlannerBitMatrix() = default;

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

  /// Copy `src_mat.Row(src_row)` into `Row(dst)` starting at bit `start_pos`.
  void CopyRow(const std::size_t dst,
               const CBCSlotPlannerBitMatrix& src_mat,
               const std::size_t src_row,
               const std::size_t start_pos = 0)
  {
    const auto start_word = start_pos / 64;
    auto* __restrict__ dst_row = Row(dst) + start_word;
    const auto* __restrict__ src_row_data = src_mat.Row(src_row) + start_word;
    const auto words_to_copy = words_per_row_ - start_word;
    std::memcpy(dst_row, src_row_data, words_to_copy * sizeof(std::uint64_t));
  }

  /// In-place OR of `src_mat.Row(src_row)` into `Row(dst)` starting at bit `start_pos`.
  void OrRows(const std::size_t dst,
              const CBCSlotPlannerBitMatrix& src_mat,
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

  /// In-place AND of `src_mat.Row(src_row)` into `Row(dst)` starting at bit `start_pos`.
  void AndRows(const std::size_t dst,
               const CBCSlotPlannerBitMatrix& src_mat,
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

  /// First set bit at or after `start_pos` on `row`; returns n_ if none.
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

  /// First set bit strictly after `pos` on `row`; returns n_ if none.
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

/**
 * Reusable scratch buffers for one CBC cell-level slot planning pass.
 *
 * Held as a `thread_local` instance inside `CBC_SPDS::ComputeMaxNumLocalPsiSlots`, so multiple
 * SPDS instances planned concurrently via `SPMD_ThreadPool` reuse per-thread allocations across
 * successive directions.
 */
struct CBCSlotPlannerWorkspace
{
  CBCSlotPlannerBitMatrix reachability;
  CBCSlotPlannerBitMatrix reuse_targets;
  std::vector<std::uint32_t> mate_u;
  std::vector<std::uint32_t> mate_v;
  std::vector<int> dist;
  std::vector<std::uint32_t> queue;
  std::vector<std::uint32_t> topo_rank;

  void Prepare(const std::size_t n)
  {
    reachability.ResizeAndClear(n);
    reuse_targets.ResizeAndClear(n);
    mate_u.assign(n, CBC_SLOT_PLANNER_INVALID_INDEX);
    mate_v.assign(n, CBC_SLOT_PLANNER_INVALID_INDEX);
    dist.assign(n, -1);

    if (queue.size() < n)
      queue.resize(n);
    if (topo_rank.size() < n)
      topo_rank.resize(n);
  }
};

/// Output of one CBC cell-level slot planning pass.
struct CBCSlotPlan
{
  /// Optimal number of distinct slots: n − |M|, where |M| is the max matching on the reuse graph
  /// (König / Dilworth). Chains of the max matching are the minimum path cover that realizes it.
  std::size_t num_dynamic_slots = 0;
  /// Number of slots produced by the chain-walk static assignment. Always equal to
  /// num_dynamic_slots by construction.
  std::size_t num_static_slots = 0;
};

/**
 * Hopcroft-Karp maximum matching on the dense task-reuse bipartite graph; a single pass yields
 * both the optimal dynamic slot count and a static per-task slot assignment that realizes it.
 *
 * **Reuse relation.** Let tasks be indexed by topological rank. Define
 *   u ~> v  iff  v ∉ succ(u)  and  v ∈ Desc(s) for every s ∈ succ(u),
 * i.e. v (not itself a local successor of u) is reachable from every local successor of u. This
 * is exactly the condition that u's slot is free by the time v writes it, because every
 * consumer of u's slot has already read before any path could reach v.
 *
 * **Dynamic count via König / Dilworth.** On the bipartite graph with left and right copies of
 * the tasks and edges u ~> v, a maximum matching M has size n − χ, where χ is the minimum
 * path-cover of the reuse DAG. The optimal number of slots is exactly χ = n − |M|.
 *
 * **Static assignment by chain walk.** Sinks (tasks with no local successors) have empty
 * reuse_targets rows, so mate_u[sink] is always unset — sinks can appear only as matching
 * targets, i.e. at chain tails. Each unmated target mate_v[i] == INVALID is the head of a
 * matching chain; walking mate_u from that head assigns one shared slot id to every task on
 * the chain. Chains are vertex-disjoint and cover all tasks, so the resulting slot count is
 * exactly n − |M| = the dynamic optimum. The same matching run drives both outputs.
 */
class CBCDenseHopcroftKarp
{
public:
  CBCDenseHopcroftKarp(const std::uint32_t num_tasks,
                       const std::vector<Task>& task_list,
                       const std::vector<std::uint32_t>& topo_order,
                       std::vector<std::uint32_t>& task_slot_ids,
                       CBCSlotPlannerWorkspace& workspace)
    : num_tasks_(num_tasks),
      task_list_(task_list),
      topo_order_(topo_order),
      task_slot_ids_(task_slot_ids),
      ws_(workspace)
  {
    ws_.Prepare(num_tasks_);
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
      ws_.topo_rank[topo_order_[i]] = i;
  }

  CBCSlotPlan Solve()
  {
    BuildReachabilityAndReuseTargets();
    const auto matching_size = ComputeMaximumMatching();
    ExtractSlotAssignments();
    const auto slot_count = static_cast<std::size_t>(num_tasks_) - matching_size;
    return CBCSlotPlan{slot_count, slot_count};
  }

private:
  /**
   * Fills `reachability.Row(i)` with Desc(topo_order[i]) (including i itself) and
   * `reuse_targets.Row(i)` with the rank-set {v : u ~> v} for u = topo_order[i].
   *
   * Reachability is built in reverse topological order so each row is the OR of its successors'
   * rows plus the self bit. `reuse_targets[i]` is then the AND of `reachability[succ_rank]` over
   * all local successors, minus the successors themselves. Only bits at rank ≥ max_succ_rank
   * can be set in the intersection (all successors have ranks ≥ their own rank), so we start
   * the CopyRow / AndRows at `max_succ_rank` to skip known-zero words.
   */
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

      ws_.reuse_targets.CopyRow(
        i, ws_.reachability, ws_.topo_rank[successors.front()], max_succ_rank);
      for (std::size_t j = 1; j < successors.size(); ++j)
        ws_.reuse_targets.AndRows(i, ws_.reachability, ws_.topo_rank[successors[j]], max_succ_rank);
      for (const auto succ : successors)
        ws_.reuse_targets.ClearBit(i, ws_.topo_rank[succ]);
    }
  }

  std::size_t ComputeMaximumMatching()
  {
    std::fill_n(ws_.mate_u.begin(), num_tasks_, CBC_SLOT_PLANNER_INVALID_INDEX);
    std::fill_n(ws_.mate_v.begin(), num_tasks_, CBC_SLOT_PLANNER_INVALID_INDEX);
    // dist is cleared at the top of each BFS pass (required: DFS leaves stale values behind).

    auto matching_size = GreedyInit();
    while (BFS())
      for (std::uint32_t i = 0; i < num_tasks_; ++i)
        if (ws_.mate_u[i] == CBC_SLOT_PLANNER_INVALID_INDEX and DFS(i))
          ++matching_size;
    return matching_size;
  }

  std::size_t GreedyInit()
  {
    std::size_t count = 0;
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_u[i] != CBC_SLOT_PLANNER_INVALID_INDEX)
        continue;

      for (auto v = ws_.reuse_targets.FindFirstSet(i, i + 1); v < num_tasks_;
           v = ws_.reuse_targets.FindNextSet(i, v))
        if (ws_.mate_v[v] == CBC_SLOT_PLANNER_INVALID_INDEX)
        {
          ws_.mate_u[i] = static_cast<std::uint32_t>(v);
          ws_.mate_v[v] = i;
          ++count;
          break;
        }
    }
    return count;
  }

  bool BFS()
  {
    std::fill_n(ws_.dist.begin(), num_tasks_, -1);
    std::size_t head = 0;
    std::size_t tail = 0;

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
      if (ws_.mate_u[i] == CBC_SLOT_PLANNER_INVALID_INDEX)
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

      for (auto v = ws_.reuse_targets.FindFirstSet(u, u + 1); v < num_tasks_;
           v = ws_.reuse_targets.FindNextSet(u, v))
      {
        const auto mate_of_v = ws_.mate_v[v];
        if (mate_of_v == CBC_SLOT_PLANNER_INVALID_INDEX)
        {
          if (dist_null_ == std::numeric_limits<int>::max())
            dist_null_ = ws_.dist[u] + 1;
        }
        else if (ws_.dist[mate_of_v] == -1)
        {
          ws_.dist[mate_of_v] = ws_.dist[u] + 1;
          ws_.queue[tail++] = mate_of_v;
        }
      }
    }
    return dist_null_ != std::numeric_limits<int>::max();
  }

  bool DFS(const std::uint32_t u)
  {
    for (auto v = ws_.reuse_targets.FindFirstSet(u, u + 1); v < num_tasks_;
         v = ws_.reuse_targets.FindNextSet(u, v))
    {
      const auto mate_of_v = ws_.mate_v[v];
      if (mate_of_v == CBC_SLOT_PLANNER_INVALID_INDEX)
      {
        if (dist_null_ == ws_.dist[u] + 1)
        {
          ws_.mate_v[v] = u;
          ws_.mate_u[u] = static_cast<std::uint32_t>(v);
          ws_.dist[u] = -1;
          return true;
        }
      }
      else if (ws_.dist[mate_of_v] == ws_.dist[u] + 1 and DFS(mate_of_v))
      {
        ws_.mate_v[v] = u;
        ws_.mate_u[u] = static_cast<std::uint32_t>(v);
        ws_.dist[u] = -1;
        return true;
      }
    }
    ws_.dist[u] = -1;
    return false;
  }

  /**
   * Walks the chains of the full max matching and assigns one slot id per chain.
   *
   * Each chain head is an unmated target (`mate_v[i] == INVALID`). Walking `mate_u` from the
   * head visits every task on that chain exactly once and terminates at the chain tail (either
   * a non-sink whose `mate_u` is unset, or a sink — sinks have empty reuse-target rows so they
   * are never chain sources). Chains are vertex-disjoint and cover all tasks, so the final slot
   * count equals the chain count, which equals (number of unmated targets) = n − |M|, matching
   * the dynamic optimum exactly.
   */
  void ExtractSlotAssignments()
  {
    task_slot_ids_.assign(num_tasks_, CBC_SLOT_PLANNER_INVALID_INDEX);
    std::uint32_t next_slot_id = 0;
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_v[i] != CBC_SLOT_PLANNER_INVALID_INDEX)
        continue;

      auto current = i;
      while (current != CBC_SLOT_PLANNER_INVALID_INDEX)
      {
        task_slot_ids_[topo_order_[current]] = next_slot_id;
        current = ws_.mate_u[current];
      }
      ++next_slot_id;
    }
  }

  std::uint32_t num_tasks_;
  const std::vector<Task>& task_list_;
  const std::vector<std::uint32_t>& topo_order_;
  std::vector<std::uint32_t>& task_slot_ids_;
  CBCSlotPlannerWorkspace& ws_;
  int dist_null_ = 0;
};

} // namespace detail
} // namespace opensn

// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace opensn::detail
{

/// Sentinel for unassigned matching or slot indices.
constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

/**
 * Dense bit-matrix with cache-line-aligned rows for vectorized set operations.
 *
 * Stores an \f$n \times n\f$ binary matrix where each row is a bit-vector
 * padded to a multiple of 512 bits (8 \c uint64_t words) to guarantee
 * alignment for compiler auto-vectorization of row-wise OR, AND, and scan
 * operations. Used by DenseHopcroftKarp to represent the transitive closure
 * and reuse-target matrices of the task DAG.
 */
class BitMatrix
{
public:
  BitMatrix() = default;

  /**
   * Resize to \p n rows and columns, zeroing all bits.
   *
   * Reuses the existing allocation when it is large enough, zeroing only
   * the required prefix to avoid redundant value-initialization on growth.
   *
   * \param n number of rows and columns
   */
  void ResizeAndClear(std::size_t n)
  {
    n_ = n;
    // Pad to multiple of 8 words (512 bits) to guarantee perfect
    // boundary alignment for compiler auto-vectorization.
    words_per_row_ = (((n + 63) / 64) + 7) & ~std::size_t{7};
    const std::size_t required_words = n * words_per_row_;

    // Elide double-initialization overhead during growth: resize() value-
    // initializes new elements, so only the pre-existing prefix needs zeroing.
    if (data_.size() < required_words)
      data_.resize(required_words, 0ULL);
    else
      std::fill_n(data_.begin(), required_words, 0ULL);
  }

  /// Number of 64-bit words per row (including padding).
  std::size_t WordsPerRow() const { return words_per_row_; }

  /// Mutable pointer to the start of row \p i.
  std::uint64_t* Row(std::size_t i) { return data_.data() + i * words_per_row_; }

  /// Const pointer to the start of row \p i.
  const std::uint64_t* Row(std::size_t i) const { return data_.data() + i * words_per_row_; }

  /// Set bit (i, j) to 1.
  void SetBit(std::size_t i, std::size_t j) { Row(i)[j / 64] |= (1ULL << (j % 64)); }

  /// Clear bit (i, j) to 0.
  void ClearBit(std::size_t i, std::size_t j) { Row(i)[j / 64] &= ~(1ULL << (j % 64)); }

  /// Test whether bit (i, j) is set.
  bool TestBit(std::size_t i, std::size_t j) const
  {
    return (Row(i)[j / 64] & (1ULL << (j % 64))) != 0ULL;
  }

  /**
   * Copy a row from another BitMatrix into this one.
   *
   * Uses \c std::memcpy for maximum throughput. The \c __restrict qualifiers
   * enable the compiler to elide alias checks.
   *
   * \param dst destination row index in this matrix
   * \param src_mat source matrix (may be \c *this)
   * \param src_row source row index
   * \param start_pos first bit position to copy from (default 0)
   */
  void
  CopyRow(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
  {
    const std::size_t start_word = start_pos / 64;
    std::uint64_t* __restrict d = Row(dst) + start_word;
    const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_copy = words_per_row_ - start_word;

    std::memcpy(d, s, words_to_copy * sizeof(std::uint64_t));
  }

  /**
   * Bitwise-OR a source row into a destination row of this matrix.
   *
   * \param dst destination row index in this matrix
   * \param src_mat source matrix
   * \param src_row source row index
   * \param start_pos first bit position to process (default 0)
   */
  void
  OrRows(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
  {
    const std::size_t start_word = start_pos / 64;
    std::uint64_t* __restrict d = Row(dst) + start_word;
    const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_process = words_per_row_ - start_word;

    for (std::size_t w = 0; w < words_to_process; ++w)
      d[w] |= s[w];
  }

  /**
   * Bitwise-AND a source row into a destination row of this matrix.
   *
   * \param dst destination row index in this matrix
   * \param src_mat source matrix
   * \param src_row source row index
   * \param start_pos first bit position to process (default 0)
   */
  void AndRows(std::size_t dst,
               const BitMatrix& src_mat,
               std::size_t src_row,
               std::size_t start_pos = 0)
  {
    const std::size_t start_word = start_pos / 64;
    std::uint64_t* __restrict d = Row(dst) + start_word;
    const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_process = words_per_row_ - start_word;

    for (std::size_t w = 0; w < words_to_process; ++w)
      d[w] &= s[w];
  }

  /**
   * Find the first set bit in \p row at or after \p start_pos.
   *
   * Uses hardware TZCNT via \c std::countr_zero for O(1) per-word scanning.
   * Padding bits are guaranteed zero, so no bounds masking is needed.
   *
   * \return bit index of the first set bit, or \c n_ if none found
   */
  std::size_t FindFirstSet(std::size_t row, std::size_t start_pos = 0) const
  {
    const std::uint64_t* __restrict r = Row(row);
    std::size_t w = start_pos / 64;

    if (w >= words_per_row_)
      return n_;

    std::uint64_t masked = r[w] & (~0ULL << (start_pos % 64));

    if (masked)
      return w * 64 + static_cast<std::size_t>(std::countr_zero(masked));

    for (++w; w < words_per_row_; ++w)
    {
      if (r[w])
        return w * 64 + static_cast<std::size_t>(std::countr_zero(r[w]));
    }
    return n_;
  }

  /**
   * Find the next set bit in \p row strictly after \p pos.
   *
   * \return bit index of the next set bit, or \c n_ if none found
   */
  std::size_t FindNextSet(std::size_t row, std::size_t pos) const
  {
    return FindFirstSet(row, pos + 1);
  }

private:
  /// Matrix dimension.
  std::size_t n_ = 0;
  /// Number of 64-bit words per row, padded for alignment.
  std::size_t words_per_row_ = 0;
  /// Flat storage for all rows.
  std::vector<std::uint64_t> data_;
};

/**
 * Thread-local scratch buffers for the dense Hopcroft-Karp solver.
 *
 * All buffers are sized to the number of tasks in the current SPDS instance
 * and reused across successive calls to \c ComputeMaxNumLocalPsiSlots on
 * the same thread, avoiding redundant heap allocations.
 */
struct ThreadLocalWorkspace
{
  /// Transitive closure of the task DAG (\f$R[i][j] = 1\f$ iff task \f$i\f$ reaches \f$j\f$).
  BitMatrix reachability;
  /// Eligible reuse targets per task (intersection of all successors' reachability sets).
  BitMatrix reuse_targets;
  /// U-side matching: \c mate_u[u] = matched V-vertex, or INVALID_INDEX.
  std::vector<std::uint32_t> mate_u;
  /// V-side matching: \c mate_v[v] = matched U-vertex, or INVALID_INDEX.
  std::vector<std::uint32_t> mate_v;
  /// BFS distance labels for the Hopcroft-Karp layered graph.
  std::vector<int> dist;
  /// BFS queue for the Hopcroft-Karp layered graph.
  std::vector<std::uint32_t> queue;
  /// Inverse topological-order map: \c topo_rank[task_id] = rank.
  std::vector<std::uint32_t> topo_rank;
  /// Per-slot last-assigned topological rank, used by the verifier.
  std::vector<std::uint32_t> last_rank_for_slot;

  /**
   * Prepare all buffers for \p n tasks.
   *
   * Zeroes the matching and distance arrays. The queue, topo_rank, and
   * last_rank_for_slot buffers are only grown (never shrunk or cleared)
   * because they are overwritten before use.
   */
  void Prepare(std::size_t n)
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
    if (last_rank_for_slot.size() < n)
      last_rank_for_slot.resize(n);
  }
};

/**
 * Optimal slot-count solver via dense Hopcroft-Karp maximum bipartite matching.
 *
 * Given a task DAG with \f$n\f$ tasks in topological order, this class computes
 * the minimum number of angular-flux storage slots \f$s^*\f$ such that every
 * task can be assigned a slot without conflict, and produces a concrete static
 * slot assignment \f$\sigma : \text{task} \to \{0, \ldots, s^*{-}1\}\f$.
 *
 * ## Algorithm
 *
 * The problem is reduced to maximum bipartite matching on a "reuse graph":
 *
 * 1. **Transitive closure.** Compute the reachability matrix
 *    \f$R \in \{0,1\}^{n \times n}\f$ of the task DAG bottom-up in reverse
 *    topological order. Each row is a bit-vector OR of successor rows.
 *
 * 2. **Reuse-target construction.** For each task \f$u\f$ with successors
 *    \f$S(u)\f$, the reuse targets are
 *    \f$T(u) = \bigl(\bigcap_{v \in S(u)} R(v)\bigr) \setminus S(u)\f$.
 *    A task \f$v \in T(u)\f$ can safely reuse \f$u\f$'s slot because \f$v\f$
 *    is reachable from every successor of \f$u\f$, guaranteeing that \f$u\f$'s
 *    data has been fully consumed before \f$v\f$ overwrites the slot.
 *
 * 3. **Hopcroft-Karp matching.** Solve the maximum matching on the bipartite
 *    graph \f$(U, V, E)\f$ where \f$U = V = \{0, \ldots, n{-}1\}\f$ (tasks in
 *    topological rank) and \f$(u, v) \in E \iff v \in T(u)\f$. Each matched
 *    edge represents one slot reuse, so the optimal slot count is
 *    \f$s^* = n - |M^*|\f$.
 *
 * 4. **Slot extraction.** Walk the matching chains to assign slot IDs:
 *    each chain head (unmatched on V-side) starts a new slot, and subsequent
 *    tasks in the chain inherit that slot.
 *
 * 5. **Verification.** A post-hoc verifier confirms that consecutive slot
 *    occupants satisfy the reuse relation, falling back to the identity
 *    assignment (one slot per cell) if any violation is detected.
 *
 * ## Complexity
 *
 * - Time: \f$O(n^2 \sqrt{n} / 64)\f$ amortized (bit-parallel matching rounds).
 * - Space: \f$O(n^2 / 8)\f$ for the two \f$n \times n\f$ bit-matrices.
 *
 * ## Memory impact
 *
 * The computed \f$s^*\f$ directly determines the size of the local angular-flux
 * buffer in CBC_FLUDS and CBCD_FLUDS. For typical unstructured meshes,
 * \f$s^* \ll n\f$, yielding a proportional reduction in memory footprint and
 * improved cache utilization during the sweep.
 */
class DenseHopcroftKarp
{
public:
  /**
   * Construct the solver for a given task graph.
   *
   * \param num_tasks number of local tasks (cells)
   * \param task_list immutable task graph with successor lists
   * \param topo_order topological ordering of task indices
   * \param task_slot_ids output slot assignment (resized and populated by Solve)
   * \param ws thread-local workspace (resized by this constructor)
   */
  DenseHopcroftKarp(std::uint32_t num_tasks,
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
      ws_.topo_rank[topo_order_[i]] = i;
  }

  /**
   * Compute the optimal slot assignment.
   *
   * \return the minimum number of slots \f$s^*\f$
   */
  std::size_t Solve()
  {
    BuildTransitiveClosure();
    BuildReuseTargets();

    std::size_t matching_size = GreedyInit();
    while (BFS())
    {
      for (std::uint32_t i = 0; i < num_tasks_; ++i)
      {
        if (ws_.mate_u[i] == INVALID_INDEX && DFS(i))
          ++matching_size;
      }
    }

    ExtractSlotAssignment();

    const std::size_t optimal_slot_count = static_cast<std::size_t>(num_tasks_) - matching_size;

    if (not VerifySlotAssignment(optimal_slot_count))
    {
      std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), std::uint32_t{0});
      return static_cast<std::size_t>(num_tasks_);
    }

    return optimal_slot_count;
  }

private:
  /**
   * Build the transitive closure bottom-up in reverse topological order.
   *
   * Each task's reachability row is the union of its successors' rows plus
   * itself: \f$R(i) = \{i\} \cup \bigcup_{j \in \text{succ}(i)} R(j)\f$.
   */
  void BuildTransitiveClosure()
  {
    for (std::uint32_t i = num_tasks_; i-- > 0;)
    {
      const std::uint32_t u = topo_order_[i];
      const auto& successors = task_list_[u].successors;

      if (successors.empty())
      {
        ws_.reachability.SetBit(i, i);
      }
      else
      {
        ws_.reachability.CopyRow(i, ws_.reachability, ws_.topo_rank[successors[0]], i);
        ws_.reachability.SetBit(i, i);

        for (std::size_t j = 1; j < successors.size(); ++j)
          ws_.reachability.OrRows(i, ws_.reachability, ws_.topo_rank[successors[j]], i);
      }
    }
  }

  /**
   * Build the reuse-targets bit-matrix.
   *
   * For each task \f$u\f$, reuse targets are the intersection of all
   * successors' reachability sets, minus the successors themselves.
   * Row operations start at the maximum successor rank to skip
   * unreachable prefixes.
   */
  void BuildReuseTargets()
  {
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      const std::uint32_t u = topo_order_[i];
      const auto& successors = task_list_[u].successors;
      if (successors.empty())
        continue;

      std::uint32_t max_succ_rank = ws_.topo_rank[successors[0]];
      for (std::size_t j = 1; j < successors.size(); ++j)
        max_succ_rank = std::max(max_succ_rank, ws_.topo_rank[successors[j]]);

      ws_.reuse_targets.CopyRow(
        i, ws_.reachability, ws_.topo_rank[successors[0]], max_succ_rank);

      for (std::size_t j = 1; j < successors.size(); ++j)
        ws_.reuse_targets.AndRows(
          i, ws_.reachability, ws_.topo_rank[successors[j]], max_succ_rank);

      for (const auto succ : successors)
        ws_.reuse_targets.ClearBit(i, ws_.topo_rank[succ]);
    }
  }

  /**
   * Walk matching chains to produce a static slot-id per task.
   *
   * Each chain head (unmatched on V-side) starts a new slot; subsequent
   * matched tasks in the chain inherit that slot.
   */
  void ExtractSlotAssignment()
  {
    task_slot_ids_.assign(num_tasks_, INVALID_INDEX);
    std::uint32_t next_slot_id = 0;

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_v[i] == INVALID_INDEX)
      {
        std::uint32_t current = i;
        while (current != INVALID_INDEX)
        {
          task_slot_ids_[topo_order_[current]] = next_slot_id;
          current = ws_.mate_u[current];
        }
        ++next_slot_id;
      }
    }
  }

  /**
   * Verify the slot assignment is conflict-free.
   *
   * Walk tasks in topological order and confirm that consecutive occupants
   * of each slot satisfy the reuse relation.
   *
   * \return true if the assignment is valid
   */
  bool VerifySlotAssignment(const std::size_t slot_count) const
  {
    for (std::uint32_t task = 0; task < num_tasks_; ++task)
    {
      if (task_slot_ids_[task] >= slot_count)
        return false;
    }

    std::fill_n(ws_.last_rank_for_slot.begin(), slot_count, INVALID_INDEX);
    for (std::uint32_t rank = 0; rank < num_tasks_; ++rank)
    {
      const auto sid = task_slot_ids_[topo_order_[rank]];
      const auto prev_rank = ws_.last_rank_for_slot[sid];
      if (prev_rank != INVALID_INDEX and not ReuseRelationHolds(prev_rank, rank))
        return false;
      ws_.last_rank_for_slot[sid] = rank;
    }
    return true;
  }

  /**
   * Check whether task at \p u_rank can safely cede its slot to the task at \p v_rank.
   *
   * The reuse relation holds iff \f$v\f$ is reachable from every successor
   * of \f$u\f$ (excluding direct successor identity).
   */
  bool ReuseRelationHolds(const std::uint32_t u_rank, const std::uint32_t v_rank) const
  {
    const auto u = topo_order_[u_rank];
    const auto& u_successors = task_list_[u].successors;

    if (u_successors.empty())
      return false;

    for (const auto succ : u_successors)
    {
      const auto succ_rank = ws_.topo_rank[succ];
      if (succ_rank == v_rank or not ws_.reachability.TestBit(succ_rank, v_rank))
        return false;
    }
    return true;
  }

  /**
   * Greedy initialization of the matching.
   *
   * Scans each free U-vertex and matches it with the first available
   * V-vertex in its reuse-target set. Provides a warm start for the
   * Hopcroft-Karp augmentation phase.
   *
   * \return number of edges in the greedy matching
   */
  std::size_t GreedyInit()
  {
    std::size_t count = 0;
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_u[i] != INVALID_INDEX)
        continue;

      for (std::size_t v = ws_.reuse_targets.FindFirstSet(i, i + 1); v < num_tasks_;
           v = ws_.reuse_targets.FindNextSet(i, v))
      {
        if (ws_.mate_v[v] == INVALID_INDEX)
        {
          ws_.mate_u[i] = static_cast<std::uint32_t>(v);
          ws_.mate_v[v] = i;
          ++count;
          break;
        }
      }
    }
    return count;
  }

  /**
   * Hopcroft-Karp BFS phase.
   *
   * Build a layered graph of shortest augmenting paths from free U-vertices.
   *
   * \return true if at least one augmenting path exists
   */
  bool BFS()
  {
    std::fill_n(ws_.dist.begin(), num_tasks_, -1);
    std::size_t head = 0, tail = 0;

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_u[i] == INVALID_INDEX)
      {
        ws_.dist[i] = 0;
        ws_.queue[tail++] = i;
      }
    }

    dist_null_ = std::numeric_limits<int>::max();

    while (head < tail)
    {
      const std::uint32_t u = ws_.queue[head++];

      if (ws_.dist[u] < dist_null_)
      {
        for (std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1); v < num_tasks_;
             v = ws_.reuse_targets.FindNextSet(u, v))
        {
          const std::uint32_t mate_of_v = ws_.mate_v[v];
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
        }
      }
    }
    return dist_null_ != std::numeric_limits<int>::max();
  }

  /**
   * Hopcroft-Karp DFS phase.
   *
   * Attempt to find an augmenting path from U-vertex \p u along the
   * layered graph built by BFS.
   *
   * \return true if an augmenting path was found and the matching was updated
   */
  bool DFS(std::uint32_t u)
  {
    for (std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1); v < num_tasks_;
         v = ws_.reuse_targets.FindNextSet(u, v))
    {
      const std::uint32_t mate_of_v = ws_.mate_v[v];
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
      else if (ws_.dist[mate_of_v] == ws_.dist[u] + 1)
      {
        if (DFS(mate_of_v))
        {
          ws_.mate_v[v] = u;
          ws_.mate_u[u] = static_cast<std::uint32_t>(v);
          ws_.dist[u] = -1;
          return true;
        }
      }
    }
    ws_.dist[u] = -1;
    return false;
  }

  /// Number of tasks (local cells) in the current SPDS instance.
  std::uint32_t num_tasks_;
  /// Immutable task graph with per-task successor lists.
  const std::vector<Task>& task_list_;
  /// Topological ordering: \c topo_order_[rank] = task_id.
  const std::vector<std::uint32_t>& topo_order_;
  /// Output slot assignment: \c task_slot_ids_[task_id] = slot_id.
  std::vector<std::uint32_t>& task_slot_ids_;
  /// Thread-local scratch workspace.
  ThreadLocalWorkspace& ws_;
  /// BFS null distance for the current augmentation round.
  int dist_null_ = 0;
};

} // namespace opensn::detail

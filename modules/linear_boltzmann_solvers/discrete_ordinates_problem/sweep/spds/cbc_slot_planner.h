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

struct SlotSolveResult
{
  std::size_t slot_count = 0;
  bool verifier_rejected = false;
};

/**
 * Dense bit-matrix with cache-line-aligned rows for vectorized set operations.
 *
 * Stores an \f$n \times n\f$ binary matrix where each row is a bit-vector
 * padded to a multiple of 512 bits (8 \c uint64_t words) to guarantee
 * alignment for compiler auto-vectorization of row-wise OR, AND, and scan
 * operations. Used to represent the task-DAG reachability relation needed by
 * the local-face slot planner.
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
 * Thread-local scratch buffers for CBC reachability and local-face matching.
 *
 * All buffers are sized to the number of tasks in the current SPDS instance
 * and reused across successive calls to \c ComputeMaxNumLocalPsiSlots on
 * the same thread, avoiding redundant heap allocations.
 */
struct ThreadLocalWorkspace
{
  /// Transitive closure of the task DAG (\f$R[i][j] = 1\f$ iff task \f$i\f$ reaches \f$j\f$).
  BitMatrix reachability;
  /// Inverse topological-order map: \c topo_rank[task_id] = rank.
  std::vector<std::uint32_t> topo_rank;
  /// U-side matching for the local-face planner.
  std::vector<std::uint32_t> face_mate_u;
  /// V-side matching for the local-face planner.
  std::vector<std::uint32_t> face_mate_v;
  /// BFS distance labels for the local-face planner.
  std::vector<int> face_dist;
  /// BFS queue for the local-face planner.
  std::vector<std::uint32_t> face_queue;
  /// Per-slot last-assigned face rank, used by the local-face verifier.
  std::vector<std::uint32_t> face_last_rank_for_slot;

  /**
   * Prepare reachability buffers for \p n tasks.
   *
   * The topological-rank buffer is only grown because it is overwritten before use.
   */
  void PrepareReachability(std::size_t n)
  {
    reachability.ResizeAndClear(n);
    if (topo_rank.size() < n)
      topo_rank.resize(n);
  }

  /// Prepare buffers for the local-face slot planner.
  void PrepareFaces(std::size_t n)
  {
    face_mate_u.assign(n, INVALID_INDEX);
    face_mate_v.assign(n, INVALID_INDEX);
    face_dist.assign(n, -1);

    if (face_queue.size() < n)
      face_queue.resize(n);
    if (face_last_rank_for_slot.size() < n)
      face_last_rank_for_slot.resize(n);
  }
};

inline void
BuildCBCReachability(const std::uint32_t num_tasks,
                     const std::vector<Task>& task_list,
                     const std::vector<std::uint32_t>& topo_order,
                     ThreadLocalWorkspace& ws)
{
  ws.PrepareReachability(num_tasks);

  for (std::uint32_t i = 0; i < num_tasks; ++i)
    ws.topo_rank[topo_order[i]] = i;

  for (std::uint32_t i = num_tasks; i-- > 0;)
  {
    const std::uint32_t u = topo_order[i];
    const auto& successors = task_list[u].successors;

    if (successors.empty())
    {
      ws.reachability.SetBit(i, i);
    }
    else
    {
      ws.reachability.CopyRow(i, ws.reachability, ws.topo_rank[successors[0]], i);
      ws.reachability.SetBit(i, i);

      for (std::size_t j = 1; j < successors.size(); ++j)
        ws.reachability.OrRows(i, ws.reachability, ws.topo_rank[successors[j]], i);
    }
  }
}

/**
 * Optimal local-face slot-count solver via dense Hopcroft-Karp matching with
 * implicit adjacency.
 *
 * The left and right vertices are the local directed faces of the CBC task DAG.
 * A left-face \f$f\f$ may match a right-face \f$g\f$ if the producer cell of
 * \f$g\f$ is reachable from, or equal to, the consumer cell of \f$f\f$. This
 * is the exact schedule-safe reuse relation for local face storage when all
 * incoming faces are read before any outgoing faces are written during a cell
 * sweep.
 *
 * Adjacency is generated on demand from the cell reachability matrix together
 * with per-producer contiguous face ranges. This avoids constructing a dense
 * face-by-face reuse matrix, which would be prohibitively large.
 */
class DenseLocalFaceHopcroftKarp
{
public:
  DenseLocalFaceHopcroftKarp(const std::vector<std::uint32_t>& face_producer_ranks,
                             const std::vector<std::uint32_t>& face_consumer_ranks,
                             const std::vector<std::uint32_t>& producer_cell_face_offsets,
                             std::vector<std::uint32_t>& face_slot_ids,
                             ThreadLocalWorkspace& ws)
    : num_faces_(static_cast<std::uint32_t>(face_producer_ranks.size())),
      face_producer_ranks_(face_producer_ranks),
      face_consumer_ranks_(face_consumer_ranks),
      producer_cell_face_offsets_(producer_cell_face_offsets),
      face_slot_ids_(face_slot_ids),
      ws_(ws)
  {
    ws_.PrepareFaces(num_faces_);
  }

  SlotSolveResult Solve()
  {
    if (num_faces_ == 0)
    {
      face_slot_ids_.clear();
      return {};
    }

    std::size_t matching_size = GreedyInit();
    while (BFS())
    {
      for (std::uint32_t i = 0; i < num_faces_; ++i)
      {
        if (ws_.face_mate_u[i] == INVALID_INDEX and DFS(i))
          ++matching_size;
      }
    }

    ExtractSlotAssignment();

    const std::size_t optimal_slot_count = static_cast<std::size_t>(num_faces_) - matching_size;
    if (not VerifySlotAssignment(optimal_slot_count))
    {
      std::iota(face_slot_ids_.begin(), face_slot_ids_.end(), std::uint32_t{0});
      return {static_cast<std::size_t>(num_faces_), true};
    }

    return {optimal_slot_count, false};
  }

private:
  template <class F>
  void ForEachCandidate(const std::uint32_t u_face_rank, F&& fn) const
  {
    const auto consumer_cell_rank = face_consumer_ranks_[u_face_rank];
    for (std::size_t producer_cell_rank =
           ws_.reachability.FindFirstSet(consumer_cell_rank, consumer_cell_rank);
         producer_cell_rank < producer_cell_face_offsets_.size() - 1;
         producer_cell_rank = ws_.reachability.FindNextSet(consumer_cell_rank, producer_cell_rank))
    {
      const auto face_begin = producer_cell_face_offsets_[producer_cell_rank];
      const auto face_end = producer_cell_face_offsets_[producer_cell_rank + 1];
      for (std::uint32_t v_face_rank = face_begin; v_face_rank < face_end; ++v_face_rank)
        fn(v_face_rank);
    }
  }

  bool ReuseRelationHolds(const std::uint32_t u_face_rank, const std::uint32_t v_face_rank) const
  {
    return ws_.reachability.TestBit(face_consumer_ranks_[u_face_rank],
                                    face_producer_ranks_[v_face_rank]);
  }

  void ExtractSlotAssignment()
  {
    face_slot_ids_.assign(num_faces_, INVALID_INDEX);
    std::uint32_t next_slot_id = 0;

    for (std::uint32_t i = 0; i < num_faces_; ++i)
    {
      if (ws_.face_mate_v[i] == INVALID_INDEX)
      {
        std::uint32_t current = i;
        while (current != INVALID_INDEX)
        {
          face_slot_ids_[current] = next_slot_id;
          current = ws_.face_mate_u[current];
        }
        ++next_slot_id;
      }
    }
  }

  bool VerifySlotAssignment(const std::size_t slot_count) const
  {
    for (std::uint32_t face = 0; face < num_faces_; ++face)
    {
      if (face_slot_ids_[face] >= slot_count)
        return false;
    }

    std::fill_n(ws_.face_last_rank_for_slot.begin(), slot_count, INVALID_INDEX);
    for (std::uint32_t rank = 0; rank < num_faces_; ++rank)
    {
      const auto sid = face_slot_ids_[rank];
      const auto prev_rank = ws_.face_last_rank_for_slot[sid];
      if (prev_rank != INVALID_INDEX and not ReuseRelationHolds(prev_rank, rank))
        return false;
      ws_.face_last_rank_for_slot[sid] = rank;
    }
    return true;
  }

  std::size_t GreedyInit()
  {
    std::size_t count = 0;
    for (std::uint32_t i = 0; i < num_faces_; ++i)
    {
      if (ws_.face_mate_u[i] != INVALID_INDEX)
        continue;

      bool matched = false;
      ForEachCandidate(
        i,
        [&](const std::uint32_t v_face_rank)
        {
          if (matched or ws_.face_mate_v[v_face_rank] != INVALID_INDEX)
            return;
          ws_.face_mate_u[i] = v_face_rank;
          ws_.face_mate_v[v_face_rank] = i;
          matched = true;
        });
      if (matched)
        ++count;
    }
    return count;
  }

  bool BFS()
  {
    std::fill_n(ws_.face_dist.begin(), num_faces_, -1);
    std::size_t head = 0;
    std::size_t tail = 0;

    for (std::uint32_t i = 0; i < num_faces_; ++i)
    {
      if (ws_.face_mate_u[i] == INVALID_INDEX)
      {
        ws_.face_dist[i] = 0;
        ws_.face_queue[tail++] = i;
      }
    }

    dist_null_ = std::numeric_limits<int>::max();

    while (head < tail)
    {
      const auto u_face_rank = ws_.face_queue[head++];
      if (ws_.face_dist[u_face_rank] >= dist_null_)
        continue;

      ForEachCandidate(
        u_face_rank,
        [&](const std::uint32_t v_face_rank)
        {
          const auto mate_of_v = ws_.face_mate_v[v_face_rank];
          if (mate_of_v == INVALID_INDEX)
          {
            if (dist_null_ == std::numeric_limits<int>::max())
              dist_null_ = ws_.face_dist[u_face_rank] + 1;
          }
          else if (ws_.face_dist[mate_of_v] == -1)
          {
            ws_.face_dist[mate_of_v] = ws_.face_dist[u_face_rank] + 1;
            ws_.face_queue[tail++] = mate_of_v;
          }
        });
    }

    return dist_null_ != std::numeric_limits<int>::max();
  }

  bool DFS(const std::uint32_t u_face_rank)
  {
    bool matched = false;
    ForEachCandidate(
      u_face_rank,
      [&](const std::uint32_t v_face_rank)
      {
        if (matched)
          return;

        const auto mate_of_v = ws_.face_mate_v[v_face_rank];
        if (mate_of_v == INVALID_INDEX)
        {
          if (dist_null_ == ws_.face_dist[u_face_rank] + 1)
          {
            ws_.face_mate_v[v_face_rank] = u_face_rank;
            ws_.face_mate_u[u_face_rank] = v_face_rank;
            ws_.face_dist[u_face_rank] = -1;
            matched = true;
          }
        }
        else if (ws_.face_dist[mate_of_v] == ws_.face_dist[u_face_rank] + 1 and DFS(mate_of_v))
        {
          ws_.face_mate_v[v_face_rank] = u_face_rank;
          ws_.face_mate_u[u_face_rank] = v_face_rank;
          ws_.face_dist[u_face_rank] = -1;
          matched = true;
        }
      });
    if (not matched)
      ws_.face_dist[u_face_rank] = -1;
    return matched;
  }

  std::uint32_t num_faces_ = 0;
  const std::vector<std::uint32_t>& face_producer_ranks_;
  const std::vector<std::uint32_t>& face_consumer_ranks_;
  const std::vector<std::uint32_t>& producer_cell_face_offsets_;
  std::vector<std::uint32_t>& face_slot_ids_;
  ThreadLocalWorkspace& ws_;
  int dist_null_ = 0;
};

} // namespace opensn::detail

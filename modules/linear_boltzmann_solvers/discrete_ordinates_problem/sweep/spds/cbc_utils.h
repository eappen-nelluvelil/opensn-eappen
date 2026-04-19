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
#include <numeric>
#include <vector>

namespace opensn::detail
{

/// Value indicating unmatched vertices and invalid assignments in slot planning.
constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

/// Result of a slot-planning solve.
struct SlotSolveResult
{
  /// Number of slots required by the computed assignment.
  std::size_t slot_count = 0;
  /// Flag indicating that the post-solve verifier rejected the computed assignment.
  bool verifier_rejected = false;
};

/**
 * Dense bit matrix with cache-aligned rows.
 *
 * Stores an ``n x n`` binary matrix whose rows are padded to a multiple of 512 bits.
 */
class BitMatrix
{
public:
  BitMatrix() = default;

  /**
   * Resize the matrix and clear all active bits.
   *
   * Reuse the allocated buffer when possible in the thread-local
   * slot-planning workspace.
   *
   * \param n Matrix dimension.
   */
  void ResizeAndClear(std::size_t n)
  {
    n_ = n;
    active_words_per_row_ = (n + 63) / 64;
    // Pad to a multiple of 8 words (512 bits) for alignment and vectorization.
    padded_words_per_row_ = (active_words_per_row_ + 7) & ~std::size_t{7};
    const std::size_t required_words = n_ * padded_words_per_row_;
    // Grow when needed; the unconditional fill below clears the active region.
    if (data_.size() < required_words)
      data_.resize(required_words);
    if (row_active_word_counts_.size() < n_)
      row_active_word_counts_.resize(n_);
    std::fill_n(data_.begin(), required_words, 0ULL);
    std::fill_n(row_active_word_counts_.begin(), n_, std::size_t{0});
  }

  /// Return the number of 64-bit words in each padded row.
  std::size_t WordsPerRow() const noexcept { return padded_words_per_row_; }

  /// Return the number of logical 64-bit words required by each row.
  std::size_t ActiveWordsPerRow() const noexcept { return active_words_per_row_; }

  /// Return a mutable pointer to the beginning of row `i`.
  std::uint64_t* Row(std::size_t i) noexcept { return data_.data() + i * padded_words_per_row_; }

  /// Return a const pointer to the beginning of row `i`.
  const std::uint64_t* Row(std::size_t i) const noexcept
  {
    return data_.data() + i * padded_words_per_row_;
  }

  /// Set the bit at `(i, j)`.
  void SetBit(std::size_t i, std::size_t j) noexcept
  {
    Row(i)[j / 64] |= (1ULL << (j % 64));
    row_active_word_counts_[i] = std::max(row_active_word_counts_[i], (j / 64) + 1);
  }

  /// Clear the bit at `(i, j)`.
  void ClearBit(std::size_t i, std::size_t j) noexcept { Row(i)[j / 64] &= ~(1ULL << (j % 64)); }

  /**
   * Test whether the bit at `(i, j)` is set.
   *
   * \return True when the bit is one.
   */
  bool TestBit(std::size_t i, std::size_t j) const noexcept
  {
    return (Row(i)[j / 64] & (1ULL << (j % 64))) != 0ULL;
  }

  /**
   * Copy one row from a source matrix into this matrix.
   *
   * \param dst Destination row index in this matrix.
   * \param src_mat Source matrix.
   * \param src_row Source row index in the source matrix.
   * \param start_pos First bit position to copy.
   */
  void CopyRow(std::size_t dst,
               const BitMatrix& src_mat,
               std::size_t src_row,
               std::size_t start_pos = 0) noexcept
  {
    CopyRowFromWord(dst, src_mat, src_row, start_pos / 64);
  }

  void CopyRowFromWord(std::size_t dst,
                       const BitMatrix& src_mat,
                       std::size_t src_row,
                       std::size_t start_word) noexcept
  {
    const std::size_t src_active_words = src_mat.row_active_word_counts_[src_row];
    if (start_word >= src_active_words)
    {
      row_active_word_counts_[dst] = std::max(row_active_word_counts_[dst], std::min(start_word, active_words_per_row_));
      return;
    }
    std::uint64_t* __restrict__ d = Row(dst) + start_word;
    const std::uint64_t* __restrict__ s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_copy = src_active_words - start_word;
    std::memcpy(d, s, words_to_copy * sizeof(std::uint64_t));
    row_active_word_counts_[dst] = src_active_words;
  }

  /**
   * Bitwise-OR a source row into a destination row.
   *
   * \param dst Destination row index in this matrix.
   * \param src_mat Source matrix.
   * \param src_row Source row index in the source matrix.
   * \param start_pos First bit position to process.
   */
  void OrRows(std::size_t dst,
              const BitMatrix& src_mat,
              std::size_t src_row,
              std::size_t start_pos = 0) noexcept
  {
    OrRowsFromWord(dst, src_mat, src_row, start_pos / 64);
  }

  void OrRowsFromWord(std::size_t dst,
                      const BitMatrix& src_mat,
                      std::size_t src_row,
                      std::size_t start_word) noexcept
  {
    const std::size_t src_active_words = src_mat.row_active_word_counts_[src_row];
    if (start_word >= src_active_words)
      return;
    std::uint64_t* __restrict__ d = Row(dst) + start_word;
    const std::uint64_t* __restrict__ s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_process = src_active_words - start_word;
    for (std::size_t w = 0; w < words_to_process; ++w)
      d[w] |= s[w];
    row_active_word_counts_[dst] = std::max(row_active_word_counts_[dst], src_active_words);
  }

  /**
   * Bitwise-AND a source row into a destination row.
   *
   * \param dst Destination row index in this matrix.
   * \param src_mat Source matrix.
   * \param src_row Source row index in the source matrix.
   * \param start_pos First bit position to process.
   */
  void AndRows(std::size_t dst,
               const BitMatrix& src_mat,
               std::size_t src_row,
               std::size_t start_pos = 0) noexcept
  {
    const std::size_t start_word = start_pos / 64;
    const std::size_t src_active_words = src_mat.row_active_word_counts_[src_row];
    const std::size_t dst_active_words = row_active_word_counts_[dst];
    if (start_word >= dst_active_words)
      return;
    std::uint64_t* __restrict__ d = Row(dst) + start_word;
    const std::uint64_t* __restrict__ s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_process =
      (start_word < src_active_words) ? (std::min(dst_active_words, src_active_words) - start_word)
                                      : 0;
    for (std::size_t w = 0; w < words_to_process; ++w)
      d[w] &= s[w];
    for (std::size_t w = start_word + words_to_process; w < dst_active_words; ++w)
      Row(dst)[w] = 0ULL;
    row_active_word_counts_[dst] = std::min(dst_active_words, src_active_words);
  }

  /**
   * Find the first set bit in a row at or after `start_pos`.
   *
   * Uses `std::countr_zero` for constant-time per-word scanning.
   *
   * \return Index of the first set bit, or `n_` if none is found.
   */
  std::size_t FindFirstSet(std::size_t row, std::size_t start_pos = 0) const noexcept
  {
    const std::uint64_t* __restrict__ r = Row(row);
    std::size_t w = start_pos / 64;
    const std::size_t active_words = row_active_word_counts_[row];
    if (w >= active_words)
      return n_;
    std::uint64_t masked = r[w] & (~0ULL << (start_pos % 64));
    if (masked)
      return w * 64 + static_cast<std::size_t>(std::countr_zero(masked));
    for (++w; w < active_words; ++w)
    {
      if (r[w])
        return w * 64 + static_cast<std::size_t>(std::countr_zero(r[w]));
    }
    return n_;
  }

  /**
   * Find the next set bit in row strictly after position `pos`.
   *
   * \return Index of the next set bit, or `n_` if none is found.
   */
  std::size_t FindNextSet(std::size_t row, std::size_t pos) const noexcept
  {
    return FindFirstSet(row, pos + 1);
  }

private:
  /// Matrix dimension.
  std::size_t n_ = 0;
  /// Number of logical 64-bit words required to cover one row.
  std::size_t active_words_per_row_ = 0;
  /// Number of 64-bit words per row after alignment padding.
  std::size_t padded_words_per_row_ = 0;
  /// Per-row exclusive high-water mark in words.
  std::vector<std::size_t> row_active_word_counts_;
  /// Flat storage for the bit matrix, row-major with padding.
  std::vector<std::uint64_t> data_;
};

/**
 * Thread-local scratch buffers for CBC reachability and local-face slot matching.
 *
 * Reuses all temporaries across successive slot-planning calls on the same worker thread
 * to avoid repeated allocations during SPDS setup.
 */
struct ThreadLocalWorkspace
{
  /// Transitive closure of the task DAG (R[i][j] = 1 <==> task i reaches task j).
  BitMatrix reachability;
  /// U-side matching for the local cell-face slots.
  std::vector<std::uint32_t> face_mate_u;
  /// V-side matching for the local cell-face slots.
  std::vector<std::uint32_t> face_mate_v;
  /// BFS distance labels for the local cell-face slot matching.
  std::vector<int> face_dist;
  /// BFS queue for the local cell-face slot matching.
  std::vector<std::uint32_t> face_queue;
  /// Per-slot last-assigned face rank used by the cell-face slot verifier.
  std::vector<std::uint32_t> face_last_rank_for_slot;

  /**
   * Prepare reachability buffers for n tasks.
   *
   * The topological-rank buffer only grows because all entries are overwritten before use.
   */
  void PrepareReachability(std::size_t n)
  {
    reachability.ResizeAndClear(n);
  }

  /// Prepare the local-face matching buffers for `n` directed faces.
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
BuildReachability(const std::uint32_t num_tasks,
                  const std::vector<std::uint32_t>& successor_rank_offsets,
                  const std::vector<std::uint32_t>& successor_ranks,
                  ThreadLocalWorkspace& ws)
{
  // Process tasks in forward topological order so each row can OR in successor rows
  // that have already been finalized in the transitive closure.
  ws.PrepareReachability(num_tasks);
  for (std::uint32_t i = 0; i < num_tasks; ++i)
  {
    const auto successor_begin = successor_ranks.begin() + successor_rank_offsets[i];
    const auto successor_end = successor_ranks.begin() + successor_rank_offsets[i + 1];
    const auto start_word = static_cast<std::size_t>(i / 64);

    // Each task reaches itself.
    ws.reachability.SetBit(i, i);

    if (successor_begin != successor_end)
    {
      ws.reachability.CopyRowFromWord(i, ws.reachability, *successor_begin, start_word);
      // Skip bits below the current topological rank because they are guaranteed to be zero.
      for (auto it = successor_begin + 1; it != successor_end; ++it)
        ws.reachability.OrRowsFromWord(i, ws.reachability, *it, start_word);
    }
  }
}

/**
 * Hopcroft-Karp solver for local-face slot reuse.
 *
 * The bipartite graph is defined over local directed faces. A left face `f` may be
 * matched to a right face `g` when the producer cell of `g` is reachable from, or
 * identical to, the consumer cell of `f`. This relation is the exact sweep execution
 * schedule-independent reuse condition for CBC local-face storage when incoming
 * faces are consumed before outgoing faces are written during a cell sweep.
 */
class LocalFaceHopcroftKarp
{
public:
  /**
   * Construct the local-face slot planner.
   *
   * \param face_producer_ranks Producer-cell topological ranks indexed by local face-task ID.
   * \param face_consumer_ranks Consumer-cell topological ranks indexed by local face-task ID.
   * \param producer_cell_face_offsets Offsets into the contiguous face ranges owned by each
   * producer rank.
   * \param face_slot_ids Output slot assignment indexed by local face-task ID.
   * \param ws Thread-local scratch buffers.
   */
  LocalFaceHopcroftKarp(const std::vector<std::uint32_t>& face_producer_ranks,
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
    ws.PrepareFaces(num_faces_);
  }

  /**
   * Solve the local-face slot assignment problem.
   *
   * \return Slot count and verifier status for the computed assignment.
   */
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

    const std::size_t max_num_slots = static_cast<std::size_t>(num_faces_) - matching_size;
    if (not VerifySlotAssignment(max_num_slots))
    {
      // Conservatively reject the slot assignment and return the upper bound.
      std::iota(face_slot_ids_.begin(), face_slot_ids_.end(), std::uint32_t{0});
      return {static_cast<std::size_t>(num_faces_), true};
    }

    return {max_num_slots, false};
  }

private:
  /**
   * Visit right-side reuse candidates for one left-side face.
   *
   * Candidates are generated on demand from the reachability closure and the
   * contiguous local-face ranges grouped by producer-cell topological rank.
   * The callback must return true to stop iteration (early exit), false to continue.
   */
  template <class F>
  void ForEachCandidate(const std::uint32_t u_face_rank, F fn) const
  {
    const auto consumer_cell_rank = face_consumer_ranks_[u_face_rank];
    const auto limit = producer_cell_face_offsets_.size() - 1;
    for (std::size_t producer_cell_rank =
           ws_.reachability.FindFirstSet(consumer_cell_rank, consumer_cell_rank);
         producer_cell_rank < limit;
         producer_cell_rank = ws_.reachability.FindNextSet(consumer_cell_rank, producer_cell_rank))
    {
      const auto face_begin = producer_cell_face_offsets_[producer_cell_rank];
      const auto face_end = producer_cell_face_offsets_[producer_cell_rank + 1];
      for (std::uint32_t v_face_rank = face_begin; v_face_rank < face_end; ++v_face_rank)
      {
        if (fn(v_face_rank))
          return;
      }
    }
  }

  /// Check whether two local faces satisfy the exact CBC reuse relation.
  bool ReuseRelationHolds(const std::uint32_t u_face_rank,
                          const std::uint32_t v_face_rank) const noexcept
  {
    return ws_.reachability.TestBit(face_consumer_ranks_[u_face_rank],
                                    face_producer_ranks_[v_face_rank]);
  }

  /// Extract a slot assignment from the final maximum matching.
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

  /// Verify that the extracted slot assignment respects the reuse relation.
  bool VerifySlotAssignment(const std::size_t slot_count) const
  {
    for (std::uint32_t face = 0; face < num_faces_; ++face)
    {
      if (face_slot_ids_[face] >= slot_count)
        return false; // Invalid slot assignment.
    }

    std::fill_n(ws_.face_last_rank_for_slot.begin(), slot_count, INVALID_INDEX);
    for (std::uint32_t rank = 0; rank < num_faces_; ++rank)
    {
      const auto sid = face_slot_ids_[rank];
      const auto prev_rank = ws_.face_last_rank_for_slot[sid];
      if ((prev_rank != INVALID_INDEX) and (not ReuseRelationHolds(prev_rank, rank)))
        return false; // Reuse relation violated by this slot assignment.
      ws_.face_last_rank_for_slot[sid] = rank;
    }
    return true; // Valid slot assignment.
  }

  /// Seed the matching with a greedy pass before the layered Hopcroft-Karp search.
  std::size_t GreedyInit()
  {
    std::size_t count = 0;
    for (std::uint32_t i = 0; i < num_faces_; ++i)
    {
      if (ws_.face_mate_u[i] != INVALID_INDEX)
        continue; // Already matched.

      ForEachCandidate(i,
                       [&](std::uint32_t v_face_rank) -> bool
                       {
                         if (ws_.face_mate_v[v_face_rank] != INVALID_INDEX)
                           return false; // Continue searching.
                         ws_.face_mate_u[i] = v_face_rank;
                         ws_.face_mate_v[v_face_rank] = i;
                         ++count;
                         return true; // Stop: matched.
                       });
    }
    return count;
  }

  /// Build one Hopcroft-Karp BFS layer graph.
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

      ForEachCandidate(u_face_rank,
                       [&](const std::uint32_t v_face_rank) -> bool
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
                         return false; // BFS must explore all candidates.
                       });
    }

    return dist_null_ != std::numeric_limits<int>::max();
  }

  /// Search one augmenting path in the current layer graph.
  bool DFS(const std::uint32_t u_face_rank)
  {
    bool matched = false;
    ForEachCandidate(u_face_rank,
                     [&](const std::uint32_t v_face_rank) -> bool
                     {
                       const auto mate_of_v = ws_.face_mate_v[v_face_rank];
                       if (mate_of_v == INVALID_INDEX)
                       {
                         if (dist_null_ == ws_.face_dist[u_face_rank] + 1)
                         {
                           ws_.face_mate_v[v_face_rank] = u_face_rank;
                           ws_.face_mate_u[u_face_rank] = v_face_rank;
                           ws_.face_dist[u_face_rank] = -1;
                           matched = true;
                           return true; // Stop: augmenting path found.
                         }
                       }
                       else if ((ws_.face_dist[mate_of_v] == ws_.face_dist[u_face_rank] + 1) and
                                DFS(mate_of_v))
                       {
                         ws_.face_mate_v[v_face_rank] = u_face_rank;
                         ws_.face_mate_u[u_face_rank] = v_face_rank;
                         ws_.face_dist[u_face_rank] = -1;
                         matched = true;
                         return true; // Stop: augmenting path found.
                       }
                       return false; // Continue searching.
                     });
    if (not matched)
      ws_.face_dist[u_face_rank] = -1; // Mark as dead end for this phase.
    return matched;
  }

  std::uint32_t num_faces_ = 0;
  const std::vector<std::uint32_t>& face_producer_ranks_;
  const std::vector<std::uint32_t>& face_consumer_ranks_;
  const std::vector<std::uint32_t>& producer_cell_face_offsets_;
  std::vector<std::uint32_t>& face_slot_ids_;
  ThreadLocalWorkspace& ws_;
  /// BFS distance label for the null vertex in the Hopcroft-Karp algorithm.
  int dist_null_ = 0;
};

} // namespace opensn::detail

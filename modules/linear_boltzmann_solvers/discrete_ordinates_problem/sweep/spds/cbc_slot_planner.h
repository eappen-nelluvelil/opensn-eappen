// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace opensn::detail
{

/// Result of an exact local-face slot-planning solve.
struct SlotSolveResult
{
  /// Exact number of reusable slots required by the computed chain cover.
  std::size_t slot_count = 0;
  /// Flag indicating that the post-solve verifier rejected the computed assignment.
  bool verifier_rejected = false;
};

/**
 * Compute the exact minimum safe local-face slot assignment.
 *
 * Let `F` denote the local directed faces, and define `u < v` when the consumer cell of
 * face `u` reaches the producer cell of face `v` in the local CBC task DAG. This is the
 * safe reuse relation: if `u < v`, then every admissible CBC or CBCD sweep consumes the
 * angular flux stored for `u` before `v` may overwrite the same slot.
 *
 * Computing the minimum number of reusable cell-face slots is equivalent to
 * the minimum chain-cover problem for the induced face poset.
 * A chain is one statically reusable slot. The minimum number of slots equals the poset
 * width (i.e. the maximum cardinality of any antichain of pairwise incomparable
 * faces). By Dilworth's theorem, this is the minimum chain-cover cardinality.
 *
 * The implementation computes the same maximum matching without materializing the transitive
 * closure. It constructs a capacitated network over the original task DAG with one unit source
 * arc for each face consumer and one unit sink arc for each face producer. Task-DAG edges have
 * capacity `|F|`. An integral source-to-sink flow path pairs exactly one left-side face with one
 * reachable right-side face; the unit terminal arcs enforce the matching constraints. Conversely,
 * every edge of the transitive-closure bipartite graph defines such a flow path. The maximum-flow
 * value therefore equals the maximum matching cardinality, and the induced minimum chain cover
 * has exactly `|F| - |M|` slots.
 *
 * Algorithm flow:
 * 1. Build the sparse residual network directly from the task-DAG CSR and face endpoint tables.
 * 2. Compute an integral maximum flow with a level-graph augmenting-path algorithm.
 * 3. Decompose the flow into certified reachable face pairs.
 * 4. Extract one slot chain per unmatched right-side face.
 * 5. Verify the flow decomposition and complete slot assignment before returning it.
 *
 * The network contains `O(|V| + |E| + |F|)` storage instead of the quadratic task-reachability
 * matrix. Flow decomposition consumes every unit on a concrete task-DAG path, providing a sparse
 * certificate for each reuse handoff. If the final slot-assignment verifier rejects the
 * decomposition, the planner conservatively returns the identity assignment (one slot per face).
 *
 * \param successor_rank_offsets Offsets into the flat successor-rank adjacency list of the
 * local CBC task DAG.
 * \param successor_ranks Flat successor-rank adjacency list of the local CBC task DAG.
 * \param face_producer_ranks Producer-cell topological rank for each local directed face.
 * \param face_consumer_ranks Consumer-cell topological rank for each local directed face.
 * \param producer_cell_face_offsets Offsets grouping local faces by producer-cell topological
 * rank.
 * \param face_slot_ids Output slot assignment keyed by local face rank.
 * \return Exact slot count and verifier status for the computed assignment.
 */
SlotSolveResult
ComputeLocalFaceSlotPlan(const std::vector<std::uint32_t>& successor_rank_offsets,
                         const std::vector<std::uint32_t>& successor_ranks,
                         const std::vector<std::uint32_t>& face_producer_ranks,
                         const std::vector<std::uint32_t>& face_consumer_ranks,
                         const std::vector<std::uint32_t>& producer_cell_face_offsets,
                         std::vector<std::uint32_t>& face_slot_ids);

} // namespace opensn::detail

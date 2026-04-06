// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace opensn
{

/**
 * Cell-by-cell sweep-plane data structure with slot-reuse planning.
 *
 * Specializes \ref SPDS for the cell-by-cell (CBC) sweep scheduler: each local cell becomes an
 * independent \ref Task whose predecessors and successors are derived from the upwind and
 * downwind neighbors implied by the angular direction \f$\vec\Omega\f$. The resulting task DAG
 * is produced by \c BuildTaskGraph and consumed by the CBC host and device sweep chunks.
 *
 * In addition to the task graph, this class exposes a cell-level angular-flux (psi) slot plan
 * that reports how few distinct storage slots would suffice if the cell's entire psi were
 * recycled as soon as all of its downstream consumers had finished reading it. The plan is
 * computed lazily by \c ComputeMaxNumLocalPsiSlots and is currently diagnostic-only: the
 * actual runtime storage consumed by the CBC FLUDS is governed by the finer-grained
 * face-node-level planner (\c cbc_local_face_slot_planner). The cell-level numbers therefore
 * serve as an upper-bound sanity check and an introspection hook for tooling, and
 * \c ComputeMaxNumLocalPsiSlots is retained for historical purposes even though it is no
 * longer invoked from the main flow.
 *
 * \par Mathematical formulation
 * Let \f$G=(V,E)\f$ be the local task DAG with \f$n=|V|\f$ tasks. Define the reuse relation
 * \f$u \rightsquigarrow v\f$ iff \f$v \notin \mathrm{succ}(u)\f$ and
 * \f$v \in \bigcap_{s \in \mathrm{succ}(u)} \mathrm{Desc}(s)\f$, i.e. every direct successor of
 * \f$u\f$ must have already dispatched before \f$v\f$ can safely reuse \f$u\f$'s slot. Because
 * this relation is consistent with the topological order, it induces a DAG whose minimum path
 * cover (by Dilworth's theorem) equals \f$n-|M^\star|\f$, where \f$M^\star\f$ is a maximum
 * bipartite matching of the reuse relation (Koenig's theorem). Each path in the cover is a
 * slot lifeline: tasks on the same path share one physical slot in topological order.
 *
 * \par Algorithm outline
 * \c ComputeMaxNumLocalPsiSlots proceeds as follows:
 *   -# Compute per-task reverse-topological descendant bitsets (reachability matrix).
 *   -# Build the reuse-targets bitmatrix row-by-row by intersecting descendant sets of each
 *      task's successors and subtracting the successor set itself.
 *   -# Run Hopcroft-Karp bipartite matching on the reuse relation in \f$O(E\sqrt V)\f$.
 *   -# The dynamic slot count equals \f$n-|M|\f$. Walk the matching to extract the chain
 *      decomposition and assign each chain a single static slot id, attaching sinks (tasks
 *      with no successors) to the tail of an existing chain whenever possible.
 *   -# Verify the assignment by checking that every consecutive pair of tasks sharing a slot
 *      satisfies the reuse relation against the raw DAG (not the cached reuse-targets
 *      bitmatrix, so construction bugs are caught). On verification failure, fall back to the
 *      safe identity assignment (one slot per cell) and emit a warning.
 *
 * \note The slot planner uses a thread-local scratch workspace, so concurrent invocations on
 * distinct \c CBC_SPDS instances from different threads are safe.
 */
class CBC_SPDS : public SPDS
{
public:
  /**
   * Construct a CBC sweep-plane data structure for the given direction and grid.
   *
   * Populates cell relationships, builds and topologically sorts the local task DAG
   * (optionally removing a feedback arc set to break cycles), constructs the \c Task list,
   * and initializes the slot outputs to the safe identity assignment. A subsequent call to
   * \c ComputeMaxNumLocalPsiSlots refines them.
   *
   * \param omega The angular direction vector.
   * \param grid Shared pointer to the mesh continuum.
   * \param allow_cycles If true, break cycles in the local cell graph via an approximate
   *                     minimum feedback arc set; otherwise throw on a cyclic graph.
   */
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  /// Return the list of tasks (one per local cell) in task-id order.
  const std::vector<Task>& GetTaskList() const noexcept;

  /**
   * Compute the cell-level psi slot plan for this direction.
   *
   * Runs the reuse-planner pipeline (reachability, reuse-targets, Hopcroft-Karp matching,
   * chain decomposition, verification) and updates \c max_num_local_psi_slots_,
   * \c num_static_local_psi_slots_, and \c task_slot_ids_. Falls back to the identity
   * assignment (one slot per cell) and emits a warning if the verifier rejects the planner's
   * output.
   *
   * \note Results are diagnostic-only at present; the CBC FLUDS runtime uses the finer-grained
   *       face-node-level planner. Retained for historical purposes; not called from the main
   *       flow.
   */
  void ComputeMaxNumLocalPsiSlots();

  /// Return the dynamic lower bound on the number of psi slots required locally.
  std::size_t GetMaxNumLocalPsiSlots() const noexcept { return max_num_local_psi_slots_; }

  /// Return the number of distinct static slot ids assigned across local tasks.
  std::size_t GetNumStaticLocalPsiSlots() const noexcept { return num_static_local_psi_slots_; }

  /// Return the per-task static slot-id assignment (one entry per local cell).
  const std::vector<std::uint32_t>& GetTaskSlotIDs() const noexcept { return task_slot_ids_; }

  ~CBC_SPDS() override = default;

private:
  /// Build the per-cell \ref Task list from the local DAG and cell-face orientations.
  void BuildTaskGraph();

  /// Topological ordering of local task ids (forward sweep order).
  std::vector<std::uint32_t> topo_order_;
  /// Per-cell tasks indexed by local cell id.
  std::vector<Task> task_list_;
  /// Static slot id assigned to each task, indexed by local cell id.
  std::vector<std::uint32_t> task_slot_ids_;
  /// Dynamic lower bound on the number of psi slots required locally (\f$n-|M|\f$).
  std::size_t max_num_local_psi_slots_ = 0;
  /// Number of distinct static slot ids actually assigned across local tasks.
  std::size_t num_static_local_psi_slots_ = 0;
};

} // namespace opensn

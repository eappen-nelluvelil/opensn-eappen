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
 * Cell-by-cell (CBC) sweep-plane data structure.
 *
 * Extends SPDS with a per-cell task graph and a static slot assignment that
 * maps each local cell to one of \f$s^*\f$ angular-flux storage slots, where
 * \f$s^*\f$ is the provably minimum number computed via Hopcroft-Karp maximum
 * bipartite matching on the task DAG's reuse graph
 * (see detail::DenseHopcroftKarp in cbc_slot_planner.h).
 *
 * The task graph encodes the local dependency structure for the given sweep
 * direction \f$\hat\Omega\f$: each task corresponds to one local cell, with
 * predecessor/successor edges derived from face orientations. The topological
 * ordering of this DAG determines the sweep execution order, and the static
 * slot assignment determines how CBC_FLUDS and CBCD_FLUDS allocate and reuse
 * angular-flux memory.
 *
 * ## Slot assignment and memory reduction
 *
 * Without slot optimization, each of the \f$n\f$ local cells requires its own
 * angular-flux buffer, yielding \f$O(n \cdot N_{\text{dof}} \cdot G \cdot A)\f$
 * memory. The static slot assignment exploits the observation that once a
 * cell's angular flux has been consumed by all of its successors, the buffer
 * can be reused by a later cell in topological order. The optimal slot count
 * \f$s^* \ll n\f$ yields a proportional reduction in FLUDS memory footprint
 * and improved cache utilization.
 *
 * The constructor initializes a safe identity assignment (one slot per cell).
 * Calling ComputeMaxNumLocalPsiSlots refines this to the optimal \f$s^*\f$.
 */
class CBC_SPDS : public SPDS
{
public:
  /**
   * Construct a CBC SPDS for a given sweep direction.
   *
   * Build the local dependency graph from face orientations, remove cyclic
   * dependencies if permitted, compute the topological ordering, and
   * construct the task graph. Initializes the slot assignment to the
   * identity (one slot per cell); call ComputeMaxNumLocalPsiSlots to
   * compute the optimal assignment.
   *
   * \param omega sweep direction vector \f$\hat\Omega\f$
   * \param grid shared mesh continuum
   * \param allow_cycles if true, remove cyclic dependencies via feedback arc set
   */
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  /// Return the immutable task graph (one Task per local cell).
  const std::vector<Task>& GetTaskList() const noexcept;

  /**
   * Compute the optimal slot count and static slot assignment.
   *
   * Invoke the dense Hopcroft-Karp maximum bipartite matching solver
   * (detail::DenseHopcroftKarp) on the task DAG to find the minimum
   * number of angular-flux storage slots \f$s^*\f$ and a conflict-free
   * mapping \f$\sigma : \text{task} \to \{0, \ldots, s^*{-}1\}\f$.
   * Updates both max_num_local_psi_slots_ and task_slot_ids_.
   *
   * Uses thread-local scratch buffers to avoid heap allocation on
   * repeated invocations from the same thread.
   */
  void ComputeMaxNumLocalPsiSlots();

  /// Optimal number of angular-flux storage slots for this SPDS instance.
  std::size_t GetMaxNumLocalPsiSlots() const noexcept { return max_num_local_psi_slots_; }

  /// Static slot assignment: \c task_slot_ids_[cell_local_id] = slot index.
  const std::vector<std::uint32_t>& GetTaskSlotIDs() const noexcept { return task_slot_ids_; }

  ~CBC_SPDS() override = default;

private:
  /**
   * Build the task graph from mesh face orientations.
   *
   * Populate task_list_ with one Task per local cell, recording predecessor
   * and successor relationships based on face orientation relative to the
   * sweep direction.
   */
  void BuildTaskGraph();

  /// Topological ordering of local cell IDs: \c topo_order_[rank] = cell_local_id.
  std::vector<std::uint32_t> topo_order_;
  /// Per-cell task descriptors with predecessor/successor adjacency lists.
  std::vector<Task> task_list_;
  /// Static slot assignment: \c task_slot_ids_[cell_local_id] = slot_id.
  std::vector<std::uint32_t> task_slot_ids_;
  /// Minimum number of angular-flux storage slots.
  std::size_t max_num_local_psi_slots_ = 0;
};

} // namespace opensn

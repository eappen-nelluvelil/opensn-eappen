// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace opensn
{

/**
 * Cell-by-cell (CBC) sweep-plane data structure.
 *
 * Extends SPDS with a per-cell task graph together with a static local-face
 * slot assignment that maps each local directed face to one of \f$s^*\f$
 * angular-flux storage slots, where \f$s^*\f$ is the provably minimum number
 * computed via Hopcroft-Karp maximum bipartite matching on the local-face
 * reuse graph (see detail::DenseLocalFaceHopcroftKarp in cbc_slot_planner.h).
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
 * Without slot optimization, each local directed face requires its own
 * angular-flux buffer, yielding \f$O(n_f \cdot N_{\text{face-dof}} \cdot G \cdot A)\f$
 * memory. The static slot assignment exploits the observation that once a
 * face's upwind data has been consumed by its downwind cell, the slot can be
 * reused by a later local directed face in topological order. The optimal slot
 * count \f$s^* \ll n_f\f$ yields a proportional reduction in FLUDS memory
 * footprint and improved cache utilization.
 *
 * The constructor initializes a safe identity assignment (one slot per local
 * directed face).
 * Calling ComputeMaxNumLocalPsiSlots refines this to the optimal \f$s^*\f$.
 */
class CBC_SPDS : public SPDS
{
public:
  static constexpr std::uint32_t INVALID_LOCAL_FACE_TASK_ID =
    std::numeric_limits<std::uint32_t>::max();

  /**
   * Construct a CBC SPDS for a given sweep direction.
   *
   * Build the local dependency graph from face orientations, remove cyclic
   * dependencies if permitted, compute the topological ordering, and
   * construct the task graph. Initializes the local-face slot assignment to the
   * identity (one slot per local directed face); call ComputeMaxNumLocalPsiSlots to
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
   * Invoke the dense local-face Hopcroft-Karp maximum bipartite matching solver
   * (detail::DenseLocalFaceHopcroftKarp) on the local directed-face reuse graph
   * to find the minimum number of angular-flux storage slots \f$s^*\f$ and a
   * conflict-free mapping \f$\sigma : \text{face-task} \to \{0, \ldots, s^*{-}1\}\f$.
   * Updates max_num_local_psi_slots_ and local_face_slot_ids_.
   *
   * Uses thread-local scratch buffers to avoid heap allocation on
   * repeated invocations from the same thread.
   */
  void ComputeMaxNumLocalPsiSlots();

  /// Optimal number of local-face angular-flux storage slots for this SPDS instance.
  std::size_t GetMaxNumLocalPsiSlots() const noexcept { return max_num_local_psi_slots_; }

  /// Static local-face slot assignment: \c local_face_slot_ids_[face_task_id] = slot index.
  const std::vector<std::uint32_t>& GetLocalFaceSlotIDs() const noexcept
  {
    return local_face_slot_ids_;
  }

  /// Return the maximum number of nodes on any local directed face.
  std::size_t GetMaxLocalFaceNodeCount() const noexcept { return max_local_face_node_count_; }

  /// Return the local directed-face task id for an outgoing local face.
  std::uint32_t GetOutgoingLocalFaceTaskID(std::uint32_t cell_local_id,
                                           unsigned int face_id) const noexcept;

  /// Return the local directed-face task id for an incoming local face.
  std::uint32_t GetIncomingLocalFaceTaskID(std::uint32_t cell_local_id,
                                           unsigned int face_id) const noexcept;

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
  void BuildLocalFaceTaskGraph();

  /// Topological ordering of local cell IDs: \c topo_order_[rank] = cell_local_id.
  std::vector<std::uint32_t> topo_order_;
  /// Per-cell task descriptors with predecessor/successor adjacency lists.
  std::vector<Task> task_list_;
  /// Flat face-table offsets indexed by cell-local-id.
  std::vector<std::uint32_t> cell_face_offsets_;
  /// Flat outgoing local-face task ids indexed by face storage index.
  std::vector<std::uint32_t> outgoing_local_face_task_ids_;
  /// Flat incoming local-face task ids indexed by face storage index.
  std::vector<std::uint32_t> incoming_local_face_task_ids_;
  /// Face-rank offsets grouped by producer-cell topological rank.
  std::vector<std::uint32_t> producer_cell_face_offsets_;
  /// Producer-cell topological rank for each local directed face.
  std::vector<std::uint32_t> local_face_producer_ranks_;
  /// Consumer-cell topological rank for each local directed face.
  std::vector<std::uint32_t> local_face_consumer_ranks_;
  /// Static slot assignment: \c local_face_slot_ids_[face_task_id] = slot_id.
  std::vector<std::uint32_t> local_face_slot_ids_;
  /// Minimum number of local-face angular-flux storage slots.
  std::size_t max_num_local_psi_slots_ = 0;
  /// Maximum number of nodes on any local directed face.
  std::size_t max_local_face_node_count_ = 0;
};

} // namespace opensn

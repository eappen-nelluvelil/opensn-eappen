// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

namespace opensn
{

/**
 * Cell-by-cell sweep-plane data structure with cycle-aware task scheduling and exact
 * local-face slot metadata.
 *
 * The class owns the local CBC task graph for one sweep direction, the local and global
 * feedback-arc-set (FAS) state used to break sweep cycles, and the local directed-face
 * tables consumed by host CBC and device CBCD FLUDS.  Delayed local and interpartition
 * dependencies created by FAS removal are tracked separately from the same-iteration task
 * graph so that compact local-face slot reuse only operates on the reduced acyclic graph.
 *
 * Slot planning is exact.  `ComputeMaxNumLocalPsiSlots` solves the minimum chain-cover of
 * the local directed-face reuse poset (equivalently, the poset width by Dilworth's
 * theorem) using a bipartite-matching reduction.  The result is the smallest number of
 * reusable angular-flux storage slots required by the local same-iteration sweep.  The
 * planner must be invoked after the global FAS is applied so that the reduced task graph
 * is final.
 */
class CBC_SPDS : public SPDS
{
public:
  /// Weighted directed edge in the interpartition CBC sweep graph.
  struct LocationEdgeWeight
  {
    /// Upstream MPI rank.
    int upstream_location = 0;
    /// Downstream MPI rank.
    int downstream_location = 0;
    /// Accumulated sweep-graph edge weight.
    double weight = 0.0;
  };

  /// Sentinel returned when a local face does not participate in the requested face-task map.
  static constexpr std::uint32_t INVALID_LOCAL_FACE_TASK_ID =
    std::numeric_limits<std::uint32_t>::max();

  /**
   * Construct a cell-by-cell sweep-plane data structure for one angular direction.
   *
   * The constructor builds the local cell dependency graph, removes a local feedback-arc
   * set when cyclic dependencies are allowed, topologically orders the local cell tasks,
   * communicates rank dependencies, builds the initial task list, and assembles the
   * non-delayed local directed-face tables that feed the slot planner.  Delayed local
   * edges are excluded from the directed-face tables so that compact slot reuse is
   * computed only over same-iteration dependencies.
   *
   * \param id Globally unique CBC SPDS identifier.
   * \param omega Angular direction vector.
   * \param grid Reference to the mesh continuum.
   * \param allow_cycles Whether cyclic dependencies in the local sweep graph may be
   * broken by lagging fluxes.
   */
  CBC_SPDS(int id,
           const Vector3& omega,
           const std::shared_ptr<MeshContinuum>& grid,
           bool allow_cycles);

  /// Return the globally unique CBC SPDS identifier.
  int GetId() const noexcept { return id_; }

  /// Return the local cell task list.
  const std::vector<Task>& GetTaskList() const;

  /// Return flattened rank pairs removed from the interpartition sweep graph.
  std::vector<int> GetGlobalSweepFAS() const { return global_sweep_fas_; }

  /// Set flattened rank pairs removed from the interpartition sweep graph.
  void SetGlobalSweepFAS(std::vector<int>& edges) { global_sweep_fas_ = edges; }

  /// Build the global feedback arc set from the interpartition sweep graph.
  void BuildGlobalSweepFAS();

  /// Apply the global feedback arc set to location dependencies and rebuild tasks.
  void ApplyGlobalSweepFAS();

  /// Compute sparse edge weights from this rank to downstream ranks.
  std::vector<LocationEdgeWeight> ComputeLocalLocationEdgeWeights() const;

  /// Store sparse global edge weights for global feedback arc set construction.
  void SetGlobalEdgeWeights(std::span<const LocationEdgeWeight> edge_weights);

  /// Return whether a local upwind-to-downwind cell dependency is delayed.
  bool IsDelayedLocalDependency(std::uint32_t upwind_local_id,
                                std::uint32_t downwind_local_id) const noexcept;

  /**
   * Compute the exact minimum number of reusable local-face psi storage slots.
   *
   * Each local directed face is one element of the reuse poset.  For two faces `u` and
   * `v`, the planner permits both faces to share the same slot when the consumer cell of
   * `u` reaches the producer cell of `v` in the reduced local task DAG: under that
   * relation, every admissible sweep consumes the psi stored for `u` before `v` may
   * overwrite the slot.  The minimum number of reusable slots is the minimum chain-cover
   * cardinality of this poset, computed exactly via a bipartite split-graph reduction and
   * Hopcroft-Karp maximum matching.  `UpdateLocalFaceSlotLayout` then sizes each slot
   * by the maximum face-node extent over the faces assigned to that slot.  If the
   * planner's verifier rejects the computed assignment, the result falls back to the
   * identity assignment (one slot per local directed face, no reuse) and a warning is
   * logged.
   *
   * The planner must be invoked after `ApplyGlobalSweepFAS` so that the reduced
   * same-iteration task graph is final.  Delayed local dependencies are already excluded
   * from the directed-face tables built by the constructor.
   */
  void ComputeMaxNumLocalPsiSlots();

  /// Return the minimum number of reusable local-face psi storage slots.
  std::size_t GetMaxNumLocalPsiSlots() const noexcept { return max_num_local_psi_slots_; }

  /// Return the static face-to-slot assignment indexed by local directed-face task ID.
  const std::vector<std::uint32_t>& GetLocalFaceSlotIDs() const noexcept
  {
    return local_face_slot_ids_;
  }

  /// Return prefix offsets into the compact local-face slot bank.
  const std::vector<std::uint32_t>& GetLocalFaceSlotNodeOffsets() const noexcept
  {
    return local_face_slot_node_offsets_;
  }

  /// Return slot-local node extents indexed by slot ID.
  const std::vector<std::uint16_t>& GetLocalFaceSlotNodeCounts() const noexcept
  {
    return local_face_slot_node_counts_;
  }

  /// Return the total number of local-face nodes spanned by the compact slot bank.
  std::size_t GetTotalLocalFaceSlotNodes() const noexcept { return total_local_face_slot_nodes_; }

  /// Return the maximum number of nodes across all local directed faces.
  std::size_t GetMaxLocalFaceNodeCount() const noexcept { return max_local_face_node_count_; }

  /// Return the local directed-face task ID for an outgoing local face.
  std::uint32_t GetOutgoingLocalFaceTaskID(std::uint32_t cell_local_id,
                                           unsigned int face_id) const noexcept;

  /// Return the local directed-face task ID for an incoming local face.
  std::uint32_t GetIncomingLocalFaceTaskID(std::uint32_t cell_local_id,
                                           unsigned int face_id) const noexcept;

protected:
  /// Build local sweep tasks from current local and delayed dependencies.
  void BuildTaskList();

  /// Build the topological-rank successor adjacency from the reduced local task DAG.
  void BuildTaskSuccessorAdjacency();

  /// Enumerate non-delayed local directed faces and map them to producer/consumer ranks.
  void BuildLocalFaceTaskGraph();

  /**
   * Recompute slot-local node extents and prefix offsets from the current slot assignment.
   *
   * Each slot is sized by the maximum face-node extent of the local directed faces
   * assigned to that slot.  The compact slot bank is the concatenation of these
   * per-slot extents, so storage is exactly the minimum required while preserving O(1)
   * indexing.
   */
  void UpdateLocalFaceSlotLayout();

  /// Globally unique CBC SPDS identifier.
  int id_ = 0;
  /// Whether cyclic dependencies may be broken by lagging fluxes.
  bool allow_cycles_ = false;
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;
  /// Incoming interpartition dependencies for each MPI rank.
  std::vector<std::vector<int>> global_dependencies_;
  /// Flattened pairs of rank edges removed from the global sweep graph.
  std::vector<int> global_sweep_fas_;
  /// Sparse transport weights keyed by directed interpartition edge.
  std::unordered_map<std::uint64_t, double> global_edge_weights_;
  /// Delayed local upwind-to-downwind cell dependencies.
  std::set<std::uint64_t> delayed_local_dependency_set_;

  /// Topological rank keyed by local cell ID; inverse of the base-class `spls_` ordering.
  std::vector<std::uint32_t> topo_rank_by_cell_local_id_;
  /// Offsets into the flat successor-rank array indexed by topological task rank.
  std::vector<std::uint32_t> task_successor_rank_offsets_;
  /// Flat successor topological ranks grouped by producer task rank.
  std::vector<std::uint32_t> task_successor_ranks_;
  /// Flat face-table offsets indexed by cell local IDs.
  std::vector<std::uint32_t> cell_face_offsets_;
  /// Flat outgoing local-face task IDs indexed by per-cell face storage index.
  std::vector<std::uint32_t> outgoing_local_face_task_ids_;
  /// Flat incoming local-face task IDs indexed by per-cell face storage index.
  std::vector<std::uint32_t> incoming_local_face_task_ids_;
  /// Face-rank offsets grouped by producer-cell topological rank.
  std::vector<std::uint32_t> producer_cell_face_offsets_;
  /// Producer-cell topological rank for each local directed face.
  std::vector<std::uint32_t> local_face_producer_ranks_;
  /// Consumer-cell topological rank for each local directed face.
  std::vector<std::uint32_t> local_face_consumer_ranks_;
  /// Number of nodes for each local directed face.
  std::vector<std::uint16_t> local_face_node_counts_;
  /// Static slot assignment: `local_face_slot_ids_[face_task_id] = slot_id`.
  std::vector<std::uint32_t> local_face_slot_ids_;
  /// Slot-local maximum node extents: `local_face_slot_node_counts_[slot_id]`.
  std::vector<std::uint16_t> local_face_slot_node_counts_;
  /// Prefix offsets into the compact local-face slot bank, one entry per slot plus a trailing total.
  std::vector<std::uint32_t> local_face_slot_node_offsets_;
  /// Minimum number of local-face angular flux storage slots.
  std::size_t max_num_local_psi_slots_ = 0;
  /// Total number of local-face nodes spanned by the compact slot bank.
  std::size_t total_local_face_slot_nodes_ = 0;
  /// Maximum number of nodes across all local directed faces.
  std::size_t max_local_face_node_count_ = 0;
};

} // namespace opensn

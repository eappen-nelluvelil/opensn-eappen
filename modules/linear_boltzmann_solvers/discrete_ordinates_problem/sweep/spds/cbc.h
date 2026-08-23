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
#include <utility>
#include <vector>

namespace opensn
{

/// CBC sweep-plane data structure.
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

  static constexpr std::uint32_t INVALID_LOCAL_FACE_TASK_ID =
    std::numeric_limits<std::uint32_t>::max();

  /**
   * Construct a cell-by-cell sweep-plane data structure for the given direction and grid.
   *
   * \param id Process-independent CBC SPDS ordinal.
   * \param omega The angular direction vector.
   * \param grid Reference to the grid.
   * \param face_neighbor_info Cached neighbor information for every local cell face.
   * \param allow_cycles Whether cycles are allowed in the local sweep dependency graph.
   */
  CBC_SPDS(int id,
           const Vector3& omega,
           const std::shared_ptr<MeshContinuum>& grid,
           const SPDSFaceNeighborInfoVec& face_neighbor_info,
           bool allow_cycles);

  /// Return the process-independent CBC SPDS ordinal.
  int GetId() const noexcept { return id_; }

  /// Return whether cyclic dependencies may be lagged.
  bool AreCyclesAllowed() const noexcept { return allow_cycles_; }

  /**
   * Remove a deterministic feedback arc set from each strongly connected
   * component (SCC).
   *
   * SCCs with 16 or fewer vertices use an exact minimum-weight dynamic program to
   * find the minimum-weight FAS. SCCs with more than 16 vertices use a deterministic
   * weighted-ordering heuristic to find an approximate minimum-weight FAS.
   */
  static std::vector<std::pair<Vertex, Vertex>> RemoveCyclicDependencies(Graph& graph);

  /// Return the local cell task list.
  const std::vector<Task>& GetTaskList() const;

  /// Return the initial unsatisfied dependency count indexed by local task ID.
  const std::vector<unsigned int>& GetInitialTaskDependencies() const noexcept
  {
    return initial_task_dependencies_;
  }

  /// Return the initially ready local task stack in local-cell order.
  const std::vector<std::uint32_t>& GetInitialReadyTasks() const noexcept
  {
    return initial_ready_tasks_;
  }

  /// Return flattened rank pairs removed from the interpartition sweep graph.
  const std::vector<int>& GetGlobalSweepFAS() const { return global_sweep_fas_; }

  /// Set flattened rank pairs removed from the interpartition sweep graph.
  void SetGlobalSweepFAS(std::vector<int> edges) { global_sweep_fas_ = std::move(edges); }

  /// Build the global feedback arc set from the interpartition sweep graph.
  void BuildGlobalSweepFAS();

  /// Apply the global feedback arc set to location dependencies and rebuild tasks.
  void ApplyGlobalSweepFAS();

  /**
   * Compute sparse local contributions to interpartition edge weights.
   *
   * Each outgoing rank edge is weighted by its total face-node count, which is proportional to the
   * lagged nonlocal angular-flux storage and copy work for this SPDS.
   */
  std::vector<LocationEdgeWeight> ComputeLocalLocationEdgeWeights() const;

  /// Store sparse global edge weights for global feedback arc set construction.
  void SetGlobalEdgeWeights(std::span<const LocationEdgeWeight> edge_weights);

  /// Store the incoming interpartition dependencies for every MPI rank.
  void SetGlobalDependencies(std::vector<std::vector<int>> global_dependencies);

  /// Return whether a local upwind-to-downwind cell dependency is delayed.
  bool IsDelayedLocalDependency(std::uint32_t upwind_local_id,
                                std::uint32_t downwind_local_id) const noexcept;

  /// Compute the minimum chain cover of the non-delayed local-face reuse poset.
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
  /// Approximate a minimum-weight FAS using a deterministic weighted vertex ordering.
  static std::vector<std::pair<Vertex, Vertex>>
  FindApproxMinimumFAS(const Graph& graph, const std::vector<Vertex>& component);

  /// Find an exact minimum-weight FAS by dynamic programming over vertex subsets.
  static std::vector<std::pair<Vertex, Vertex>>
  FindExactMinimumFAS(const Graph& graph, const std::vector<Vertex>& component);

  /// Build local sweep tasks from current local and delayed dependencies.
  void BuildTaskList();

  void BuildTaskSuccessorAdjacency();
  void BuildLocalFaceTaskGraph();
  void UpdateLocalFaceSlotLayout();

  /// Process-independent ordinal used to order CBC collectives.
  int id_ = 0;
  /// Whether cyclic dependencies may be broken by lagging fluxes.
  bool allow_cycles_ = false;
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;
  /// Contiguous initial dependency counts used to reset every angle-set sweep.
  std::vector<unsigned int> initial_task_dependencies_;
  /// Initially ready tasks in local-cell order.
  std::vector<std::uint32_t> initial_ready_tasks_;
  /// Incoming interpartition dependencies for each MPI rank.
  std::vector<std::vector<int>> global_dependencies_;
  /// Flattened pairs of rank edges removed from the global sweep graph.
  std::vector<int> global_sweep_fas_;
  /// Sparse transport weights keyed by directed interpartition edge.
  std::unordered_map<std::uint64_t, double> global_edge_weights_;
  /// Delayed local upwind-to-downwind cell dependencies.
  std::set<std::uint64_t> delayed_local_dependency_set_;
  /// MPI-rank-indexed flags for delayed incoming location dependencies.
  std::vector<unsigned char> delayed_location_dependency_flags_;

  std::vector<std::uint32_t> topo_rank_by_cell_local_id_;
  std::vector<std::uint32_t> task_successor_rank_offsets_;
  std::vector<std::uint32_t> task_successor_ranks_;
  std::vector<std::uint32_t> cell_face_offsets_;
  std::vector<std::uint32_t> outgoing_local_face_task_ids_;
  std::vector<std::uint32_t> incoming_local_face_task_ids_;
  std::vector<std::uint32_t> producer_cell_face_offsets_;
  std::vector<std::uint32_t> local_face_producer_ranks_;
  std::vector<std::uint32_t> local_face_consumer_ranks_;
  std::vector<std::uint16_t> local_face_node_counts_;
  std::vector<std::uint32_t> local_face_slot_ids_;
  std::vector<std::uint16_t> local_face_slot_node_counts_;
  std::vector<std::uint32_t> local_face_slot_node_offsets_;
  std::size_t max_num_local_psi_slots_ = 0;
  std::size_t total_local_face_slot_nodes_ = 0;
  std::size_t max_local_face_node_count_ = 0;
};

} // namespace opensn

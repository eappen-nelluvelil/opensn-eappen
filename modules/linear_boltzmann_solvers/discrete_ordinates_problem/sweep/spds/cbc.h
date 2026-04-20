// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

namespace opensn
{

/**
 * Cell-by-cell sweep plane data structure.
 *
 * Stores the local CBC task graph together with the precomputed metadata needed to
 * allocate and index minimally sized local-face angular-flux storage.
 * Each local cell contributes one task. Local directed faces are numbered separately
 * so that incoming and outgoing local-face accesses can be mapped onto a compact slot bank.
 */
class CBC_SPDS : public SPDS
{
public:
  /// Value returned when a local face does not participate in the requested face-task map.
  static constexpr std::uint32_t INVALID_LOCAL_FACE_TASK_ID =
    std::numeric_limits<std::uint32_t>::max();

  /**
   * Construct the CBC sweep plane data structure for one angular direction.
   *
   * \param omega Angular sweep direction.
   * \param grid Grid on which the local sweep graph is built.
   * \param allow_cycles Allow cycles in the local dependency graph when the mesh or
   * sweep ordering requires them.
   */
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  /// Returns the id of this SPDS.
  int GetId() const noexcept { return id_; }

  /// Sets the id of this SPDS.
  void SetId(int id) noexcept { id_ = id; }

  /// Return the local CBC task list.
  const std::vector<Task>& GetTaskList() const noexcept;

  /// Returns the global sweep FAS as a vector of edges.
  std::vector<int> GetGlobalSweepFAS() const { return global_sweep_fas_; }

  /// Sets the global sweep FAS.
  void SetGlobalSweepFAS(std::vector<int>& edges) { global_sweep_fas_ = edges; }

  /// Builds the Feedback Arc Set (FAS) for the global sweep.
  void BuildGlobalSweepFAS();

  /// Builds the Task Dependency Graph (TDG) for the global sweep.
  void BuildGlobalSweepTDG();

  /// Returns the locally accumulated location-to-location edge weights.
  std::vector<double> ComputeLocalLocationEdgeWeights() const;

  /// Sets the global location-to-location edge weights.
  void SetGlobalEdgeWeights(std::vector<double>& weights)
  {
    global_edge_weights_ = std::move(weights);
  }

  /**
   * Compute the minimum number of reusable local-face psi slots and assign faces to slots.
   *
   * Builds the local-face reuse relation implied by the task DAG and records a
   * static slot assignment that is safe for every admissible CBC sweep execution.
   */
  void ComputeMaxNumLocalPsiSlots();

  /// Return the number of local-face psi slots required by the static assignment.
  std::size_t GetMaxNumLocalPsiSlots() const noexcept { return max_num_local_psi_slots_; }

  /// Return the per-local-face slot assignment indexed by local face-task ID.
  const std::vector<std::uint32_t>& GetLocalFaceSlotIDs() const noexcept
  {
    return local_face_slot_ids_;
  }

  /// Return the base node offset of each local-face slot in the compact bank.
  const std::vector<std::uint32_t>& GetLocalFaceSlotNodeOffsets() const noexcept
  {
    return local_face_slot_node_offsets_;
  }

  /// Return the node extent of each local-face slot.
  const std::vector<std::uint16_t>& GetLocalFaceSlotNodeCounts() const noexcept
  {
    return local_face_slot_node_counts_;
  }

  /// Return the total number of local-face nodes stored in the compact slot bank.
  std::size_t GetTotalLocalFaceSlotNodes() const noexcept { return total_local_face_slot_nodes_; }

  /// Return the maximum number of nodes over all local directed faces.
  std::size_t GetMaxLocalFaceNodeCount() const noexcept { return max_local_face_node_count_; }

  /**
   * Return the local directed-face task ID for an outgoing local face.
   *
   * \param cell_local_id Local cell ID.
   * \param face_id Local face ID on the cell.
   * \return Local face-task ID, or INVALID_LOCAL_FACE_TASK_ID when the face is not
   * an outgoing local face in the CBC ordering.
   */
  std::uint32_t GetOutgoingLocalFaceTaskID(std::uint32_t cell_local_id,
                                           unsigned int face_id) const noexcept;

  /**
   * Return the local directed-face task ID for an incoming local face.
   *
   * \param cell_local_id Local cell ID.
   * \param face_id Local face ID on the cell.
   * \return Local face-task ID, or INVALID_LOCAL_FACE_TASK_ID when the face is not
   * an incoming local face in the CBC ordering.
   */
  std::uint32_t GetIncomingLocalFaceTaskID(std::uint32_t cell_local_id,
                                           unsigned int face_id) const noexcept;

  ~CBC_SPDS() override = default;

private:
  /**
   * Build the per-cell CBC task graph from the face orientations.
   */
  void BuildTaskGraph();

  /**
   * Build the local directed-face indexing used by CBC and CBCD FLUDS.
   *
   * Enumerates incoming and outgoing local faces, records their producer/consumer
   * task ranks, and prepares the compact metadata consumed by the slot planner.
   */
  void BuildLocalFaceTaskGraph();

  /// Topological ordering of local cell IDs: topo_order_[rank] = cell_local_id.
  std::vector<std::uint32_t> topo_order_;
  /// Unique identifier for this SPDS.
  int id_ = 0;
  /// Flag indicating whether cycles are allowed in the dependency graphs.
  bool allow_cycles_ = false;
  /// Per-cell task descriptors with successor adjacency lists.
  std::vector<Task> task_list_;
  /// Location-to-location global sweep dependencies.
  std::vector<std::vector<int>> global_dependencies_;
  /// Vector of edges representing the FAS used to break cycles in the global sweep graph.
  std::vector<int> global_sweep_fas_;
  /// Flattened comm_size x comm_size matrix of global edge weights.
  std::vector<double> global_edge_weights_;
  /// Set of local delayed dependency edges encoded as packed (upwind, downwind) pairs.
  std::set<std::uint64_t> delayed_local_dependency_set_;
  /// Offsets into the flat successor-rank array indexed by topological task rank.
  std::vector<std::uint32_t> task_successor_rank_offsets_;
  /// Flat successor topological ranks grouped by producer task rank.
  std::vector<std::uint32_t> task_successor_ranks_;
  /// Flat face-table offsets indexed by cell local IDs.
  std::vector<std::uint32_t> cell_face_offsets_;
  /// Flat outgoing local-face task IDs indexed by face storage index.
  std::vector<std::uint32_t> outgoing_local_face_task_ids_;
  /// Flat incoming local-face task IDs indexed by face storage index.
  std::vector<std::uint32_t> incoming_local_face_task_ids_;
  /// Face-rank offsets grouped by producer-cell topological rank.
  std::vector<std::uint32_t> producer_cell_face_offsets_;
  /// Producer-cell topological rank for each local directed face.
  std::vector<std::uint32_t> local_face_producer_ranks_;
  /// Consumer-cell topological rank for each local directed face.
  std::vector<std::uint32_t> local_face_consumer_ranks_;
  /// Number of nodes for each local directed face task.
  std::vector<std::uint16_t> local_face_node_counts_;
  /// Static slot assignment: local_face_slot_ids_[face_task_id] = slot_id.
  std::vector<std::uint32_t> local_face_slot_ids_;
  /// Slot-local node extents: local_face_slot_node_counts_[slot_id] = max nodes in that slot.
  std::vector<std::uint16_t> local_face_slot_node_counts_;
  /// Prefix offsets into the compact local-face slot bank.
  std::vector<std::uint32_t> local_face_slot_node_offsets_;
  /// Minimum number of local-face angular flux storage slots.
  std::size_t max_num_local_psi_slots_ = 0;
  /// Total number of local-face nodes in the compact slot bank.
  std::size_t total_local_face_slot_nodes_ = 0;
  /// Maximum number of nodes across all local directed faces.
  std::size_t max_local_face_node_count_ = 0;

  /// Recompute slot-local node extents and prefix offsets from the current slot assignment.
  void UpdateLocalFaceSlotLayout();
};

} // namespace opensn

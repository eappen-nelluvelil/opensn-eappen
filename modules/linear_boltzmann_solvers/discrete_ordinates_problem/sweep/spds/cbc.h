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

class CBC_SPDS : public SPDS
{
public:
  static constexpr std::uint32_t INVALID_LOCAL_FACE_TASK_ID =
    std::numeric_limits<std::uint32_t>::max();

  /**
   * Constructs a cell-by-cell sweep-plane data strcture (SPDS) with the given direction and grid.
   *
   * \param omega The angular direction vector.
   * \param grid Reference to the grid.
   * \param allow_cycles Whether cycles are allowed in the local sweep dependency graph.
   */
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  /// Returns the cell-by-cell task list.
  const std::vector<Task>& GetTaskList() const noexcept;

  /**
   * Compute the maximum number of local cell-face psi slots and static cell-face slot assignment.
   */
  void ComputeMaxNumLocalPsiSlots();

  std::size_t GetMaxNumLocalPsiSlots() const noexcept { return max_num_local_psi_slots_; }

  const std::vector<std::uint32_t>& GetLocalFaceSlotIDs() const noexcept
  {
    return local_face_slot_ids_;
  }

  std::size_t GetMaxLocalFaceNodeCount() const noexcept { return max_local_face_node_count_; }

  /// Returns the local directed-face task ID for an outgoing local face.
  std::uint32_t GetOutgoingLocalFaceTaskID(std::uint32_t cell_local_id,
                                           unsigned int face_id) const noexcept;

  /// Returns the local directed-face task ID for an incoming local face.
  std::uint32_t GetIncomingLocalFaceTaskID(std::uint32_t cell_local_id,
                                           unsigned int face_id) const noexcept;

  ~CBC_SPDS() override = default;

private:
  /**
   * Buid the task graph from mesh face orientations.
   *
   * Populate task_list_ with one Task per local cell, recording successor
   * relationships based on face orientation relative to the sweep direction.
   */
  void BuildTaskGraph();
  void BuildLocalFaceTaskGraph();

  /// Topological ordering of local cell IDs: topo_order_[rank] = cell_local_id.
  std::vector<std::uint32_t> topo_order_;
  /// Per-cell task descriptors with successor adjacency lists.
  std::vector<Task> task_list_;
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
  /// Static slot assignment: local_face_slot_ids_[face_task_id] = slot_id.
  std::vector<std::uint32_t> local_face_slot_ids_;
  /// Minimum number of local-face angular flux storage slots.
  std::size_t max_num_local_psi_slots_ = 0;
  /// Maximum number of nodes across all local directed faces.
  std::size_t max_local_face_node_count_ = 0;
};

} // namespace opensn

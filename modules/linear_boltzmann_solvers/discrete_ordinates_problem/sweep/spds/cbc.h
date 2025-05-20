// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"

namespace opensn
{

class CBC_SPDS : public SPDS
{
public:
  /**
   * Constructs a cell-by-cell sweep-plane data strcture (SPDS) with the given direction and grid.
   *
   * \param omega The angular direction vector.
   * \param grid Reference to the grid.
   * \param allow_cycles Whether cycles are allowed in the local sweep dependency graph.
   */
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum> grid, bool allow_cycles);

  /// Returns the cell-by-cell task list.
  const std::vector<Task>& GetTaskList() const;

  const std::vector<int>& GetSPLSToTaskIndexMap() const;

  const std::vector<std::vector<int>>&
  GetTaskLocalPredecessorsMap() const; // new_task_idx -> list of pred_new_task_idx

  const Task& GetTaskByNewIndex(int new_task_idx) const; // Access task by its spls-ordered index

  int GetPeakLivenessCount() const { return peak_liveness_count_; }

  const std::vector<int>& GetPsiStoreTimestepMap() const { return psi_store_timestep_; }

  const std::vector<int>& GetPsiDiscardTimestepMap() const { return psi_discard_timestep_; }

protected:
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;

  std::vector<int> map_original_local_id_new_task_index_;

  std::vector<std::vector<int>> task_local_predecessors_map_;

  int peak_liveness_count_ = 0;
  std::vector<int> psi_store_timestep_;   // Index by original_local_id
  std::vector<int> psi_discard_timestep_; // Index by original_local_id
};

} // namespace opensn

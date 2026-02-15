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
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  /// Returns the cell-by-cell task list.
  const std::vector<Task>& GetTaskList() const;

  /// Returns the maximum number of slots needed for CBC_FLUDS pool allocator.
  unsigned int GetMaxNumSlots() const { return max_num_slots_; }

  /**
   * Computes the maximum number of slots needed for CBC_FLUDS pool allocator 
   * by constructing the reflexive transitive closure of the local sweep dependency graph,
   * constructing a bipartite graph from the closure, and finding the maximum matching of the bipartite graph.
   */
  void SimulateLocalSweep();

protected:
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;
  /// Maximum number of slots needed for CBC_FLUDS pool allocator.
  unsigned int max_num_slots_ = 0;
};

} // namespace opensn

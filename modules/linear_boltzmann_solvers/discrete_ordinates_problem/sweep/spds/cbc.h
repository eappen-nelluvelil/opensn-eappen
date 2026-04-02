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
  const std::vector<Task>& GetTaskList() const noexcept;

  /// Compute the exact minimum number of pool slots required for local CBC angular flux storage.
  void ComputeMinNumLocalPsiSlots();

  /// Return the exact minimum number of pool slots required for local CBC angular flux storage.
  std::size_t GetMinNumLocalPsiSlots() const noexcept { return min_num_local_psi_slots_; }

protected:
  /// Topological ordering of the local task graph using local cell ids.
  std::vector<std::uint32_t> topo_order_;
  /// CSR row offsets for the local successor graph used by exact slot counting.
  std::vector<std::uint32_t> local_successor_offsets_;
  /// CSR column indices for the local successor graph used by exact slot counting.
  std::vector<std::uint32_t> local_successors_;
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;
  /// Exact minimum number of local CBC pool allocator slots.
  std::size_t min_num_local_psi_slots_ = 0;
};

} // namespace opensn

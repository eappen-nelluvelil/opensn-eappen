// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/sweep.h"

#include <cstddef>
#include <sys/types.h>
#include <unordered_map> // For the task list index map for cell local IDs
#include <cstdint>       // For uint64_t

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

  const std::unordered_map<uint64_t, size_t>& GetLocalIDToTaskIndexMap() const
  {
    return local_id_to_task_map_;
  }

  // const getter for the vector where each entry `k` stores the index in the
  // topological sort (`spls_`) after which the data for cell `spls_[k]`
  // is no longer needed by any local successor.
  const std::vector<size_t>& GetLocalCellLastUseIndices() const 
  { 
    return local_cell_last_use_index_; 
  }

  // const getter for the release schedule map
  // The key is the index in the topological sort, and the value is the list of
  // cell local IDs to release after that index
  const std::unordered_map<size_t, std::vector<uint64_t>>& GetReleaseSchedule() 
  const 
  { 
    return release_schedule_; 
  }

  // Returns the calculated maximum number of cells concurrently active during the sweep
  // for this specific sweep ordering.
  size_t GetMaxActiveCells() const { return max_active_cells_; }

protected:
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;

  // Map from cell local ID to task index
  std::unordered_map<uint64_t, size_t> local_id_to_task_map_;

  /// Stores the index in the topological sort (`spls_`) which represents
  /// the last time the data associated with cell `spls_[k]` (where k is the
  /// index into this vector) is needed by a local successor cell.
  std::vector<size_t> local_cell_last_use_index_;

  // Maps an index `k` in the topological sort `spls_` to a list 
  // of cell local IDs whose data can be released AFTER processing
  // the cell at index `k`
  std::unordered_map<size_t, std::vector<uint64_t>> release_schedule_;

  // Stores the calculated maximum number of cells concurrently active during 
  // the sweep.
  size_t max_active_cells_ = 0;
};

} // namespace opensn

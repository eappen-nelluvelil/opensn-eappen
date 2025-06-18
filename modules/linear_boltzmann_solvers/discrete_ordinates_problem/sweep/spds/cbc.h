// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <queue>

namespace opensn
{

class CBC_SPDS : public SPDS
{
public:
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum> grid, bool allow_cycles);

  // Returns the cell-by-cell task list.
  const std::vector<Task>& GetTaskList() const;

  size_t GetMaxWavefrontSize() const { return max_wavefront_size_; }

protected:
  // Cell-by-cell task list.
  std::vector<Task> task_list_;

  // Maximum number of concurrently "live" psi blocks required
  // (number of cells in the largest wavefront)
  size_t max_wavefront_size_ = 0;
};

} // namespace opensn

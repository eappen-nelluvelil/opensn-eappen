// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace opensn
{

class CBC_SPDS : public SPDS
{
public:
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  const std::vector<Task>& GetTaskList() const noexcept;

  void ComputeMaxNumLocalPsiSlots();

  std::size_t GetMaxNumLocalPsiSlots() const noexcept { return max_num_local_psi_slots_; }

  const std::vector<std::uint32_t>& GetTaskSlotIDs() const noexcept { return task_slot_ids_; }

  ~CBC_SPDS() override = default;

private:
  void BuildTaskGraph();

  std::vector<std::uint32_t> topo_order_;
  std::vector<Task> task_list_;
  std::vector<std::uint32_t> task_slot_ids_;
  std::size_t max_num_local_psi_slots_ = 0;
};

} // namespace opensn

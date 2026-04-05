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

struct CBCDeviceTaskGraph
{
  const std::uint32_t* reference_ids = nullptr;
  const std::uint32_t* successor_offsets = nullptr;
  const std::uint32_t* successors = nullptr;
  const std::uint32_t* predecessor_offsets = nullptr;
  const std::uint32_t* predecessors = nullptr;
  const int* initial_dependencies = nullptr;
  const std::uint32_t* initial_successors_to_retire = nullptr;
  std::uint32_t num_tasks = 0;
};

class CBC_SPDS : public SPDS
{
public:
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  const std::vector<Task>& GetTaskList() const noexcept;

  void ComputeMaxNumLocalPsiSlots();

  std::size_t GetMaxNumLocalPsiSlots() const noexcept { return max_num_local_psi_slots_; }
  std::size_t GetNumStaticLocalPsiSlots() const noexcept { return num_static_local_psi_slots_; }

  const std::vector<std::uint32_t>& GetTaskSlotIDs() const noexcept { return task_slot_ids_; }

  void CopyTaskGraphDataOnDevice() const;
  void FreeDeviceData() const;

  const CBCDeviceTaskGraph& GetDeviceTaskGraph() const noexcept { return device_task_graph_; }

  ~CBC_SPDS() override;

private:
  void BuildTaskGraph();

  std::vector<std::uint32_t> topo_order_;
  std::vector<std::uint32_t> local_successor_offsets_;
  std::vector<std::uint32_t> local_successors_;
  std::vector<std::uint32_t> initial_successors_to_retire_;
  std::vector<Task> task_list_;
  std::vector<std::uint32_t> task_slot_ids_;
  std::size_t max_num_local_psi_slots_ = 0;
  std::size_t num_static_local_psi_slots_ = 0;
  mutable CBCDeviceTaskGraph device_task_graph_;
};

} // namespace opensn

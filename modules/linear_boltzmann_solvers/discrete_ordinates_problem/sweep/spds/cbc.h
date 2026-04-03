// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <cstddef>
#include <cstdint>

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

  /// Compute the exact maximum number of pool slots required for local CBC angular flux storage.
  void ComputeMaxNumLocalPsiSlots();

  /// Return the exact maximum number of pool slots required for local CBC angular flux storage.
  std::size_t GetMaxNumLocalPsiSlots() const noexcept { return max_num_local_psi_slots_; }

  /// Copy immutable CBC task-graph data to device memory on demand.
  void CopyTaskGraphDataOnDevice() const;

  /// Release immutable CBC task-graph device data.
  void FreeDeviceData() const;

  /// Return the device-visible CBC task-graph view.
  const CBCDeviceTaskGraph& GetDeviceTaskGraph() const noexcept { return device_task_graph_; }

  ~CBC_SPDS() override;

private:
  void BuildTaskGraph();

  /// Topological ordering of the local task graph using local cell ids.
  std::vector<std::uint32_t> topo_order_;
  /// CSR row offsets for the local successor graph used by exact slot counting.
  std::vector<std::uint32_t> local_successor_offsets_;
  /// CSR column indices for the local successor graph used by exact slot counting.
  std::vector<std::uint32_t> local_successors_;
  /// Initial successor-retirement countdown per task.
  std::vector<std::uint32_t> initial_successors_to_retire_;
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;
  /// Exact maximum number of local CBC pool allocator slots.
  std::size_t max_num_local_psi_slots_ = 0;
  /// Immutable CBC task-graph data mirrored on device for CBCD-specific kernels.
  mutable CBCDeviceTaskGraph device_task_graph_;
};

} // namespace opensn

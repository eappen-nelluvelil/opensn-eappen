// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace opensn
{

/// CBC sweep-plane data structure.
class CBC_SPDS : public SPDS
{
public:
  CBC_SPDS(int id,
           const Vector3& omega,
           const std::shared_ptr<MeshContinuum>& grid,
           bool allow_cycles);

  int GetId() const noexcept { return id_; }

  const std::vector<Task>& GetTaskList() const;

  std::vector<int> GetGlobalSweepFAS() const { return global_sweep_fas_; }

  void SetGlobalSweepFAS(std::vector<int>& edges) { global_sweep_fas_ = edges; }

  void BuildGlobalSweepFAS();

  void ApplyGlobalSweepFAS();

  std::vector<double> ComputeLocalLocationEdgeWeights() const;

  void SetGlobalEdgeWeights(std::vector<double>& weights)
  {
    global_edge_weights_ = std::move(weights);
  }

  bool IsDelayedLocalDependency(std::uint32_t upwind_local_id,
                                std::uint32_t downwind_local_id) const noexcept;

protected:
  void BuildTaskList();

  int id_ = 0;
  bool allow_cycles_ = false;
  std::vector<Task> task_list_;
  std::vector<std::vector<int>> global_dependencies_;
  std::vector<int> global_sweep_fas_;
  std::vector<double> global_edge_weights_;
  std::set<std::uint64_t> delayed_local_dependency_set_;
};

} // namespace opensn

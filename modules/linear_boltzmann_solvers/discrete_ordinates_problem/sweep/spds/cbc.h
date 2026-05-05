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
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  int GetId() const noexcept { return id_; }

  void SetId(int id) noexcept { id_ = id; }

  const std::vector<Task>& GetTaskList() const;

  std::vector<int> GetGlobalSweepFAS() const { return global_sweep_fas_; }

  void SetGlobalSweepFAS(std::vector<int>& edges) { global_sweep_fas_ = edges; }

  void BuildGlobalSweepFAS();

  void BuildGlobalSweepTDG();

  std::vector<double> ComputeLocalLocationEdgeWeights() const;

  void SetGlobalEdgeWeights(std::vector<double>& weights)
  {
    global_edge_weights_ = std::move(weights);
  }

  bool IsDelayedLocalDependency(std::uint32_t upwind_local_id,
                                std::uint32_t downwind_local_id) const noexcept;

  bool IsDelayedLocationDependency(int location_id) const noexcept;

protected:
  void BuildTaskList();

  int id_ = 0;
  bool allow_cycles_ = false;
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;
  /// Location-to-location dependencies on all MPI ranks.
  std::vector<std::vector<int>> global_dependencies_;
  /// Packed global feedback arc set used to lag cyclic MPI dependencies.
  std::vector<int> global_sweep_fas_;
  /// Flattened comm_size x comm_size global edge weights.
  std::vector<double> global_edge_weights_;
  /// Local feedback arc set encoded as packed `(upwind, downwind)` local cell IDs.
  std::set<std::uint64_t> delayed_local_dependency_set_;
  /// MPI-rank-indexed delayed location dependency flags.
  std::vector<unsigned char> delayed_location_dependency_flags_;
};

} // namespace opensn

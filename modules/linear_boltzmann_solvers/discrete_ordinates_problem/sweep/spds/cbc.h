// // SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// // SPDX-License-Identifier: MIT

// #pragma once

// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
// #include <unordered_set>
// #include <utility>

// namespace opensn
// {

// class CBC_SPDS : public SPDS
// {
// public:
//   /**
//    * Constructs a cell-by-cell sweep-plane data strcture (SPDS) with the given direction and grid.
//    *
//    * \param omega The angular direction vector.
//    * \param grid Reference to the grid.
//    * \param allow_cycles Whether cycles are allowed in the local sweep dependency graph.
//    */
//   CBC_SPDS(int id, const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

//   bool GetID() const noexcept { return id_; }

//   /// Returns the cell-by-cell task list.
//   const std::vector<Task>& GetTaskList() const noexcept { return task_list_; };

//   std::vector<double> ComputeLocalLocationEdgeWeights() const;
//   void SetGlobalEdgeWeights(std::vector<double>&& weights) { global_edge_weights_ = std::move(weights); }

//   void BuildGlobalSweepFAS();
//   const std::vector<int>& GetGlobalSweepFAS() const noexcept { return global_sweep_fas_; }
//   void SetGlobalSweepFAS(const std::vector<int>& edges) { global_sweep_fas_ = edges; }
//   void BuildGlobalSweepTDG();

//   bool IsDelayedLocalDependency(std::uint32_t upwind_local_id, std::uint32_t downwind_local_id) const noexcept;

// private:
//   static std::uint64_t PackEdge(std::uint32_t upwind_local_id, std::uint32_t downwind_local_id) noexcept
//   {
//     return (static_cast<std::uint64_t>(upwind_local_id) << 32) | static_cast<std::uint64_t>(downwind_local_id);
//   }

//   void BuildTaskList();

//   int id_;
//   bool allow_cycles_;
//   std::vector<std::vector<int>> global_dependencies_;
//   std::vector<int> global_sweep_fas_;
//   std::vector<double> global_edge_weights_;
//   std::unordered_set<std::uint64_t> delayed_local_dependency_set_;
//   /// Cell-by-cell task list.
//   std::vector<Task> task_list_;
// };

// } // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <unordered_set>
#include <utility>

namespace opensn
{

class CBC_SPDS : public SPDS
{
  
public:
  CBC_SPDS(int id,
            const Vector3& omega,
            const std::shared_ptr<MeshContinuum>& grid,
            bool allow_cycles);

  int GetId() const noexcept { return id_; }

  const std::vector<Task>& GetTaskList() const noexcept { return task_list_; }

  std::vector<double> ComputeLocalLocationEdgeWeights() const;
  void SetGlobalEdgeWeights(std::vector<double>&& weights) { global_edge_weights_ = std::move(weights); }

  void BuildGlobalSweepFAS();
  const std::vector<int>& GetGlobalSweepFAS() const noexcept { return global_sweep_fas_; }
  void SetGlobalSweepFAS(const std::vector<int>& edges) { global_sweep_fas_ = edges; }
  void BuildGlobalSweepTDG();

  bool IsDelayedLocalDependency(std::uint32_t upwind_local_id,
                                std::uint32_t downwind_local_id) const noexcept;

private:
  static std::uint64_t PackEdge(std::uint32_t upwind_local_id, std::uint32_t downwind_local_id) noexcept
  {
    return (static_cast<std::uint64_t>(upwind_local_id) << 32) |
            static_cast<std::uint64_t>(downwind_local_id);
  }

  void BuildTaskList();

private:
  int id_ = 0;
  bool allow_cycles_ = false;
  std::vector<std::vector<int>> global_dependencies_;
  std::vector<int> global_sweep_fas_;
  std::vector<double> global_edge_weights_;
  std::unordered_set<std::uint64_t> delayed_local_dependency_set_;
  std::vector<Task> task_list_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <set>

namespace opensn
{

class CBC_SPDS : public SPDS
{
public:
  /**
   * Constructs a cell-by-cell sweep-plane data structure (SPDS) with the given direction and grid.
   *
   * \param omega The angular direction vector.
   * \param grid Reference to the grid.
   * \param allow_cycles Whether cycles are allowed in the local and global sweep dependency graphs.
   */
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  /// Returns the cell-by-cell task list.
  const std::vector<Task>& GetTaskList() const;

  /// Returns the id of this SPDS.
  int GetId() const { return id_; }

  /// Sets the id of this SPDS.
  void SetId(int id) { id_ = id; }

  /// Returns the global sweep FAS as a vector of edges.
  std::vector<int> GetGlobalSweepFAS() { return global_sweep_fas_; }

  /// Sets the global sweep FAS.
  void SetGlobalSweepFAS(std::vector<int>& edges) { global_sweep_fas_ = edges; }

  /// Builds the Feedback Arc Set (FAS) for the global sweep.
  void BuildGlobalSweepFAS();

  /// Builds the Task Dependency Graph (TDG) for the global sweep.
  void BuildGlobalSweepTDG();

  /// Returns the locally accumulated location-to-location edge weights.
  std::vector<double> ComputeLocalLocationEdgeWeights() const;

  /// Sets the global location-to-location edge weights.
  void SetGlobalEdgeWeights(std::vector<double>& weights)
  {
    global_edge_weights_ = std::move(weights);
  }

  /// Returns true if the given local face is a delayed local (FAS) edge on the incoming side.
  bool IsDelayedLocalIncomingFace(uint32_t cell_local_id, uint32_t face_idx) const
  {
    return delayed_local_incoming_faces_.count({cell_local_id, face_idx}) > 0;
  }

  /// Returns true if the given local face is a delayed local (FAS) edge on the outgoing side.
  bool IsDelayedLocalOutgoingFace(uint32_t cell_local_id, uint32_t face_idx) const
  {
    return delayed_local_outgoing_faces_.count({cell_local_id, face_idx}) > 0;
  }

  /// Returns true if the given non-local face is from a delayed location dependency.
  bool IsDelayedNonlocalIncomingFace(uint32_t cell_local_id, uint32_t face_idx) const
  {
    return delayed_nonlocal_incoming_faces_.count({cell_local_id, face_idx}) > 0;
  }

  /// Returns the set of delayed local incoming faces.
  const std::set<std::pair<uint32_t, uint32_t>>& GetDelayedLocalIncomingFaces() const
  {
    return delayed_local_incoming_faces_;
  }

  /// Returns the set of delayed nonlocal incoming faces.
  const std::set<std::pair<uint32_t, uint32_t>>& GetDelayedNonlocalIncomingFaces() const
  {
    return delayed_nonlocal_incoming_faces_;
  }

protected:
  /// Unique identifier for this SPDS.
  int id_ = 0;
  /// Flag indicating whether cycles are allowed in the dependency graphs.
  bool allow_cycles_;
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;
  /// Location-to-location global sweep dependencies.
  std::vector<std::vector<int>> global_dependencies_;
  /// Vector of edges representing the FAS used to break cycles in the global sweep graph.
  std::vector<int> global_sweep_fas_;
  /// Flattened comm_size x comm_size matrix of global edge weights.
  std::vector<double> global_edge_weights_;

  /// Builds the cell-by-cell task list, accounting for delayed edges.
  void BuildTaskList();

  /// Set of (cell_local_id, face_idx) pairs for delayed local incoming faces (FAS edges).
  std::set<std::pair<uint32_t, uint32_t>> delayed_local_incoming_faces_;
  /// Set of (cell_local_id, face_idx) pairs for delayed local outgoing faces (FAS edges).
  std::set<std::pair<uint32_t, uint32_t>> delayed_local_outgoing_faces_;
  /// Set of (cell_local_id, face_idx) pairs for non-local incoming faces from delayed locations.
  std::set<std::pair<uint32_t, uint32_t>> delayed_nonlocal_incoming_faces_;
};

} // namespace opensn

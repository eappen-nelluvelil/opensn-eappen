// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"

namespace opensn
{
class CellMapping;

class CBCSweepChunk : public SweepChunk
{
public:
  CBCSweepChunk(std::vector<double>& destination_phi,
                std::vector<double>& destination_psi,
                const std::shared_ptr<MeshContinuum> grid,
                const SpatialDiscretization& discretization,
                const std::vector<UnitCellMatrices>& unit_cell_matrices,
                std::vector<CellLBSView>& cell_transport_views,
                const std::vector<double>& densities,
                const std::vector<double>& source_moments,
                const LBSGroupset& groupset,
                const std::map<int, std::shared_ptr<MultiGroupXS>>& xs,
                int num_moments,
                int max_num_cell_dofs);

  void SetAngleSet(AngleSet& angle_set) override;

  void SetCell(Cell const* cell_ptr, AngleSet& angle_set) override;

  // psi_out_block: pointer to pre-allocated memory block where outgoing angular
  // for the current cell should be stored
  void Sweep(AngleSet& angle_set, double* psi_out_block);

  void Sweep(AngleSet& angle_set) override {}

private:
  CBC_FLUDS* fluds_; // Pointer to the CBC_FLUDS for the current AngleSet.
  size_t gs_size_;   // Number of energy groups in the parent LBSGroupset.
  int gs_gi_;        // Global starting group index of the parent LBSGroupset.

  // --- Strides for accessing local cell angular fluxes in `CBC_FLUDS` ---

  // Number of angular directions managed by the current AngleSet.
  size_t num_angles_in_set_local_;

  // Stride to jump between data for different angles (local to AngleSet)
  // for the same spatial DOF
  // Equal to `gs_size_` (number of groups).
  size_t local_compact_angle_stride_;

  // Stride to jump between data for different spatial DOFs (nodes)
  // Equal to `num_angles_in_set_local_ * gs_size_`.
  size_t local_compact_node_stride_;

  // --- Strides for MPI communication buffers (e.g., `psi_dnwnd_data` for outgoing) ---

  // These buffers are structured to contain data for all angles in THIS AngleSet,
  // for all nodes on a particular face.
  // Layout: face spatial DOF major -> angle in set major -> group major.

  // Number of angular directions in the current AngleSet (same as local, for clarity).
  size_t num_angles_in_set_remote_;

  // Stride between angles within a single face-node's data block in an MPI buffer.
  // Equal to `gs_size_`.
  size_t remote_angle_stride_;

  // Stride between different face-nodes' data blocks in an MPI buffer.
  // Equal to `num_angles_in_set_remote_ * gs_size_`.
  size_t remote_node_stride_;

  bool surface_source_active_;

  const Cell* cell_;
  uint64_t cell_local_id_;
  const CellMapping* cell_mapping_;
  CellLBSView* cell_transport_view_;
  size_t cell_num_faces_;
  size_t cell_num_nodes_;

  DenseMatrix<Vector3> G_;
  DenseMatrix<double> M_;
  std::vector<DenseMatrix<double>> M_surf_;
  std::vector<Vector<double>> IntS_shapeI_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep_chunks/sweep_chunk.h"

namespace opensn
{
class CellMapping;

class CbcSweepChunk : public SweepChunk
{
public:
  CbcSweepChunk(std::vector<double>& destination_phi,
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

  void Sweep(AngleSet& angle_set) override;

private:
  CBC_FLUDS* fluds_;
  size_t gs_size_;
  int gs_gi_;

  // --- NEW:
  // Strides based on the compact data layout for the current AngleSet
  // size_t num_angles_in_set_;    // Number of angles in the current AngleSet
  // size_t compact_angle_stride_; // Stride to get to next angle ( = gs_size_)
  // size_t compact_node_stride_;  // Stride to get to next node ( = num_angles_in_set_ * gs_size_)

  // --- FIX? For local/remote buffer sizing issues
  // --- Strides for COMPACT local_psi_data_ ---
  size_t num_angles_in_set_local_;    // Angles in this AngleSet (used for local_psi_data_ layout)
  size_t local_compact_angle_stride_; // = gs_size_
  size_t local_compact_node_stride_;  // = num_angles_in_set_local_ * gs_size_

  // // --- Strides for REMOTE communication buffer layout ---
  size_t num_angles_in_set_remote_; // Angles in this AngleSet (used for remote buffer layout)
  size_t remote_angle_stride_; // = gs_size_ (stride between angles within a node's block in remote
                               // buffer)
  size_t remote_node_stride_;  // = num_angles_in_set_remote_ * gs_size_ (stride between nodes in
                               // remote buffer)

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

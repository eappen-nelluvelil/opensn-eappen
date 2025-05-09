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
  size_t gs_size_; // Number of groups in the LBSGroupset
  int gs_gi_;

  // Strides for accessing the COMPACT local_psi_data_ in CBC_FLUDS
  size_t num_angles_in_set_local_; // Number of angles in THIS AngleSet
  size_t
    local_compact_angle_stride_; // Stride between angles within a spatial DOF's block (= gs_size_)
  size_t local_compact_node_stride_; // Stride between spatial DOFs' blocks (=
                                     // num_angles_in_set_local_ * gs_size_)

  // Strides for defining the layout and accessing REMOTE communication buffers (psi_dnwnd_data)
  // This buffer contains data for all angles in THIS AngleSet for all nodes on a face.
  size_t
    num_angles_in_set_remote_; // Number of angles in THIS AngleSet (same as local, for clarity)
  size_t remote_angle_stride_; // Stride between angles within a face-node's block in the remote
                               // buffer (= gs_size_)
  size_t remote_node_stride_;  // Stride between face-nodes' blocks in the remote buffer (=
                               // num_angles_in_set_remote_ * gs_size_)

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

// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"

namespace opensn
{
class CellMapping;
class DiscreteOrdinatesProblem;

/**
 * Time-dependent variant of the CBC sweep chunk.
 *
 * Extends CBCSweepChunk with time-dependent terms for transient transport:
 *   - Adds tau = 1/(v * theta * dt) to the total cross-section (time absorption)
 *   - Adds tau * psi_old to the RHS source term
 *   - Applies theta-method reconstruction when saving angular flux:
 *     psi_new = (1/theta) * (psi_sol + (theta - 1) * psi_old)
 */
class CBCSweepChunkTD : public SweepChunk
{
public:
  CBCSweepChunkTD(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  void SetAngleSet(AngleSet& angle_set) override;

  void SetCell(Cell const* cell_ptr, AngleSet& angle_set) override;

  void Sweep(AngleSet& angle_set) override;

  bool IsTimeDependent() const override { return true; }

private:
  DiscreteOrdinatesProblem& problem_;
  const std::vector<double>& psi_old_;

  CBC_FLUDS* fluds_;
  size_t gs_size_;
  unsigned int gs_gi_;
  size_t num_angles_in_as_;
  unsigned int group_stride_;
  size_t group_angle_stride_;
  bool surface_source_active_;

  const Cell* cell_;
  std::uint32_t cell_local_id_;
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

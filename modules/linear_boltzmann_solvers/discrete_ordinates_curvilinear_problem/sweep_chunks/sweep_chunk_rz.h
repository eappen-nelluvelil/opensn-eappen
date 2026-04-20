// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include <memory>
#include <vector>

namespace opensn
{

class CurvilinearProductQuadrature;
class DiscreteOrdinatesProblem;
class LBSGroupset;

/// Shared curvilinear RZ sweep-chunk state for host AAH and CBC implementations.
class SweepChunkRZ : public SweepChunk
{
public:
  SweepChunkRZ(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

protected:
  unsigned int PolarLevel(unsigned int direction_num) const noexcept
  {
    return polar_level_of_direction_[direction_num];
  }

  /// Secondary spatial discretization cell matrices.
  const std::vector<UnitCellMatrices>& secondary_unit_cell_matrices_;
  /// Validated curvilinear product quadrature used by this chunk.
  std::shared_ptr<CurvilinearProductQuadrature> curvilinear_product_quadrature_;
  /// Unknown manager for quantities that depend on polar level.
  UnknownManager unknown_manager_;
  /// Sweeping dependency angular intensity.
  std::vector<double> psi_sweep_;
  /// Dense direction-to-polar-level lookup.
  std::vector<unsigned int> polar_level_of_direction_;
  /// Normal vector used to detect the symmetric boundary.
  Vector3 normal_vector_boundary_;
  /// Reusable cell-system matrix scratch.
  DenseMatrix<double> Amat_;
  /// Reusable cell-system matrix scratch.
  DenseMatrix<double> Atemp_;
  /// Reusable group right-hand-side scratch.
  std::vector<Vector<double>> b_;
  /// Reusable source scratch.
  std::vector<double> source_;
};

} // namespace opensn

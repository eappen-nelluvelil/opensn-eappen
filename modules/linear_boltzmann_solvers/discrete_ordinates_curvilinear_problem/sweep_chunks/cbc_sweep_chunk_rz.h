// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include <map>
#include <vector>

namespace opensn
{

class LBSGroupset;
class DiscreteOrdinatesProblem;

/// A CBC sweep-chunk in axial-symmetric cylindrical coordinates.
class CBCSweepChunkRZ : public SweepChunk
{
public:
  CBCSweepChunkRZ(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  void SetAngleSet(AngleSet& angle_set) override;
  void SetCell(const Cell* cell_ptr, AngleSet& angle_set) override;
  void Sweep(AngleSet& angle_set) override;

private:
  /// Secondary spatial discretization cell matrices.
  const std::vector<UnitCellMatrices>& secondary_unit_cell_matrices_;
  /// Unknown manager for quantities that depend on polar level.
  UnknownManager unknown_manager_;
  /// Sweeping dependency angular intensity.
  std::vector<double> psi_sweep_;
  /// Mapping from direction linear index to direction polar level.
  std::map<unsigned int, unsigned int> map_polar_level_;
  /// Normal vector used to detect the symmetric boundary.
  Vector3 normal_vector_boundary_;
  /// Current cell selected by the CBC scheduler.
  const Cell* current_cell_ = nullptr;
};

} // namespace opensn

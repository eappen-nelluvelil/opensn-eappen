// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_curvilinear_problem/sweep_chunks/sweep_chunk_rz.h"

namespace opensn
{

class LBSGroupset;
class DiscreteOrdinatesProblem;

/// A CBC sweep-chunk in axial-symmetric cylindrical coordinates.
class CBCSweepChunkRZ : public SweepChunkRZ
{
public:
  CBCSweepChunkRZ(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  void SetAngleSet(AngleSet& angle_set) override;
  void SetCell(const Cell* cell_ptr, AngleSet& angle_set) override;
  void Sweep(AngleSet& angle_set) override;

private:
  /// Current cell selected by the CBC scheduler.
  const Cell* current_cell_ = nullptr;
};

} // namespace opensn

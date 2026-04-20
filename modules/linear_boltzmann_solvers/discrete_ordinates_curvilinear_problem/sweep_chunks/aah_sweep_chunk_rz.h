// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_curvilinear_problem/sweep_chunks/sweep_chunk_rz.h"

namespace opensn
{

class LBSGroupset;
class DiscreteOrdinatesProblem;

/// A sweep-chunk in point-symmetric and axial-symmetric curvilinear coordinates.
class AAHSweepChunkRZ : public SweepChunkRZ
{
public:
  AAHSweepChunkRZ(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  void Sweep(AngleSet& angle_set) override;
};

} // namespace opensn

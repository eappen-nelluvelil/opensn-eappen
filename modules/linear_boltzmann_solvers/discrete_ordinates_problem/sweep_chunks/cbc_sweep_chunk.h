// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"

namespace opensn
{
class CellMapping;
class DiscreteOrdinatesProblem;

class CBCSweepChunk : public SweepChunk
{
public:
  CBCSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  void SetAngleSet(AngleSet& angle_set) override;

  void SetCell(Cell const* cell_ptr, AngleSet& angle_set) override;

  void Sweep(AngleSet& angle_set) override;

protected:
  DiscreteOrdinatesProblem& problem_;
  unsigned int group_block_size_;
  CBC_FLUDS* fluds_ = nullptr;
  const Cell* cell_ = nullptr;

private:
  using SweepFunc = void (CBCSweepChunk::*)(AngleSet&);
  SweepFunc sweep_impl_ = nullptr;

  void Sweep_Generic(AngleSet& angle_set);
  template <unsigned int NumNodes>
  void Sweep_FixedN(AngleSet& angle_set);
};

} // namespace opensn

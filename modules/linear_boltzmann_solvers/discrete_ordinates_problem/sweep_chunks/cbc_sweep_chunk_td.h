// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"

namespace opensn
{
class CellMapping;
class DiscreteOrdinatesProblem;

class CBCSweepChunkTD : public SweepChunk
{
public:
  CBCSweepChunkTD(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  ~CBCSweepChunkTD() override;

  void SetAngleSet(AngleSet& angle_set) override;

  void SetCell(Cell const* cell_ptr, AngleSet& angle_set) override;

  void Sweep(AngleSet& angle_set) override;

  bool IsTimeDependent() const override { return true; }

protected:
  void Sweep_Generic(AngleSet& angle_set);
  template <int NumNodes>
  void Sweep_FixedN(AngleSet& angle_set);

  DiscreteOrdinatesProblem& problem_;
  const std::vector<double>& psi_old_;
  unsigned int group_block_size_ = 0;
  bool use_fixed_n_ = false;
  unsigned int fixed_num_nodes_ = 0;
  CBC_FLUDS* fluds_ = nullptr;
  const Cell* cell_ = nullptr;
};

} // namespace opensn

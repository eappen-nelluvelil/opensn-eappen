// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/avx_sweep_chunk_utils.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk_shared.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"

namespace opensn
{

/// Time-dependent sweep chunk for the host cell-by-cell sweep algorithm.
class CBCSweepChunkTD : public SweepChunk
{
public:
  /// Construct a time-dependent host CBC sweep chunk.
  CBCSweepChunkTD(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);
  ~CBCSweepChunkTD() override = default;

  /// Set the current angle set.
  void SetAngleSet(AngleSet& angle_set) override;

  /// Set the current cell to be swept.
  void SetCell(const Cell* cell_ptr, AngleSet& angle_set) override;

  /// Sweep the currently bound cell for all angles and groups in the active angle set.
  void Sweep(AngleSet& angle_set) override;

  /// Return whether this sweep chunk includes transient terms.
  bool IsTimeDependent() const override { return true; }

protected:
  using SweepFunc = void (CBCSweepChunkTD::*)(AngleSet&);

  /// Sweep using the generic dense-kernel path.
  void Sweep_Generic(AngleSet& angle_set);

  /// Sweep using a fixed-node-count dense-kernel path.
  template <unsigned int NumNodes>
  void Sweep_FixedN(AngleSet& angle_set);

  /// Owning discrete ordinates problem.
  DiscreteOrdinatesProblem& problem_;

  /// Previous-step angular flux vector.
  const std::vector<double>& psi_old_;

  /// Number of groups solved in one block.
  unsigned int group_block_size_ = 0;

  /// Reusable CBC sweep context.
  CBCSweepChunkContext ctx_;

private:
  /// Selected time-dependent sweep implementation.
  SweepFunc sweep_impl_td_ = nullptr;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/unknown_manager/unknown_manager.h"
#include <memory>
#include <vector>

namespace opensn
{

class DiscreteOrdinatesProblem;
class PieceWiseLinearContinuous;

/**
 * Consistent edge-based DSA for steady-state diamond-difference host CBC slabs.
 *
 * Supports one isotropic classic Richardson groupset with self-scattering and
 * adjacent downscatter, prescribed incoming boundaries, and no delayed angular
 * DOFs. Transport sources use the previous iterate in every group. Corrections
 * are solved in increasing group order with shared edge DOFs. Each cell uses
 * the diamond-consistent h/4 mass matrix, not the usual PWLC mass matrix.
 * Construction, Apply, and destruction are collective over opensn::mpi_comm.
 * The problem must outlive this accelerator. Owned PETSc matrices, vectors,
 * solvers, and per-group communicators are released before MPI finalization.
 */
class DiamondDifferenceDSA
{
public:
  explicit DiamondDifferenceDSA(DiscreteOrdinatesProblem& problem);
  ~DiamondDifferenceDSA();

  DiamondDifferenceDSA(const DiamondDifferenceDSA&) = delete;
  DiamondDifferenceDSA& operator=(const DiamondDifferenceDSA&) = delete;

  /// Add the scalar-flux correction after an exact diamond-difference CBC sweep.
  void Apply();

private:
  struct GroupSystem;
  DiscreteOrdinatesProblem& problem_;
  std::shared_ptr<PieceWiseLinearContinuous> edges_;
  UnknownManager unknowns_;
  std::vector<std::unique_ptr<GroupSystem>> systems_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_curvilinear_problem/discrete_ordinates_curvilinear_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_curvilinear_problem/sweep_chunks/sweep_chunk_rz.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/math/quadratures/angular/curvilinear_product_quadrature.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include <stdexcept>

namespace opensn
{

SweepChunkRZ::SweepChunkRZ(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset)
  : SweepChunk(problem.GetPhiNewLocal(),
               problem.GetPsiNewLocal()[groupset.id],
               problem.GetGrid(),
               problem.GetSpatialDiscretization(),
               problem.GetUnitCellMatrices(),
               problem.GetCellTransportViews(),
               problem.GetQMomentsLocal(),
               groupset,
               problem.GetBlockID2XSMap(),
               problem.GetNumMoments(),
               problem.GetMaxCellDOFCount(),
               problem.GetMinCellDOFCount()),
    secondary_unit_cell_matrices_(dynamic_cast<const DiscreteOrdinatesCurvilinearProblem&>(problem)
                                    .GetSecondaryUnitCellMatrices()),
    curvilinear_product_quadrature_(
      std::dynamic_pointer_cast<CurvilinearProductQuadrature>(groupset_.quadrature)),
    unknown_manager_(),
    psi_sweep_(),
    polar_level_of_direction_(groupset_.quadrature->omegas.size(), 0),
    normal_vector_boundary_(),
    Amat_(max_num_cell_dofs_, max_num_cell_dofs_),
    Atemp_(max_num_cell_dofs_, max_num_cell_dofs_),
    b_(groupset_.GetNumGroups(), Vector<double>(max_num_cell_dofs_)),
    source_(max_num_cell_dofs_)
{
  if (curvilinear_product_quadrature_ == nullptr)
    throw std::invalid_argument("SweepChunkRZ: invalid angular quadrature");

  const size_t dir_map_size = curvilinear_product_quadrature_->GetDirectionMap().size();
  for (size_t m = 0; m < dir_map_size; ++m)
    unknown_manager_.AddUnknown(UnknownType::VECTOR_N, groupset_.GetNumGroups());

  psi_sweep_.resize(discretization_.GetNumLocalDOFs(unknown_manager_));

  for (const auto& dir_set : curvilinear_product_quadrature_->GetDirectionMap())
    for (const auto& dir_idx : dir_set.second)
      polar_level_of_direction_[dir_idx] = dir_set.first;

  const auto d = (grid_->GetDimension() == 1) ? 2 : 0;
  normal_vector_boundary_ = Vector3(0.0, 0.0, 0.0);
  normal_vector_boundary_(d) = 1.0;
}

} // namespace opensn

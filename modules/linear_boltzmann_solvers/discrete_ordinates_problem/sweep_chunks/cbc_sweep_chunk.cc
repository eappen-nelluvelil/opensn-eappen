// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_kernels.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mesh/cell/cell.h"
#include "framework/logging/log.h"
#include "caliper/cali.h"

namespace opensn
{

CBCSweepChunk::CBCSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset)
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
    problem_(problem),
    sweep_impl_(&CBCSweepChunk::Sweep_Generic)
{
  if (min_num_cell_dofs_ == max_num_cell_dofs_ and min_num_cell_dofs_ >= 2 and
      min_num_cell_dofs_ <= 8)
  {
    switch (min_num_cell_dofs_)
    {
      case 2:
        sweep_impl_ = &CBCSweepChunk::Sweep_FixedN<2>;
        break;
      case 3:
        sweep_impl_ = &CBCSweepChunk::Sweep_FixedN<3>;
        break;
      case 4:
        sweep_impl_ = &CBCSweepChunk::Sweep_FixedN<4>;
        break;
      case 5:
        sweep_impl_ = &CBCSweepChunk::Sweep_FixedN<5>;
        break;
      case 6:
        sweep_impl_ = &CBCSweepChunk::Sweep_FixedN<6>;
        break;
      case 7:
        sweep_impl_ = &CBCSweepChunk::Sweep_FixedN<7>;
        break;
      case 8:
        sweep_impl_ = &CBCSweepChunk::Sweep_FixedN<8>;
        break;
      default:
        break;
    }
  }
  group_block_size_ = ComputeGroupBlockSize(groupset_.GetNumGroups());
}

void
CBCSweepChunk::SetAngleSet(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunk::SetAngleSet");
  fluds_ = &dynamic_cast<CBC_FLUDS&>(angle_set.GetFLUDS());

  // Build persistent sweep data (only cell pointer changes per-cell)
  sweep_data_ = CBCSweepData{&discretization_,
                              &unit_cell_matrices_,
                              &cell_transport_views_,
                              &source_moments_,
                              &groupset_,
                              &xs_,
                              num_moments_,
                              max_num_cell_dofs_,
                              SaveAngularFluxEnabled(),
                              groupset_angle_group_stride_,
                              groupset_group_stride_,
                              &destination_phi_,
                              &destination_psi_,
                              surface_source_active_,
                              include_rhs_time_term_,
                              nullptr,
                              nullptr,
                              group_block_size_,
                              fluds_,
                              nullptr};
}

void
CBCSweepChunk::SetCell(const Cell* cell_ptr, AngleSet& angle_set)
{
  cell_ = cell_ptr;
}

void
CBCSweepChunk::Sweep(AngleSet& angle_set)
{
  (this->*sweep_impl_)(angle_set);
}

void
CBCSweepChunk::Sweep_Generic(AngleSet& angle_set)
{
  sweep_data_.cell = cell_;
  CBC_Sweep_CellKernel_Generic<false>(sweep_data_, angle_set);
}

} // namespace opensn

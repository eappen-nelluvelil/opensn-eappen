// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk_td.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_kernels.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mesh/cell/cell.h"
#include "framework/logging/log.h"
#include "caliper/cali.h"
#include <stdexcept>

namespace opensn
{

CBCSweepChunkTD::CBCSweepChunkTD(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset)
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
    psi_old_(problem.GetPsiOldLocal()[groupset.id])
{
  if (problem.UseGPUs())
    throw std::runtime_error("Time-dependent calculations do not yet support GPUs.\n");

  if (min_num_cell_dofs_ == max_num_cell_dofs_ and min_num_cell_dofs_ >= 2 and
      min_num_cell_dofs_ <= 8)
  {
    use_fixed_n_ = true;
    fixed_num_nodes_ = min_num_cell_dofs_;
  }

  group_block_size_ = ComputeGroupBlockSize(groupset_.GetNumGroups());
}

void
CBCSweepChunkTD::SetAngleSet(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunkTD::SetAngleSet");
  fluds_ = &dynamic_cast<CBC_FLUDS&>(angle_set.GetFLUDS());
}

void
CBCSweepChunkTD::SetCell(const Cell* cell_ptr, AngleSet& angle_set)
{
  cell_ = cell_ptr;
}

void
CBCSweepChunkTD::Sweep(AngleSet& angle_set)
{
  if (use_fixed_n_)
  {
    switch (fixed_num_nodes_)
    {
      case 2:
        Sweep_FixedN<2>(angle_set);
        break;
      case 3:
        Sweep_FixedN<3>(angle_set);
        break;
      case 4:
        Sweep_FixedN<4>(angle_set);
        break;
      case 5:
        Sweep_FixedN<5>(angle_set);
        break;
      case 6:
        Sweep_FixedN<6>(angle_set);
        break;
      case 7:
        Sweep_FixedN<7>(angle_set);
        break;
      case 8:
        Sweep_FixedN<8>(angle_set);
        break;
      default:
        Sweep_Generic(angle_set);
        break;
    }
  }
  else
  {
    Sweep_Generic(angle_set);
  }
}

void
CBCSweepChunkTD::Sweep_Generic(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunkTD::Sweep");

  CBCSweepData data{discretization_,
                    unit_cell_matrices_,
                    cell_transport_views_,
                    source_moments_,
                    groupset_,
                    xs_,
                    num_moments_,
                    max_num_cell_dofs_,
                    SaveAngularFluxEnabled(),
                    groupset_angle_group_stride_,
                    groupset_group_stride_,
                    destination_phi_,
                    destination_psi_,
                    surface_source_active_,
                    include_rhs_time_term_,
                    &problem_,
                    &psi_old_,
                    group_block_size_,
                    fluds_,
                    cell_};

  CBC_Sweep_CellKernel_Generic<true>(data, angle_set);
}

CBCSweepChunkTD::~CBCSweepChunkTD() = default;

} // namespace opensn

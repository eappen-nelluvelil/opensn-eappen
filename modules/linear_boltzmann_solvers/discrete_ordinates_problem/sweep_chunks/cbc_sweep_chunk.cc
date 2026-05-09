// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"
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
               problem.GetCellOutflowViews(),
               problem.GetQMomentsLocal(),
               groupset,
               problem.GetBlockID2XSMap(),
               problem.GetNumMoments(),
               problem.GetMaxCellDOFCount(),
               problem.GetMinCellDOFCount()),
    sweep_impl_(&CBCSweepChunk::Sweep_Generic)
{
  if ((min_num_cell_dofs_ == max_num_cell_dofs_) and (min_num_cell_dofs_ >= 2) and
      (min_num_cell_dofs_ <= 8))
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
  async_comm_ = &dynamic_cast<CBC_AsynchronousCommunicator&>(*angle_set.GetCommunicator());
  gs_size_ = groupset_.GetNumGroups();
  gs_gi_ = groupset_.first_group;
  num_angles_in_as_ = angle_set.GetNumAngles();
  group_stride_ = angle_set.GetNumGroups();
  group_angle_stride_ = group_stride_ * num_angles_in_as_;
}

void
CBCSweepChunk::SetCell(const Cell* cell_ptr, AngleSet&)
{
  cell_ = cell_ptr;
  cell_local_id_ = cell_ptr->local_id;
  cell_mapping_ = &discretization_.GetCellMapping(*cell_);
  cell_transport_view_ = &cell_transport_views_[cell_local_id_];
  cell_outflow_view_ = &cell_outflow_views_[cell_local_id_];
  cell_num_faces_ = cell_->faces.size();
  cell_num_nodes_ = cell_mapping_->GetNumNodes();

  const auto& unit_mats = unit_cell_matrices_[cell_local_id_];
  G_ = &unit_mats.intV_shapeI_gradshapeJ;
  M_ = &unit_mats.intV_shapeI_shapeJ;
  M_surf_ = &unit_mats.intS_shapeI_shapeJ;
  IntS_shapeI_ = &unit_mats.intS_shapeI;
}

void
CBCSweepChunk::Sweep(AngleSet& angle_set)
{
  (this->*sweep_impl_)(angle_set);
}

void
CBCSweepChunk::Sweep_Generic(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunk::Sweep_Generic");

  CBC_Sweep_Generic<false>(*this, angle_set);
}

} // namespace opensn

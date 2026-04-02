// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_kernels.h"

namespace opensn
{

struct CBCSweepChunkContext
{
  CBC_FLUDS* fluds = nullptr;

  size_t gs_size = 0;
  unsigned int gs_gi = 0;
  size_t num_angles_in_as = 0;
  unsigned int group_stride = 0;
  size_t group_angle_stride = 0;
  bool surface_source_active = false;

  const Cell* cell = nullptr;
  std::uint32_t cell_local_id = 0;
  const CellMapping* cell_mapping = nullptr;
  CellLBSView* cell_transport_view = nullptr;
  size_t cell_num_faces = 0;
  size_t cell_num_nodes = 0;

  const DenseMatrix<Vector3>* G = nullptr;
  const DenseMatrix<double>* M = nullptr;
  const std::vector<DenseMatrix<double>>* M_surf = nullptr;
  const std::vector<Vector<double>>* IntS_shapeI = nullptr;

  void BindAngleSet(const LBSGroupset& groupset, const bool has_surface_source, AngleSet& angle_set)
  {
    fluds = &dynamic_cast<CBC_FLUDS&>(angle_set.GetFLUDS());
    gs_size = groupset.GetNumGroups();
    gs_gi = groupset.first_group;
    surface_source_active = has_surface_source;
    num_angles_in_as = angle_set.GetNumAngles();
    group_stride = angle_set.GetNumGroups();
    group_angle_stride = group_stride * num_angles_in_as;
  }

  void BindCell(const SpatialDiscretization& discretization,
                const std::vector<UnitCellMatrices>& unit_cell_matrices,
                std::vector<CellLBSView>& cell_transport_views,
                const Cell* cell_ptr)
  {
    cell = cell_ptr;
    cell_local_id = cell_ptr->local_id;
    cell_mapping = &discretization.GetCellMapping(*cell);
    cell_transport_view = &cell_transport_views[cell->local_id];
    cell_num_faces = cell->faces.size();
    cell_num_nodes = cell_mapping->GetNumNodes();

    const auto& unit_mats = unit_cell_matrices[cell_local_id];
    G = &unit_mats.intV_shapeI_gradshapeJ;
    M = &unit_mats.intV_shapeI_shapeJ;
    M_surf = &unit_mats.intS_shapeI_shapeJ;
    IntS_shapeI = &unit_mats.intS_shapeI;
  }
};

} // namespace opensn

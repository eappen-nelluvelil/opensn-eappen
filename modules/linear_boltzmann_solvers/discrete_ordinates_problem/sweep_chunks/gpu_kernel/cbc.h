// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/main.h"

namespace opensn::gpu_kernel
{

__global__ void
SweepKernelCBC(Arguments<SweepType::CBC> args,
               const std::uint32_t* cells_to_sweep,
               const unsigned int num_cells,
               double* saved_psi)
{
  (void)args.cbc_task_graph;

  unsigned int cell_idx = threadIdx.y + blockDim.y * blockIdx.y;
  unsigned int angle_group_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (cell_idx >= num_cells || angle_group_idx >= args.flud_data.stride_size)
    return;

  unsigned int angle_idx = angle_group_idx / args.groupset_size;
  unsigned int group_idx = angle_group_idx - angle_idx * args.groupset_size;
  const std::uint32_t cell_local_idx = cells_to_sweep[cell_idx];

  CellView cell;
  MeshView(args.mesh_data).GetCellView(cell, cell_local_idx);
  if (cell.num_nodes == 0)
    return;

  auto [cell_edge_data, _] = GetCellDataIndex(args.flud_index, cell_local_idx);

  std::uint32_t num_moments;
  const std::uint32_t direction_num = args.directions[angle_idx];
  DirectionView direction;
  {
    QuadratureView quadrature(args.quad_data);
    num_moments = quadrature.num_moments;
    quadrature.GetDirectionView(direction, direction_num);
  }

  SweepDispatch<SweepType::CBC>(cell.num_nodes,
                                args,
                                cell,
                                direction,
                                cell_edge_data,
                                angle_group_idx,
                                group_idx,
                                num_moments,
                                saved_psi);
}

} // namespace opensn::gpu_kernel

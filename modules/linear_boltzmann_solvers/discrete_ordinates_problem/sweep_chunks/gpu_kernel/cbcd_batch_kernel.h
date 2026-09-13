// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/cbcd_batch.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/main.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/round_up.h"

namespace opensn::gpu_kernel
{

__CRB_GLOBAL_FUNC__ void
CBCDBatchKernel(const CBCDBatch* batches, std::uint32_t num_batches, std::uint64_t num_blocks)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  const auto thread = threadIdx.x;
  const auto first_block = blockIdx.x;
  const auto block_stride = gridDim.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const auto thread = work_index.get_local_id(2);
  const auto first_block = work_index.get_group(2);
  const auto block_stride = work_index.get_group_range(2);
#endif
  for (std::uint64_t block = first_block; block < num_blocks; block += block_stride)
  {
    std::uint32_t begin = 0;
    std::uint32_t end = num_batches;
    while (begin < end)
    {
      const auto mid = begin + (end - begin) / 2;
      if (block < batches[mid].block_end)
        end = mid;
      else
        begin = mid + 1;
    }
    const auto& batch = batches[begin];
    // RoundUp can produce a block width that does not divide the launch threshold.
    if (thread >= batch.block_size_x * (threshold / batch.block_size_x))
      continue;
    const auto local_block = block - (begin == 0 ? 0 : batches[begin - 1].block_end);
    auto args = *batch.arguments;
    const auto cell_idx = thread / batch.block_size_x + (threshold / batch.block_size_x) *
                                                          (local_block / batch.num_stride_blocks);
    const auto angle_group_idx = static_cast<unsigned int>(
      thread % batch.block_size_x + batch.block_size_x * (local_block % batch.num_stride_blocks));
    if (cell_idx >= batch.num_cells or angle_group_idx >= args.flud_data.stride_size)
      continue;

    const auto angle_idx = angle_group_idx / args.groupset_size;
    auto group_idx = angle_group_idx - angle_idx * args.groupset_size;
    const auto cell_local_idx = batch.cells[cell_idx];
    CellView cell;
    MeshView(args.mesh_data).GetCellView(cell, cell_local_idx);
    if (cell.num_nodes == 0)
      continue;
    auto [cell_edge_data, _] = GetCellDataIndex(args.flud_index, cell_local_idx);
    QuadratureView quadrature(args.quad_data);
    auto num_moments = quadrature.num_moments;
    DirectionView direction;
    quadrature.GetDirectionView(direction, args.directions[angle_idx]);
    auto saved_psi = batch.saved_psi;
    auto stride_idx = angle_group_idx;
    SweepDispatch<SweepKind::CBC>(cell.num_nodes,
                                  args,
                                  cell,
                                  direction,
                                  cell_edge_data,
                                  stride_idx,
                                  group_idx,
                                  num_moments,
                                  saved_psi);
  }
}

} // namespace opensn::gpu_kernel

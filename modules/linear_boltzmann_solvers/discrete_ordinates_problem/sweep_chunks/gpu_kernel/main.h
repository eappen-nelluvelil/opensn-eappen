// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/buffer.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "caribou/main.hpp"
#include <limits>
#include <utility>
#include <type_traits>

namespace crb = caribou;

namespace opensn::gpu_kernel
{

template <std::uint32_t... I, class F>
__CRB_DEVICE_FUNC__ void
ForDOFs1ToN(std::integer_sequence<std::uint32_t, I...> sequence, F& f)
{
  static_cast<void>(sequence);
  (f(std::integral_constant<std::uint32_t, I + 1>{}), ...);
}

template <class F>
__CRB_DEVICE_FUNC__ void
ForDOFs1ToMax(F& f)
{
  ForDOFs1ToN(std::make_integer_sequence<std::uint32_t, LBSProblem::max_dofs_gpu>{}, f);
}

template <SweepKind k, class... Args>
__CRB_DEVICE_FUNC__ void
SweepDispatch(std::uint32_t n, Args&... args)
{
  bool done = false;
  auto dispatch = [&](auto dof_c)
  {
    constexpr std::uint32_t dof = decltype(dof_c)::value;
    if (!done && n == dof)
    {
      gpu_kernel::Sweep<dof, k>(args...);
      done = true;
    }
  };
  ForDOFs1ToMax(dispatch);
}

template <SweepKind k>
__CRB_DEVICE_FUNC__ void
SweepCell(const Arguments<k>& args,
          const std::uint32_t cell_local_idx,
          const unsigned int angle_group_idx,
          double* saved_psi)
{
  unsigned int angle_idx = angle_group_idx / args.groupset_size;
  unsigned int group_idx = angle_group_idx - angle_idx * args.groupset_size;
  CellView cell;
  MeshView(args.mesh_data).GetCellView(cell, cell_local_idx);
  if (cell.num_nodes == 0)
    return;
  auto [cell_edge_data, _] = GetCellDataIndex(args.flud_index, cell_local_idx);
  std::uint32_t num_moments;
  std::uint32_t direction_num = args.directions[angle_idx];
  DirectionView direction;
  {
    QuadratureView quadrature(args.quad_data);
    num_moments = quadrature.num_moments;
    quadrature.GetDirectionView(direction, direction_num);
  }
  opensn::gpu_kernel::SweepDispatch<k>(cell.num_nodes,
                                       args,
                                       cell,
                                       direction,
                                       cell_edge_data,
                                       angle_group_idx,
                                       group_idx,
                                       num_moments,
                                       saved_psi);
}

template <SweepKind k>
__CRB_GLOBAL_FUNC__ void
SweepKernel(Arguments<k> args,
            const std::uint32_t* cells_to_sweep,
            unsigned int num_cells,
            double* saved_psi)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  unsigned int cell_idx = threadIdx.y + blockDim.y * blockIdx.y;
  unsigned int angle_group_idx = threadIdx.x + blockDim.x * blockIdx.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  unsigned int cell_idx = work_index.get_global_id(1);
  unsigned int angle_group_idx = work_index.get_global_id(2);
#endif
  if (cell_idx >= num_cells || angle_group_idx >= args.flud_data.stride_size)
    return;
  SweepCell(args, cells_to_sweep[cell_idx], angle_group_idx, saved_psi);
}

#if defined(__NVCC__) || defined(__HIPCC__)

__CRB_DEVICE_FUNC__ std::uint32_t
CBCDDequeueCell(CBCDDeviceQueueState& state, const std::uint32_t* cell_queue)
{
  if (state.head == state.tail)
    return std::numeric_limits<std::uint32_t>::max();
  return cell_queue[state.head++];
}

__CRB_DEVICE_FUNC__ void
CBCDEnqueueCell(CBCDDeviceQueueState& state,
                std::uint32_t* cell_queue,
                const std::uint32_t cell_local_id)
{
  cell_queue[state.tail++] = cell_local_id;
}

/// Sweep the locally reachable closure of a CBCD cell-task DAG.
template <SweepKind k>
__CRB_GLOBAL_FUNC__ void
CBCDClosureKernel(Arguments<k> args,
                  std::uint32_t* completed_cell_ids,
                  const std::uint32_t initial_queue_size,
                  double* saved_psi,
                  CBCDDeviceScheduler scheduler)
{
  static_assert(k == SweepKind::CBC);
  extern __shared__ std::uint32_t cell_local_ids[];
  __shared__ std::uint32_t num_cells;
  std::uint32_t completed_count = 0;
  std::uint32_t publication_count = 0;

  if (threadIdx.x == 0 and threadIdx.y == 0)
    *scheduler.queue_state = {0, initial_queue_size};
  __syncthreads();

  while (true)
  {
    if (threadIdx.x == 0 and threadIdx.y == 0)
    {
      num_cells = 0;
      while (num_cells < blockDim.y)
      {
        const auto cell_local_id = CBCDDequeueCell(*scheduler.queue_state, scheduler.cell_queue);
        if (cell_local_id == std::numeric_limits<std::uint32_t>::max())
          break;
        cell_local_ids[num_cells++] = cell_local_id;
      }
    }
    __syncthreads();

    if (num_cells == 0)
      break;

    if (threadIdx.y < num_cells)
      for (std::uint32_t angle_group_idx = threadIdx.x;
           angle_group_idx < args.flud_data.stride_size;
           angle_group_idx += blockDim.x)
        SweepCell(args, cell_local_ids[threadIdx.y], angle_group_idx, saved_psi);
    __syncthreads();

    if (threadIdx.x == 0 and threadIdx.y == 0)
    {
      for (std::uint32_t cell = 0; cell < num_cells; ++cell)
      {
        const auto cell_local_id = cell_local_ids[cell];
        ++completed_count;
        if (scheduler.requires_publication[cell_local_id] != 0)
          completed_cell_ids[publication_count++] = cell_local_id;
        const auto successor_begin = scheduler.successor_offsets[cell_local_id];
        const auto successor_end = scheduler.successor_offsets[cell_local_id + 1];
        for (auto successor_index = successor_begin; successor_index < successor_end;
             ++successor_index)
        {
          const auto successor = scheduler.successors[successor_index];
          if (--scheduler.remaining_local_dependencies[successor] == 0)
          {
            if (scheduler.initial_remote_dependencies[successor] == 0)
              CBCDEnqueueCell(*scheduler.queue_state, scheduler.cell_queue, successor);
            else
            {
              scheduler.locally_ready[successor] = 1;
              if (scheduler.remaining_remote_dependencies[successor] == 0)
                CBCDEnqueueCell(*scheduler.queue_state, scheduler.cell_queue, successor);
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 and threadIdx.y == 0)
  {
    *scheduler.completed_count = completed_count;
    *scheduler.publication_count = publication_count;
  }
}

#endif

} // namespace opensn::gpu_kernel

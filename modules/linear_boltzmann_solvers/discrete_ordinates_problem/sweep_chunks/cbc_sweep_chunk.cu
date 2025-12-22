// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_gpu_kernel/main.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set_helpers.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/memory_pinner.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/outflow_carrier.h"
#include "caliper/cali.h"
#include "caribou/caribou.h"
#include <algorithm>

namespace crb = caribou;

namespace opensn
{

namespace cbc_gpu_kernel
{

template<typename ArgType>
__global__ void
CBCSweepKernel(const ArgType args)
{
  Index idx;
  {
    std::uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= args.batch_size)
      return;
    idx.Compute(thread_idx, args.angleset_size, args.groupset_size);
  }

  std::uint64_t cell_local_idx;
  if constexpr (std::is_same_v<ArgType, Arguments>)
    cell_local_idx = args.cell_local_ids[idx.cell_idx];
  else
    cell_local_idx = args.cell_local_id;
  // const std::uint64_t cell_local_idx = args.cell_local_ids[idx.cell_idx];

  CellView cell;
  MeshView(args.mesh_data).GetCellView(cell, cell_local_idx);
  if (cell.num_nodes == 0)
    return;

  auto [cell_edge_data, _] = GetCBCDCellFaceDataIndex(args.flud_index, cell_local_idx);

  std::uint32_t num_moments;
  std::uint32_t direction_num = args.directions[idx.angle_idx];
  DirectionView direction;
  {
    QuadratureView quadrature(args.quad_data);
    num_moments = quadrature.num_moments;
    quadrature.GetDirectionView(direction, direction_num);
  }

  const std::uint64_t angle_group_idx = idx.angle_idx * args.groupset_size + idx.group_idx;

  cbc_sweep_spec_map[cell.num_nodes - 1](
    args, cell, direction, cell_edge_data, angle_group_idx, idx.group_idx, num_moments);
}

} // namespace cbc_gpu_kernel

void
CBCSweepChunk::CopyPhiAndSrcToDevice()
{
  reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(0))->CopyToDevice();
  reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(1))->CopyToDevice();
}

void
CBCSweepChunk::CopyOutflowAndPhiFromDevice()
{
  reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(1))->CopyFromDevice();
  auto* outflow = reinterpret_cast<OutflowCarrier*>(problem_.GetCarrier(1));
  outflow->AccumulateBack(cell_transport_views_);
  outflow->Reset();
}

void
CBCSweepChunk::BuildCUDAGraphArguments(std::vector<CBC_AngleSet*>& angle_sets, std::vector<std::vector<std::any>>& cuda_graph_arguments)
{
  cuda_graph_arguments.resize(angle_sets.size());
  for (size_t i = 0; i < angle_sets.size(); ++i)
  {
    auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(*angle_sets[i]);
    auto& current_task_list = dynamic_cast<const CBC_SPDS&>(cbc_angle_set.GetSPDS()).GetTaskList();
    auto& cbcd_fluds = dynamic_cast<CBCD_FLUDS&>(cbc_angle_set.GetFLUDS());

    auto& graph_args = cuda_graph_arguments[i];
    graph_args.resize(current_task_list.size());

    for (size_t j = 0; j < current_task_list.size(); ++j)
      graph_args[j] = std::any_cast<cbc_gpu_kernel::GraphArguments>(cbc_gpu_kernel::GraphArguments(
        problem_, groupset_, cbc_angle_set, cbcd_fluds, current_task_list[j].reference_id));
  }
}

void
CBCSweepChunk::GPUSweep(AngleSet& angle_set, std::vector<Task*>& tasks_to_execute)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunk::GPUSweep");

  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  auto& cbcd_fluds = dynamic_cast<CBCD_FLUDS&>(angle_set.GetFLUDS());

  auto& host_cell_local_ids = cbcd_fluds.GetLocalCellIDs().GetHostVector();
  // for (size_t idx = 0; idx < tasks_to_execute.size(); ++idx)
  //   host_cell_local_ids[idx] = tasks_to_execute[idx]->reference_id;

  std::transform(tasks_to_execute.begin(),
                 tasks_to_execute.end(),
                 host_cell_local_ids.begin(),
                 [](const Task* task) { return task->reference_id; });

  caribou::Stream& stream = GetCBCAngleSetStream(cbc_angle_set);
  crb::copy_async(cbcd_fluds.GetLocalCellIDs().GetDeviceMemory(),
                  host_cell_local_ids,
                  tasks_to_execute.size(),
                  stream);

  cbc_gpu_kernel::Arguments args(problem_, groupset_, angle_set, cbcd_fluds, tasks_to_execute);

  const std::uint32_t threads_per_block = 128;
  const std::uint32_t num_blocks = (args.batch_size + threads_per_block - 1) / threads_per_block;

  cbc_gpu_kernel::CBCSweepKernel<<<num_blocks, threads_per_block, 0, stream.get()>>>(args);
}

} // namespace opensn
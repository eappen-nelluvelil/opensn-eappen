// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_gpu_kernel/arguments.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.cu"
#include "external/caribou/caribou.h"
#include "external/caribou/cuda/stream.hpp"
#include <unordered_map>

namespace opensn
{
void
CBC_AngleSet::CreateStream()
{
  if (stream_ == nullptr)
  {
    caribou::Stream* stream = new caribou::Stream();
    stream_ = stream;
  }
}

void
CBC_AngleSet::InitializeBoundaryEvent()
{
  if (boundary_event_.has_value())
    return;

  cudaEvent_t event;
  cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  boundary_event_ = event;
}

void
CBC_AngleSet::DestroyBoundaryEvent()
{
  if (boundary_event_.has_value())
  {
    cudaEvent_t event = std::any_cast<cudaEvent_t>(boundary_event_);
    cudaEventDestroy(event);
    boundary_event_.reset();
  }
}

void
CBC_AngleSet::CreateCUDAGraph()
{
  cudaGraph_t graph{};
  cudaGraphCreate(&graph, 0);
  cuda_graph_ = graph;
}

void 
CBC_AngleSet::BuildAndInstantiateCUDAGraph(std::vector<cbc_gpu_kernel::GraphArguments>& graph_args)
{
  // Initialize boundary wait event
  InitializeBoundaryEvent();
  cudaEvent_t boundary_event = std::any_cast<cudaEvent_t>(boundary_event_);

  // Create boundary wait node
  // (This node halts execution of its dependent nodes until boundary_event is recorded)
  cudaGraphNode_t boundary_wait_node;
  cudaGraphAddEventWaitNode(&boundary_wait_node, std::any_cast<cudaGraph_t>(cuda_graph_), nullptr, 0, boundary_event);

  const auto& current_task_list = dynamic_cast<const CBC_SPDS&>(GetSPDS()).GetTaskList();

  // Map from cell local ID to its corresponding kernel node
  std::unordered_map<std::uint64_t, cudaGraphNode_t> cell_local_id_to_node_map;
  cell_local_id_to_node_map.reserve(current_task_list.size());

  // Loop through tasks, and create kernel nodes for each task
  for (size_t i = 0; i < current_task_list.size(); ++i)
  {
    const auto& task = current_task_list[i];
    const auto& args = graph_args[i];

    cudaKernelNodeParams kernel_node_params{};
    void* kernel_args[] = {const_cast<void*>(reinterpret_cast<const void*>(&args))};
    kernel_node_params.func = (void*)cbc_gpu_kernel::CBCSweepKernel<cbc_gpu_kernel::GraphArguments>; // Need to update CBCSweepKernel to take `const GraphArguments args`
    kernel_node_params.gridDim = dim3((args.batch_size + 127) / 128, 1, 1);
    kernel_node_params.blockDim = dim3(128, 1, 1);
    kernel_node_params.sharedMemBytes = 0;
    kernel_node_params.kernelParams = kernel_args;
    kernel_node_params.extra = nullptr;

    // Append kernel node to kernel_nodes vector
    cudaGraphNode_t kernel_node;
    cudaGraphAddKernelNode(&kernel_node, std::any_cast<cudaGraph_t>(cuda_graph_), nullptr, 0, &kernel_node_params);

    // kernel_nodes[i] = kernel_node;
    cell_local_id_to_node_map[task.reference_id] = kernel_node;
  }

  // Now, set up dependencies between kernel nodes based on task dependencies
  // Also need to create edges from incoming boundary event wait node to kernel nodes
  // corresponding to cells with incoming boundary faces
  // A cell has incoming boundary faces if its corresponding task has (task.has_incoming_boundary_faces == true)

  for (const auto& task : current_task_list)
  {
    cudaGraphNode_t current_kernel_node = cell_local_id_to_node_map[task.reference_id];
    std::vector<cudaGraphNode_t> predecessor_nodes;

    // If this task has incoming boundary faces, add dependency from boundary wait node
    if (task.has_incoming_boundary_faces)
    {
      predecessor_nodes.push_back(boundary_wait_node);
    }

    // Create edges from current kernel node to its successor kernel nodes
    if (!task.predecessors.empty())
    {
      for (const auto& pred_local_id : task.predecessors)
      {
        if (cell_local_id_to_node_map.find(pred_local_id) != cell_local_id_to_node_map.end())
        {
          cudaGraphNode_t pred_kernel_node = cell_local_id_to_node_map[pred_local_id];
          predecessor_nodes.push_back(pred_kernel_node);
        }
      }
    }

    // Finally, add dependencies from predecessor nodes to current kernel node
    // if (!predecessor_nodes.empty())
    // {
    //   cudaGraphAddDependencies(std::any_cast<cudaGraph_t>(cuda_graph_), predecessor_nodes.data(), &current_kernel_node, nullptr, predecessor_nodes.size());
    // }
    if (!predecessor_nodes.empty())
    {
      for (auto& pred_node : predecessor_nodes)
      {
        cudaGraphAddDependencies(std::any_cast<cudaGraph_t>(cuda_graph_), &pred_node, &current_kernel_node, nullptr, 1);
      }
      // cudaGraphAddDependencies(std::any_cast<cudaGraph_t>(cuda_graph_), predecessor_nodes.data(), &current_kernel_node, nullptr, predecessor_nodes.size());
    }
  }

  // Instantiate the CUDA graph
  cudaGraphExec_t graph_exec;
  cudaGraphInstantiate(&graph_exec, std::any_cast<cudaGraph_t>(cuda_graph_), nullptr, nullptr, 0);
  cuda_graph_exec_ = std::make_any<cudaGraphExec_t>(graph_exec);
}

void
CBC_AngleSet::DestroyStream()
{
  if (stream_ != nullptr)
  {
    caribou::Stream* stream = static_cast<caribou::Stream*>(stream_);
    delete stream;
    stream_ = nullptr;
  }
}

void
CBC_AngleSet::DestroyCUDAGraph()
{
  if (cuda_graph_.has_value())
  {
    cudaGraph_t graph = std::any_cast<cudaGraph_t>(cuda_graph_);
    cudaGraphDestroy(graph);
    cuda_graph_.reset();
  }

  if (cuda_graph_exec_.has_value())
  {
    cudaGraphExec_t graph_exec = std::any_cast<cudaGraphExec_t>(cuda_graph_exec_);
    cudaGraphExecDestroy(graph_exec);
    cuda_graph_exec_.reset();
  }

  DestroyBoundaryEvent();
}

void
CBC_AngleSet::AssociateAngleSetWithFLUDS()
{
  CBCD_FLUDS* cbcd_fluds = dynamic_cast<CBCD_FLUDS*>(fluds_.get());
  cbcd_fluds->SetAngleSet(*this);
}

} // namespace opensn
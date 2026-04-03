// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "caribou/main.hpp"
#include <type_traits>

namespace crb = caribou;

namespace opensn
{

namespace
{

template <class T>
T*
CopyVectorToDevice(const std::vector<T>& values)
{
  if (values.empty())
    return nullptr;

  crb::HostVector<T> host_values(values.begin(), values.end());
  crb::DeviceMemory<T> device_values(values.size());
  crb::copy(device_values, host_values, host_values.size());
  return device_values.release();
}

template <class T>
void
FreeDevicePointer(T*& ptr)
{
  if (ptr == nullptr)
    return;

  using ValueType = std::remove_const_t<T>;
  crb::DeviceMemory<ValueType> device_values(const_cast<ValueType*>(ptr));
  device_values.reset();
  ptr = nullptr;
}

} // namespace

void
CBC_SPDS::CopyTaskGraphDataOnDevice() const
{
  if (device_task_graph_.reference_ids != nullptr)
    return;

  std::vector<std::uint32_t> predecessor_offsets(task_list_.size() + 1, 0);
  std::vector<std::uint32_t> predecessors;
  std::size_t num_predecessors = 0;
  for (std::size_t task_idx = 0; task_idx < task_list_.size(); ++task_idx)
  {
    num_predecessors += task_list_[task_idx].predecessors.size();
    predecessor_offsets[task_idx + 1] = static_cast<std::uint32_t>(num_predecessors);
  }
  predecessors.reserve(num_predecessors);
  for (const auto& task : task_list_)
    predecessors.insert(predecessors.end(), task.predecessors.begin(), task.predecessors.end());

  std::vector<std::uint32_t> reference_ids(task_list_.size(), 0);
  std::vector<int> initial_dependencies(task_list_.size(), 0);
  for (std::size_t task_idx = 0; task_idx < task_list_.size(); ++task_idx)
  {
    reference_ids[task_idx] = static_cast<std::uint32_t>(task_list_[task_idx].reference_id);
    initial_dependencies[task_idx] = static_cast<int>(task_list_[task_idx].num_dependencies);
  }

  device_task_graph_.reference_ids = CopyVectorToDevice(reference_ids);
  device_task_graph_.successor_offsets = CopyVectorToDevice(local_successor_offsets_);
  device_task_graph_.successors = CopyVectorToDevice(local_successors_);
  device_task_graph_.predecessor_offsets = CopyVectorToDevice(predecessor_offsets);
  device_task_graph_.predecessors = CopyVectorToDevice(predecessors);
  device_task_graph_.initial_dependencies = CopyVectorToDevice(initial_dependencies);
  device_task_graph_.initial_successors_to_retire =
    CopyVectorToDevice(initial_successors_to_retire_);
  device_task_graph_.num_tasks = static_cast<std::uint32_t>(task_list_.size());
}

void
CBC_SPDS::FreeDeviceData() const
{
  FreeDevicePointer(device_task_graph_.reference_ids);
  FreeDevicePointer(device_task_graph_.successor_offsets);
  FreeDevicePointer(device_task_graph_.successors);
  FreeDevicePointer(device_task_graph_.predecessor_offsets);
  FreeDevicePointer(device_task_graph_.predecessors);
  FreeDevicePointer(device_task_graph_.initial_dependencies);
  FreeDevicePointer(device_task_graph_.initial_successors_to_retire);
  device_task_graph_.num_tasks = 0;
}

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/main.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/round_up.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/device_vector_mirror.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cstdint>
#include <set>
#include <unordered_map>

namespace opensn
{

CBCDSweepChunk::DispatchState::DispatchState(const std::size_t stride,
                                             const crb::Dim3 threads,
                                             const unsigned int stride_blocks)
  : stride_size(stride), threads_per_block(threads), num_stride_blocks(stride_blocks)
{
}

CBCDSweepChunk::CBCDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset)
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
    problem_(problem)
{
  std::vector<CBCD_FLUDS*> fluds_list;
  for (auto& as : *(groupset.angle_agg))
  {
    auto* angle_set = dynamic_cast<CBCD_AngleSet*>(as.get());
    auto* fluds = dynamic_cast<CBCD_FLUDS*>(&(angle_set->GetFLUDS()));
    angle_sets_.push_back(angle_set);
    fluds_list.push_back(fluds);

    angle_set->RefreshDeviceData();

    gpu_kernel::Arguments<SweepKind::CBC> args(
      problem_, groupset_, *angle_set, *fluds, IsSurfaceSourceActive());
    const auto stride_size =
      gpu_kernel::RoundUp(static_cast<unsigned int>(args.flud_data.stride_size));
    const auto block_size_x = std::min(stride_size, gpu_kernel::threshold);
    const auto block_size_y = gpu_kernel::threshold / block_size_x;
    const auto grid_size_x = (stride_size + gpu_kernel::threshold - 1) / gpu_kernel::threshold;
    kernel_launches_.push_back({args,
                                crb::Dim3(block_size_x, block_size_y),
                                grid_size_x,
                                fluds,
                                fluds->GetSavedPsiDevicePointer()});
    host_launch_data_.push_back({args, fluds->GetSavedPsiDevicePointer()});
  }

  if (not host_launch_data_.empty())
  {
    device_launch_data_ = crb::DeviceMemory<gpu_kernel::CBCDLaunchData>(host_launch_data_.size());
    crb::copy(device_launch_data_, host_launch_data_, host_launch_data_.size());
  }

  if (not angle_sets_.empty())
  {
    profiler_ = CBCDProfiler::Create(angle_sets_.size());
    std::vector<std::vector<int>> incoming_source_partitions_by_angle_set;
    incoming_source_partitions_by_angle_set.reserve(angle_sets_.size());
    std::unordered_map<int, std::vector<std::size_t>> section_bytes_by_source;
    std::vector<AngleSetCommunicationBounds> communication_bounds(angle_sets_.size());
    for (std::size_t angle_set_id = 0; angle_set_id < angle_sets_.size(); ++angle_set_id)
    {
      const auto stride = fluds_list[angle_set_id]->GetStrideSize();
      const auto& common_data = fluds_list[angle_set_id]->GetCommonData();
      std::unordered_map<int, DestinationQueueBounds> outgoing_bounds_by_destination;
      incoming_source_partitions_by_angle_set.push_back(common_data.GetIncomingSourcePartitions());
      auto& bounds = communication_bounds[angle_set_id];
      bounds.incoming_mailbox_capacity = common_data.GetNumIncomingNonlocalFaces();
      for (std::size_t cell_local_id = 0; cell_local_id < common_data.GetNumLocalCells();
           ++cell_local_id)
      {
        for (const auto& face_info : common_data.GetOutgoingNonlocalFaces(cell_local_id))
        {
          const int destination_rank =
            common_data.GetDestinationRanks()[face_info.destination_index];
          auto& destination = outgoing_bounds_by_destination[destination_rank];
          destination.destination_rank = destination_rank;
          ++destination.num_faces;
        }
      }
      bounds.outgoing_queue_bounds.reserve(outgoing_bounds_by_destination.size());
      for (const auto& [_, destination] : outgoing_bounds_by_destination)
        bounds.outgoing_queue_bounds.push_back(destination);

      std::unordered_map<std::uint32_t, std::size_t> incoming_faces_by_source;
      std::unordered_map<std::uint32_t, std::size_t> incoming_values_by_source;
      for (std::size_t cell_local_id = 0; cell_local_id < common_data.GetNumLocalCells();
           ++cell_local_id)
      {
        for (const auto& face_info : common_data.GetIncomingNonlocalFaces(cell_local_id))
        {
          if (face_info.num_face_nodes == 0)
            continue;
          ++incoming_faces_by_source[face_info.source_partition_index];
          incoming_values_by_source[face_info.source_partition_index] +=
            static_cast<std::size_t>(face_info.num_face_nodes) * stride;
          const auto source_partition =
            common_data.GetIncomingSourcePartitions()[face_info.source_partition_index];
          auto& section_bytes_by_angle_set = section_bytes_by_source[source_partition];
          if (section_bytes_by_angle_set.empty())
            section_bytes_by_angle_set.assign(angle_sets_.size(), 0);
          section_bytes_by_angle_set[angle_set_id] +=
            sizeof(std::uint32_t) + sizeof(std::size_t) +
            static_cast<std::size_t>(face_info.num_face_nodes) * stride * sizeof(double);
        }
      }
      for (const auto& [_, count] : incoming_faces_by_source)
        bounds.max_incoming_faces_per_batch = std::max(bounds.max_incoming_faces_per_batch, count);
      for (const auto& [_, values] : incoming_values_by_source)
        bounds.max_incoming_values_per_batch =
          std::max(bounds.max_incoming_values_per_batch, values);
    }

    std::size_t max_message_bytes = 0;
    for (const auto& [_, section_bytes_by_angle_set] : section_bytes_by_source)
    {
      std::size_t message_bytes = sizeof(std::size_t);
      for (const auto& section_bytes : section_bytes_by_angle_set)
      {
        if (section_bytes == 0)
          continue;
        message_bytes += 2 * sizeof(std::size_t) + section_bytes;
      }
      max_message_bytes = std::max(max_message_bytes, message_bytes);
    }

    std::vector<AngleSet*> base_angle_sets(angle_sets_.begin(), angle_sets_.end());
    async_comm_ =
      std::make_unique<CBCD_AsynchronousCommunicator>(base_angle_sets,
                                                      angle_sets_.front()->GetCommunicatorSet(),
                                                      incoming_source_partitions_by_angle_set,
                                                      max_message_bytes,
                                                      communication_bounds,
                                                      profiler_.get());
    for (auto* angle_set : angle_sets_)
      angle_set->SetCommunicator(*async_comm_);
  }
}

CBCDSweepChunk::~CBCDSweepChunk()
{
  StopCommunicator();
}

void
CBCDSweepChunk::StartCommunicator(const std::size_t num_workers)
{
  ConfigureWorkerDispatches(num_workers);
  if (async_comm_)
    async_comm_->Start(num_workers);
}

void
CBCDSweepChunk::StopCommunicator()
{
  if (async_comm_)
    async_comm_->Stop();
}

void
CBCDSweepChunk::RefreshKernelArguments()
{
  CALI_CXX_MARK_SCOPE("CBCDSweepChunk::RefreshKernelArguments");

  for (std::size_t angle_set_id = 0; angle_set_id < angle_sets_.size(); ++angle_set_id)
  {
    auto& launch = kernel_launches_[angle_set_id];
    {
      CALI_CXX_MARK_SCOPE("CBCDSweepChunk::Sweep::ArgsRefresh");
      launch.arguments = gpu_kernel::Arguments<SweepKind::CBC>(
        problem_, groupset_, *angle_sets_[angle_set_id], *launch.fluds, IsSurfaceSourceActive());
      launch.device_saved_psi = launch.fluds->GetSavedPsiDevicePointer();
      host_launch_data_[angle_set_id] = {launch.arguments, launch.device_saved_psi};
    }
  }
  if (not host_launch_data_.empty())
    crb::copy(device_launch_data_, host_launch_data_, host_launch_data_.size());
}

void
CBCDSweepChunk::ConfigureWorkerDispatches(const std::size_t num_workers)
{
  if (configured_workers_ == num_workers and angle_set_dispatches_.size() == angle_sets_.size())
    return;

  configured_workers_ = num_workers;
  dispatch_storage_.clear();
  worker_dispatches_.assign(num_workers, {});
  worker_angle_set_ids_.assign(num_workers, {});
  angle_set_dispatches_.assign(angle_sets_.size(), nullptr);
  angle_set_dispatch_status_.assign(angle_sets_.size(), {});

  for (std::size_t worker_id = 0; worker_id < num_workers; ++worker_id)
  {
    auto& dispatches = worker_dispatches_[worker_id];
    auto& angle_set_ids = worker_angle_set_ids_[worker_id];
    angle_set_ids.reserve((angle_sets_.size() + num_workers - 1) / num_workers);
    for (std::size_t angle_set_id = worker_id; angle_set_id < angle_sets_.size();
         angle_set_id += num_workers)
    {
      const auto& launch = kernel_launches_[angle_set_id];
      auto dispatch_it =
        std::find_if(dispatches.begin(),
                     dispatches.end(),
                     [&launch](const DispatchState* dispatch)
                     {
                       return dispatch->stride_size == launch.fluds->GetStrideSize() and
                              dispatch->threads_per_block.x == launch.threads_per_block.x and
                              dispatch->threads_per_block.y == launch.threads_per_block.y and
                              dispatch->num_stride_blocks == launch.num_stride_blocks;
                     });
      if (dispatch_it == dispatches.end())
      {
        auto dispatch = std::make_unique<DispatchState>(
          launch.fluds->GetStrideSize(), launch.threads_per_block, launch.num_stride_blocks);
        dispatches.push_back(dispatch.get());
        dispatch_storage_.push_back(std::move(dispatch));
        dispatch_it = std::prev(dispatches.end());
      }

      auto* dispatch = *dispatch_it;
      ++dispatch->angle_set_capacity;
      angle_set_dispatches_[angle_set_id] = dispatch;
      angle_set_ids.push_back(angle_set_id);
    }
  }

  for (auto& dispatch : dispatch_storage_)
  {
    dispatch->host_batches.reserve(dispatch->angle_set_capacity);
    dispatch->device_batches =
      crb::DeviceMemory<gpu_kernel::CBCDBatchDescriptor>(dispatch->angle_set_capacity);
    dispatch->ready_angle_sets.reserve(dispatch->angle_set_capacity);
    dispatch->active_angle_set_ids.reserve(dispatch->angle_set_capacity);
  }
}

bool
CBCDSweepChunk::PollWorkerDispatches(const std::size_t worker_id)
{
  bool completed_any = false;
  for (auto* dispatch : worker_dispatches_[worker_id])
  {
    if (dispatch->active and dispatch->stream.is_completed())
    {
      for (const auto angle_set_id : dispatch->active_angle_set_ids)
        angle_set_dispatch_status_[angle_set_id].complete = true;
      dispatch->active_angle_set_ids.clear();
      dispatch->active = false;
      completed_any = true;
    }
  }
  for (const auto angle_set_id : worker_angle_set_ids_[worker_id])
  {
    auto& status = angle_set_dispatch_status_[angle_set_id];
    if (status.kind == DispatchKind::SINGLE and (not status.complete) and
        angle_sets_[angle_set_id]->GetStream().is_completed())
    {
      status.complete = true;
      completed_any = true;
    }
  }
  return completed_any;
}

bool
CBCDSweepChunk::IsDispatchComplete(const std::size_t angle_set_id) const
{
  return angle_set_dispatch_status_[angle_set_id].complete;
}

void
CBCDSweepChunk::LaunchSingleBatch(const std::size_t worker_id,
                                  const std::size_t angle_set_id,
                                  const std::span<std::uint32_t> local_cell_ids)
{
  auto& status = angle_set_dispatch_status_[angle_set_id];
  status.kind = DispatchKind::SINGLE;
  status.complete = false;
  if (profiler_)
    profiler_->RecordDeviceDispatch(worker_id, 1, local_cell_ids.size());
  Sweep(static_cast<std::uint32_t>(local_cell_ids.size()), angle_set_id, local_cell_ids.data());
}

void
CBCDSweepChunk::LaunchFusedBatch(const std::size_t worker_id, DispatchState& dispatch)
{
  crb::copy(dispatch.device_batches,
            dispatch.host_batches,
            dispatch.host_batches.size(),
            0,
            0,
            dispatch.stream);
  std::uint32_t num_cell_blocks = 0;
  for (const auto& batch : dispatch.host_batches)
    num_cell_blocks =
      std::max(num_cell_blocks,
               (batch.num_cells + dispatch.threads_per_block.y - 1) / dispatch.threads_per_block.y);
  crb::Dim3 grid_size(dispatch.num_stride_blocks,
                      num_cell_blocks,
                      static_cast<std::uint32_t>(dispatch.host_batches.size()));
#if defined(__NVCC__) || defined(__HIPCC__)
  gpu_kernel::CBCDFusedSweepKernel<SweepKind::CBC>
    <<<grid_size, dispatch.threads_per_block, 0, dispatch.stream>>>(device_launch_data_.get(),
                                                                    dispatch.device_batches.get());
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto* launch_data = device_launch_data_.get();
  auto* batches = dispatch.device_batches.get();
  dispatch.stream.parallel_for(
    sycl::nd_range<3>(grid_size * dispatch.threads_per_block, dispatch.threads_per_block),
    [=](sycl::nd_item<3>)
    { gpu_kernel::CBCDFusedSweepKernel<SweepKind::CBC>(launch_data, batches); });
#endif
  if (profiler_)
  {
    std::uint64_t num_cells = 0;
    for (const auto& batch : dispatch.host_batches)
      num_cells += batch.num_cells;
    profiler_->RecordDeviceDispatch(worker_id, dispatch.host_batches.size(), num_cells);
  }
  dispatch.active = true;
}

bool
CBCDSweepChunk::DispatchReadyAngleSets(const std::size_t worker_id,
                                       const std::span<CBCD_AngleSet*> ready_angle_sets)
{
  if (ready_angle_sets.empty())
    return false;

  for (auto* dispatch : worker_dispatches_[worker_id])
    dispatch->ready_angle_sets.clear();
  for (auto* angle_set : ready_angle_sets)
    angle_set_dispatches_[angle_set->GetID()]->ready_angle_sets.push_back(angle_set);

  bool dispatched_any = false;
  for (auto* dispatch : worker_dispatches_[worker_id])
  {
    auto& ready = dispatch->ready_angle_sets;
    if (ready.empty())
      continue;

    if (dispatch->active or ready.size() == 1)
    {
      for (auto* angle_set : ready)
        LaunchSingleBatch(worker_id, angle_set->GetID(), angle_set->PrepareReadyBatch());
      dispatched_any = true;
      continue;
    }

    dispatch->host_batches.clear();
    for (auto* angle_set : ready)
    {
      const auto cell_ids = angle_set->PrepareReadyBatch();
      dispatch->host_batches.push_back({cell_ids.data(),
                                        static_cast<std::uint32_t>(angle_set->GetID()),
                                        static_cast<std::uint32_t>(cell_ids.size())});
      dispatch->active_angle_set_ids.push_back(angle_set->GetID());
    }

    if (dispatch->host_batches.size() == 1)
    {
      const auto angle_set_id = dispatch->host_batches.front().angle_set_id;
      auto& status = angle_set_dispatch_status_[angle_set_id];
      status.kind = DispatchKind::SINGLE;
      status.complete = false;
      Sweep(dispatch->host_batches.front().num_cells,
            angle_set_id,
            dispatch->host_batches.front().cell_ids);
      if (profiler_)
        profiler_->RecordDeviceDispatch(worker_id, 1, dispatch->host_batches.front().num_cells);
      dispatch->active_angle_set_ids.clear();
    }
    else if (not dispatch->host_batches.empty())
    {
      for (const auto angle_set_id : dispatch->active_angle_set_ids)
      {
        auto& status = angle_set_dispatch_status_[angle_set_id];
        status.kind = DispatchKind::FUSED;
        status.complete = false;
      }
      LaunchFusedBatch(worker_id, *dispatch);
    }
    dispatched_any = true;
  }
  return dispatched_any;
}

void
CBCDSweepChunk::Sweep(std::uint32_t num_ready_cells,
                      std::size_t angle_set_id,
                      std::uint32_t* local_cell_ids)
{
  CALI_CXX_MARK_SCOPE("CBCDSweepChunk::Sweep");

  auto& launch = kernel_launches_[angle_set_id];
  auto& stream = angle_sets_[angle_set_id]->GetStream();
  const auto grid_size_y =
    (num_ready_cells + launch.threads_per_block.y - 1) / launch.threads_per_block.y;
  crb::Dim3 grid_size(launch.num_stride_blocks, grid_size_y);
  {
    CALI_CXX_MARK_SCOPE("CBCDSweepChunk::Sweep::KernelLaunch");
#if defined(__NVCC__) || defined(__HIPCC__)
    gpu_kernel::SweepKernel<SweepKind::CBC><<<grid_size, launch.threads_per_block, 0, stream>>>(
      launch.arguments, local_cell_ids, num_ready_cells, launch.device_saved_psi);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    stream.synchronize();
    stream.parallel_for(
      sycl::nd_range<3>(grid_size * launch.threads_per_block, launch.threads_per_block),
      [=](sycl::nd_item<3> work_index)
      {
        gpu_kernel::SweepKernel<SweepKind::CBC>(
          launch.arguments, local_cell_ids, num_ready_cells, launch.device_saved_psi);
      });
#endif
  }
}

} // namespace opensn

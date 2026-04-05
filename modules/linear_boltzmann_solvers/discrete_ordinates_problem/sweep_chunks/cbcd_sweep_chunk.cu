// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/main.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/round_up.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/memory_pinner.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "caliper/cali.h"
#include <algorithm>
#include <unordered_map>

namespace opensn
{

CBCDSweepChunk::CBCDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset)
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
    problem_(problem)
{
  for (auto& as : *(groupset.angle_agg))
  {
    auto* angle_set = static_cast<CBCD_AngleSet*>(as.get());
    auto* fluds = static_cast<CBCD_FLUDS*>(&(angle_set->GetFLUDS()));
    angle_sets_.push_back(angle_set);
    fluds_list_.push_back(fluds);
    streams_list_.push_back(angle_set->GetStream());
    gpu_kernel::Arguments<gpu_kernel::SweepType::CBC> args(problem_, groupset_, *angle_set, *fluds);
    kernel_args_list_.push_back(args);
    unsigned int stride_size =
      gpu_kernel::RoundUp(static_cast<unsigned int>(args.flud_data.stride_size));
    unsigned int block_size_x = std::min(stride_size, gpu_kernel::threshold);
    unsigned int block_size_y = gpu_kernel::threshold / block_size_x;
    unsigned int grid_size_x = (stride_size + gpu_kernel::threshold - 1) / gpu_kernel::threshold;
    block_sizes_.push_back(::dim3(block_size_x, block_size_y));
    grid_size_x_list_.push_back(grid_size_x);
  }

  if (not angle_sets_.empty())
  {
    struct PerSourceAngleSetInfo
    {
      size_t num_entries = 0;
      size_t psi_bytes = 0;
    };

    std::unordered_map<int, std::unordered_map<size_t, PerSourceAngleSetInfo>> source_as_info;
    for (size_t as_idx = 0; as_idx < angle_sets_.size(); ++as_idx)
    {
      auto& fluds = *fluds_list_[as_idx];
      const auto stride = fluds.GetStrideSize();
      const auto& common_data = fluds.GetCommonData();
      for (size_t cell_local_id = 0; cell_local_id < common_data.GetNumLocalCells(); ++cell_local_id)
      {
        for (const auto& face_info : common_data.GetIncomingNonlocalFaces(cell_local_id))
        {
          if (face_info.num_nodes == 0)
            continue;
          auto& info = source_as_info[face_info.source_partition][as_idx];
          ++info.num_entries;
          info.psi_bytes += sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(size_t) +
                            static_cast<size_t>(face_info.num_nodes) * stride * sizeof(double);
        }
      }
    }

    size_t max_message_bytes = 0;
    for (const auto& [source_partition, as_map] : source_as_info)
    {
      (void)source_partition;
      size_t msg_size_in_bytes = sizeof(size_t);
      for (const auto& [as_idx, info] : as_map)
      {
        (void)as_idx;
        msg_size_in_bytes += sizeof(size_t) + sizeof(size_t);
        msg_size_in_bytes += info.psi_bytes;
      }
      max_message_bytes = std::max(max_message_bytes, msg_size_in_bytes);
    }

    std::vector<AngleSet*> base_angle_sets(angle_sets_.begin(), angle_sets_.end());
    async_comm_ = std::make_unique<CBCD_AsynchronousCommunicator>(
      base_angle_sets, angle_sets_.front()->GetCommunicatorSet(), max_message_bytes);
    for (auto* angle_set : angle_sets_)
      angle_set->SetCommunicator(*async_comm_);
    for (auto* fluds : fluds_list_)
      fluds->InitializeQueueIndices(*async_comm_);
  }
}

CBCDSweepChunk::~CBCDSweepChunk()
{
  StopCommunicator();
}

void
CBCDSweepChunk::StartCommunicator()
{
  if (async_comm_)
    async_comm_->Start();
}

void
CBCDSweepChunk::StopCommunicator()
{
  if (async_comm_)
    async_comm_->Stop();
}

void
CBCDSweepChunk::Sweep(const std::vector<std::uint32_t>& cell_local_ids, size_t angle_set_id)
{
  CALI_CXX_MARK_SCOPE("CBCDSweepChunk::Sweep");

  auto* fluds = fluds_list_[angle_set_id];
  auto* device_saved_psi = fluds->GetSavedAngularFluxDevicePointer();
  const auto& stream = streams_list_[angle_set_id];
  auto& host_cell_local_ids = fluds->GetLocalCellIDs();
  std::copy(cell_local_ids.begin(), cell_local_ids.end(), host_cell_local_ids.begin());
  const auto& args = kernel_args_list_[angle_set_id];
  ::dim3 block_size = block_sizes_[angle_set_id];
  unsigned int num_ready_cells = static_cast<unsigned int>(cell_local_ids.size());
  unsigned int grid_size_x = grid_size_x_list_[angle_set_id];
  unsigned int grid_size_y = (num_ready_cells + block_size.y - 1) / block_size.y;
  ::dim3 grid_size{grid_size_x, grid_size_y};
  gpu_kernel::SweepKernel<gpu_kernel::SweepType::CBC><<<grid_size, block_size, 0, stream>>>(
    args, host_cell_local_ids.data(), num_ready_cells, device_saved_psi);
}

} // namespace opensn

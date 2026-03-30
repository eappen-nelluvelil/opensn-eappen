// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_aggregated_comm.h"
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
    problem_(problem),
    angle_sets_(),
    fluds_list_()
{
  for (auto& as : *(groupset_.angle_agg))
  {
    auto* angle_set = static_cast<CBCD_AngleSet*>(as.get());
    angle_sets_.push_back(angle_set);
    fluds_list_.push_back(static_cast<CBCD_FLUDS*>(&(angle_set->GetFLUDS())));
  }

  const auto& grid = *problem_.GetGrid();

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
      const auto grouped_faces = common_data.GetIncomingNonlocalFaces(cell_local_id);
      for (const auto& face_info : grouped_faces)
      {
        if (face_info.num_nodes == 0)
          continue;
        auto& info = source_as_info[face_info.source_partition][as_idx];
        info.num_entries += 1;
        info.psi_bytes += sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(size_t) +
                          static_cast<size_t>(face_info.num_nodes) * stride * sizeof(double);
      }
    }
  }

  size_t max_message_bytes = 0;
  for (const auto& [source_partition, as_map] : source_as_info)
  {
    size_t msg_size_in_bytes = sizeof(size_t);
    for (const auto& [as_idx, info] : as_map)
    {
      msg_size_in_bytes += sizeof(size_t) + sizeof(size_t);
      msg_size_in_bytes += info.psi_bytes;
    }
    max_message_bytes = std::max(max_message_bytes, msg_size_in_bytes);
  }

  std::vector<AngleSet*> base_angle_sets(angle_sets_.begin(), angle_sets_.end());
  agg_comm_ = std::make_unique<CBCD_AggregatedCommunicator>(base_angle_sets,
                                                            problem_.GetMPICommunicatorSet(),
                                                            max_message_bytes);
  for (auto* as : angle_sets_)
  {
    as->SetAggregatedCommunicator(agg_comm_.get());
    as->SetSweepChunk(this);
  }
  for (auto* fluds : fluds_list_)
    fluds->InitializeQueueIndices(*agg_comm_);

  cached_kernel_params_.reserve(angle_sets_.size());
  for (size_t i = 0; i < angle_sets_.size(); ++i)
  {
    gpu_kernel::Arguments<gpu_kernel::SweepType::CBC> args(
      problem_, groupset_, *angle_sets_[i], *fluds_list_[i]);
    unsigned int stride_size =
      gpu_kernel::RoundUp(static_cast<unsigned int>(args.flud_data.stride_size));
    unsigned int block_size_x = std::min(stride_size, gpu_kernel::threshold);
    unsigned int block_size_y = gpu_kernel::threshold / block_size_x;
    unsigned int grid_size_x = (stride_size + gpu_kernel::threshold - 1) / gpu_kernel::threshold;
    double* device_saved_psi = fluds_list_[i]->GetSavedAngularFluxDevicePointer();
    cached_kernel_params_.emplace_back(args,
                                       ::dim3{block_size_x, block_size_y},
                                       grid_size_x,
                                       device_saved_psi);
  }
}

CBCDSweepChunk::~CBCDSweepChunk() = default;

void
CBCDSweepChunk::StartCommunicator()
{
  agg_comm_->Start();
}

void
CBCDSweepChunk::StopCommunicator()
{
  agg_comm_->Stop();
}

void
CBCDSweepChunk::CopySavedPsiToDestinationPsi()
{
  for (auto* fluds : fluds_list_)
    fluds->CopySavedPsiToHost();

  for (size_t i = 0; i < angle_sets_.size(); ++i)
    fluds_list_[i]->CopySavedPsiToDestinationPsi(*this, *angle_sets_[i]);
}

CBCD_AggregatedCommunicator&
CBCDSweepChunk::GetAggregatedCommunicator()
{
  return *agg_comm_;
}

void
CBCDSweepChunk::Sweep(CBCD_AngleSet& angle_set, unsigned int num_ready_cells)
{
  CALI_CXX_MARK_SCOPE("CBCDSweepChunk::Sweep");

  auto id = angle_set.GetID();
  auto& ck = cached_kernel_params_[id];
  auto& stream = angle_set.GetStream();
  unsigned int grid_size_y = (num_ready_cells + ck.block_size.y - 1) / ck.block_size.y;
  ::dim3 grid_size{ck.grid_size_x, grid_size_y};
  gpu_kernel::SweepKernel<gpu_kernel::SweepType::CBC><<<grid_size, ck.block_size, 0, stream>>>(
    ck.args, fluds_list_[id]->GetLocalCellIDs().data(), num_ready_cells, ck.device_saved_psi);
}

} // namespace opensn

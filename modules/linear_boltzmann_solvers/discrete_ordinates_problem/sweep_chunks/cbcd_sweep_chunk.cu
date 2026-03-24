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

namespace opensn
{

CBCDSweepChunk::CBCDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset)
  : SweepChunk(problem.GetPhiNewLocal(),
               problem.GetPsiNewLocal()[groupset.id],
               problem.GetGrid(),
               problem.GetSpatialDiscretization(),
               problem.GetUnitCellMatrices(),
               problem.GetCellTransportViews(),
               problem.GetDensitiesLocal(),
               problem.GetQMomentsLocal(),
               groupset,
               problem.GetBlockID2XSMap(),
               problem.GetNumMoments(),
               problem.GetMaxCellDOFCount(),
               problem.GetMinCellDOFCount()),
    problem_(problem),
    angle_sets_(),
    fluds_list_(),
    streams_list_()
{
  for (auto& as : *(groupset_.angle_agg))
  {
    auto* angle_set = static_cast<CBCD_AngleSet*>(as.get());
    auto* fluds = static_cast<CBCD_FLUDS*>(&(angle_set->GetFLUDS()));
    angle_sets_.push_back(angle_set);
    fluds_list_.push_back(fluds);
    streams_list_.push_back(&(angle_set->GetStream()));
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

  // Compute exact per-source worst-case message size for the receive buffer.
  // This is stored and used later when per-worker communicators are created.
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
    const auto& incoming_map = fluds.GetCommonData().GetIncomingNonlocalNodeMap();

    std::unordered_map<int, std::map<std::pair<std::uint64_t, unsigned int>, size_t>>
      source_face_node_counts;

    for (const auto& [cell_local_id, nodes] : incoming_map)
    {
      for (const auto& node : nodes)
      {
        int source_partition = grid.cells[node.cell_global_id].partition_id;
        source_face_node_counts[source_partition][{node.cell_global_id, node.face_id}]++;
      }
    }

    for (const auto& [source_partition, face_counts] : source_face_node_counts)
    {
      auto& info = source_as_info[source_partition][as_idx];
      for (const auto& [face_key, num_nodes] : face_counts)
      {
        info.num_entries += 1;
        info.psi_bytes += sizeof(std::uint64_t) + sizeof(unsigned int) +
                          sizeof(size_t) + num_nodes * stride * sizeof(double);
      }
    }
  }

  max_message_size_ = 0;
  for (const auto& [source_partition, as_map] : source_as_info)
  {
    size_t msg_size_in_bytes = sizeof(size_t);
    for (const auto& [as_idx, info] : as_map)
    {
      msg_size_in_bytes += sizeof(size_t) + sizeof(size_t);
      msg_size_in_bytes += info.psi_bytes;
    }
    max_message_size_ = std::max(max_message_size_, msg_size_in_bytes);
  }

  // Note: communicators are NOT created here. They are created lazily by
  // SetupPerWorkerCommunicators() once the number of workers is known.
}

CBCDSweepChunk::~CBCDSweepChunk() = default;

void
CBCDSweepChunk::SetupPerWorkerCommunicators(size_t num_workers)
{
  if (setup_num_workers_ == num_workers)
    return; // Already set up for this worker count.

  const size_t num_angle_sets = angle_sets_.size();
  const size_t total_num_angle_sets = num_angle_sets; // Used for MPI tag base

  // Clear any previous communicators
  agg_comms_.clear();
  agg_comms_.resize(num_workers);

  for (size_t w = 0; w < num_workers; ++w)
  {
    // Compute this worker's contiguous angle set range [begin, end)
    const size_t chunk_size = (num_angle_sets + num_workers - 1) / num_workers;
    const size_t begin = w * chunk_size;
    const size_t end = std::min(begin + chunk_size, num_angle_sets);

    if (begin >= num_angle_sets)
    {
      // This worker has no angle sets (more workers than angle sets).
      // Create a dummy communicator with an empty angle set list.
      std::vector<AngleSet*> empty_angle_sets;
      agg_comms_[w] = std::make_unique<CBCD_AggregatedCommunicator>(
        empty_angle_sets,
        problem_.GetMPICommunicatorSet(),
        0,
        begin,
        static_cast<int>(total_num_angle_sets + w));
      continue;
    }

    // Build the angle set list for this worker
    std::vector<AngleSet*> worker_angle_sets;
    worker_angle_sets.reserve(end - begin);
    for (size_t i = begin; i < end; ++i)
      worker_angle_sets.push_back(angle_sets_[i]);

    // MPI tag: total_num_angle_sets + worker_id
    // This ensures no collision with per-angle-set tags [0, N-1]
    // and uniqueness across workers.
    int tag = static_cast<int>(total_num_angle_sets + w);

    agg_comms_[w] = std::make_unique<CBCD_AggregatedCommunicator>(
      worker_angle_sets,
      problem_.GetMPICommunicatorSet(),
      max_message_size_,
      begin,
      tag);

    // Set each angle set's communicator pointer to this worker's communicator
    for (size_t i = begin; i < end; ++i)
      angle_sets_[i]->SetAggregatedCommunicator(agg_comms_[w].get());
  }

  setup_num_workers_ = num_workers;
}

void
CBCDSweepChunk::StartCommunicators()
{
  for (auto& comm : agg_comms_)
    if (comm)
      comm->Start();
}

void
CBCDSweepChunk::StopCommunicators()
{
  // Request stop on all communicators first, then join.
  // Each comm thread independently checks its own stop_requested_ flag,
  // so they can wind down in parallel.
  for (auto& comm : agg_comms_)
    if (comm)
      comm->Stop();
}

CBCD_AggregatedCommunicator&
CBCDSweepChunk::GetAggregatedCommunicator(size_t worker_id)
{
  return *agg_comms_[worker_id];
}

void
CBCDSweepChunk::Sweep(CBCD_AngleSet& angle_set, const std::vector<std::uint32_t>& cell_local_ids)
{
  CALI_CXX_MARK_SCOPE("CBCDSweepChunk::Sweep");

  auto& fluds = fluds_list_[angle_set.GetID()];
  auto* device_saved_psi = fluds->GetSavedAngularFluxDevicePointer();
  const auto& stream = streams_list_[angle_set.GetID()];
  auto& host_cell_local_ids = fluds->GetLocalCellIDs();
  std::copy(cell_local_ids.begin(), cell_local_ids.end(), host_cell_local_ids.begin());
  const auto& args = kernel_args_list_[angle_set.GetID()];
  ::dim3 block_size = block_sizes_[angle_set.GetID()];
  unsigned int num_ready_cells = static_cast<unsigned int>(cell_local_ids.size());
  unsigned int grid_size_x = grid_size_x_list_[angle_set.GetID()];
  unsigned int grid_size_y = (num_ready_cells + block_size.y - 1) / block_size.y;
  ::dim3 grid_size{grid_size_x, grid_size_y};
  gpu_kernel::SweepKernel<gpu_kernel::SweepType::CBC><<<grid_size, block_size, 0, *stream>>>(
    args, host_cell_local_ids.data(), num_ready_cells, device_saved_psi);
}

} // namespace opensn

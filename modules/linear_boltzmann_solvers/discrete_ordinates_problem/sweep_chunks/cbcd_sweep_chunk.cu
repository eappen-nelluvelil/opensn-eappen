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

  // Compute exact per-source worst-case message size for the receive buffer
  // For each dependency location J, sum the wire-format size of all incoming
  // non-local face data from J across all angle sets.
  // The maximum over all J gives the tightest possible reservation for
  // persistent_recv_buffer_.
  const auto& grid = *problem_.GetGrid();

  // Per-source, per-angle-set: accumulate (num_entries, total_psi_bytes).
  // Key: source partition ID
  // Value: per-angle-set entry counts and data sizes
  struct PerSourceAngleSetInfo
  {
    size_t num_entries = 0;
    size_t psi_bytes = 0;
  };

  // Source partition -> (angle_set_id -> per-angle-set info)
  std::unordered_map<int, std::unordered_map<size_t, PerSourceAngleSetInfo>> source_as_info;
  for (size_t as_idx = 0; as_idx < angle_sets_.size(); ++as_idx)
  {
    auto& fluds = *fluds_list_[as_idx];
    const auto stride = fluds.GetStrideSize();
    const auto& incoming_map = fluds.GetCommonData().GetIncomingNonlocalNodeMap();

    // Group incoming nodes by (source_partition, cell_global_id, face_id) to get
    // the face node count for each distinct face entry.
    // face_key: (cell_global_id, face_id) -> node count
    std::unordered_map<int, std::map<std::pair<std::uint64_t, unsigned int>, size_t>>
      source_face_node_counts;

    for (const auto& [cell_local_id, nodes] : incoming_map)
    {
      for (const auto& node : nodes)
      {
        // Resolve source partition from cell_global_id
        int source_partition = grid.cells[node.cell_global_id].partition_id;
        source_face_node_counts[source_partition][{node.cell_global_id, node.face_id}]++;
      }
    }

    // Convert per-face counts to per-source per-angleset totals
    for (const auto& [source_partition, face_counts] : source_face_node_counts)
    {
      auto& info = source_as_info[source_partition][as_idx];
      for (const auto& [face_key, num_nodes] : face_counts)
      {
        info.num_entries += 1; // Each face corresponds to one entry in the message
        // Per entry: cell_global_id + face_id + data_size + psi_data
        info.psi_bytes += sizeof(std::uint64_t) + sizeof(unsigned int) + 
                          sizeof(size_t) + num_nodes * stride * sizeof(double);
      }
    }
  }

  // Compute the worst-case message size: max over all sources
  size_t max_single_message_size_in_bytes = 0;
  for (const auto& [source_partition, as_map] : source_as_info)
  {
    // num_active_angle_sets header
    size_t msg_size_in_bytes = sizeof(size_t);
    for (const auto& [as_idx, info] : as_map)
    {
      // Per active angle set: as_id + num_entries
      msg_size_in_bytes += sizeof(size_t) + sizeof(size_t);
      // Data for all entries in this angle set
      msg_size_in_bytes += info.psi_bytes;
    }
    max_single_message_size_in_bytes = std::max(max_single_message_size_in_bytes, msg_size_in_bytes);
  }

  // Create aggregated communicator and set it on all angle sets
  std::vector<AngleSet*> base_angle_sets(angle_sets_.begin(), angle_sets_.end());
  agg_comm_ = std::make_unique<CBCD_AggregatedCommunicator>(base_angle_sets,
                                                            problem_.GetMPICommunicatorSet(),
                                                            max_single_message_size_in_bytes);
  for (auto* as : angle_sets_)
    as->SetAggregatedCommunicator(agg_comm_.get());
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

CBCD_AggregatedCommunicator&
CBCDSweepChunk::GetAggregatedCommunicator()
{
  return *agg_comm_;
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

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/main.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/round_up.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/device_vector_mirror.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "caliper/cali.h"
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

namespace opensn
{

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
    auto* angle_set = static_cast<CBCD_AngleSet*>(as.get());
    auto* fluds = static_cast<CBCD_FLUDS*>(&(angle_set->GetFLUDS()));
    angle_sets_.push_back(angle_set);
    fluds_list.push_back(fluds);

    gpu_kernel::Arguments<SweepKind::CBC> args(
      problem_, groupset_, *angle_set, *fluds, IsSurfaceSourceActive());
    const auto stride_size =
      gpu_kernel::RoundUp(static_cast<unsigned int>(args.flud_data.stride_size));
    const auto block_size_x = std::min(stride_size, gpu_kernel::threshold);
    const auto block_size_y = gpu_kernel::threshold / block_size_x;
    const auto grid_size_x = (stride_size + gpu_kernel::threshold - 1) / gpu_kernel::threshold;
    cached_params_.push_back({args,
                              crb::Dim3(block_size_x, block_size_y),
                              grid_size_x,
                              fluds,
                              fluds->GetCommonData().HasDelayedFluxes(),
                              fluds->GetSavedAngularFluxDevicePointer()});
  }

  if (not angle_sets_.empty())
  {
    std::vector<std::vector<int>> incoming_source_partitions_by_angle_set;
    std::vector<std::vector<int>> delayed_incoming_source_partitions_by_angle_set;
    incoming_source_partitions_by_angle_set.reserve(angle_sets_.size());
    delayed_incoming_source_partitions_by_angle_set.reserve(angle_sets_.size());
    std::vector<AngleSetCapacity> capacities(angle_sets_.size());
    std::map<int, DestinationCapacity> destination_capacity_map;
    constexpr std::size_t section_header_bytes = sizeof(std::uint8_t) + 2 * sizeof(std::size_t);
    const auto checked_add = [](const std::size_t lhs,
                                const std::size_t rhs,
                                const char* const description)
    {
      if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
        throw std::overflow_error(description);
      return lhs + rhs;
    };
    const auto checked_multiply = [](const std::size_t lhs,
                                     const std::size_t rhs,
                                     const char* const description)
    {
      if (lhs != 0 and rhs > std::numeric_limits<std::size_t>::max() / lhs)
        throw std::overflow_error(description);
      return lhs * rhs;
    };
    const auto account_outgoing_face = [&destination_capacity_map,
                                        &checked_add,
                                        &checked_multiply](
                                         const int dest_rank,
                                         const CBCDMessageKind kind,
                                         const std::size_t payload_values)
    {
      auto& capacity = destination_capacity_map[dest_rank];
      capacity.dest_rank = dest_rank;
      const auto kind_index = static_cast<std::size_t>(kind);
      auto& records = capacity.records[kind_index];
      auto& builder_bytes = capacity.builder_bytes[kind_index];
      if (records == 0)
        builder_bytes = sizeof(std::size_t);
      const auto payload_bytes = checked_multiply(
        payload_values, sizeof(double), "CBCD communicator: static face-payload size overflow.");
      const auto record_bytes = checked_add(sizeof(std::uint32_t) + sizeof(std::size_t),
                                            payload_bytes,
                                            "CBCD communicator: static record size overflow.");
      builder_bytes = checked_add(
        builder_bytes,
        checked_add(section_header_bytes,
                    record_bytes,
                    "CBCD communicator: static section size overflow."),
        "CBCD communicator: static destination size overflow.");
      records = checked_add(records, 1, "CBCD communicator: static record-count overflow.");
    };
    for (std::size_t as_ss_idx = 0; as_ss_idx < angle_sets_.size(); ++as_ss_idx)
    {
      const auto stride = fluds_list[as_ss_idx]->GetStrideSize();
      const auto& common_data = fluds_list[as_ss_idx]->GetCommonData();
      incoming_source_partitions_by_angle_set.push_back(common_data.GetIncomingSourcePartitions());
      delayed_incoming_source_partitions_by_angle_set.push_back(
        common_data.GetDelayedIncomingSourcePartitions());
      // Outgoing queue capacity includes normal and delayed face payloads. Delayed
      // completion is tracked by the exact number of received delayed face records.
      capacities[as_ss_idx].outgoing_faces = common_data.GetNumOutgoingNonlocalFaces() +
                                             common_data.GetNumDelayedOutgoingNonlocalFaces();
      capacities[as_ss_idx].incoming_faces = common_data.GetNumIncomingNonlocalFaces() +
                                             common_data.GetNumDelayedIncomingNonlocalFaces();
      capacities[as_ss_idx].incoming_faces_by_source.assign(
        common_data.GetIncomingSourcePartitions().size(), 0);
      capacities[as_ss_idx].delayed_incoming_faces =
        common_data.GetNumDelayedIncomingNonlocalFaces();
      capacities[as_ss_idx].delayed_incoming_faces_by_source.assign(
        common_data.GetDelayedIncomingSourcePartitions().size(), 0);
      for (std::size_t cell_local_id = 0; cell_local_id < common_data.GetNumLocalCells();
           ++cell_local_id)
      {
        for (const auto& face_info : common_data.GetOutgoingNonlocalFaces(cell_local_id))
        {
          const auto payload_values = checked_multiply(
            face_info.num_face_nodes, stride, "CBCD communicator: static face-value overflow.");
          capacities[as_ss_idx].max_outgoing_face_values =
            std::max(capacities[as_ss_idx].max_outgoing_face_values,
                     payload_values);
          account_outgoing_face(common_data.GetOutgoingLocalities()[face_info.dest_slot],
                                CBCDMessageKind::NORMAL_FACE_PSI,
                                payload_values);
        }
        for (const auto& face_info : common_data.GetIncomingNonlocalFaces(cell_local_id))
          ++capacities[as_ss_idx].incoming_faces_by_source[face_info.source_slot];
        for (const auto& face_info : common_data.GetDelayedOutgoingNonlocalFaces(cell_local_id))
        {
          const auto payload_values = checked_multiply(
            face_info.num_face_nodes, stride, "CBCD communicator: static face-value overflow.");
          capacities[as_ss_idx].max_outgoing_face_values =
            std::max(capacities[as_ss_idx].max_outgoing_face_values,
                     payload_values);
          account_outgoing_face(
            common_data.GetDelayedOutgoingLocalities()[face_info.dest_slot],
            CBCDMessageKind::DELAYED_FACE_PSI,
            payload_values);
        }
        for (const auto& face_info : common_data.GetDelayedIncomingNonlocalFaces(cell_local_id))
          ++capacities[as_ss_idx].delayed_incoming_faces_by_source[face_info.source_slot];
      }
    }

    std::vector<DestinationCapacity> destination_capacities;
    destination_capacities.reserve(destination_capacity_map.size());
    for (const auto& [dest_rank, capacity] : destination_capacity_map)
      destination_capacities.push_back(capacity);

    std::vector<AngleSet*> base_angle_sets(angle_sets_.begin(), angle_sets_.end());
    async_comm_ = std::make_unique<CBCD_AsynchronousCommunicator>(
      base_angle_sets,
      angle_sets_.front()->GetCommunicatorSet(),
      incoming_source_partitions_by_angle_set,
      delayed_incoming_source_partitions_by_angle_set,
      capacities,
      destination_capacities);
    for (auto* angle_set : angle_sets_)
      angle_set->SetCommunicator(*async_comm_);
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
CBCDSweepChunk::RefreshCachedKernelArgs()
{
  CALI_CXX_MARK_SCOPE("CBCDSweepChunk::RefreshCachedKernelArgs");

  for (std::size_t angle_set_id = 0; angle_set_id < angle_sets_.size(); ++angle_set_id)
  {
    auto& ck = cached_params_[angle_set_id];
    {
      CALI_CXX_MARK_SCOPE("CBCDSweepChunk::Sweep::ArgsRefresh");
      ck.args = gpu_kernel::Arguments<SweepKind::CBC>(
        problem_, groupset_, *angle_sets_[angle_set_id], *ck.fluds, IsSurfaceSourceActive());
      ck.device_saved_psi = ck.fluds->GetSavedAngularFluxDevicePointer();
    }
  }
}

void
CBCDSweepChunk::Sweep(std::uint32_t num_ready_cells,
                      std::size_t angle_set_id,
                      const std::uint32_t* local_cell_ids)
{
  CALI_CXX_MARK_SCOPE("CBCDSweepChunk::Sweep");

  auto& ck = cached_params_[angle_set_id];
  auto& stream = angle_sets_[angle_set_id]->GetStream();
  const auto grid_size_y = (num_ready_cells + ck.block_size.y - 1) / ck.block_size.y;
  crb::Dim3 grid_size(ck.grid_size_x, grid_size_y);
  {
    CALI_CXX_MARK_SCOPE("CBCDSweepChunk::Sweep::KernelLaunch");
#if defined(__NVCC__) || defined(__HIPCC__)
    if (ck.use_delayed_fluxes)
      gpu_kernel::SweepKernel<SweepKind::CBC, true><<<grid_size, ck.block_size, 0, stream>>>(
        ck.args, local_cell_ids, num_ready_cells, ck.device_saved_psi);
    else
      gpu_kernel::SweepKernel<SweepKind::CBC, false><<<grid_size, ck.block_size, 0, stream>>>(
        ck.args, local_cell_ids, num_ready_cells, ck.device_saved_psi);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    stream.synchronize();
    if (ck.use_delayed_fluxes)
      stream.parallel_for(sycl::nd_range<3>(grid_size * ck.block_size, ck.block_size),
                          [=](sycl::nd_item<3> work_index)
                          {
                            gpu_kernel::SweepKernel<SweepKind::CBC, true>(
                              ck.args, local_cell_ids, num_ready_cells, ck.device_saved_psi);
                          });
    else
      stream.parallel_for(sycl::nd_range<3>(grid_size * ck.block_size, ck.block_size),
                          [=](sycl::nd_item<3> work_index)
                          {
                            gpu_kernel::SweepKernel<SweepKind::CBC, false>(
                              ck.args, local_cell_ids, num_ready_cells, ck.device_saved_psi);
                          });
#endif
  }
}

} // namespace opensn

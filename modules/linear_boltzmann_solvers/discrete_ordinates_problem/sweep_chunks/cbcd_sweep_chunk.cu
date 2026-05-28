// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/main.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/round_up.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/device_vector_mirror.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "caliper/cali.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace opensn
{

namespace
{

std::size_t
CheckedAdd(const std::size_t lhs, const std::size_t rhs, const char* const description)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::overflow_error(description);
  return lhs + rhs;
}

std::size_t
CheckedMultiply(const std::size_t lhs, const std::size_t rhs, const char* const description)
{
  if (lhs != 0 and rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::overflow_error(description);
  return lhs * rhs;
}

constexpr std::size_t message_header_bytes = sizeof(std::size_t);
constexpr std::size_t section_header_bytes = sizeof(std::uint8_t) + 2 * sizeof(std::size_t);
constexpr std::size_t entry_header_bytes = sizeof(std::uint32_t) + sizeof(std::size_t);
constexpr auto mpi_count_limit = static_cast<std::size_t>(std::numeric_limits<int>::max());

void
ValidateWireEntrySize(const std::size_t entry_bytes)
{
  if (entry_bytes > mpi_count_limit - message_header_bytes - section_header_bytes)
    throw std::overflow_error("CBCD sweep chunk: one face payload exceeds the MPI count range.");
}

} // namespace

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
                              fluds->GetSavedAngularFluxDevicePointer()});
  }

  const bool has_nonlocal_communication =
    std::any_of(fluds_list.begin(),
                fluds_list.end(),
                [](const CBCD_FLUDS* fluds)
                {
                  const auto& common_data = fluds->GetCommonData();
                  return common_data.GetNumIncomingNonlocalFaces() > 0 or
                         common_data.GetNumOutgoingNonlocalFaces() > 0 or
                         common_data.GetNumDelayedIncomingNonlocalFaces() > 0 or
                         common_data.GetNumDelayedOutgoingNonlocalFaces() > 0;
                });
  if (has_nonlocal_communication)
  {
    std::vector<std::vector<int>> incoming_source_partitions_by_angle_set;
    std::vector<std::vector<int>> delayed_incoming_source_partitions_by_angle_set;
    incoming_source_partitions_by_angle_set.reserve(angle_sets_.size());
    delayed_incoming_source_partitions_by_angle_set.reserve(angle_sets_.size());
    // Payload bytes keyed by destination partition, message kind, and angle set. This is
    // the exact upper bound relevant to this rank's outgoing message serialization.
    std::unordered_map<int, std::array<std::vector<std::size_t>, 3>> destination_section_bytes;
    std::vector<AngleSetCapacity> capacities(angle_sets_.size());
    for (std::size_t as_ss_idx = 0; as_ss_idx < angle_sets_.size(); ++as_ss_idx)
    {
      const auto stride = fluds_list[as_ss_idx]->GetStrideSize();
      const auto& common_data = fluds_list[as_ss_idx]->GetCommonData();
      std::unordered_map<int, std::size_t> outgoing_records_by_destination;
      incoming_source_partitions_by_angle_set.push_back(common_data.GetIncomingSourcePartitions());
      delayed_incoming_source_partitions_by_angle_set.push_back(
        common_data.GetDelayedIncomingSourcePartitions());
      // Outgoing queue capacity includes (a) normal outgoing face payloads, (b) delayed
      // outgoing face payloads, and (c) one delayed-completion marker per
      // delayed-destination locality.  Each item occupies one queue slot.
      capacities[as_ss_idx].outgoing_faces =
        CheckedAdd(CheckedAdd(common_data.GetNumOutgoingNonlocalFaces(),
                              common_data.GetNumDelayedOutgoingNonlocalFaces(),
                              "CBCD sweep chunk: outgoing face-count overflow."),
                   common_data.GetDelayedOutgoingLocalities().size(),
                   "CBCD sweep chunk: outgoing record-count overflow.");
      capacities[as_ss_idx].incoming_faces =
        CheckedAdd(common_data.GetNumIncomingNonlocalFaces(),
                   common_data.GetNumDelayedIncomingNonlocalFaces(),
                   "CBCD sweep chunk: incoming face-count overflow.");
      for (std::size_t cell_local_id = 0; cell_local_id < common_data.GetNumLocalCells();
           ++cell_local_id)
      {
        for (const auto& face_info : common_data.GetOutgoingNonlocalFaces(cell_local_id))
        {
          if (face_info.dest_slot >= common_data.GetOutgoingLocalities().size())
            throw std::logic_error("CBCD sweep chunk: invalid outgoing destination slot.");
          const int dest_rank = common_data.GetOutgoingLocalities()[face_info.dest_slot];
          outgoing_records_by_destination[dest_rank] =
            CheckedAdd(outgoing_records_by_destination[dest_rank],
                       1,
                       "CBCD sweep chunk: outgoing record-count overflow.");
          const auto num_values =
            CheckedMultiply(static_cast<std::size_t>(face_info.num_face_nodes),
                            stride,
                            "CBCD sweep chunk: outgoing face-value count overflow.");
          auto& per_as_bytes =
            destination_section_bytes[dest_rank]
                                     [static_cast<std::size_t>(CBCDMessageKind::NORMAL_FACE_PSI)];
          if (per_as_bytes.empty())
            per_as_bytes.assign(angle_sets_.size(), 0);
          const auto entry_bytes = CheckedAdd(
            entry_header_bytes,
            CheckedMultiply(
              num_values, sizeof(double), "CBCD sweep chunk: outgoing face-payload size overflow."),
            "CBCD sweep chunk: outgoing wire-entry size overflow.");
          ValidateWireEntrySize(entry_bytes);
          per_as_bytes[as_ss_idx] =
            CheckedAdd(per_as_bytes[as_ss_idx],
                       entry_bytes,
                       "CBCD sweep chunk: outgoing wire-section size overflow.");
        }
        for (const auto& face_info : common_data.GetDelayedOutgoingNonlocalFaces(cell_local_id))
        {
          if (face_info.dest_slot >= common_data.GetDelayedOutgoingLocalities().size())
            throw std::logic_error("CBCD sweep chunk: invalid delayed destination slot.");
          const int dest_rank = common_data.GetDelayedOutgoingLocalities()[face_info.dest_slot];
          outgoing_records_by_destination[dest_rank] =
            CheckedAdd(outgoing_records_by_destination[dest_rank],
                       1,
                       "CBCD sweep chunk: delayed outgoing record-count overflow.");
          const auto num_values =
            CheckedMultiply(static_cast<std::size_t>(face_info.num_face_nodes),
                            stride,
                            "CBCD sweep chunk: delayed outgoing face-value count overflow.");
          auto& per_as_bytes =
            destination_section_bytes[dest_rank]
                                     [static_cast<std::size_t>(CBCDMessageKind::DELAYED_FACE_PSI)];
          if (per_as_bytes.empty())
            per_as_bytes.assign(angle_sets_.size(), 0);
          const auto entry_bytes = CheckedAdd(
            entry_header_bytes,
            CheckedMultiply(num_values,
                            sizeof(double),
                            "CBCD sweep chunk: delayed outgoing face-payload size overflow."),
            "CBCD sweep chunk: delayed outgoing wire-entry size overflow.");
          ValidateWireEntrySize(entry_bytes);
          per_as_bytes[as_ss_idx] =
            CheckedAdd(per_as_bytes[as_ss_idx],
                       entry_bytes,
                       "CBCD sweep chunk: delayed outgoing wire-section size overflow.");
        }
      }
      for (const int dest_rank : common_data.GetDelayedOutgoingLocalities())
      {
        outgoing_records_by_destination[dest_rank] =
          CheckedAdd(outgoing_records_by_destination[dest_rank],
                     1,
                     "CBCD sweep chunk: completion record-count overflow.");
        auto& completion_sections =
          destination_section_bytes[dest_rank]
                                   [static_cast<std::size_t>(CBCDMessageKind::DELAYED_COMPLETION)];
        if (completion_sections.empty())
          completion_sections.assign(angle_sets_.size(), 0);
        completion_sections[as_ss_idx] = 1;
      }
      capacities[as_ss_idx].outgoing_faces_by_destination.reserve(
        outgoing_records_by_destination.size());
      for (const auto& [dest_rank, face_count] : outgoing_records_by_destination)
        capacities[as_ss_idx].outgoing_faces_by_destination.push_back({dest_rank, face_count});

      std::unordered_map<std::uint32_t, std::size_t> incoming_entries_by_source_slot;
      std::unordered_map<std::uint32_t, std::size_t> incoming_values_by_source_slot;
      for (std::size_t cell_local_id = 0; cell_local_id < common_data.GetNumLocalCells();
           ++cell_local_id)
      {
        for (const auto& face_info : common_data.GetIncomingNonlocalFaces(cell_local_id))
        {
          if (face_info.num_nodes == 0)
            continue;
          if (face_info.source_slot >= common_data.GetIncomingSourcePartitions().size())
            throw std::logic_error("CBCD sweep chunk: invalid incoming source slot.");
          ++incoming_entries_by_source_slot[face_info.source_slot];
          const auto num_values =
            CheckedMultiply(static_cast<std::size_t>(face_info.num_nodes),
                            stride,
                            "CBCD sweep chunk: incoming face-value count overflow.");
          incoming_values_by_source_slot[face_info.source_slot] =
            CheckedAdd(incoming_values_by_source_slot[face_info.source_slot],
                       num_values,
                       "CBCD sweep chunk: incoming batch-value count overflow.");
        }
        // Mirror the sizing loop for delayed incoming faces so the receiver's mailbox
        // batches are big enough to hold delayed-face-psi sections too.  The
        // delayed-source-slot indices are independent from normal source-slot indices. The
        // message model retains a distinct section for each kind because normal, delayed,
        // and delayed-completion records may be aggregated into the same MPI payload.
        for (const auto& face_info : common_data.GetDelayedIncomingNonlocalFaces(cell_local_id))
        {
          if (face_info.num_nodes == 0)
            continue;
          if (face_info.source_slot >= common_data.GetDelayedIncomingSourcePartitions().size())
            throw std::logic_error("CBCD sweep chunk: invalid delayed source slot.");
          ++incoming_entries_by_source_slot[face_info.source_slot];
          const auto num_values =
            CheckedMultiply(static_cast<std::size_t>(face_info.num_nodes),
                            stride,
                            "CBCD sweep chunk: delayed face-value count overflow.");
          incoming_values_by_source_slot[face_info.source_slot] =
            CheckedAdd(incoming_values_by_source_slot[face_info.source_slot],
                       num_values,
                       "CBCD sweep chunk: delayed batch-value count overflow.");
        }
      }
      for (const auto& [_, count] : incoming_entries_by_source_slot)
        capacities[as_ss_idx].max_incoming_batch_entries =
          std::max(capacities[as_ss_idx].max_incoming_batch_entries, count);
      for (const auto& [_, values] : incoming_values_by_source_slot)
        capacities[as_ss_idx].max_incoming_batch_values =
          std::max(capacities[as_ss_idx].max_incoming_batch_values, values);
    }

    std::size_t max_message_bytes = 0;
    for (const auto& [_, per_kind_bytes] : destination_section_bytes)
    {
      std::size_t msg_size_in_bytes = message_header_bytes;
      for (std::size_t kind_index = 0; kind_index < per_kind_bytes.size(); ++kind_index)
      {
        const bool is_completion =
          kind_index == static_cast<std::size_t>(CBCDMessageKind::DELAYED_COMPLETION);
        const auto& per_as_bytes = per_kind_bytes[kind_index];
        for (const auto& section_bytes : per_as_bytes)
        {
          if (section_bytes == 0)
            continue;
          const auto wire_section_bytes =
            CheckedAdd(section_header_bytes,
                       is_completion ? 0 : section_bytes,
                       "CBCD sweep chunk: wire-section size overflow.");
          msg_size_in_bytes = wire_section_bytes > mpi_count_limit - msg_size_in_bytes
                                ? mpi_count_limit
                                : msg_size_in_bytes + wire_section_bytes;
        }
      }
      max_message_bytes = std::max(max_message_bytes, std::min(msg_size_in_bytes, mpi_count_limit));
    }

    std::vector<AngleSet*> base_angle_sets(angle_sets_.begin(), angle_sets_.end());
    async_comm_ = std::make_unique<CBCD_AsynchronousCommunicator>(
      base_angle_sets,
      angle_sets_.front()->GetCommunicatorSet(),
      incoming_source_partitions_by_angle_set,
      delayed_incoming_source_partitions_by_angle_set,
      max_message_bytes,
      capacities);
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
    gpu_kernel::SweepKernel<SweepKind::CBC><<<grid_size, ck.block_size, 0, stream>>>(
      ck.args, local_cell_ids, num_ready_cells, ck.device_saved_psi);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    stream.synchronize();
    stream.parallel_for(sycl::nd_range<3>(grid_size * ck.block_size, ck.block_size),
                        [=](sycl::nd_item<3> work_index)
                        {
                          gpu_kernel::SweepKernel<SweepKind::CBC>(
                            ck.args, local_cell_ids, num_ready_cells, ck.device_saved_psi);
                        });
#endif
  }
}

} // namespace opensn

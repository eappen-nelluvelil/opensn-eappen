// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_aggregated_comm.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "framework/math/unknown_manager/unknown_manager.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/utils/error.h"
#include <algorithm>
#include <cstring>

namespace opensn
{

CBCD_FLUDS::CBCD_FLUDS(size_t num_groups,
                       size_t num_angles,
                       size_t num_local_cells,
                       const CBCD_FLUDSCommonData& common_data,
                       const UnknownManager& psi_uk_man,
                       const SpatialDiscretization& sdm,
                       bool save_angular_flux)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(psi_uk_man),
    local_psi_data_size_((sdm.GetNumLocalDOFs(psi_uk_man) / psi_uk_man.GetNumberOfUnknowns() /
                          num_groups) *
                         num_groups_and_angles_),
    incoming_boundary_psi_(common_data_.GetNumIncomingBoundaryNodes() * num_groups_and_angles_),
    outgoing_boundary_psi_(common_data_.GetNumOutgoingBoundaryNodes() * num_groups_and_angles_),
    incoming_nonlocal_psi_(common_data_.GetNumIncomingNonlocalNodes() * num_groups_and_angles_),
    outgoing_nonlocal_psi_(common_data_.GetNumOutgoingNonlocalNodes() * num_groups_and_angles_),
    local_cell_ids_(num_local_cells),
    save_angular_flux_(save_angular_flux)
{
  grid_ptr_ = GetSPDS().GetGrid().get();

  const auto& outgoing_localities = common_data_.GetOutgoingLocalities();
  outgoing_destinations_.reserve(outgoing_localities.size());
  for (const int locality : outgoing_localities)
    outgoing_destinations_.push_back({locality, -1});

  outgoing_node_memcpy_plan_.reserve(common_data_.GetNumOutgoingNonlocalNodes());
  outgoing_face_pack_plans_.resize(common_data_.GetNumOutgoingNonlocalFaces());
  for (size_t cell_local_id = 0; cell_local_id < common_data_.GetNumLocalCells(); ++cell_local_id)
  {
    for (const auto& face_info : common_data_.GetOutgoingNonlocalFaces(cell_local_id))
    {
      outgoing_face_pack_plans_[face_info.pack_plan_index].payload_doubles =
        static_cast<size_t>(face_info.num_face_nodes) * num_groups_and_angles_;
      for (const auto& node : common_data_.GetOutgoingNodeCopies(face_info))
      {
        outgoing_node_memcpy_plan_.push_back(
          {static_cast<size_t>(node.storage_index) * num_groups_and_angles_,
           static_cast<size_t>(node.face_node) * num_groups_and_angles_});
      }
    }
  }

  const size_t num_dests = outgoing_destinations_.size();
  scratch_dest_face_counts_.resize(num_dests, 0);
  scratch_dest_touched_.resize(num_dests, 0);
  active_dest_indices_.reserve(num_dests);
  dest_buffers_.resize(num_dests);
}

CBCD_FLUDS::~CBCD_FLUDS()
{
  local_psi_.async_free(stream_);
  if (not host_saved_psi_.empty())
  {
    host_saved_psi_.clear();
    device_saved_psi_.async_free(stream_);
  }
  local_cell_ids_.clear();
  incoming_boundary_psi_.clear();
  outgoing_boundary_psi_.clear();
  incoming_nonlocal_psi_.clear();
  outgoing_nonlocal_psi_.clear();
}

void
CBCD_FLUDS::AllocateLocalAndSavedPsi()
{
  local_psi_ = crb::DeviceMemory<double>(local_psi_data_size_, stream_);
  if (save_angular_flux_ and host_saved_psi_.empty())
  {
    host_saved_psi_ = crb::HostVector<double>(local_psi_data_size_);
    device_saved_psi_ = crb::DeviceMemory<double>(local_psi_data_size_, stream_);
  }
  CreatePointerSet();
}

void
CBCD_FLUDS::InitializeQueueIndices(const CBCD_AggregatedCommunicator& agg_comm)
{
  for (auto& dest : outgoing_destinations_)
    dest.queue_index = agg_comm.GetQueueIndex(dest.locality);
}

void
CBCD_FLUDS::InitializeReflectingBoundaryNodes(
  const std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries)
{
  const auto num_local_cells = common_data_.GetNumLocalCells();
  reflecting_outgoing_boundary_node_offsets_.assign(num_local_cells + 1, 0);
  reflecting_outgoing_boundary_nodes_.clear();
  reflecting_outgoing_boundary_nodes_.reserve(common_data_.GetNumOutgoingBoundaryNodes());

  for (size_t cell_local_id = 0; cell_local_id < num_local_cells; ++cell_local_id)
  {
    reflecting_outgoing_boundary_node_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(reflecting_outgoing_boundary_nodes_.size());
    for (const auto& node : common_data_.GetOutgoingBoundaryNodes(cell_local_id))
    {
      const auto boundary_it = boundaries.find(node.boundary_id);
      if (boundary_it != boundaries.end() and boundary_it->second->IsReflecting())
        reflecting_outgoing_boundary_nodes_.push_back(node);
    }
    reflecting_outgoing_boundary_node_offsets_[cell_local_id + 1] =
      static_cast<std::uint32_t>(reflecting_outgoing_boundary_nodes_.size());
  }
}

void
CBCD_FLUDS::CreatePointerSet()
{
  pointer_set_.local_psi = local_psi_.get();
  if (local_psi_data_size_ > 0)
    assert(pointer_set_.local_psi != nullptr);

  pointer_set_.incoming_boundary_psi = incoming_boundary_psi_.data();
  if (common_data_.GetNumIncomingBoundaryNodes() > 0)
    assert(pointer_set_.incoming_boundary_psi != nullptr);

  pointer_set_.outgoing_boundary_psi = outgoing_boundary_psi_.data();
  if (common_data_.GetNumOutgoingBoundaryNodes() > 0)
    assert(pointer_set_.outgoing_boundary_psi != nullptr);

  pointer_set_.nonlocal_incoming_psi = incoming_nonlocal_psi_.data();
  if (common_data_.GetNumIncomingNonlocalNodes() > 0)
    assert(pointer_set_.nonlocal_incoming_psi != nullptr);

  pointer_set_.nonlocal_outgoing_psi = outgoing_nonlocal_psi_.data();
  if (common_data_.GetNumOutgoingNonlocalNodes() > 0)
    assert(pointer_set_.nonlocal_outgoing_psi != nullptr);

  pointer_set_.stride_size = num_groups_and_angles_;
}

void
CBCD_FLUDS::CopyIncomingBoundaryPsiToDevice(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set)
{
  const auto& angle_indices = angle_set->GetAngleIndices();
  const auto& num_angles = angle_indices.size();
  const auto& incoming_boundary_node_map = common_data_.GetIncomingBoundaryNodeMap();
  const size_t groups_bytes = num_groups_ * sizeof(double);

  for (const auto& node : incoming_boundary_node_map)
  {
    for (size_t as_ss_idx = 0; as_ss_idx < num_angles; ++as_ss_idx)
    {
      auto direction_num = angle_indices[as_ss_idx];
      double* dst_psi = incoming_boundary_psi_.data() +
                        node.storage_index * num_groups_and_angles_ + as_ss_idx * num_groups_;
      const double* src_psi = angle_set->PsiBoundary(node.boundary_id,
                                                     direction_num,
                                                     node.cell_local_id,
                                                     node.face_id,
                                                     node.face_node,
                                                     sweep_chunk.GetGroupsetGroupIndex(),
                                                     sweep_chunk.IsSurfaceSourceActive());
      std::memcpy(dst_psi, src_psi, groups_bytes);
    }
  }
}

uint64_t
CBCD_FLUDS::ScatterReceivedFaceData(uint64_t cell_global_id,
                                    unsigned int face_id,
                                    const double* psi_data)
{
  const auto [cell_local_id, face_info] = common_data_.FindIncomingNonlocalFace(cell_global_id, face_id);
  OpenSnLogicalErrorIf(face_info == nullptr,
                       "CBCD_FLUDS::ScatterReceivedFaceData: incoming face metadata not found.");

  for (const auto& node : common_data_.GetIncomingFaceNodes(*face_info))
  {
    double* dst = incoming_nonlocal_psi_.data() + node.storage_index * num_groups_and_angles_;
    const double* src = psi_data + node.face_node_mapped * num_groups_and_angles_;
    std::memcpy(dst, src, num_groups_and_angles_ * sizeof(double));
  }
  return cell_local_id;
}

void
CBCD_FLUDS::CopyOutgoingPsiBackToHost(CBCD_AngleSet* angle_set,
                                      const std::vector<std::uint64_t>& cell_local_ids)
{
  if (common_data_.GetNumOutgoingBoundaryNodes() == 0 and outgoing_destinations_.empty())
    return;

  auto* agg_comm = angle_set->GetAggregatedCommunicator();
  const auto& angle_indices = angle_set->GetAngleIndices();
  const auto num_angles = angle_indices.size();
  const auto angle_set_id = angle_set->GetID();
  const size_t groups_bytes = num_groups_ * sizeof(double);
  const size_t stride_bytes = num_groups_and_angles_ * sizeof(double);

  constexpr size_t section_header_size = sizeof(size_t) + sizeof(size_t);
  constexpr size_t entry_header_size =
    sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(size_t);

  active_dest_indices_.clear();

  const auto initialize_dest_buffer =
    [this, angle_set_id](const size_t dest_index)
  {
    scratch_dest_touched_[dest_index] = 1;
    active_dest_indices_.push_back(static_cast<std::uint32_t>(dest_index));
    scratch_dest_face_counts_[dest_index] = 0;
    auto& data = dest_buffers_[dest_index].Data();
    data.clear();
    data.resize(section_header_size);
    std::memcpy(data.data(), &angle_set_id, sizeof(size_t));
  };

  for (const auto& cell_local_id : cell_local_ids)
  {
    const auto boundary_nodes = GetReflectingOutgoingBoundaryNodes(cell_local_id);
    if (not boundary_nodes.empty())
    {
      for (const auto& node : boundary_nodes)
      {
        for (size_t as_ss_idx = 0; as_ss_idx < num_angles; ++as_ss_idx)
        {
          auto direction_num = angle_indices[as_ss_idx];
          double* dst_psi = angle_set->PsiReflected(
            node.boundary_id, direction_num, node.cell_local_id, node.face_id, node.face_node);
          const double* src_psi = outgoing_boundary_psi_.data() +
                                  node.storage_index * num_groups_and_angles_ +
                                  as_ss_idx * num_groups_;
          std::memcpy(dst_psi, src_psi, groups_bytes);
        }
      }
    }

    const auto grouped_faces = common_data_.GetOutgoingNonlocalFaces(cell_local_id);
    for (const auto& face_info : grouped_faces)
    {
      const size_t dest_index = face_info.dest_slot;
      const auto& pack_plan = outgoing_face_pack_plans_[face_info.pack_plan_index];
      const size_t face_data_size = pack_plan.payload_doubles;
      if (not scratch_dest_touched_[dest_index])
        initialize_dest_buffer(dest_index);
      scratch_dest_face_counts_[dest_index]++;
      auto& data = dest_buffers_[dest_index].Data();
      const size_t offset = data.size();
      data.resize(offset + entry_header_size + face_data_size * sizeof(double));
      auto* base = data.data();

      std::memcpy(base + offset,
                  face_info.entry_header_prefix.data(),
                  face_info.entry_header_prefix.size());
      std::memcpy(base + offset + face_info.entry_header_prefix.size(),
                  &face_data_size,
                  sizeof(size_t));

      auto* psi_dst = reinterpret_cast<double*>(base + offset + entry_header_size);
      const auto* node_plan = outgoing_node_memcpy_plan_.data() + face_info.node_copy_offset;
      const auto* node_plan_end = node_plan + face_info.num_node_copies;
      for (; node_plan != node_plan_end; ++node_plan)
      {
        double* dst = psi_dst + node_plan->dst_offset;
        const double* src = outgoing_nonlocal_psi_.data() + node_plan->src_offset;
        std::memcpy(dst, src, stride_bytes);
      }
    }
  }

  for (const auto dest_index_u32 : active_dest_indices_)
  {
    const size_t dest_index = dest_index_u32;
    std::memcpy(dest_buffers_[dest_index].Data().data() + sizeof(size_t),
                &scratch_dest_face_counts_[dest_index],
                sizeof(size_t));
    agg_comm->EnqueuePrepackedByIndex(outgoing_destinations_[dest_index].queue_index,
                                      std::move(dest_buffers_[dest_index]));
    scratch_dest_touched_[dest_index] = 0;
  }
}

void
CBCD_FLUDS::CopySavedPsiToDestinationPsi(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set)
{
  if (not save_angular_flux_)
    return;
  crb::copy(host_saved_psi_, device_saved_psi_, host_saved_psi_.size(), 0, 0, stream_);
  stream_.synchronize();
  DiscreteOrdinatesProblem& problem = sweep_chunk.GetProblem();
  auto* mesh = problem.GetMeshCarrier();
  auto& groupset = sweep_chunk.GetGroupset();
  auto& destination_psi = problem.GetPsiNewLocal()[groupset.id];
  const auto& discretization = problem.GetSpatialDiscretization();
  const std::size_t groupset_angle_group_stride =
    groupset.psi_uk_man_.GetNumberOfUnknowns() * groupset.GetNumGroups();
  const auto& angle_indices = angle_set->GetAngleIndices();
  const auto& num_angles = angle_set->GetNumAngles();
  const size_t groups_bytes = num_groups_ * sizeof(double);
  for (const auto& cell : grid_ptr_->local_cells)
  {
    double* dst_psi = &destination_psi[discretization.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0)];
    double* src_psi =
      host_saved_psi_.data() + mesh->saved_psi_offset[cell.local_id] * GetStrideSize();
    std::uint32_t cell_num_nodes = discretization.GetCellMapping(cell).GetNumNodes();
    for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
    {
      for (std::uint32_t as_ss_idx = 0; as_ss_idx < num_angles; ++as_ss_idx)
      {
        auto direction_num = angle_indices[as_ss_idx];
        double* dst = dst_psi + direction_num * num_groups_;
        double* src = src_psi + as_ss_idx * num_groups_;
        std::memcpy(dst, src, groups_bytes);
      }
      dst_psi += groupset_angle_group_stride;
      src_psi += num_groups_and_angles_;
    }
  }
}

} // namespace opensn

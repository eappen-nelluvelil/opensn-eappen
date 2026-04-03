// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/math/unknown_manager/unknown_manager.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <unordered_map>
#include <utility>

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
    sdm_(sdm),
    num_angles_in_gs_quadrature_(psi_uk_man_.GetNumberOfUnknowns()),
    num_quadrature_local_dofs_(sdm_.GetNumLocalDOFs(psi_uk_man_)),
    num_local_spatial_dofs_(num_quadrature_local_dofs_ / num_angles_in_gs_quadrature_ /
                            num_groups_),
    num_local_psi_slots_(
      static_cast<const CBC_SPDS&>(common_data.GetSPDS()).GetMaxNumLocalPsiSlots()),
    local_psi_slot_stride_(
      [&]()
      {
        std::size_t max_num_nodes = 0;
        for (const auto& cell : common_data.GetSPDS().GetGrid()->local_cells)
          max_num_nodes = std::max(max_num_nodes, sdm_.GetCellMapping(cell).GetNumNodes());
        return max_num_nodes;
      }()),
    local_psi_data_size_(num_local_psi_slots_ * local_psi_slot_stride_ * num_groups_and_angles_),
    saved_psi_data_size_(num_local_spatial_dofs_ * num_groups_and_angles_),
    incoming_boundary_node_map_(common_data_.GetIncomingBoundaryNodeMap()),
    incoming_boundary_psi_(common_data_.GetNumIncomingBoundaryNodes() * num_groups_and_angles_),
    outgoing_boundary_psi_(common_data_.GetNumOutgoingBoundaryNodes() * num_groups_and_angles_),
    incoming_nonlocal_psi_(common_data_.GetNumIncomingNonlocalNodes() * num_groups_and_angles_),
    outgoing_nonlocal_psi_(common_data_.GetNumOutgoingNonlocalNodes() * num_groups_and_angles_),
    local_cell_ids_(num_local_cells),
    local_slot_offsets_(num_local_cells, INVALID_SLOT_OFFSET),
    save_angular_flux_(save_angular_flux)
{
  grid_ptr_ = GetSPDS().GetGrid().get();
  deplocs_outgoing_messages_.reserve(common_data.GetNumIncomingNonlocalFaces());
  free_slot_stack_.resize(num_local_psi_slots_);
  for (std::uint32_t slot = 0; slot < num_local_psi_slots_; ++slot)
    free_slot_stack_[slot] = slot;

  const auto& outgoing_localities = common_data_.GetOutgoingLocalities();
  outgoing_destinations_.reserve(outgoing_localities.size());
  for (const int locality : outgoing_localities)
    outgoing_destinations_.push_back({locality, -1});

  outgoing_node_memcpy_plan_.reserve(common_data_.GetNumOutgoingNonlocalNodes());
  outgoing_face_payload_sizes_.resize(common_data_.GetNumOutgoingNonlocalFaces());
  for (size_t cell_local_id = 0; cell_local_id < common_data_.GetNumLocalCells(); ++cell_local_id)
  {
    for (const auto& face_info : common_data_.GetOutgoingNonlocalFaces(cell_local_id))
    {
      outgoing_face_payload_sizes_[face_info.pack_plan_index] =
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
  local_slot_offsets_.clear();
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
    host_saved_psi_ = crb::HostVector<double>(saved_psi_data_size_);
    device_saved_psi_ = crb::DeviceMemory<double>(saved_psi_data_size_, stream_);
  }
  CreatePointerSet();
}

void
CBCD_FLUDS::InitializeQueueIndices(const CBCD_AsynchronousCommunicator& async_comm)
{
  for (auto& dest : outgoing_destinations_)
    dest.queue_index = async_comm.GetQueueIndex(dest.locality);
}

void
CBCD_FLUDS::InitializeReflectingBoundaryNodes(
  const std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries)
{
  const auto num_local_cells = common_data_.GetNumLocalCells();
  reflecting_outgoing_boundary_face_offsets_.assign(num_local_cells + 1, 0);
  reflecting_boundary_face_plans_.clear();
  reflecting_boundary_face_plans_.reserve(common_data_.GetNumOutgoingBoundaryNodes());

  for (size_t cell_local_id = 0; cell_local_id < num_local_cells; ++cell_local_id)
  {
    reflecting_outgoing_boundary_face_offsets_[cell_local_id] =
      static_cast<std::uint32_t>(reflecting_boundary_face_plans_.size());

    const auto boundary_nodes = common_data_.GetOutgoingBoundaryNodes(cell_local_id);
    for (size_t i = 0; i < boundary_nodes.size();)
    {
      const auto& first_node = boundary_nodes[i];
      const auto boundary_it = boundaries.find(first_node.boundary_id);
      if (boundary_it == boundaries.end() or not boundary_it->second->IsReflecting())
      {
        ++i;
        continue;
      }

      size_t num_nodes = 1;
      while (i + num_nodes < boundary_nodes.size())
      {
        const auto& node = boundary_nodes[i + num_nodes];
        if (node.boundary_id != first_node.boundary_id or node.cell_local_id != first_node.cell_local_id or
            node.face_id != first_node.face_id or
            node.storage_index != first_node.storage_index + num_nodes or
            node.face_node != first_node.face_node + num_nodes)
          break;
        ++num_nodes;
      }

      reflecting_boundary_face_plans_.push_back(
        {first_node.boundary_id,
         static_cast<std::uint32_t>(first_node.cell_local_id),
         first_node.face_id,
         static_cast<std::uint16_t>(first_node.face_node),
         static_cast<size_t>(first_node.storage_index) * num_groups_and_angles_,
         static_cast<std::uint16_t>(num_nodes)});
      i += num_nodes;
    }

    reflecting_outgoing_boundary_face_offsets_[cell_local_id + 1] =
      static_cast<std::uint32_t>(reflecting_boundary_face_plans_.size());
  }
}

void
CBCD_FLUDS::CreatePointerSet()
{
  pointer_set_.local_psi = local_psi_.get();
  if (local_psi_data_size_ > 0)
    assert(pointer_set_.local_psi != nullptr);
  pointer_set_.local_slot_offsets = local_slot_offsets_.data();
  if (not local_slot_offsets_.empty())
    assert(pointer_set_.local_slot_offsets != nullptr);

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
CBCD_FLUDS::AllocateSlots(const std::vector<std::uint32_t>& cell_local_ids)
{
  for (const auto cell_local_id : cell_local_ids)
  {
    assert(local_slot_offsets_[cell_local_id] == INVALID_SLOT_OFFSET);
    assert(not free_slot_stack_.empty());

    const auto slot = free_slot_stack_.back();
    free_slot_stack_.pop_back();
    local_slot_offsets_[cell_local_id] = slot * static_cast<std::uint32_t>(local_psi_slot_stride_);
  }
}

void
CBCD_FLUDS::DeallocateSlots(const std::vector<std::uint32_t>& cell_local_ids)
{
  for (const auto cell_local_id : cell_local_ids)
  {
    const auto slot_offset = local_slot_offsets_[cell_local_id];
    assert(slot_offset != INVALID_SLOT_OFFSET);

    free_slot_stack_.push_back(slot_offset / static_cast<std::uint32_t>(local_psi_slot_stride_));
    local_slot_offsets_[cell_local_id] = INVALID_SLOT_OFFSET;
  }
}

void
CBCD_FLUDS::CopyIncomingBoundaryPsiToDevice(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set)
{
  const auto& angle_indices = angle_set->GetAngleIndices();
  const auto& num_angles = angle_indices.size();
  const size_t groups_bytes = num_groups_ * sizeof(double);

  for (const auto& node : incoming_boundary_node_map_)
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

void
CBCD_FLUDS::CopyIncomingNonlocalPsiToDevice(CBCD_AngleSet* angle_set,
                                            const std::vector<std::uint32_t>& cell_local_ids)
{
  (void)angle_set;
  (void)cell_local_ids;
}

void
CBCD_FLUDS::CopyOutgoingPsiBackToHost(CBCDSweepChunk& sweep_chunk,
                                      CBCD_AngleSet* angle_set,
                                      const std::vector<std::uint32_t>& cell_local_ids)
{
  (void)sweep_chunk;

  if (common_data_.GetNumOutgoingBoundaryNodes() == 0 and outgoing_destinations_.empty())
    return;

  const auto& angle_indices = angle_set->GetAngleIndices();
  const auto num_angles = angle_indices.size();
  const auto& grid = *(GetSPDS().GetGrid());
  const auto angle_set_id = angle_set->GetID();
  const size_t groups_bytes = num_groups_ * sizeof(double);
  const size_t stride_bytes = num_groups_and_angles_ * sizeof(double);
  constexpr size_t section_header_size = 2 * sizeof(size_t);
  constexpr size_t entry_header_size =
    sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(size_t);

  active_dest_indices_.clear();
  const auto initialize_dest_buffer = [this, angle_set_id](const size_t dest_index)
  {
    scratch_dest_touched_[dest_index] = 1;
    active_dest_indices_.push_back(static_cast<std::uint32_t>(dest_index));
    scratch_dest_face_counts_[dest_index] = 0;
    auto& data = dest_buffers_[dest_index].Data();
    data.clear();
    data.resize(section_header_size);
    std::memcpy(data.data(), &angle_set_id, sizeof(size_t));
    size_t num_entries = 0;
    std::memcpy(data.data() + sizeof(size_t), &num_entries, sizeof(size_t));
  };

  for (const auto& cell_local_id : cell_local_ids)
  {
    const auto reflecting_faces = GetReflectingOutgoingBoundaryFaces(cell_local_id);
    for (const auto& face_plan : reflecting_faces)
    {
      for (size_t as_ss_idx = 0; as_ss_idx < num_angles; ++as_ss_idx)
      {
        const auto direction_num = static_cast<unsigned int>(angle_indices[as_ss_idx]);
        const double* src_face =
          outgoing_boundary_psi_.data() + face_plan.src_base_offset + as_ss_idx * num_groups_;
        for (size_t n = 0; n < face_plan.num_nodes; ++n)
        {
          double* dst = angle_set->PsiReflected(face_plan.boundary_id,
                                                direction_num,
                                                face_plan.cell_local_id,
                                                face_plan.face_id,
                                                static_cast<unsigned int>(face_plan.first_face_node + n));
          std::memcpy(dst,
                      src_face + n * num_groups_and_angles_,
                      groups_bytes);
        }
      }
    }

    for (const auto& face_info : common_data_.GetOutgoingNonlocalFaces(cell_local_id))
    {
      const size_t dest_index = face_info.dest_slot;
      const size_t face_data_size = outgoing_face_payload_sizes_[face_info.pack_plan_index];
      if (not scratch_dest_touched_[dest_index])
        initialize_dest_buffer(dest_index);
      ++scratch_dest_face_counts_[dest_index];

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
    auto& data = dest_buffers_[dest_index].Data();
    std::memcpy(data.data() + sizeof(size_t),
                &scratch_dest_face_counts_[dest_index],
                sizeof(size_t));
    angle_set->GetAsyncCommunicator().EnqueuePrepackedByIndex(
      outgoing_destinations_[dest_index].queue_index, angle_set_id, std::move(dest_buffers_[dest_index]));
    scratch_dest_touched_[dest_index] = 0;
    scratch_dest_face_counts_[dest_index] = 0;
  }
}

void
CBCD_FLUDS::CopySavedPsiFromDevice()
{
  if (not save_angular_flux_)
    return;
  crb::copy(host_saved_psi_, device_saved_psi_, host_saved_psi_.size(), 0, 0, stream_);
}

void
CBCD_FLUDS::CopySavedPsiToDestinationPsi(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set)
{
  if (not save_angular_flux_)
    return;

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

std::uint64_t
CBCD_FLUDS::ScatterReceivedFaceData(std::uint64_t cell_global_id,
                                    unsigned int face_id,
                                    const double* psi_data)
{
  const auto& face_info = common_data_.FindIncomingNonlocalFace(cell_global_id, face_id);
  double* dst = incoming_nonlocal_psi_.data() +
                static_cast<size_t>(face_info.base_storage_index) * num_groups_and_angles_;
  const size_t face_bytes =
    static_cast<size_t>(face_info.num_nodes) * num_groups_and_angles_ * sizeof(double);
  std::memcpy(dst, psi_data, face_bytes);
  return face_info.cell_local_id;
}

void
CBCD_FLUDS::ClearLocalAndReceivePsi()
{
  deplocs_outgoing_messages_.clear();
  std::fill(local_slot_offsets_.begin(), local_slot_offsets_.end(), INVALID_SLOT_OFFSET);
  free_slot_stack_.resize(num_local_psi_slots_);
  for (std::uint32_t slot = 0; slot < num_local_psi_slots_; ++slot)
    free_slot_stack_[slot] = slot;
}

} // namespace opensn

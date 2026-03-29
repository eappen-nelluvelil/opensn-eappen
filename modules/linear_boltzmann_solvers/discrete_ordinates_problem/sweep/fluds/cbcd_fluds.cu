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
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include <utility>
#include <map>
#include <algorithm>

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
  const auto& grid = *grid_ptr_;

  cell_to_face_grouped_outgoing_.resize(num_local_cells);
  cell_to_face_grouped_incoming_.resize(num_local_cells);

  // Group outgoing nodes by face, pre-resolve dest_index per face, and build
  // the outgoing_destinations_ table.  After this loop, the locality→index
  // mapping is discarded — dest_index is embedded in each FaceOutgoingInfo,
  // eliminating two unordered_map lookups per face in CopyOutgoingPsiBackToHost.
  std::unordered_map<int, size_t> locality_to_dest_index;
  const auto& outgoing_nonlocal_map = common_data_.GetOutgoingNonlocalNodeMap();
  for (size_t cell_id = 0; cell_id < num_local_cells; ++cell_id)
  {
    const auto& nodes = outgoing_nonlocal_map[cell_id];
    if (nodes.empty())
      continue;

    const auto& cell = grid.local_cells[cell_id];
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    std::map<unsigned int, std::vector<const NonlocalNodeInfo*>> by_face;
    for (const auto& node : nodes)
      by_face[node.face_id].push_back(&node);

    auto& grouped = cell_to_face_grouped_outgoing_[cell_id];
    for (auto& [fid, fnodes] : by_face)
    {
      const auto& face = cell.faces[fid];
      const auto& face_nodal_mapping = common_data_.GetFaceNodalMapping(cell_id, fid);
      size_t face_data_size = cell_mapping.GetNumFaceNodes(fid) * num_groups_and_angles_;
      num_outgoing_faces_++;

      int locality = grid.cells[face.neighbor_id].partition_id;
      auto [it, inserted] =
        locality_to_dest_index.try_emplace(locality, outgoing_destinations_.size());
      if (inserted)
        outgoing_destinations_.push_back({locality, -1});

      grouped.push_back({fid,
                         std::move(fnodes),
                         face_data_size,
                         it->second,
                         face.neighbor_id,
                         static_cast<unsigned int>(face_nodal_mapping.associated_face_)});
    }
  }

  // Group incoming nodes by face for fast linear scattering
  const auto& incoming_nonlocal_map = common_data_.GetIncomingNonlocalNodeMap();
  for (size_t cell_id = 0; cell_id < num_local_cells; ++cell_id)
  {
    const auto& nodes = incoming_nonlocal_map[cell_id];
    if (nodes.empty())
      continue;

    // Build fast global→local lookup for cells that receive nonlocal data
    uint64_t global_id = grid.local_cells[cell_id].global_id;
    incoming_global_to_local_[global_id] = cell_id;

    std::map<unsigned int, std::vector<const NonlocalNodeInfo*>> by_face;
    for (const auto& node : nodes)
      by_face[node.face_id].push_back(&node);

    auto& grouped = cell_to_face_grouped_incoming_[cell_id];
    for (auto& [fid, fnodes] : by_face)
    {
      num_incoming_faces_++;
      grouped.push_back({fid, std::move(fnodes)});
    }
  }

  // Pre-allocate scratch and destination buffers for CopyOutgoingPsiBackToHost.
  const size_t num_dests = outgoing_destinations_.size();
  scratch_dest_face_counts_.resize(num_dests, 0);
  scratch_dest_psi_bytes_.resize(num_dests, 0);
  scratch_dest_offsets_.resize(num_dests, 0);
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
      std::copy(src_psi, src_psi + num_groups_, dst_psi);
    }
  }
}

uint64_t
CBCD_FLUDS::ScatterReceivedFaceData(uint64_t cell_global_id,
                                    unsigned int face_id,
                                    const double* psi_data)
{
  uint64_t cell_local_id = incoming_global_to_local_.find(cell_global_id)->second;
  const auto& grouped = cell_to_face_grouped_incoming_[cell_local_id];

  for (const auto& face_info : grouped)
  {
    if (face_info.face_id == face_id)
    {
      for (const auto* node : face_info.nodes)
      {
        double* dst = incoming_nonlocal_psi_.data() + node->storage_index * num_groups_and_angles_;
        const double* src = psi_data + node->face_node_mapped * num_groups_and_angles_;
        std::memcpy(dst, src, num_groups_and_angles_ * sizeof(double));
      }
      break;
    }
  }
  return cell_local_id;
}

void
CBCD_FLUDS::CopyOutgoingPsiBackToHost(CBCDSweepChunk& sweep_chunk,
                                      CBCD_AngleSet* angle_set,
                                      const std::vector<std::uint64_t>& cell_local_ids)
{
  if (common_data_.GetNumOutgoingBoundaryNodes() == 0 and outgoing_destinations_.empty())
    return;

  auto* agg_comm = angle_set->GetAggregatedCommunicator();
  const auto& angle_indices = angle_set->GetAngleIndices();
  const auto num_angles = angle_indices.size();
  const auto angle_set_id = angle_set->GetID();

  // Lazily resolve queue indices on first call (agg_comm not available at construction time).
  if (not outgoing_destinations_.empty() and outgoing_destinations_[0].queue_index < 0)
  {
    for (auto& dest : outgoing_destinations_)
      dest.queue_index = agg_comm->GetQueueIndex(dest.locality);
  }

  const auto& outgoing_boundary_map = common_data_.GetOutgoingBoundaryNodeMap();

  // Per-destination: count faces and total psi bytes in a first pass (using scratch buffers).
  std::fill(scratch_dest_face_counts_.begin(), scratch_dest_face_counts_.end(), 0);
  std::fill(scratch_dest_psi_bytes_.begin(), scratch_dest_psi_bytes_.end(), 0);

  for (const auto& cell_local_id : cell_local_ids)
  {
    // Handle outgoing boundary (reflecting) faces — no wire format needed.
    const auto& boundary_nodes = outgoing_boundary_map[cell_local_id];
    if (not boundary_nodes.empty())
    {
      const auto& cell = grid_ptr_->local_cells[cell_local_id];
      for (const auto& node : boundary_nodes)
      {
        const auto& face = cell.faces[node.face_id];
        if (angle_set->GetBoundaries().at(face.neighbor_id)->IsReflecting())
        {
          for (size_t as_ss_idx = 0; as_ss_idx < num_angles; ++as_ss_idx)
          {
            auto direction_num = angle_indices[as_ss_idx];
            double* dst_psi = angle_set->PsiReflected(
              face.neighbor_id, direction_num, node.cell_local_id, node.face_id, node.face_node);
            const double* src_psi = outgoing_boundary_psi_.data() +
                                    node.storage_index * num_groups_and_angles_ +
                                    as_ss_idx * num_groups_;
            std::copy(src_psi, src_psi + num_groups_, dst_psi);
          }
        }
      }
    }

    // Count outgoing non-local faces per destination (pre-resolved dest_index).
    const auto& grouped_nodes = cell_to_face_grouped_outgoing_[cell_local_id];
    for (const auto& face_info : grouped_nodes)
    {
      scratch_dest_face_counts_[face_info.dest_index]++;
      scratch_dest_psi_bytes_[face_info.dest_index] += face_info.face_data_size * sizeof(double);
    }
  }

  // Pack wire-format sections into reusable ByteArrays (one per destination).
  // Section: [angle_set_id : size_t][num_entries : size_t][entries...]
  // Entry:   [cell_global_id : uint64_t][face_id : uint][data_size : size_t][psi doubles]
  constexpr size_t entry_header_size =
    sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(size_t);
  constexpr size_t section_header_size = sizeof(size_t) + sizeof(size_t);

  std::fill(scratch_dest_offsets_.begin(), scratch_dest_offsets_.end(), 0);

  for (size_t d = 0; d < outgoing_destinations_.size(); ++d)
  {
    if (scratch_dest_face_counts_[d] == 0)
    {
      dest_buffers_[d].Data().clear();
      continue;
    }
    size_t buf_size = section_header_size + scratch_dest_face_counts_[d] * entry_header_size +
                      scratch_dest_psi_bytes_[d];
    dest_buffers_[d].Data().resize(buf_size);

    auto* base = dest_buffers_[d].Data().data();
    std::memcpy(base, &angle_set_id, sizeof(size_t));
    std::memcpy(base + sizeof(size_t), &scratch_dest_face_counts_[d], sizeof(size_t));
    scratch_dest_offsets_[d] = section_header_size;
  }

  // Second pass: pack entry data from outgoing_nonlocal_psi_ into the ByteArrays.
  // The zero-fill is omitted: every face-node position [0, num_face_nodes) is
  // written by the inner loop (CBCD_FLUDSCommonData adds ALL face nodes of each
  // outgoing non-local face), so no position is left uninitialized.
  for (const auto& cell_local_id : cell_local_ids)
  {
    const auto& grouped_nodes = cell_to_face_grouped_outgoing_[cell_local_id];
    for (const auto& face_info : grouped_nodes)
    {
      auto* base = dest_buffers_[face_info.dest_index].Data().data();
      size_t& offset = scratch_dest_offsets_[face_info.dest_index];

      std::memcpy(base + offset, &face_info.neighbor_global_id, sizeof(std::uint64_t));
      offset += sizeof(std::uint64_t);
      std::memcpy(base + offset, &face_info.associated_face, sizeof(unsigned int));
      offset += sizeof(unsigned int);
      std::memcpy(base + offset, &face_info.face_data_size, sizeof(size_t));
      offset += sizeof(size_t);

      auto* psi_dst = reinterpret_cast<double*>(base + offset);
      for (const auto* node : face_info.nodes)
      {
        double* dst = psi_dst + node->face_node * num_groups_and_angles_;
        const double* src =
          outgoing_nonlocal_psi_.data() + node->storage_index * num_groups_and_angles_;
        std::memcpy(dst, src, num_groups_and_angles_ * sizeof(double));
      }
      offset += face_info.face_data_size * sizeof(double);
    }
  }

  // Enqueue one pre-packed section per destination. The ByteArray content is
  // moved to the Treiber stack; the dest_buffers_ slot retains its capacity
  // for reuse on the next call.
  for (size_t d = 0; d < outgoing_destinations_.size(); ++d)
  {
    if (scratch_dest_face_counts_[d] == 0)
      continue;
    agg_comm->EnqueuePrepackedByIndex(outgoing_destinations_[d].queue_index,
                                      std::move(dest_buffers_[d]));
  }
}

// void
// CBCD_FLUDS::CopySavedPsiFromDevice()
// {
//   if (not save_angular_flux_)
//     return;
//   crb::copy(host_saved_psi_, device_saved_psi_, host_saved_psi_.size(), 0, 0, stream_);
//   stream_.synchronize();
// }

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
        std::copy(src, src + num_groups_, dst);
      }
      dst_psi += groupset_angle_group_stride;
      src_psi += num_groups_and_angles_;
    }
  }
}

} // namespace opensn

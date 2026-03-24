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
    local_psi_data_size_(num_local_spatial_dofs_ * num_groups_and_angles_),
    incoming_boundary_node_map_(common_data_.GetIncomingBoundaryNodeMap()),
    cell_to_outgoing_boundary_nodes_(common_data_.GetOutgoingBoundaryNodeMap()),
    cell_to_incoming_nonlocal_nodes_(common_data_.GetIncomingNonlocalNodeMap()),
    cell_to_outgoing_nonlocal_nodes_(common_data_.GetOutgoingNonlocalNodeMap()),
    incoming_boundary_psi_(common_data_.GetNumIncomingBoundaryNodes() * num_groups_and_angles_),
    outgoing_boundary_psi_(common_data_.GetNumOutgoingBoundaryNodes() * num_groups_and_angles_),
    incoming_nonlocal_psi_(common_data_.GetNumIncomingNonlocalNodes() * num_groups_and_angles_),
    outgoing_nonlocal_psi_(common_data_.GetNumOutgoingNonlocalNodes() * num_groups_and_angles_),
    local_cell_ids_(num_local_cells),
    save_angular_flux_(save_angular_flux)
{
  // Cache grid pointer to avoid shared_ptr copy on the hot path.
  grid_ptr_ = GetSPDS().GetGrid().get();
  const auto& grid = *grid_ptr_;

  // Pre-compute face-grouped outgoing nonlocal nodes once at construction.
  // Caches face_data_size, locality, neighbor_global_id, and associated_face so that
  // CopyOutgoingPsiBackToHost needs no per-call mesh queries, virtual dispatches, or map lookups.
  for (const auto& [cell_id, nodes] : cell_to_outgoing_nonlocal_nodes_)
  {
    const auto& cell = grid.local_cells[cell_id];
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    std::map<unsigned int, std::vector<const NonlocalNodeInfo*>> by_face;
    for (const auto& node : nodes)
      by_face[node.face_id].push_back(&node);
    auto& grouped = cell_to_face_grouped_outgoing_[cell_id];
    for (auto& [fid, fnodes] : by_face)
    {
      const auto& face = cell.faces[fid];
      const auto& face_nodal_mapping = common_data_.GetFaceNodalMapping(cell_id, fid);
      const auto num_face_nodes = cell_mapping.GetNumFaceNodes(fid);
      const int locality = grid.cells[face.neighbor_id].partition_id;
      grouped.push_back({fid,
                         std::move(fnodes),
                         num_face_nodes * num_groups_and_angles_,
                         locality,
                         face.neighbor_id,
                         static_cast<unsigned int>(face_nodal_mapping.associated_face_)});

      // Track unique destinations
      if (locality_to_dest_index_.find(locality) == locality_to_dest_index_.end())
      {
        locality_to_dest_index_[locality] = outgoing_destinations_.size();
        outgoing_destinations_.push_back({locality, -1}); // queue_index resolved later
      }
    }
  }

  // Pre-compute incoming nonlocal scatter lookup: (cell_global_id, face_id) -> nodes.
  // Enables O(1) direct scatter into incoming_nonlocal_psi_ at MPI receive time.
  for (const auto& [cell_id, nodes] : cell_to_incoming_nonlocal_nodes_)
    for (const auto& node : nodes)
      incoming_nonlocal_face_lookup_[{node.cell_global_id, node.face_id}].push_back(&node);
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
      std::copy(src_psi, src_psi + num_groups_, dst_psi);
    }
  }
}

void
CBCD_FLUDS::ScatterReceivedFaceData(uint64_t cell_global_id,
                                    unsigned int face_id,
                                    const std::vector<double>& psi_data)
{
  auto it = incoming_nonlocal_face_lookup_.find({cell_global_id, face_id});
  if (it == incoming_nonlocal_face_lookup_.end())
    return;
  for (const auto* node : it->second)
  {
    double* dst = incoming_nonlocal_psi_.data() +
                  node->storage_index * num_groups_and_angles_;
    const double* src = psi_data.data() +
                        node->face_node_mapped * num_groups_and_angles_;
    std::copy(src, src + num_groups_and_angles_, dst);
  }
}

void
CBCD_FLUDS::CopyOutgoingPsiBackToHost(CBCDSweepChunk& sweep_chunk,
                                      CBCD_AngleSet* angle_set,
                                      const std::vector<std::uint32_t>& cell_local_ids)
{
  if (cell_to_outgoing_boundary_nodes_.empty() and cell_to_face_grouped_outgoing_.empty())
    return;
  auto* agg_comm = angle_set->GetAggregatedCommunicator();
  const auto& angle_indices = angle_set->GetAngleIndices();
  const auto num_angles = angle_indices.size();
  const auto angle_set_id = angle_set->GetID();

  // Resolve queue indices on first call (agg_comm is not available at FLUDS construction time).
  if (not outgoing_destinations_.empty() and outgoing_destinations_[0].queue_index < 0)
  {
    for (auto& dest : outgoing_destinations_)
      dest.queue_index = agg_comm->GetQueueIndex(dest.locality);
  }

  // Per-destination batch accumulators, index by locality_to_dest_index_.
  // Allocated once per call, reused across all cells in this kernel completion.
  std::vector<std::vector<CBCD_AggregatedCommunicator::OutgoingEntry>> dest_batches(
    outgoing_destinations_.size()
  );

  for (const auto& cell_local_id : cell_local_ids)
  {
    auto boundary_it = cell_to_outgoing_boundary_nodes_.find(cell_local_id);
    if (boundary_it != cell_to_outgoing_boundary_nodes_.end())
    {
      const auto& cell = grid_ptr_->local_cells[cell_local_id];
      for (const auto& node : boundary_it->second)
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
    // Use pre-computed face grouping with cached metadata — no per-call mesh queries needed.
    // Handle outgoing non-local faces (accumulate into per-destination batches)
    auto grouped_it = cell_to_face_grouped_outgoing_.find(cell_local_id);
    if (grouped_it != cell_to_face_grouped_outgoing_.end())
    {
      for (const auto& face_info : grouped_it->second)
      {
        std::vector<double> psi_data(face_info.face_data_size, 0.0);
        for (const auto* node : face_info.nodes)
        {
          double* dst = psi_data.data() + node->face_node * num_groups_and_angles_;
          const double* src = outgoing_nonlocal_psi_.data() +
                              node->storage_index * num_groups_and_angles_;
          std::copy(src, src + num_groups_and_angles_, dst);
        }

        // Accumulate into destination batch
        auto dest_index = locality_to_dest_index_.at(face_info.locality);
        dest_batches[dest_index].push_back({angle_set_id,
                                            face_info.neighbor_global_id,
                                            face_info.associated_face,
                                            std::move(psi_data)});
      }
    }
  }

  // Push one batch per destination
  // (1 Treiber node per destination per kernel completion).
  for (size_t d = 0; d < outgoing_destinations_.size(); ++d)
  {
    if (dest_batches[d].empty())
      continue;
    agg_comm->EnqueueOutgoingBatchByIndex(outgoing_destinations_[d].queue_index,
                                          std::move(dest_batches[d]));
  }
}

void
CBCD_FLUDS::CopySavedPsiFromDevice()
{
  if (not save_angular_flux_)
    return;
  crb::copy(host_saved_psi_, device_saved_psi_, host_saved_psi_.size(), 0, 0, stream_);
  stream_.synchronize();
}

void
CBCD_FLUDS::CopySavedPsiToDestinationPsi(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet* angle_set)
{
  if (not save_angular_flux_)
    return;
  DiscreteOrdinatesProblem& problem = sweep_chunk.GetProblem();
  auto* mesh = problem.GetMeshCarrier();
  auto grid = problem.GetGrid();
  auto& groupset = sweep_chunk.GetGroupset();
  auto& destination_psi = problem.GetPsiNewLocal()[groupset.id];
  const auto& discretization = problem.GetSpatialDiscretization();
  const std::size_t groupset_angle_group_stride =
    groupset.psi_uk_man_.GetNumberOfUnknowns() * groupset.GetNumGroups();
  const auto& angle_indices = angle_set->GetAngleIndices();
  const auto& num_angles = angle_set->GetNumAngles();
  for (const auto& cell : grid->local_cells)
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

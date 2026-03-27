// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"

namespace opensn
{

CBC_FLUDS::CBC_FLUDS(unsigned int num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     const UnknownManager& psi_uk_man,
                     const SpatialDiscretization& sdm)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(psi_uk_man),
    sdm_(sdm),
    num_angles_in_gs_quadrature_(psi_uk_man_.GetNumberOfUnknowns()),
    num_quadrature_local_dofs_(sdm_.GetNumLocalDOFs(psi_uk_man_)),
    num_local_spatial_dofs_(num_quadrature_local_dofs_ / num_angles_in_gs_quadrature_ /
                            num_groups_),
    local_psi_data_size_(num_local_spatial_dofs_ * num_groups_and_angles_),
    local_psi_data_(local_psi_data_size_)
{
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

double*
CBC_FLUDS::UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx)
{
  // Map to face neighbor cell's first spatial DOF index
  // (0 to (num_local_spatial_dofs_ - 1))
  const size_t face_nbr_spatial_dof_0_index =
    (sdm_.MapDOFLocal(face_neighbor, 0, psi_uk_man_, 0, 0) / num_angles_in_gs_quadrature_ /
     num_groups_);

  // Index to start of neighbor cell's data block in local_psi_data_
  const size_t face_nbr_data_start_index = face_nbr_spatial_dof_0_index * num_groups_and_angles_;
  const size_t addr_offset = adj_cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  const size_t face_nbr_data_index = face_nbr_data_start_index + addr_offset;

  assert((face_nbr_data_index >= 0) and (face_nbr_data_index < local_psi_data_.size()));

  return &local_psi_data_[face_nbr_data_index];
}

double*
CBC_FLUDS::OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx)
{
  // Map to current cell's first spatial DOF index
  // (0 to (num_local_spatial_dofs_ - 1))
  const size_t cur_cell_spatial_dof_0_index =
    (sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0) / num_angles_in_gs_quadrature_ / num_groups_);

  // Index to start of current cell's data block in local_psi_data_
  const size_t cur_cell_data_start_index = cur_cell_spatial_dof_0_index * num_groups_and_angles_;
  const size_t addr_offset = cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  const size_t cur_cell_data_index = cur_cell_data_start_index + addr_offset;

  assert((cur_cell_data_index >= 0) and (cur_cell_data_index < local_psi_data_.size()));

  return &local_psi_data_[cur_cell_data_index];
}

double*
CBC_FLUDS::NLUpwindPsi(uint64_t cell_global_id,
                       unsigned int face_id,
                       unsigned int face_node_mapped,
                       size_t as_ss_idx)
{
  std::vector<double>& psi = deplocs_outgoing_messages_.at({cell_global_id, face_id});
  const size_t dof_map =
    face_node_mapped * num_groups_and_angles_ + //  Offset to start of data for face_node_mapped
    as_ss_idx * num_groups_;                    // Offset to start of data for angle_set_index

  assert((dof_map >= 0) and (dof_map < psi.size()));
  return &psi[dof_map];
}

double*
CBC_FLUDS::NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing,
                         size_t face_node,
                         size_t as_ss_idx)
{
  assert(psi_nonlocal_outgoing != nullptr);
  const size_t addr_offset = face_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return &(*psi_nonlocal_outgoing)[addr_offset];
}

// ===== Delayed local psi (local FAS edges) =====

double*
CBC_FLUDS::DelayedLocalUpwindPsi(uint64_t cell_global_id,
                                 unsigned int face_id,
                                 unsigned int face_node_mapped,
                                 size_t as_ss_idx)
{
  CellFaceKey key{cell_global_id, face_id};
  auto it = delayed_local_face_lookup_.find(key);
  assert(it != delayed_local_face_lookup_.end());

  const size_t base_offset = it->second.offset;
  const size_t dof_map = face_node_mapped * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return &delayed_local_psi_old_data_[base_offset + dof_map];
}

double*
CBC_FLUDS::DelayedLocalOutgoingPsi(uint64_t cell_global_id,
                                   unsigned int face_id,
                                   unsigned int face_node_mapped,
                                   size_t as_ss_idx)
{
  CellFaceKey key{cell_global_id, face_id};
  auto it = delayed_local_face_lookup_.find(key);
  assert(it != delayed_local_face_lookup_.end());

  const size_t base_offset = it->second.offset;
  const size_t dof_map = face_node_mapped * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return &delayed_local_psi_data_[base_offset + dof_map];
}

// ===== Delayed nonlocal psi (delayed location dependencies) =====

double*
CBC_FLUDS::DelayedNLUpwindPsi(uint64_t cell_global_id,
                              unsigned int face_id,
                              unsigned int face_node_mapped,
                              size_t as_ss_idx)
{
  CellFaceKey key{cell_global_id, face_id};
  auto it = delayed_nonlocal_face_lookup_.find(key);
  assert(it != delayed_nonlocal_face_lookup_.end());

  const auto& info = it->second;
  const size_t dof_map = face_node_mapped * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return &delayed_prelocI_psi_old_data_[info.dep_idx][info.offset + dof_map];
}

void
CBC_FLUDS::StoreDelayedNonlocalData(uint64_t cell_global_id,
                                    unsigned int face_id,
                                    const double* data,
                                    size_t data_size)
{
  CellFaceKey key{cell_global_id, face_id};
  auto it = delayed_nonlocal_face_lookup_.find(key);
  if (it == delayed_nonlocal_face_lookup_.end())
    return;

  const auto& info = it->second;
  auto& buffer = delayed_prelocI_psi_data_[info.dep_idx];
  const size_t copy_size = std::min(data_size, buffer.size() - info.offset);
  std::copy(data, data + copy_size, buffer.begin() + info.offset);
}

// ===== Allocation =====

void
CBC_FLUDS::AllocateDelayedLocalPsi()
{
  const auto* cbc_spds = dynamic_cast<const CBC_SPDS*>(&spds_);
  if (not cbc_spds)
    return;

  const auto& delayed_incoming = cbc_spds->GetDelayedLocalIncomingFaces();
  if (delayed_incoming.empty())
    return;

  const auto& grid = *spds_.GetGrid();

  // Compute total size and build lookup table
  size_t total_size = 0;
  for (const auto& [cell_lid, face_idx] : delayed_incoming)
  {
    const auto& cell = grid.local_cells[cell_lid];
    const auto& mapping = sdm_.GetCellMapping(cell);
    const size_t num_face_nodes = mapping.GetNumFaceNodes(face_idx);

    CellFaceKey key{cell.global_id, face_idx};
    delayed_local_face_lookup_[key] = {total_size, num_face_nodes};
    total_size += num_face_nodes * num_groups_and_angles_;
  }

  delayed_local_psi_data_.resize(total_size, 0.0);
  delayed_local_psi_old_data_.resize(total_size, 0.0);
  delayed_local_psi_view_ = std::span<double>(delayed_local_psi_data_);
  delayed_local_psi_old_view_ = std::span<double>(delayed_local_psi_old_data_);

}

void
CBC_FLUDS::AllocateDelayedPrelocIOutgoingPsi()
{
  const auto* cbc_spds = dynamic_cast<const CBC_SPDS*>(&spds_);
  if (not cbc_spds)
    return;

  const auto& delayed_deps = spds_.GetDelayedLocationDependencies();
  const auto& delayed_nl_faces = cbc_spds->GetDelayedNonlocalIncomingFaces();
  if (delayed_deps.empty())
    return;
  if (delayed_nl_faces.empty())
    return;

  const auto& grid = *spds_.GetGrid();

  // Build map from partition_id → dep_idx
  std::map<int, size_t> partition_to_dep_idx;
  for (size_t i = 0; i < delayed_deps.size(); ++i)
    partition_to_dep_idx[delayed_deps[i]] = i;

  // First pass: compute size per dependency
  const size_t num_deps = delayed_deps.size();
  std::vector<size_t> dep_sizes(num_deps, 0);

  for (const auto& [cell_lid, face_idx] : delayed_nl_faces)
  {
    const auto& cell = grid.local_cells[cell_lid];
    const auto& face = cell.faces[face_idx];
    const auto& neighbor = grid.cells[face.neighbor_id];
    const int partition = neighbor.partition_id;

    auto dep_it = partition_to_dep_idx.find(partition);
    if (dep_it == partition_to_dep_idx.end())
      continue;

    const auto& mapping = sdm_.GetCellMapping(cell);
    const size_t num_face_nodes = mapping.GetNumFaceNodes(face_idx);

    CellFaceKey key{cell.global_id, face_idx};
    delayed_nonlocal_face_lookup_[key] = {
      dep_it->second, dep_sizes[dep_it->second], num_face_nodes};
    dep_sizes[dep_it->second] += num_face_nodes * num_groups_and_angles_;
  }

  // Allocate per-dependency buffers
  delayed_prelocI_psi_data_.resize(num_deps);
  delayed_prelocI_psi_old_data_.resize(num_deps);
  for (size_t i = 0; i < num_deps; ++i)
  {
    delayed_prelocI_psi_data_[i].resize(dep_sizes[i], 0.0);
    delayed_prelocI_psi_old_data_[i].resize(dep_sizes[i], 0.0);
  }

  UpdateRange(delayed_prelocI_psi_data_, delayed_prelocI_outgoing_psi_view_);
  UpdateRange(delayed_prelocI_psi_old_data_, delayed_prelocI_outgoing_psi_old_view_);

}

// ===== Copy operations =====

void
CBC_FLUDS::SetDelayedLocalPsiNewToOld()
{
  delayed_local_psi_old_data_ = delayed_local_psi_data_;
  delayed_local_psi_old_view_ = std::span<double>(delayed_local_psi_old_data_);
}

void
CBC_FLUDS::SetDelayedLocalPsiOldToNew()
{
  delayed_local_psi_data_ = delayed_local_psi_old_data_;
  delayed_local_psi_view_ = std::span<double>(delayed_local_psi_data_);
}

void
CBC_FLUDS::SetDelayedOutgoingPsiNewToOld()
{
  delayed_prelocI_psi_old_data_ = delayed_prelocI_psi_data_;
  UpdateRange(delayed_prelocI_psi_old_data_, delayed_prelocI_outgoing_psi_old_view_);
}

void
CBC_FLUDS::SetDelayedOutgoingPsiOldToNew()
{
  delayed_prelocI_psi_data_ = delayed_prelocI_psi_old_data_;
  UpdateRange(delayed_prelocI_psi_data_, delayed_prelocI_outgoing_psi_view_);
}

void
CBC_FLUDS::UpdateRange(std::vector<std::vector<double>>& data,
                       std::vector<std::span<double>>& spans)
{
  spans.resize(data.size());
  for (std::size_t i = 0; i < data.size(); ++i)
    spans[i] = std::span<double>(data[i]);
}

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"

namespace opensn
{

CBC_FLUDS::CBC_FLUDS(unsigned int num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     size_t max_cell_dof_count,
                     bool use_gpus)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data)
{
  if (not use_gpus)
  {
    num_local_cells_ = common_data.GetSPDS().GetGrid()->local_cells.size();
    slot_size_ = max_cell_dof_count * num_groups_and_angles_;
    num_slots_ = static_cast<const CBC_SPDS&>(common_data.GetSPDS()).GetMaxNumSlots();
    cell_to_slot_ptrs_.resize(num_local_cells_, nullptr);
    local_psi_backing_buffer_.resize(num_slots_ * slot_size_);
    local_psi_pool_.add_block(local_psi_backing_buffer_.data(),
                            (num_slots_ * slot_size_) * sizeof(double),
                            slot_size_ * sizeof(double));
  }
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

void
CBC_FLUDS::AllocateSlot(std::uint64_t cell_local_id)
{
  assert(cell_to_slot_ptrs_[cell_local_id] == nullptr);
  void* cell_slot = local_psi_pool_.malloc();
  assert(cell_slot != nullptr);
  cell_to_slot_ptrs_[cell_local_id] = static_cast<double*>(cell_slot);
}

void
CBC_FLUDS::DeallocateSlot(std::uint64_t cell_local_id)
{
  assert(cell_to_slot_ptrs_[cell_local_id] != nullptr);
  local_psi_pool_.free(cell_to_slot_ptrs_[cell_local_id]);
  cell_to_slot_ptrs_[cell_local_id] = nullptr;
}

double*
CBC_FLUDS::UpwindPsi(std::uint64_t neighbor_cell_local_id, unsigned int adj_cell_node, size_t as_ss_idx)
{
  assert(cell_to_slot_ptrs_[neighbor_cell_local_id] != nullptr);
  const size_t addr_offset = adj_cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return cell_to_slot_ptrs_[neighbor_cell_local_id] + addr_offset;
}

double*
CBC_FLUDS::OutgoingPsi(std::uint64_t cell_local_id, unsigned int cell_node, size_t as_ss_idx)
{
  assert(cell_to_slot_ptrs_[cell_local_id] != nullptr);
  const size_t addr_offset = cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return cell_to_slot_ptrs_[cell_local_id] + addr_offset;
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

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/cell/cell.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/utils/error.h"
#include <algorithm>
#include <cstring>

namespace opensn
{

CBC_FLUDS::CBC_FLUDS(unsigned int num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     size_t max_cell_dof_count)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    num_slots_(static_cast<const CBC_SPDS&>(common_data.GetSPDS()).GetMinNumLocalPsiSlots()),
    slot_size_(max_cell_dof_count * num_groups_and_angles_),
    cell_slot_indices_(common_data.GetSPDS().GetGrid()->local_cells.size(), INVALID_SLOT),
    cell_slot_bases_(common_data.GetSPDS().GetGrid()->local_cells.size(), nullptr),
    free_slot_stack_(num_slots_),
    local_psi_buffer_(num_slots_ * slot_size_),
    incoming_nonlocal_psi_data_(common_data.GetNumIncomingNonlocalFaceNodes() *
                                num_groups_and_angles_)
{
  for (std::uint32_t slot = 0; slot < num_slots_; ++slot)
    free_slot_stack_[slot] = slot;
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

void
CBC_FLUDS::AllocateSlot(std::uint64_t cell_local_id)
{
  OpenSnLogicalErrorIf(cell_slot_indices_[cell_local_id] != INVALID_SLOT,
                       "CBC_FLUDS attempted to allocate an already assigned slot.");
  OpenSnLogicalErrorIf(free_slot_stack_.empty(),
                       "CBC_FLUDS pool allocator exhausted during a local sweep.");

  const auto slot = free_slot_stack_.back();
  free_slot_stack_.pop_back();
  cell_slot_indices_[cell_local_id] = slot;
  cell_slot_bases_[cell_local_id] =
    local_psi_buffer_.data() + static_cast<size_t>(slot) * slot_size_;
}

void
CBC_FLUDS::DeallocateSlot(std::uint64_t cell_local_id)
{
  const auto slot = cell_slot_indices_[cell_local_id];
  OpenSnLogicalErrorIf(slot == INVALID_SLOT,
                       "CBC_FLUDS attempted to release a slot that is not assigned.");

  free_slot_stack_.push_back(slot);
  cell_slot_indices_[cell_local_id] = INVALID_SLOT;
  cell_slot_bases_[cell_local_id] = nullptr;
}

double*
CBC_FLUDS::UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx)
{
  auto* const slot_base = cell_slot_bases_[face_neighbor.local_id];
  OpenSnLogicalErrorIf(slot_base == nullptr,
                       "CBC_FLUDS missing local upwind storage for a swept neighbor cell.");

  const size_t offset = adj_cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return slot_base + offset;
}

double*
CBC_FLUDS::OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx)
{
  auto* const slot_base = cell_slot_bases_[cell.local_id];
  OpenSnLogicalErrorIf(slot_base == nullptr,
                       "CBC_FLUDS missing local output storage for the current cell.");

  const size_t offset = cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return slot_base + offset;
}

double*
CBC_FLUDS::NLUpwindPsi(uint64_t cell_global_id,
                       unsigned int face_id,
                       unsigned int face_node_mapped,
                       size_t as_ss_idx)
{
  const auto* face_info = common_data_.FindIncomingNonlocalFaceInfo(cell_global_id, face_id);
  if (face_info == nullptr)
    return nullptr;

  return NLUpwindPsi(*face_info, face_node_mapped, as_ss_idx);
}

double*
CBC_FLUDS::NLUpwindPsi(const CBC_FLUDSCommonData::IncomingNonlocalFaceInfo& face_info,
                       unsigned int face_node_mapped,
                       size_t as_ss_idx)
{
  const size_t dof_map =
    (static_cast<size_t>(face_info.face_node_offset) + face_node_mapped) * num_groups_and_angles_ +
    as_ss_idx * num_groups_;
  return incoming_nonlocal_psi_data_.data() + dof_map;
}

double*
CBC_FLUDS::NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing,
                         size_t face_node,
                         size_t as_ss_idx)
{
  OpenSnLogicalErrorIf(psi_nonlocal_outgoing == nullptr,
                       "CBC_FLUDS received a null nonlocal outgoing psi buffer.");

  return NLOutgoingPsi(psi_nonlocal_outgoing->data(), face_node, as_ss_idx);
}

double*
CBC_FLUDS::NLOutgoingPsi(double* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx)
{
  OpenSnLogicalErrorIf(psi_nonlocal_outgoing == nullptr,
                       "CBC_FLUDS received a null nonlocal outgoing psi buffer.");

  const size_t addr_offset = face_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return psi_nonlocal_outgoing + addr_offset;
}

void
CBC_FLUDS::StoreIncomingFaceData(uint64_t cell_global_id,
                                 unsigned int face_id,
                                 const double* psi_data,
                                 size_t data_size)
{
  const auto* face_info = common_data_.FindIncomingNonlocalFaceInfo(cell_global_id, face_id);
  OpenSnLogicalErrorIf(
    face_info == nullptr,
    "CBC_FLUDS received incoming nonlocal angular flux for an unknown cell-face pair.");

  const auto expected_size =
    static_cast<size_t>(face_info->num_face_nodes) * num_groups_and_angles_;
  OpenSnLogicalErrorIf(
    data_size != expected_size,
    "CBC_FLUDS received incoming nonlocal angular flux with an unexpected size.");

  const size_t base = static_cast<size_t>(face_info->face_node_offset) * num_groups_and_angles_;
  std::memcpy(incoming_nonlocal_psi_data_.data() + base, psi_data, data_size * sizeof(double));
}

void
CBC_FLUDS::ClearLocalAndReceivePsi()
{
  std::fill(cell_slot_indices_.begin(), cell_slot_indices_.end(), INVALID_SLOT);
  std::fill(cell_slot_bases_.begin(), cell_slot_bases_.end(), nullptr);
  free_slot_stack_.resize(num_slots_);
  for (std::uint32_t slot = 0; slot < num_slots_; ++slot)
    free_slot_stack_[slot] = slot;
}

} // namespace opensn

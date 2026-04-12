// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include <algorithm>
#include <cassert>
#include <limits>

namespace opensn
{

CBC_FLUDS::CBC_FLUDS(unsigned int num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    local_psi_data_size_(common_data_.GetTotalLocalFaceSlotNodes() * num_groups_and_angles_),
    local_psi_data_(local_psi_data_size_),
    incoming_nonlocal_psi_(common_data_.GetTotalIncomingNonlocalFaceNodes() *
                           num_groups_and_angles_),
    incoming_nonlocal_psi_generation_(common_data.GetNumIncomingNonlocalFaces(), 0)
{
}

const CBC_FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

double*
CBC_FLUDS::UpwindPsi(std::uint32_t cell_local_id,
                     unsigned int face_id,
                     unsigned int face_node_mapped,
                     size_t as_ss_idx)
{
  const auto slot_node_offset =
    common_data_.GetLocalFaceSlotNodeOffsetByLocalFace(cell_local_id, face_id);
  assert(slot_node_offset != CBC_FLUDSCommonData::INVALID_FACE_SLOT);
  const auto index =
    (slot_node_offset + face_node_mapped) * num_groups_and_angles_ + as_ss_idx * num_groups_;
  assert(index < local_psi_data_.size());
  return &local_psi_data_[index];
}

double*
CBC_FLUDS::OutgoingPsi(std::uint32_t cell_local_id,
                       unsigned int face_id,
                       size_t face_node,
                       size_t as_ss_idx)
{
  const auto slot_node_offset =
    common_data_.GetLocalFaceSlotNodeOffsetByLocalFace(cell_local_id, face_id);
  assert(slot_node_offset != CBC_FLUDSCommonData::INVALID_FACE_SLOT);
  const auto index =
    (slot_node_offset + face_node) * num_groups_and_angles_ + as_ss_idx * num_groups_;
  assert(index < local_psi_data_.size());
  return &local_psi_data_[index];
}

double*
CBC_FLUDS::NLUpwindPsi(size_t incoming_face_slot, unsigned int face_node_mapped, size_t as_ss_idx)
{
  if (incoming_face_slot == CBC_FLUDSCommonData::INVALID_FACE_SLOT or
      incoming_nonlocal_psi_generation_[incoming_face_slot] !=
        incoming_nonlocal_psi_current_generation_)
    return nullptr;

  const auto slot_offset =
    common_data_.GetIncomingNonlocalFaceNodeOffset(incoming_face_slot) * num_groups_and_angles_;
  const auto dof_map =
    face_node_mapped * num_groups_and_angles_ + //  Offset to start of data for face_node_mapped
    as_ss_idx * num_groups_;                    // Offset to start of data for angle_set_index

  assert(slot_offset + dof_map < incoming_nonlocal_psi_.size());
  return &incoming_nonlocal_psi_[slot_offset + dof_map];
}

double*
CBC_FLUDS::NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing,
                         size_t face_node,
                         size_t as_ss_idx)
{
  assert(psi_nonlocal_outgoing != nullptr);
  const auto addr_offset = face_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  assert(addr_offset < psi_nonlocal_outgoing->size());
  return &(*psi_nonlocal_outgoing)[addr_offset];
}

void
CBC_FLUDS::ClearLocalAndReceivePsi()
{
  if (incoming_nonlocal_psi_current_generation_ == std::numeric_limits<std::uint32_t>::max())
  {
    std::fill(
      incoming_nonlocal_psi_generation_.begin(), incoming_nonlocal_psi_generation_.end(), 0);
    incoming_nonlocal_psi_current_generation_ = 1;
  }
  else
    ++incoming_nonlocal_psi_current_generation_;
}

CBC_FLUDS::IncomingNonlocalPsi
CBC_FLUDS::PrepareIncomingNonlocalPsiBySlot(size_t incoming_face_slot, size_t data_size)
{
  assert(incoming_face_slot != CBC_FLUDSCommonData::INVALID_FACE_SLOT);
  assert(incoming_face_slot < incoming_nonlocal_psi_generation_.size());

  const auto slot_begin =
    common_data_.GetIncomingNonlocalFaceNodeOffset(incoming_face_slot) * num_groups_and_angles_;
  const auto slot_end =
    common_data_.GetIncomingNonlocalFaceNodeOffset(incoming_face_slot + 1) *
    num_groups_and_angles_;
  assert((slot_end - slot_begin) == data_size);

  incoming_nonlocal_psi_generation_[incoming_face_slot] = incoming_nonlocal_psi_current_generation_;

  return {std::span<double>(incoming_nonlocal_psi_.data() + slot_begin, data_size),
          common_data_.GetIncomingNonlocalFaceLocalCell(incoming_face_slot)};
}

} // namespace opensn

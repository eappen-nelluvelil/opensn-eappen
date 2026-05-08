// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include <algorithm>
#include <cassert>
#include <limits>

namespace opensn
{

namespace
{

void
UpdateSpan(std::vector<double>& data, std::span<double>& view)
{
  view = std::span<double>(data);
}

void
UpdateSpanVector(std::vector<std::vector<double>>& data, std::vector<std::span<double>>& views)
{
  views.resize(data.size());
  for (std::size_t i = 0; i < data.size(); ++i)
    views[i] = std::span<double>(data[i]);
}

} // namespace

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
    local_psi_data_(local_psi_data_size_),
    incoming_nonlocal_psi_offsets_(common_data.GetNumIncomingNonlocalFaces() + 1, 0),
    incoming_nonlocal_psi_generation_(common_data.GetNumIncomingNonlocalFaces(), 0)
{
  const auto& grid = *spds_.GetGrid();
  cell_psi_start_.resize(grid.local_cells.size());
  for (const auto& cell : grid.local_cells)
  {
    cell_psi_start_[cell.local_id] =
      (sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0) / num_angles_in_gs_quadrature_ / num_groups_) *
      num_groups_and_angles_;

    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto slot = common_data_.GetIncomingNonlocalFaceSlotByLocalFace(
        cell.local_id, static_cast<unsigned int>(f));
      if (slot == CBC_FLUDSCommonData::INVALID_FACE_SLOT)
        continue;

      incoming_nonlocal_psi_offsets_[slot + 1] =
        sdm_.GetCellMapping(cell).GetNumFaceNodes(f) * num_groups_and_angles_;
    }
  }

  for (std::size_t slot = 0; slot + 1 < incoming_nonlocal_psi_offsets_.size(); ++slot)
    incoming_nonlocal_psi_offsets_[slot + 1] += incoming_nonlocal_psi_offsets_[slot];
  incoming_nonlocal_psi_.resize(incoming_nonlocal_psi_offsets_.back());
}

const CBC_FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

double*
CBC_FLUDS::UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx)
{
  const auto index = cell_psi_start_[face_neighbor.local_id] +
                     adj_cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  assert(index < local_psi_data_.size());
  return &local_psi_data_[index];
}

double*
CBC_FLUDS::UpwindPsi(std::uint32_t cell_local_id,
                     unsigned int face_id,
                     unsigned int face_node_mapped,
                     size_t as_ss_idx)
{
  const auto& info = common_data_.GetDelayedLocalFaceInfo(cell_local_id, face_id);
  const auto index =
    (info.slot_address + face_node_mapped) * num_groups_and_angles_ + as_ss_idx * num_groups_;
  assert(index < delayed_local_psi_old_.size());
  return &delayed_local_psi_old_[index];
}

double*
CBC_FLUDS::OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx)
{
  const auto index =
    cell_psi_start_[cell.local_id] + cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  assert(index < local_psi_data_.size());
  return &local_psi_data_[index];
}

double*
CBC_FLUDS::OutgoingPsi(std::uint32_t cell_local_id,
                       unsigned int face_id,
                       unsigned int face_node,
                       size_t as_ss_idx)
{
  const auto& info = common_data_.GetDelayedLocalFaceInfo(cell_local_id, face_id);
  const auto index =
    (info.slot_address + face_node) * num_groups_and_angles_ + as_ss_idx * num_groups_;
  assert(index < delayed_local_psi_.size());
  return &delayed_local_psi_[index];
}

double*
CBC_FLUDS::NLUpwindPsi(size_t incoming_face_slot, unsigned int face_node_mapped, size_t as_ss_idx)
{
  if (incoming_face_slot == CBC_FLUDSCommonData::INVALID_FACE_SLOT or
      incoming_nonlocal_psi_generation_[incoming_face_slot] !=
        incoming_nonlocal_psi_current_generation_)
    return nullptr;

  const auto slot_offset = incoming_nonlocal_psi_offsets_[incoming_face_slot];
  const auto dof_map =
    face_node_mapped * num_groups_and_angles_ + //  Offset to start of data for face_node_mapped
    as_ss_idx * num_groups_;                    // Offset to start of data for angle_set_index

  assert(slot_offset + dof_map < incoming_nonlocal_psi_.size());
  return &incoming_nonlocal_psi_[slot_offset + dof_map];
}

double*
CBC_FLUDS::NLUpwindPsi(const CBC_FLUDSCommonData::DelayedNonlocalFaceInfo& info,
                       unsigned int face_node_mapped,
                       size_t as_ss_idx)
{
  assert(info.prelocI < delayed_prelocI_outgoing_psi_old_.size());
  auto& psi = delayed_prelocI_outgoing_psi_old_[info.prelocI];
  const auto index =
    (info.slot_address + face_node_mapped) * num_groups_and_angles_ + as_ss_idx * num_groups_;
  assert(index < psi.size());
  return &psi[index];
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

void
CBC_FLUDS::AllocateDelayedLocalPsi()
{
  const auto size = common_data_.GetNumDelayedLocalFaceNodes() * num_groups_and_angles_;
  delayed_local_psi_.assign(size, 0.0);
  delayed_local_psi_old_.assign(size, 0.0);
  UpdateSpan(delayed_local_psi_, delayed_local_psi_view_);
  UpdateSpan(delayed_local_psi_old_, delayed_local_psi_old_view_);
}

void
CBC_FLUDS::AllocateDelayedPrelocIOutgoingPsi()
{
  const auto num_delayed_dependencies = spds_.GetDelayedLocationDependencies().size();
  delayed_prelocI_outgoing_psi_.resize(num_delayed_dependencies);
  delayed_prelocI_outgoing_psi_old_.resize(num_delayed_dependencies);

  for (size_t prelocI = 0; prelocI < num_delayed_dependencies; ++prelocI)
  {
    const auto size = common_data_.GetDelayedPrelocIFaceNodeCount(prelocI) * num_groups_and_angles_;
    delayed_prelocI_outgoing_psi_[prelocI].assign(size, 0.0);
    delayed_prelocI_outgoing_psi_old_[prelocI].assign(size, 0.0);
  }

  UpdateSpanVector(delayed_prelocI_outgoing_psi_, delayed_prelocI_outgoing_psi_view_);
  UpdateSpanVector(delayed_prelocI_outgoing_psi_old_, delayed_prelocI_outgoing_psi_old_view_);
}

void
CBC_FLUDS::SetDelayedLocalPsiOldToNew()
{
  delayed_local_psi_ = delayed_local_psi_old_;
  UpdateSpan(delayed_local_psi_, delayed_local_psi_view_);
}

void
CBC_FLUDS::SetDelayedLocalPsiNewToOld()
{
  delayed_local_psi_old_ = delayed_local_psi_;
  UpdateSpan(delayed_local_psi_old_, delayed_local_psi_old_view_);
}

void
CBC_FLUDS::SetDelayedOutgoingPsiOldToNew()
{
  delayed_prelocI_outgoing_psi_ = delayed_prelocI_outgoing_psi_old_;
  UpdateSpanVector(delayed_prelocI_outgoing_psi_, delayed_prelocI_outgoing_psi_view_);
}

void
CBC_FLUDS::SetDelayedOutgoingPsiNewToOld()
{
  delayed_prelocI_outgoing_psi_old_ = delayed_prelocI_outgoing_psi_;
  UpdateSpanVector(delayed_prelocI_outgoing_psi_old_, delayed_prelocI_outgoing_psi_old_view_);
}

CBC_FLUDS::IncomingNonlocalPsi
CBC_FLUDS::PrepareIncomingNonlocalPsiBySlot(size_t incoming_face_slot, size_t data_size)
{
  assert(incoming_face_slot != CBC_FLUDSCommonData::INVALID_FACE_SLOT);
  assert(incoming_face_slot < incoming_nonlocal_psi_generation_.size());

  const auto slot_begin = incoming_nonlocal_psi_offsets_[incoming_face_slot];
  assert((incoming_nonlocal_psi_offsets_[incoming_face_slot + 1] - slot_begin) == data_size);

  if (incoming_nonlocal_psi_generation_[incoming_face_slot] ==
      incoming_nonlocal_psi_current_generation_)
    throw std::logic_error("CBC_FLUDS received duplicate non-local psi for a cell-face slot.");

  incoming_nonlocal_psi_generation_[incoming_face_slot] = incoming_nonlocal_psi_current_generation_;

  return {std::span<double>(incoming_nonlocal_psi_.data() + slot_begin, data_size),
          common_data_.GetIncomingNonlocalFaceLocalCell(incoming_face_slot)};
}

size_t
CBC_FLUDS::GetIncomingNonlocalPsiSize(size_t incoming_face_slot) const
{
  if (incoming_face_slot == CBC_FLUDSCommonData::INVALID_FACE_SLOT or
      incoming_face_slot + 1 >= incoming_nonlocal_psi_offsets_.size())
    throw std::logic_error("CBC_FLUDS received non-local psi for an unknown cell-face slot.");

  return incoming_nonlocal_psi_offsets_[incoming_face_slot + 1] -
         incoming_nonlocal_psi_offsets_[incoming_face_slot];
}

size_t
CBC_FLUDS::GetDelayedNonlocalPsiSize(size_t delayed_face_slot) const
{
  return common_data_.GetDelayedNonlocalFaceNodeCount(delayed_face_slot) * num_groups_and_angles_;
}

std::span<double>
CBC_FLUDS::PrepareIncomingDelayedNonlocalPsiBySlot(size_t delayed_face_slot, size_t data_size)
{
  if (data_size != GetDelayedNonlocalPsiSize(delayed_face_slot))
    throw std::logic_error(
      "CBC_FLUDS received delayed non-local psi with an unexpected payload size.");

  const auto& info = common_data_.GetDelayedNonlocalFaceInfoBySlot(delayed_face_slot);
  if (info.prelocI >= delayed_prelocI_outgoing_psi_.size())
    throw std::logic_error("CBC_FLUDS received delayed non-local psi for an unknown dependency.");

  auto& psi = delayed_prelocI_outgoing_psi_[info.prelocI];
  const auto begin = info.slot_address * num_groups_and_angles_;
  assert(begin + data_size <= psi.size());
  return std::span<double>(psi.data() + begin, data_size);
}

} // namespace opensn

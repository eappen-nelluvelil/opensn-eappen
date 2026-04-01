// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
  // SPDX-License-Identifier: MIT

  #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
  #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
  #include "framework/math/spatial_discretization/spatial_discretization.h"
  #include "framework/mesh/mesh_continuum/mesh_continuum.h"
  #include <cstring>

  namespace opensn
  {

  namespace
  {
  void UpdateRange(std::vector<std::vector<double>>& target, std::vector<std::span<double>>& view)
  {
    view.resize(target.size());
    for (std::size_t i = 0; i < target.size(); ++i)
      view[i] = std::span<double>(target[i]);
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
      local_psi_data_(local_psi_data_size_)
  {
    const auto& grid = *spds_.GetGrid();
    cell_psi_start_.resize(grid.local_cells.size());
    for (const auto& cell : grid.local_cells)
    {
      cell_psi_start_[cell.local_id] =
        (sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0) / num_angles_in_gs_quadrature_ / num_groups_) *
        num_groups_and_angles_;
    }

    deplocs_outgoing_messages_.reserve(common_data.GetNumIncomingNonlocalFaces());
  }

  const FLUDSCommonData&
  CBC_FLUDS::GetCommonData() const
  {
    return common_data_;
  }

  double*
  CBC_FLUDS::UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx)
  {
    const size_t index = cell_psi_start_[face_neighbor.local_id] +
                         adj_cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
    return &local_psi_data_[index];
  }

  double*
  CBC_FLUDS::OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx)
  {
    const size_t index =
      cell_psi_start_[cell.local_id] + cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
    return &local_psi_data_[index];
  }

  double*
  CBC_FLUDS::DelayedLocalUpwindPsi(const Cell& face_neighbor,
                                   unsigned int adj_cell_node,
                                   size_t as_ss_idx)
  {
    if (delayed_local_psi_old_data_.empty())
      return nullptr;

    const size_t index = cell_psi_start_[face_neighbor.local_id] +
                         adj_cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
    return &delayed_local_psi_old_data_[index];
  }

  double*
  CBC_FLUDS::DelayedLocalOutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx)
  {
    if (delayed_local_psi_data_.empty())
      return nullptr;

    const size_t index =
      cell_psi_start_[cell.local_id] + cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
    return &delayed_local_psi_data_[index];
  }

  double*
  CBC_FLUDS::NLUpwindPsi(uint64_t cell_global_id,
                         unsigned int face_id,
                         unsigned int face_node_mapped,
                         size_t as_ss_idx)
  {
    auto it = deplocs_outgoing_messages_.find({cell_global_id, face_id});
    if (it == deplocs_outgoing_messages_.end())
      return nullptr;

    auto& psi = it->second;
    const size_t dof_map = face_node_mapped * num_groups_and_angles_ + as_ss_idx * num_groups_;
    return &psi[dof_map];
  }

  double*
  CBC_FLUDS::DelayedNLUpwindPsi(const CBC_FLUDSCommonData::DelayedNonlocalFaceInfo& info,
                                unsigned int face_node_mapped,
                                size_t as_ss_idx)
  {
    auto& psi = delayed_preloc_outgoing_psi_old_[info.prelocI];
    const size_t dof_map =
      (info.slot_address + face_node_mapped) * num_groups_and_angles_ + as_ss_idx * num_groups_;
    return &psi[dof_map];
  }

  double*
  CBC_FLUDS::NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing,
                           size_t face_node,
                           size_t as_ss_idx)
  {
    const size_t addr_offset = face_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
    return &(*psi_nonlocal_outgoing)[addr_offset];
  }

  void
  CBC_FLUDS::StoreIncomingFaceData(uint64_t cell_global_id,
                                   unsigned int face_id,
                                   std::vector<double>&& psi_data)
  {
    deplocs_outgoing_messages_[{cell_global_id, face_id}] = std::move(psi_data);
  }

  void
  CBC_FLUDS::StoreDelayedIncomingFaceData(uint64_t cell_global_id,
                                          unsigned int face_id,
                                          const double* psi_data,
                                          size_t data_size)
  {
    CBC_FLUDSCommonData::DelayedNonlocalFaceInfo info;
    if (not common_data_.TryGetDelayedNonlocalFaceInfo(cell_global_id, face_id, info))
      return;

    auto& buffer = delayed_preloc_outgoing_psi_[info.prelocI];
    const size_t begin = info.slot_address * num_groups_and_angles_;
    std::memcpy(buffer.data() + begin, psi_data, data_size * sizeof(double));
  }

  void
  CBC_FLUDS::AllocateDelayedLocalPsi()
  {
    if (not common_data_.HasDelayedLocalDependencies())
      return;

    delayed_local_psi_data_.assign(local_psi_data_size_, 0.0);
    delayed_local_psi_old_data_.assign(local_psi_data_size_, 0.0);
    delayed_local_psi_view_ = std::span<double>(delayed_local_psi_data_);
    delayed_local_psi_old_view_ = std::span<double>(delayed_local_psi_old_data_);
  }

  void
  CBC_FLUDS::AllocateDelayedPrelocIOutgoingPsi()
  {
    const auto num_delayed = spds_.GetDelayedLocationDependencies().size();
    delayed_preloc_outgoing_psi_.resize(num_delayed);
    delayed_preloc_outgoing_psi_old_.resize(num_delayed);

    for (size_t prelocI = 0; prelocI < num_delayed; ++prelocI)
    {
      const auto count = common_data_.GetDelayedPrelocIFaceNodeCount(prelocI);
      delayed_preloc_outgoing_psi_[prelocI].assign(count * num_groups_and_angles_, 0.0);
      delayed_preloc_outgoing_psi_old_[prelocI].assign(count * num_groups_and_angles_, 0.0);
    }

    UpdateRange(delayed_preloc_outgoing_psi_, delayed_prelocI_outgoing_psi_view_);
    UpdateRange(delayed_preloc_outgoing_psi_old_, delayed_prelocI_outgoing_psi_old_view_);
  }

  void
  CBC_FLUDS::SetDelayedOutgoingPsiOldToNew()
  {
    delayed_preloc_outgoing_psi_ = delayed_preloc_outgoing_psi_old_;
    UpdateRange(delayed_preloc_outgoing_psi_, delayed_prelocI_outgoing_psi_view_);
  }

  void
  CBC_FLUDS::SetDelayedOutgoingPsiNewToOld()
  {
    delayed_preloc_outgoing_psi_old_ = delayed_preloc_outgoing_psi_;
    UpdateRange(delayed_preloc_outgoing_psi_old_, delayed_prelocI_outgoing_psi_old_view_);
  }

  void
  CBC_FLUDS::SetDelayedLocalPsiOldToNew()
  {
    delayed_local_psi_data_ = delayed_local_psi_old_data_;
    delayed_local_psi_view_ = std::span<double>(delayed_local_psi_data_);
  }

  void
  CBC_FLUDS::SetDelayedLocalPsiNewToOld()
  {
    delayed_local_psi_old_data_ = delayed_local_psi_data_;
    delayed_local_psi_old_view_ = std::span<double>(delayed_local_psi_old_data_);
  }

  } // namespace opensn
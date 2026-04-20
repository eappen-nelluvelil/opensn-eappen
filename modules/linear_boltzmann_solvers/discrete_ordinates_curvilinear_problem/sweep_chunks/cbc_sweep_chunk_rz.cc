// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_curvilinear_problem/discrete_ordinates_curvilinear_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_curvilinear_problem/sweep_chunks/cbc_sweep_chunk_rz.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "framework/math/quadratures/angular/curvilinear_product_quadrature.h"
#include <stdexcept>

namespace opensn
{

CBCSweepChunkRZ::CBCSweepChunkRZ(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset)
  : SweepChunkRZ(problem, groupset)
{
}

void
CBCSweepChunkRZ::SetAngleSet(AngleSet& angle_set)
{
  (void)angle_set;
}

void
CBCSweepChunkRZ::SetCell(const Cell* cell_ptr, AngleSet& angle_set)
{
  current_cell_ = cell_ptr;
}

void
CBCSweepChunkRZ::Sweep(AngleSet& angle_set)
{
  if (current_cell_ == nullptr)
    throw std::logic_error("CBCSweepChunkRZ::Sweep called without an active cell.");

  const auto gs_size = groupset_.GetNumGroups();
  const auto gs_gi = groupset_.first_group;

  auto& fluds = dynamic_cast<CBC_FLUDS&>(angle_set.GetFLUDS());
  const auto& cbc_common = dynamic_cast<const CBC_FLUDSCommonData&>(fluds.GetCommonData());
  auto* const async_comm = dynamic_cast<CBC_AsynchronousCommunicator*>(angle_set.GetCommunicator());
  const auto& m2d_op = groupset_.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset_.quadrature->GetDiscreteToMomentOperator();

  const auto& spds = angle_set.GetSPDS();
  const auto cell_local_id = current_cell_->local_id;
  const auto& cell = *current_cell_;
  const auto& cell_mapping = discretization_.GetCellMapping(cell);
  auto& cell_transport_view = cell_transport_views_[cell_local_id];
  const auto cell_num_faces = cell.faces.size();
  const auto cell_num_nodes = cell_mapping.GetNumNodes();

  const auto& face_orientations = spds.GetCellFaceOrientations()[cell_local_id];
  std::vector<double> face_mu_values(cell_num_faces);

  const auto& sigma_t = xs_.at(cell.block_id)->GetSigmaTotal();
  const auto& G = unit_cell_matrices_[cell_local_id].intV_shapeI_gradshapeJ;
  const auto& M = unit_cell_matrices_[cell_local_id].intV_shapeI_shapeJ;
  const auto& M_surf = unit_cell_matrices_[cell_local_id].intS_shapeI_shapeJ;
  const auto& Maux = secondary_unit_cell_matrices_[cell_local_id].intV_shapeI_shapeJ;

  const std::vector<std::uint32_t>& as_angle_indices = angle_set.GetAngleIndices();
  for (size_t as_ss_idx = 0; as_ss_idx < as_angle_indices.size(); ++as_ss_idx)
  {
    const auto direction_num = as_angle_indices[as_ss_idx];
    const auto omega = groupset_.quadrature->omegas[direction_num];
    const auto wt = groupset_.quadrature->weights[direction_num];

    const auto polar_level = PolarLevel(direction_num);
    const auto fac_diamond_difference =
      curvilinear_product_quadrature_->GetDiamondDifferenceFactor()[direction_num];
    const auto fac_streaming_operator =
      curvilinear_product_quadrature_->GetStreamingOperatorFactor()[direction_num];

    for (size_t gsg = 0; gsg < gs_size; ++gsg)
      for (size_t i = 0; i < cell_num_nodes; ++i)
        b_[gsg](i) = 0.0;

    for (size_t i = 0; i < cell_num_nodes; ++i)
    {
      for (size_t j = 0; j < cell_num_nodes; ++j)
      {
        const auto jr = discretization_.MapDOFLocal(cell, j, unknown_manager_, polar_level, gs_gi);
        for (size_t gsg = 0; gsg < gs_size; ++gsg)
          b_[gsg](i) += fac_streaming_operator * Maux(i, j) * psi_sweep_[jr + gsg];
      }
    }

    for (size_t i = 0; i < cell_num_nodes; ++i)
      for (size_t j = 0; j < cell_num_nodes; ++j)
        Amat_(i, j) = omega.Dot(G(i, j)) + fac_streaming_operator * Maux(i, j);

    for (size_t f = 0; f < cell_num_faces; ++f)
      face_mu_values[f] = omega.Dot(cell.faces[f].normal);

    for (size_t f = 0; f < cell_num_faces; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& cell_face = cell.faces[f];
      const auto* face_nodal_mapping = &fluds.GetCommonData().GetFaceNodalMapping(cell_local_id, f);
      const auto face_kind = cbc_common.GetIncomingFaceKind(cell_local_id, f);
      const bool is_delayed_local_face =
        face_kind == CBC_FLUDSCommonData::IncomingFaceKind::DELAYED_LOCAL;
      const bool is_delayed_nonlocal_face =
        face_kind == CBC_FLUDSCommonData::IncomingFaceKind::DELAYED_NONLOCAL;
      const bool is_normal_local_face =
        face_kind == CBC_FLUDSCommonData::IncomingFaceKind::NORMAL_LOCAL;
      const bool is_normal_nonlocal_face =
        face_kind == CBC_FLUDSCommonData::IncomingFaceKind::NORMAL_NONLOCAL;

      const auto delayed_nonlocal_face_info =
        is_delayed_nonlocal_face
          ? cbc_common.GetDelayedNonlocalFaceInfo(cell_local_id, static_cast<unsigned int>(f))
          : CBC_FLUDSCommonData::DelayedNonlocalFaceInfo{};

      const bool is_boundary_face = not cell_face.has_neighbor;
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping.MapFaceNode(f, fi);

        for (size_t fj = 0; fj < num_face_nodes; ++fj)
        {
          const int j = cell_mapping.MapFaceNode(f, fj);
          const double mu_Nij = -face_mu_values[f] * M_surf[f](i, j);
          Amat_(i, j) += mu_Nij;

          const double* psi = nullptr;
          if (is_delayed_local_face)
            psi = fluds.DelayedLocalUpwindPsi(cell_local_id,
                                              static_cast<unsigned int>(f),
                                              face_nodal_mapping->face_node_mapping_[fj],
                                              as_ss_idx);
          else if (is_delayed_nonlocal_face)
            psi = fluds.DelayedNLUpwindPsi(
              delayed_nonlocal_face_info, face_nodal_mapping->face_node_mapping_[fj], as_ss_idx);
          else if (is_normal_local_face)
            psi = fluds.UpwindPsi(cell_local_id,
                                  static_cast<unsigned int>(f),
                                  face_nodal_mapping->face_node_mapping_[fj],
                                  as_ss_idx);
          else if (is_normal_nonlocal_face)
            psi = fluds.NLUpwindPsi(cell_local_id,
                                    static_cast<unsigned int>(f),
                                    face_nodal_mapping->face_node_mapping_[fj],
                                    as_ss_idx);
          else if (not is_boundary_face)
            continue;
          else
          {
            const bool incident_on_symmetric_boundary =
              (cell_face.normal.Dot(normal_vector_boundary_) < -0.999999);
            if (not incident_on_symmetric_boundary)
            {
              psi = angle_set.PsiBoundary(cell_face.neighbor_id,
                                          direction_num,
                                          cell_local_id,
                                          f,
                                          fj,
                                          gs_gi,
                                          IsSurfaceSourceActive());
            }
          }

          if (psi == nullptr)
            continue;

          for (size_t gsg = 0; gsg < gs_size; ++gsg)
            b_[gsg](i) += psi[gsg] * mu_Nij;
        }
      }
    }

    const auto row_offset = static_cast<size_t>(direction_num) * static_cast<size_t>(num_moments_);
    const double* m2d_row = m2d_op.data() + row_offset;
    const double* d2m_row = d2m_op.data() + row_offset;

    for (size_t gsg = 0; gsg < gs_size; ++gsg)
    {
      const double sigma_tg = sigma_t[gs_gi + gsg];

      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        double temp_src = 0.0;
        for (unsigned int m = 0; m < num_moments_; ++m)
        {
          const auto ir = cell_transport_view.MapDOF(i, m, gs_gi + gsg);
          temp_src += m2d_row[m] * source_moments_[ir];
        }
        source_[i] = temp_src;
      }

      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        double temp = 0.0;
        for (size_t j = 0; j < cell_num_nodes; ++j)
        {
          const double Mij = M(i, j);
          Atemp_(i, j) = Amat_(i, j) + Mij * sigma_tg;
          temp += Mij * source_[j];
        }
        b_[gsg](i) += temp;
      }

      GaussElimination(Atemp_, b_[gsg], static_cast<int>(cell_num_nodes));
    }

    for (unsigned int m = 0; m < num_moments_; ++m)
    {
      const double wn_d2m = d2m_row[m];
      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        const auto ir = cell_transport_view.MapDOF(i, m, gs_gi);
        for (size_t gsg = 0; gsg < gs_size; ++gsg)
          destination_phi_[ir + gsg] += wn_d2m * b_[gsg](i);
      }
    }

    if (SaveAngularFluxEnabled())
    {
      double* cell_psi_data =
        &destination_psi_[discretization_.MapDOFLocal(cell, 0, groupset_.psi_uk_man_, 0, 0)];

      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        const size_t imap =
          i * groupset_angle_group_stride_ + direction_num * groupset_group_stride_;
        for (size_t gsg = 0; gsg < gs_size; ++gsg)
          cell_psi_data[imap + gsg] = b_[gsg](i);
      }
    }

    const auto f0 = 1.0 / fac_diamond_difference;
    const auto f1 = f0 - 1.0;

    for (size_t f = 0; f < cell_num_faces; ++f)
    {
      if (face_orientations[f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face = cell.faces[f];
      const auto* face_nodal_mapping = &fluds.GetCommonData().GetFaceNodalMapping(cell_local_id, f);
      const auto face_kind = cbc_common.GetOutgoingFaceKind(cell_local_id, f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        is_boundary_face and angle_set.GetBoundaries()[face.neighbor_id]->IsReflecting();
      const auto& IntF_shapeI = unit_cell_matrices_[cell_local_id].intS_shapeI[f];

      double* psi_nonlocal_outgoing = nullptr;
      if (face_kind == CBC_FLUDSCommonData::OutgoingFaceKind::NORMAL_NONLOCAL)
      {
        const auto& outgoing_nonlocal_face_info =
          cbc_common.GetOutgoingNonlocalFaceInfo(cell_local_id, static_cast<unsigned int>(f));
        const size_t data_size_for_msg =
          static_cast<size_t>(outgoing_nonlocal_face_info.num_face_nodes) *
          groupset_.GetNumGroups() * as_angle_indices.size();
        psi_nonlocal_outgoing =
          async_comm
            ->InitGetDownwindMessageData(outgoing_nonlocal_face_info.locality,
                                         outgoing_nonlocal_face_info.cell_global_id,
                                         outgoing_nonlocal_face_info.associated_face,
                                         angle_set.GetID(),
                                         data_size_for_msg)
            .data();
      }

      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping.MapFaceNode(f, fi);

        if (is_boundary_face)
        {
          for (size_t gsg = 0; gsg < gs_size; ++gsg)
            cell_transport_view.AddOutflow(
              f, gs_gi + gsg, wt * face_mu_values[f] * b_[gsg](i) * IntF_shapeI(i));
        }

        double* psi = nullptr;
        if (face_kind == CBC_FLUDSCommonData::OutgoingFaceKind::DELAYED_LOCAL)
        {
          const auto delayed_local_cell_local_id = face.GetNeighborLocalID(grid_.get());
          const auto delayed_local_face_id =
            static_cast<unsigned int>(face_nodal_mapping->associated_face_);
          psi = fluds.DelayedLocalOutgoingPsi(delayed_local_cell_local_id,
                                              delayed_local_face_id,
                                              static_cast<unsigned int>(fi),
                                              as_ss_idx);
        }
        else if (face_kind == CBC_FLUDSCommonData::OutgoingFaceKind::NORMAL_LOCAL)
          psi = fluds.OutgoingPsi(cell_local_id, static_cast<unsigned int>(f), fi, as_ss_idx);
        else if (face_kind == CBC_FLUDSCommonData::OutgoingFaceKind::NORMAL_NONLOCAL)
          psi = fluds.NLOutgoingPsi(psi_nonlocal_outgoing, fi, as_ss_idx);
        else if (is_reflecting_boundary_face)
          psi = angle_set.PsiReflected(face.neighbor_id, direction_num, cell_local_id, f, fi);

        if ((not is_boundary_face or is_reflecting_boundary_face) and psi != nullptr)
        {
          for (size_t gsg = 0; gsg < gs_size; ++gsg)
            psi[gsg] = b_[gsg](i);
        }
      }
    }

    for (size_t i = 0; i < cell_num_nodes; ++i)
    {
      const auto ir = discretization_.MapDOFLocal(cell, i, unknown_manager_, polar_level, gs_gi);
      for (size_t gsg = 0; gsg < gs_size; ++gsg)
        psi_sweep_[ir + gsg] = f0 * b_[gsg](i) - f1 * psi_sweep_[ir + gsg];
    }
  }
}

} // namespace opensn

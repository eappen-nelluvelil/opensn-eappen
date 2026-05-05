// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_curvilinear_problem/sweep_chunks/cbc_sweep_chunk_rz.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_curvilinear_problem/discrete_ordinates_curvilinear_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/math/quadratures/angular/curvilinear_product_quadrature.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/utils/error.h"
#include "caliper/cali.h"
#include <stdexcept>

namespace opensn
{
namespace
{

const CurvilinearProductQuadrature&
RequireCurvilinearProductQuadrature(const LBSGroupset& groupset)
{
  const auto* quadrature =
    dynamic_cast<const CurvilinearProductQuadrature*>(groupset.quadrature.get());
  if (quadrature == nullptr)
    throw std::invalid_argument("CBCSweepChunkRZ requires a curvilinear product quadrature.");
  return *quadrature;
}

} // namespace

CBCSweepChunkRZ::CBCSweepChunkRZ(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset)
  : SweepChunk(problem.GetPhiNewLocal(),
               problem.GetPsiNewLocal()[groupset.id],
               problem.GetGrid(),
               problem.GetSpatialDiscretization(),
               problem.GetUnitCellMatrices(),
               problem.GetCellTransportViews(),
               problem.GetQMomentsLocal(),
               groupset,
               problem.GetBlockID2XSMap(),
               problem.GetNumMoments(),
               problem.GetMaxCellDOFCount(),
               problem.GetMinCellDOFCount()),
    curvilinear_quadrature_(RequireCurvilinearProductQuadrature(groupset)),
    secondary_unit_cell_matrices_(dynamic_cast<const DiscreteOrdinatesCurvilinearProblem&>(problem)
                                    .GetSecondaryUnitCellMatrices()),
    unknown_manager_(),
    psi_sweep_(),
    direction_polar_level_(groupset.quadrature->abscissae.size(), 0),
    normal_vector_boundary_()
{
  const auto& direction_map = curvilinear_quadrature_.GetDirectionMap();
  const auto num_polar_levels = direction_map.size();
  for (size_t m = 0; m < num_polar_levels; ++m)
    unknown_manager_.AddUnknown(UnknownType::VECTOR_N, groupset_.GetNumGroups());

  psi_sweep_.assign(discretization_.GetNumLocalDOFs(unknown_manager_), 0.0);

  for (const auto& [polar_level, direction_indices] : direction_map)
  {
    for (const auto direction_index : direction_indices)
    {
      OpenSnLogicalErrorIf(direction_index >= direction_polar_level_.size(),
                           "CBCSweepChunkRZ received an invalid quadrature direction index.");
      direction_polar_level_[direction_index] = polar_level;
    }
  }

  const auto d = (grid_->GetDimension() == 1) ? 2 : 0;
  normal_vector_boundary_ = Vector3(0.0, 0.0, 0.0);
  normal_vector_boundary_(d) = 1.0;
}

void
CBCSweepChunkRZ::SetAngleSet(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunkRZ::SetAngleSet");

  CBCBindAngleSetContext(ctx_, groupset_, IsSurfaceSourceActive(), angle_set);
}

void
CBCSweepChunkRZ::SetCell(const Cell* cell_ptr, AngleSet& /*angle_set*/)
{
  CBCBindCellContext(ctx_, discretization_, unit_cell_matrices_, cell_transport_views_, cell_ptr);
}

void
CBCSweepChunkRZ::PrepareOutgoingNonlocalFaceBuffers(
  const std::vector<FaceOrientation>& face_orientations)
{
  auto& buffers = ctx_.outgoing_nonlocal_face_buffers;
  auto& buffer_by_face = ctx_.outgoing_nonlocal_face_buffer_by_face;
  buffers.reserve(ctx_.cell_num_faces);
  buffer_by_face.assign(ctx_.cell_num_faces, nullptr);
  ctx_.num_outgoing_nonlocal_face_buffers = 0;

  auto& fluds = *ctx_.fluds;
  const auto& common_data = fluds.GetCommonData();

  for (size_t f = 0; f < ctx_.cell_num_faces; ++f)
  {
    if (face_orientations[f] != FaceOrientation::OUTGOING)
      continue;

    const auto& face = ctx_.cell->faces[f];
    if ((not face.has_neighbor) or ctx_.cell_transport_view->IsFaceLocal(f))
      continue;

    const auto buffer_index = ctx_.num_outgoing_nonlocal_face_buffers++;
    if (buffer_index == buffers.size())
      buffers.emplace_back();

    auto& buffer = buffers[buffer_index];
    buffer.incoming_face_slot = common_data.GetOutgoingNonlocalFaceSlotByLocalFace(
      ctx_.cell_local_id, static_cast<unsigned int>(f));
    buffer.peer_index = common_data.GetOutgoingNonlocalFacePeerIndexByLocalFace(
      ctx_.cell_local_id, static_cast<unsigned int>(f));

    OpenSnLogicalErrorIf(buffer.incoming_face_slot == CBC_FLUDSCommonData::INVALID_FACE_SLOT,
                         "CBCSweepChunkRZ missing an outgoing non-local face slot.");
    OpenSnLogicalErrorIf(buffer.peer_index == CBC_FLUDSCommonData::INVALID_PEER_INDEX,
                         "CBCSweepChunkRZ missing an outgoing non-local peer index.");

    buffer.Prepare(ctx_.cell_mapping->GetNumFaceNodes(f) * ctx_.group_angle_stride);
    buffer_by_face[f] = &buffer;
  }
}

void
CBCSweepChunkRZ::QueueOutgoingNonlocalFaceBuffers()
{
  auto& async_comm = *ctx_.async_comm;
  for (size_t i = 0; i < ctx_.num_outgoing_nonlocal_face_buffers; ++i)
  {
    const auto& buffer = ctx_.outgoing_nonlocal_face_buffers[i];
    async_comm.QueueDownwindMessage(buffer.peer_index,
                                    buffer.incoming_face_slot,
                                    std::span<const double>(buffer.data.data(), buffer.data_size));
  }
}

void
CBCSweepChunkRZ::Sweep(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunkRZ::Sweep");

  OpenSnLogicalErrorIf(ctx_.cell == nullptr,
                       "CBCSweepChunkRZ::Sweep called before a cell was bound.");
  OpenSnLogicalErrorIf(ctx_.fluds == nullptr,
                       "CBCSweepChunkRZ::Sweep called before an angle set was bound.");

  auto& fluds = *ctx_.fluds;
  const auto& common_data = fluds.GetCommonData();
  const auto& groupset = groupset_;
  const auto& m2d_op = groupset.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset.quadrature->GetDiscreteToMomentOperator();
  const auto& diamond_difference_factor = curvilinear_quadrature_.GetDiamondDifferenceFactor();
  const auto& streaming_operator_factor = curvilinear_quadrature_.GetStreamingOperatorFactor();

  const auto& cell = *ctx_.cell;
  const auto& cell_mapping = *ctx_.cell_mapping;
  auto& cell_transport_view = *ctx_.cell_transport_view;
  const auto& face_orientations =
    angle_set.GetSPDS().GetCellFaceOrientations()[ctx_.cell_local_id];
  const auto& sigma_t = xs_.at(cell.block_id)->GetSigmaTotal();

  const auto& G = *ctx_.G;
  const auto& M = *ctx_.M;
  const auto& M_surf = *ctx_.M_surf;
  const auto& IntS_shapeI = *ctx_.IntS_shapeI;
  const auto& Maux = secondary_unit_cell_matrices_[ctx_.cell_local_id].intV_shapeI_shapeJ;

  DenseMatrix<double> Amat(max_num_cell_dofs_, max_num_cell_dofs_);
  DenseMatrix<double> Atemp(max_num_cell_dofs_, max_num_cell_dofs_);
  std::vector<Vector<double>> b(ctx_.gs_size, Vector<double>(max_num_cell_dofs_));
  std::vector<double> source(max_num_cell_dofs_);

  ctx_.face_mu_values.assign(ctx_.cell_num_faces, 0.0);
  PrepareOutgoingNonlocalFaceBuffers(face_orientations);

  const auto& as_angle_indices = angle_set.GetAngleIndices();
  for (size_t as_ss_idx = 0; as_ss_idx < ctx_.num_angles_in_as; ++as_ss_idx)
  {
    const auto direction_num = as_angle_indices[as_ss_idx];
    const auto& omega = groupset.quadrature->omegas[direction_num];
    const auto wt = groupset.quadrature->weights[direction_num];

    const auto polar_level = direction_polar_level_[direction_num];
    const auto fac_diamond_difference = diamond_difference_factor[direction_num];
    const auto fac_streaming_operator = streaming_operator_factor[direction_num];

    for (size_t gsg = 0; gsg < ctx_.gs_size; ++gsg)
      for (size_t i = 0; i < ctx_.cell_num_nodes; ++i)
        b[gsg](i) = 0.0;

    for (size_t i = 0; i < ctx_.cell_num_nodes; ++i)
    {
      for (size_t j = 0; j < ctx_.cell_num_nodes; ++j)
      {
        const auto jr =
          discretization_.MapDOFLocal(cell, j, unknown_manager_, polar_level, ctx_.gs_gi);
        for (size_t gsg = 0; gsg < ctx_.gs_size; ++gsg)
          b[gsg](i) += fac_streaming_operator * Maux(i, j) * psi_sweep_[jr + gsg];
      }
    }

    for (size_t i = 0; i < ctx_.cell_num_nodes; ++i)
      for (size_t j = 0; j < ctx_.cell_num_nodes; ++j)
        Amat(i, j) = omega.Dot(G(i, j)) + fac_streaming_operator * Maux(i, j);

    for (size_t f = 0; f < ctx_.cell_num_faces; ++f)
      ctx_.face_mu_values[f] = omega.Dot(cell.faces[f].normal);

    for (size_t f = 0; f < ctx_.cell_num_faces; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& face = cell.faces[f];
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const auto* face_nodal_mapping = is_boundary_face
                                         ? nullptr
                                         : &common_data.GetFaceNodalMapping(
                                             ctx_.cell_local_id, static_cast<unsigned int>(f));
      const auto incoming_nonlocal_slot =
        (is_boundary_face or is_local_face)
          ? CBC_FLUDSCommonData::INVALID_FACE_SLOT
          : common_data.GetIncomingNonlocalFaceSlotByLocalFace(ctx_.cell_local_id,
                                                               static_cast<unsigned int>(f));

      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping.MapFaceNode(f, fi);

        for (size_t fj = 0; fj < num_face_nodes; ++fj)
        {
          const int j = cell_mapping.MapFaceNode(f, fj);
          const double mu_Nij = -ctx_.face_mu_values[f] * M_surf[f](i, j);
          Amat(i, j) += mu_Nij;

          const double* psi = nullptr;
          if (is_local_face)
          {
            psi = fluds.UpwindPsi(*cell_transport_view.FaceNeighbor(f),
                                  face_nodal_mapping->cell_node_mapping_[fj],
                                  as_ss_idx);
          }
          else if (not is_boundary_face)
          {
            psi = fluds.NLUpwindPsi(
              incoming_nonlocal_slot, face_nodal_mapping->face_node_mapping_[fj], as_ss_idx);
          }
          else
          {
            const bool incident_on_symmetric_boundary =
              (face.normal.Dot(normal_vector_boundary_) < -0.999999);
            if (not incident_on_symmetric_boundary)
            {
              psi = angle_set.PsiBoundary(face.neighbor_id,
                                          direction_num,
                                          ctx_.cell_local_id,
                                          static_cast<unsigned int>(f),
                                          static_cast<unsigned int>(fj),
                                          ctx_.gs_gi,
                                          IsSurfaceSourceActive());
            }
          }

          if (psi != nullptr)
            for (size_t gsg = 0; gsg < ctx_.gs_size; ++gsg)
              b[gsg](i) += psi[gsg] * mu_Nij;
        }
      }
    }

    const auto dir_moment_offset =
      static_cast<size_t>(direction_num) * static_cast<size_t>(num_moments_);
    const double* m2d_row = m2d_op.data() + dir_moment_offset;
    const double* d2m_row = d2m_op.data() + dir_moment_offset;

    for (size_t gsg = 0; gsg < ctx_.gs_size; ++gsg)
    {
      const double sigma_tg = sigma_t[ctx_.gs_gi + gsg];

      for (size_t i = 0; i < ctx_.cell_num_nodes; ++i)
      {
        double temp_src = 0.0;
        for (unsigned int m = 0; m < num_moments_; ++m)
        {
          const auto ir = cell_transport_view.MapDOF(i, m, ctx_.gs_gi + gsg);
          temp_src += m2d_row[m] * source_moments_[ir];
        }
        source[i] = temp_src;
      }

      for (size_t i = 0; i < ctx_.cell_num_nodes; ++i)
      {
        double temp = 0.0;
        for (size_t j = 0; j < ctx_.cell_num_nodes; ++j)
        {
          const double Mij = M(i, j);
          Atemp(i, j) = Amat(i, j) + Mij * sigma_tg;
          temp += Mij * source[j];
        }
        b[gsg](i) += temp;
      }

      GaussElimination(Atemp, b[gsg], static_cast<int>(ctx_.cell_num_nodes));
    }

    for (unsigned int m = 0; m < num_moments_; ++m)
    {
      const double wn_d2m = d2m_row[m];
      for (size_t i = 0; i < ctx_.cell_num_nodes; ++i)
      {
        const auto ir = cell_transport_view.MapDOF(i, m, ctx_.gs_gi);
        for (size_t gsg = 0; gsg < ctx_.gs_size; ++gsg)
          destination_phi_[ir + gsg] += wn_d2m * b[gsg](i);
      }
    }

    if (SaveAngularFluxEnabled())
    {
      double* cell_psi_data =
        &destination_psi_[discretization_.MapDOFLocal(cell, 0, groupset.psi_uk_man_, 0, 0)];

      for (size_t i = 0; i < ctx_.cell_num_nodes; ++i)
      {
        const size_t imap =
          i * groupset_angle_group_stride_ + direction_num * groupset_group_stride_;
        for (size_t gsg = 0; gsg < ctx_.gs_size; ++gsg)
          cell_psi_data[imap + gsg] = b[gsg](i);
      }
    }

    for (size_t f = 0; f < ctx_.cell_num_faces; ++f)
    {
      if (face_orientations[f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face = cell.faces[f];
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face and angle_set.GetBoundaries()[face.neighbor_id]->IsReflecting());
      const auto& int_f_shape_i = IntS_shapeI[f];

      std::vector<double>* psi_nonlocal_outgoing = nullptr;
      if (not is_boundary_face and not is_local_face)
        psi_nonlocal_outgoing = &ctx_.outgoing_nonlocal_face_buffer_by_face[f]->data;

      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping.MapFaceNode(f, fi);

        if (is_boundary_face)
        {
          for (size_t gsg = 0; gsg < ctx_.gs_size; ++gsg)
            cell_transport_view.AddOutflow(
              f, ctx_.gs_gi + gsg, wt * ctx_.face_mu_values[f] * b[gsg](i) * int_f_shape_i(i));
        }

        double* psi = nullptr;
        if (is_local_face)
          psi = fluds.OutgoingPsi(cell, i, as_ss_idx);
        else if (not is_boundary_face)
          psi = fluds.NLOutgoingPsi(psi_nonlocal_outgoing, fi, as_ss_idx);
        else if (is_reflecting_boundary_face)
        {
          psi = angle_set.PsiReflected(face.neighbor_id,
                                       direction_num,
                                       ctx_.cell_local_id,
                                       static_cast<unsigned int>(f),
                                       static_cast<unsigned int>(fi));
        }

        if (psi != nullptr)
          for (size_t gsg = 0; gsg < ctx_.gs_size; ++gsg)
            psi[gsg] = b[gsg](i);
      }
    }

    const auto f0 = 1.0 / fac_diamond_difference;
    const auto f1 = f0 - 1.0;
    for (size_t i = 0; i < ctx_.cell_num_nodes; ++i)
    {
      const auto ir = discretization_.MapDOFLocal(cell, i, unknown_manager_, polar_level, ctx_.gs_gi);
      for (size_t gsg = 0; gsg < ctx_.gs_size; ++gsg)
        psi_sweep_[ir + gsg] = f0 * b[gsg](i) - f1 * psi_sweep_[ir + gsg];
    }
  }

  QueueOutgoingNonlocalFaceBuffers();
}

} // namespace opensn

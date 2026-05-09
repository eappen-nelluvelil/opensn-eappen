// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/dense_matrix.h"
#include "framework/data_types/vector.h"
#include "framework/mesh/cell/cell.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include <algorithm>
#include <cassert>
#include <span>

namespace opensn
{

/// Staging buffer for one outgoing nonlocal CBC face payload.
struct CBCOutgoingFaceBuffer
{
  size_t incoming_face_slot = CBC_FLUDSCommonData::INVALID_FACE_SLOT;
  size_t peer_index = CBC_FLUDSCommonData::INVALID_PEER_INDEX;
  size_t data_size = 0;
  std::vector<double> data;

  void Prepare(size_t size)
  {
    data_size = size;
    if (data.size() != size)
      data.resize(size);
  }
};

struct CBCSweepScratch
{
  std::vector<CBCOutgoingFaceBuffer> outgoing_nonlocal_face_buffers;
  std::vector<CBCOutgoingFaceBuffer*> outgoing_nonlocal_face_buffer_by_face;
  size_t num_outgoing_nonlocal_face_buffers = 0;
  std::vector<double> fixed_rhs_buffer;
  std::vector<double> fixed_sigma_block;
  std::vector<size_t> fixed_moment_dof_map;
  std::vector<double> face_mu_values;
};

/// Prepare reusable outgoing nonlocal face payload buffers for the current cell.
template <class SweepChunkT>
inline void
PrepareOutgoingNonlocalFaceBuffers(SweepChunkT& sweep_chunk,
                                   const std::vector<FaceOrientation>& face_orientations)
{
  auto& scratch = sweep_chunk.scratch_;
  auto& buffers = scratch.outgoing_nonlocal_face_buffers;
  auto& buffer_by_face = scratch.outgoing_nonlocal_face_buffer_by_face;
  buffers.reserve(sweep_chunk.cell_num_faces_);
  buffer_by_face.assign(sweep_chunk.cell_num_faces_, nullptr);
  scratch.num_outgoing_nonlocal_face_buffers = 0;

  for (size_t f = 0; f < sweep_chunk.cell_num_faces_; ++f)
  {
    if (face_orientations[f] != FaceOrientation::OUTGOING)
      continue;

    const auto& face = sweep_chunk.cell_->faces[f];
    if ((not face.has_neighbor) or sweep_chunk.cell_transport_view_->IsFaceLocal(f))
      continue;

    const auto buffer_index = scratch.num_outgoing_nonlocal_face_buffers++;
    if (buffer_index == buffers.size())
      buffers.emplace_back();

    auto& buffer = buffers[buffer_index];
    buffer.incoming_face_slot =
      sweep_chunk.fluds_->GetCommonData().GetOutgoingNonlocalFaceSlotByLocalFace(
        sweep_chunk.cell_local_id_, static_cast<unsigned int>(f));
    buffer.peer_index =
      sweep_chunk.fluds_->GetCommonData().GetOutgoingNonlocalFacePeerIndexByLocalFace(
        sweep_chunk.cell_local_id_, static_cast<unsigned int>(f));
    assert(buffer.incoming_face_slot != CBC_FLUDSCommonData::INVALID_FACE_SLOT);
    assert(buffer.peer_index != CBC_FLUDSCommonData::INVALID_PEER_INDEX);
    buffer.Prepare(sweep_chunk.cell_mapping_->GetNumFaceNodes(f) * sweep_chunk.group_angle_stride_);
    buffer_by_face[f] = &buffer;
  }
}

/// Queue prepared outgoing nonlocal face payloads.
template <class SweepChunkT>
inline void
QueueOutgoingNonlocalFaceBuffers(SweepChunkT& sweep_chunk)
{
  const auto& scratch = sweep_chunk.scratch_;
  for (size_t i = 0; i < scratch.num_outgoing_nonlocal_face_buffers; ++i)
  {
    const auto& buffer = scratch.outgoing_nonlocal_face_buffers[i];
    sweep_chunk.async_comm_->QueueDownwindMessage(
      buffer.peer_index,
      buffer.incoming_face_slot,
      std::span<const double>(buffer.data.data(), buffer.data_size));
  }
}

/**
 * Sweep one host CBC cell using the generic dense-kernel path.
 * \tparam time_dependent Whether transient time terms are assembled.
 */
template <bool time_dependent, class SweepChunkT>
inline void
CBC_Sweep_Generic(SweepChunkT& sweep_chunk, AngleSet& angle_set)
{
  const auto& groupset = sweep_chunk.groupset_;
  const auto& cell = *sweep_chunk.cell_;
  const auto& cell_mapping = *sweep_chunk.cell_mapping_;
  const auto& cell_transport_view = *sweep_chunk.cell_transport_view_;
  auto& cell_outflow_view = *sweep_chunk.cell_outflow_view_;
  auto& fluds = *sweep_chunk.fluds_;
  const auto& G = *sweep_chunk.G_;
  const auto& M = *sweep_chunk.M_;
  const auto& M_surf = *sweep_chunk.M_surf_;
  const auto& IntS_shapeI = *sweep_chunk.IntS_shapeI_;
  const auto& m2d_op = groupset.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset.quadrature->GetDiscreteToMomentOperator();

  DenseMatrix<double> Amat(sweep_chunk.max_num_cell_dofs_, sweep_chunk.max_num_cell_dofs_);
  DenseMatrix<double> Atemp(sweep_chunk.max_num_cell_dofs_, sweep_chunk.max_num_cell_dofs_);
  std::vector<Vector<double>> b(sweep_chunk.gs_size_,
                                Vector<double>(sweep_chunk.max_num_cell_dofs_));
  std::vector<double> source(sweep_chunk.max_num_cell_dofs_);
  std::vector<double> face_mu_values(sweep_chunk.cell_num_faces_);

  const auto& face_orientations =
    angle_set.GetSPDS().GetCellFaceOrientations()[sweep_chunk.cell_local_id_];
  const auto& sigma_t = sweep_chunk.xs_.at(cell.block_id)->GetSigmaTotal();

  std::vector<double> tau_gsg;
  if constexpr (time_dependent)
  {
    const auto& inv_velg = sweep_chunk.xs_.at(cell.block_id)->GetInverseVelocity();
    const double theta = sweep_chunk.problem_.GetTheta();
    const double inv_theta = 1.0 / theta;
    const double dt = sweep_chunk.problem_.GetTimeStep();
    const double inv_dt = 1.0 / dt;

    tau_gsg.assign(sweep_chunk.gs_size_, 0.0);
    for (size_t gsg = 0; gsg < sweep_chunk.gs_size_; ++gsg)
      tau_gsg[gsg] = inv_velg[sweep_chunk.gs_gi_ + gsg] * inv_theta * inv_dt;
  }

  const double* psi_old = nullptr;
  if constexpr (time_dependent)
    psi_old =
      &sweep_chunk
         .psi_old_[sweep_chunk.discretization_.MapDOFLocal(cell, 0, groupset.psi_uk_man_, 0, 0)];

  const auto& as_angle_indices = angle_set.GetAngleIndices();
  PrepareOutgoingNonlocalFaceBuffers(sweep_chunk, face_orientations);

  for (size_t as_ss_idx = 0; as_ss_idx < sweep_chunk.num_angles_in_as_; ++as_ss_idx)
  {
    const auto direction_num = as_angle_indices[as_ss_idx];
    const auto& omega = groupset.quadrature->GetOmega(direction_num);
    const auto wt = groupset.quadrature->GetWeight(direction_num);

    for (size_t gsg = 0; gsg < sweep_chunk.gs_size_; ++gsg)
      for (size_t i = 0; i < sweep_chunk.cell_num_nodes_; ++i)
        b[gsg](i) = 0.0;

    for (size_t i = 0; i < sweep_chunk.cell_num_nodes_; ++i)
      for (size_t j = 0; j < sweep_chunk.cell_num_nodes_; ++j)
        Amat(i, j) = omega.Dot(G(i, j));

    for (size_t f = 0; f < sweep_chunk.cell_num_faces_; ++f)
      face_mu_values[f] = omega.Dot(cell.faces[f].normal);

    for (size_t f = 0; f < sweep_chunk.cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& face = cell.faces[f];
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const auto* face_nodal_mapping =
        is_boundary_face ? nullptr
                         : &fluds.GetCommonData().GetFaceNodalMapping(sweep_chunk.cell_local_id_,
                                                                      static_cast<unsigned int>(f));
      const auto incoming_nonlocal_slot =
        (is_boundary_face or is_local_face)
          ? CBC_FLUDSCommonData::INVALID_FACE_SLOT
          : fluds.GetCommonData().GetIncomingNonlocalFaceSlotByLocalFace(
              sweep_chunk.cell_local_id_, static_cast<unsigned int>(f));

      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping.MapFaceNode(f, fi);

        for (size_t fj = 0; fj < num_face_nodes; ++fj)
        {
          const int j = cell_mapping.MapFaceNode(f, fj);
          const double mu_Nij = -face_mu_values[f] * M_surf[f](i, j);
          Amat(i, j) += mu_Nij;

          const double* psi = nullptr;

          if (is_local_face)
            psi = fluds.UpwindPsi(*cell_transport_view.FaceNeighbor(f),
                                  face_nodal_mapping->cell_node_mapping_[fj],
                                  as_ss_idx);
          else if (not is_boundary_face)
            psi = fluds.NLUpwindPsi(
              incoming_nonlocal_slot, face_nodal_mapping->face_node_mapping_[fj], as_ss_idx);
          else
            psi = angle_set.PsiBoundary(face.neighbor_id,
                                        direction_num,
                                        sweep_chunk.cell_local_id_,
                                        f,
                                        fj,
                                        0,
                                        sweep_chunk.IsSurfaceSourceActive());

          if (psi != nullptr)
            for (size_t gsg = 0; gsg < sweep_chunk.gs_size_; ++gsg)
              b[gsg](i) += psi[gsg] * mu_Nij;
        }
      }
    }

    const auto dir_moment_offset =
      static_cast<std::size_t>(direction_num) * static_cast<std::size_t>(sweep_chunk.num_moments_);
    const double* m2d_row = m2d_op.data() + dir_moment_offset;
    const double* d2m_row = d2m_op.data() + dir_moment_offset;

    for (unsigned int gsg = 0; gsg < sweep_chunk.gs_size_; ++gsg)
    {
      double sigma_tg = sigma_t[sweep_chunk.gs_gi_ + gsg];
      if constexpr (time_dependent)
        sigma_tg += tau_gsg[gsg];

      for (size_t i = 0; i < sweep_chunk.cell_num_nodes_; ++i)
      {
        double temp_src = 0.0;
        for (unsigned int m = 0; m < sweep_chunk.num_moments_; ++m)
        {
          const auto ir = cell_transport_view.MapDOF(i, m, sweep_chunk.gs_gi_ + gsg);
          temp_src += m2d_row[m] * sweep_chunk.source_moments_[ir];
        }

        if constexpr (time_dependent)
        {
          const size_t imap = i * sweep_chunk.groupset_angle_group_stride_ +
                              direction_num * sweep_chunk.groupset_group_stride_;
          if (sweep_chunk.include_rhs_time_term_ and psi_old)
            temp_src += tau_gsg[gsg] * psi_old[imap + gsg];
        }

        source[i] = temp_src;
      }

      for (size_t i = 0; i < sweep_chunk.cell_num_nodes_; ++i)
      {
        double temp = 0.0;
        for (size_t j = 0; j < sweep_chunk.cell_num_nodes_; ++j)
        {
          const double Mij = M(i, j);
          Atemp(i, j) = Amat(i, j) + Mij * sigma_tg;
          temp += Mij * source[j];
        }
        b[gsg](i) += temp;
      }

      GaussElimination(Atemp, b[gsg], static_cast<int>(sweep_chunk.cell_num_nodes_));
    }

    for (unsigned int m = 0; m < sweep_chunk.num_moments_; ++m)
    {
      const auto wn_d2m = d2m_row[m];
      for (size_t i = 0; i < sweep_chunk.cell_num_nodes_; ++i)
      {
        const auto ir = cell_transport_view.MapDOF(i, m, sweep_chunk.gs_gi_);
        for (size_t gsg = 0; gsg < sweep_chunk.gs_size_; ++gsg)
          sweep_chunk.destination_phi_[ir + gsg] += wn_d2m * b[gsg](i);
      }
    }

    if (sweep_chunk.SaveAngularFluxEnabled())
    {
      double* psi_new = &sweep_chunk.destination_psi_[sweep_chunk.discretization_.MapDOFLocal(
        cell, 0, groupset.psi_uk_man_, 0, 0)];

      double theta = 1.0;
      double inv_theta = 1.0;
      if constexpr (time_dependent)
      {
        theta = sweep_chunk.problem_.GetTheta();
        inv_theta = 1.0 / theta;
      }

      for (size_t i = 0; i < sweep_chunk.cell_num_nodes_; ++i)
      {
        const size_t imap = i * sweep_chunk.groupset_angle_group_stride_ +
                            direction_num * sweep_chunk.groupset_group_stride_;

        for (size_t gsg = 0; gsg < sweep_chunk.gs_size_; ++gsg)
        {
          const double psi_sol = b[gsg](i);
          if constexpr (time_dependent)
          {
            const double psi_old_val = psi_old ? psi_old[imap + gsg] : 0.0;
            psi_new[imap + gsg] = inv_theta * (psi_sol + (theta - 1.0) * psi_old_val);
          }
          else
            psi_new[imap + gsg] = psi_sol;
        }
      }
    }

    for (size_t f = 0; f < sweep_chunk.cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face = cell.faces[f];
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face and angle_set.GetBoundaries()[face.neighbor_id]->IsReflecting());
      const auto& IntF_shapeI = IntS_shapeI[f];

      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      std::vector<double>* psi_nonlocal_outgoing = nullptr;
      if (not is_boundary_face and not is_local_face)
        psi_nonlocal_outgoing =
          &sweep_chunk.scratch_.outgoing_nonlocal_face_buffer_by_face[f]->data;

      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping.MapFaceNode(f, fi);

        if (is_boundary_face)
        {
          for (size_t gsg = 0; gsg < sweep_chunk.gs_size_; ++gsg)
            cell_outflow_view.Add(
              f, sweep_chunk.gs_gi_ + gsg, wt * face_mu_values[f] * b[gsg](i) * IntF_shapeI(i));
        }

        double* psi = nullptr;
        if (is_local_face)
          psi = fluds.OutgoingPsi(cell, i, as_ss_idx);
        else if (not is_boundary_face)
          psi = fluds.NLOutgoingPsi(psi_nonlocal_outgoing, fi, as_ss_idx);
        else if (is_reflecting_boundary_face)
          psi = angle_set.PsiReflected(
            face.neighbor_id, direction_num, sweep_chunk.cell_local_id_, f, fi);

        if (psi != nullptr)
          for (size_t gsg = 0; gsg < sweep_chunk.gs_size_; ++gsg)
            psi[gsg] = b[gsg](i);
      }
    }
  }

  QueueOutgoingNonlocalFaceBuffers(sweep_chunk);
}

/**
 * Sweep one host CBC cell using a fixed-node-count dense-kernel path.
 * \tparam NumNodes Number of cell nodes.
 * \tparam time_dependent Whether transient time terms are assembled.
 */
template <unsigned int NumNodes, bool time_dependent, class SweepChunkT>
void CBC_Sweep_FixedN(SweepChunkT& sweep_chunk, AngleSet& angle_set);

} // namespace opensn

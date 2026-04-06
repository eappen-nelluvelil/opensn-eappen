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

namespace opensn
{

struct CBCSweepData
{
  const SpatialDiscretization& discretization;
  const std::vector<double>& source_moments;
  const LBSGroupset& groupset;
  unsigned int num_moments;
  unsigned int max_num_cell_dofs;
  bool save_angular_flux;
  size_t groupset_angle_group_stride;
  size_t groupset_group_stride;
  std::vector<double>& destination_phi;
  std::vector<double>& destination_psi;
  bool surface_source_active;
  bool include_rhs_time_term;
  DiscreteOrdinatesProblem& problem;
  const std::vector<double>* psi_old;
  unsigned int group_block_size;

  CBC_FLUDS& fluds;
  const Cell& cell;
  std::uint32_t cell_local_id;
  const CellMapping& cell_mapping;
  CellLBSView& cell_transport_view;
  size_t cell_num_faces;
  size_t cell_num_nodes;

  size_t gs_size;
  unsigned int gs_gi;
  size_t num_angles_in_as;
  unsigned int group_stride;
  size_t group_angle_stride;

  const DenseMatrix<Vector3>& G;
  const DenseMatrix<double>& M;
  const std::vector<DenseMatrix<double>>& M_surf;
  const std::vector<Vector<double>>& IntS_shapeI;
};

struct CBCIncomingFaceData
{
  bool is_local_face = false;
  bool is_boundary_face = false;
  const FaceNodalMapping* face_nodal_mapping = nullptr;
  const CBC_FLUDSCommonData::IncomingNonlocalFaceInfo* incoming_nonlocal_face_info = nullptr;
};

struct CBCOutgoingFaceData
{
  bool is_local_face = false;
  bool is_boundary_face = false;
  bool is_reflecting_boundary_face = false;
  const FaceNodalMapping* face_nodal_mapping = nullptr;
  const CBC_FLUDSCommonData::OutgoingNonlocalFaceInfo* outgoing_nonlocal_face_info = nullptr;
};

struct CBCGenericSweepScratch
{
  DenseMatrix<double> Amat;
  DenseMatrix<double> Atemp;
  std::vector<Vector<double>> b;
  std::vector<double> source;
  std::vector<double> face_mu_values;
  std::vector<double> tau_gsg;
  std::vector<CBCIncomingFaceData> incoming_face_data;
  std::vector<CBCOutgoingFaceData> outgoing_face_data;
  std::vector<size_t> moment_dof_map;

  void
  EnsureCapacity(const size_t max_num_cell_dofs, const size_t gs_size, const size_t cell_num_faces)
  {
    if (Amat.Rows() != max_num_cell_dofs or Amat.Columns() != max_num_cell_dofs)
    {
      Amat = DenseMatrix<double>(max_num_cell_dofs, max_num_cell_dofs);
      Atemp = DenseMatrix<double>(max_num_cell_dofs, max_num_cell_dofs);
    }

    if (b.size() != gs_size)
      b.assign(gs_size, Vector<double>(max_num_cell_dofs));
    else
      for (auto& vec : b)
        if (vec.Rows() != max_num_cell_dofs)
          vec = Vector<double>(max_num_cell_dofs);

    if (source.size() != max_num_cell_dofs)
      source.assign(max_num_cell_dofs, 0.0);

    if (face_mu_values.size() != cell_num_faces)
      face_mu_values.assign(cell_num_faces, 0.0);

    if (incoming_face_data.size() != cell_num_faces)
      incoming_face_data.assign(cell_num_faces, CBCIncomingFaceData{});

    if (outgoing_face_data.size() != cell_num_faces)
      outgoing_face_data.assign(cell_num_faces, CBCOutgoingFaceData{});
  }
};

template <bool time_dependent>
inline void
CBC_Sweep_Generic(CBCSweepData& data, CBCGenericSweepScratch& scratch, AngleSet& angle_set)
{
  const auto& groupset = data.groupset;
  const auto& m2d_op = groupset.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset.quadrature->GetDiscreteToMomentOperator();
  scratch.EnsureCapacity(data.max_num_cell_dofs, data.gs_size, data.cell_num_faces);
  auto& Amat = scratch.Amat;
  auto& Atemp = scratch.Atemp;
  auto& b = scratch.b;
  auto& source = scratch.source;
  auto& face_mu_values = scratch.face_mu_values;

  const auto& face_orientations = angle_set.GetSPDS().GetCellFaceOrientations()[data.cell_local_id];
  const auto& cell_xs = data.cell_transport_view.GetXS();
  const auto& sigma_t = cell_xs.GetSigmaTotal();

  scratch.tau_gsg.clear();
  if constexpr (time_dependent)
  {
    const auto& inv_velg = cell_xs.GetInverseVelocity();
    const double theta = data.problem.GetTheta();
    const double inv_theta = 1.0 / theta;
    const double dt = data.problem.GetTimeStep();
    const double inv_dt = 1.0 / dt;

    auto& tau_gsg = scratch.tau_gsg;
    tau_gsg.assign(data.gs_size, 0.0);
    for (size_t gsg = 0; gsg < data.gs_size; ++gsg)
      tau_gsg[gsg] = inv_velg[data.gs_gi + gsg] * inv_theta * inv_dt;
  }

  const double* psi_old =
    (time_dependent and data.psi_old)
      ? &(*data.psi_old)[data.discretization.MapDOFLocal(data.cell, 0, groupset.psi_uk_man_, 0, 0)]
      : nullptr;

  const auto& as_angle_indices = angle_set.GetAngleIndices();
  const auto& cbc_common = static_cast<const CBC_FLUDSCommonData&>(data.fluds.GetCommonData());
  auto* const async_comm = static_cast<CBC_AsynchronousCommunicator*>(angle_set.GetCommunicator());
  auto& incoming_face_data = scratch.incoming_face_data;
  auto& outgoing_face_data = scratch.outgoing_face_data;
  for (size_t f = 0; f < data.cell_num_faces; ++f)
  {
    incoming_face_data[f] = CBCIncomingFaceData{};
    const auto& face = data.cell.faces[f];
    const bool is_local_face = data.cell_transport_view.IsFaceLocal(f);
    const bool is_boundary_face = not face.has_neighbor;
    const auto* face_nodal_mapping =
      &data.fluds.GetCommonData().GetFaceNodalMapping(data.cell_local_id, f);

    if (face_orientations[f] == FaceOrientation::INCOMING)
    {
      auto& face_data = incoming_face_data[f];
      face_data.is_local_face = is_local_face;
      face_data.is_boundary_face = is_boundary_face;
      face_data.face_nodal_mapping = face_nodal_mapping;
      if (not is_local_face and not is_boundary_face)
        face_data.incoming_nonlocal_face_info =
          &cbc_common.GetIncomingNonlocalFaceInfo(data.cell_local_id, static_cast<unsigned int>(f));
    }

    if (face_orientations[f] == FaceOrientation::OUTGOING)
    {
      auto& face_data = outgoing_face_data[f];
      face_data.is_local_face = is_local_face;
      face_data.is_boundary_face = is_boundary_face;
      face_data.is_reflecting_boundary_face =
        is_boundary_face and angle_set.GetBoundaries()[face.neighbor_id]->IsReflecting();
      face_data.face_nodal_mapping = face_nodal_mapping;
      if (not is_local_face and not is_boundary_face)
        face_data.outgoing_nonlocal_face_info =
          &cbc_common.GetOutgoingNonlocalFaceInfo(data.cell_local_id, static_cast<unsigned int>(f));
    }
  }

  auto& moment_dof_map = scratch.moment_dof_map;
  moment_dof_map.resize(static_cast<size_t>(data.num_moments) * data.cell_num_nodes);
  for (unsigned int m = 0; m < data.num_moments; ++m)
    for (size_t i = 0; i < data.cell_num_nodes; ++i)
      moment_dof_map[static_cast<size_t>(m) * data.cell_num_nodes + i] =
        data.cell_transport_view.MapDOF(i, m, data.gs_gi);

  double* psi_new_base = nullptr;
  double theta = 1.0;
  double inv_theta = 1.0;
  if (data.save_angular_flux)
  {
    psi_new_base = &data.destination_psi[data.discretization.MapDOFLocal(
      data.cell, 0, groupset.psi_uk_man_, 0, 0)];
    if constexpr (time_dependent)
    {
      theta = data.problem.GetTheta();
      inv_theta = 1.0 / theta;
    }
  }

  for (size_t as_ss_idx = 0; as_ss_idx < data.num_angles_in_as; ++as_ss_idx)
  {
    const auto direction_num = as_angle_indices[as_ss_idx];
    const auto omega = groupset.quadrature->omegas[direction_num];
    const auto wt = groupset.quadrature->weights[direction_num];

    for (size_t gsg = 0; gsg < data.gs_size; ++gsg)
      for (size_t i = 0; i < data.cell_num_nodes; ++i)
        b[gsg](i) = 0.0;

    for (size_t i = 0; i < data.cell_num_nodes; ++i)
      for (size_t j = 0; j < data.cell_num_nodes; ++j)
        Amat(i, j) = omega.Dot(data.G(i, j));

    for (size_t f = 0; f < data.cell_num_faces; ++f)
      face_mu_values[f] = omega.Dot(data.cell.faces[f].normal);

    for (size_t f = 0; f < data.cell_num_faces; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& face = data.cell.faces[f];
      const auto& face_data = incoming_face_data[f];
      const bool is_local_face = face_data.is_local_face;
      const bool is_boundary_face = face_data.is_boundary_face;
      const auto* face_nodal_mapping = face_data.face_nodal_mapping;

      const size_t num_face_nodes = data.cell_mapping.GetNumFaceNodes(f);
      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = data.cell_mapping.MapFaceNode(f, fi);

        for (size_t fj = 0; fj < num_face_nodes; ++fj)
        {
          const int j = data.cell_mapping.MapFaceNode(f, fj);
          const double mu_Nij = -face_mu_values[f] * data.M_surf[f](i, j);
          Amat(i, j) += mu_Nij;

          const double* psi = nullptr;

          if (is_local_face)
            psi = data.fluds.UpwindPsi(data.cell_transport_view.FaceNeighbor(f)->local_id,
                                       face_nodal_mapping->cell_node_mapping_[fj],
                                       as_ss_idx);
          else if (not is_boundary_face)
            psi = data.fluds.NLUpwindPsi(data.cell_local_id,
                                         static_cast<unsigned int>(f),
                                         face_nodal_mapping->face_node_mapping_[fj],
                                         as_ss_idx);
          else
            psi = angle_set.PsiBoundary(face.neighbor_id,
                                        direction_num,
                                        data.cell_local_id,
                                        f,
                                        fj,
                                        data.gs_gi,
                                        data.surface_source_active);

          if (psi != nullptr)
            for (size_t gsg = 0; gsg < data.gs_size; ++gsg)
              b[gsg](i) += psi[gsg] * mu_Nij;
        }
      }
    }

    const auto dir_moment_offset =
      static_cast<std::size_t>(direction_num) * static_cast<std::size_t>(data.num_moments);
    const double* m2d_row = m2d_op.data() + dir_moment_offset;
    const double* d2m_row = d2m_op.data() + dir_moment_offset;

    for (unsigned int gsg = 0; gsg < data.gs_size; ++gsg)
    {
      double sigma_tg = sigma_t[data.gs_gi + gsg];
      if constexpr (time_dependent)
      {
        const auto& tau_gsg = scratch.tau_gsg;
        sigma_tg += tau_gsg[gsg];
      }

      for (size_t i = 0; i < data.cell_num_nodes; ++i)
      {
        double temp_src = 0.0;
        for (unsigned int m = 0; m < data.num_moments; ++m)
        {
          const auto ir = moment_dof_map[static_cast<size_t>(m) * data.cell_num_nodes + i] + gsg;
          temp_src += m2d_row[m] * data.source_moments[ir];
        }

        if constexpr (time_dependent)
        {
          const auto& tau_gsg = scratch.tau_gsg;
          const size_t imap =
            i * data.groupset_angle_group_stride + direction_num * data.groupset_group_stride;
          if (data.include_rhs_time_term and psi_old)
            temp_src += tau_gsg[gsg] * psi_old[imap + gsg];
        }

        source[i] = temp_src;
      }

      for (size_t i = 0; i < data.cell_num_nodes; ++i)
      {
        double temp = 0.0;
        for (size_t j = 0; j < data.cell_num_nodes; ++j)
        {
          const double Mij = data.M(i, j);
          Atemp(i, j) = Amat(i, j) + Mij * sigma_tg;
          temp += Mij * source[j];
        }
        b[gsg](i) += temp;
      }

      GaussElimination(Atemp, b[gsg], static_cast<int>(data.cell_num_nodes));
    }

    for (unsigned int m = 0; m < data.num_moments; ++m)
    {
      const auto wn_d2m = d2m_row[m];
      for (size_t i = 0; i < data.cell_num_nodes; ++i)
      {
        const auto ir = moment_dof_map[static_cast<size_t>(m) * data.cell_num_nodes + i];
        for (size_t gsg = 0; gsg < data.gs_size; ++gsg)
          data.destination_phi[ir + gsg] += wn_d2m * b[gsg](i);
      }
    }

    if (data.save_angular_flux)
    {
      for (size_t i = 0; i < data.cell_num_nodes; ++i)
      {
        const size_t imap =
          i * data.groupset_angle_group_stride + direction_num * data.groupset_group_stride;

        for (size_t gsg = 0; gsg < data.gs_size; ++gsg)
        {
          const double psi_sol = b[gsg](i);
          if constexpr (time_dependent)
          {
            const double psi_old_val = psi_old ? psi_old[imap + gsg] : 0.0;
            psi_new_base[imap + gsg] = inv_theta * (psi_sol + (theta - 1.0) * psi_old_val);
          }
          else
            psi_new_base[imap + gsg] = psi_sol;
        }
      }
    }

    for (size_t f = 0; f < data.cell_num_faces; ++f)
    {
      if (face_orientations[f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face = data.cell.faces[f];
      const auto& face_data = outgoing_face_data[f];
      const bool is_local_face = face_data.is_local_face;
      const bool is_boundary_face = face_data.is_boundary_face;
      const bool is_reflecting_boundary_face = face_data.is_reflecting_boundary_face;
      const auto& IntF_shapeI = data.IntS_shapeI[f];

      const size_t num_face_nodes = data.cell_mapping.GetNumFaceNodes(f);
      double* psi_nonlocal_outgoing = nullptr;

      if (not is_boundary_face and not is_local_face)
      {
        const auto& outgoing_nonlocal_face_info = *face_data.outgoing_nonlocal_face_info;
        const size_t data_size_for_msg =
          static_cast<size_t>(outgoing_nonlocal_face_info.num_face_nodes) * data.group_angle_stride;
        psi_nonlocal_outgoing =
          async_comm
            ->InitGetDownwindMessageData(outgoing_nonlocal_face_info.locality,
                                         outgoing_nonlocal_face_info.cell_global_id,
                                         outgoing_nonlocal_face_info.associated_face,
                                         angle_set.GetID(),
                                         data_size_for_msg)
            .data();
      }

      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = data.cell_mapping.MapFaceNode(f, fi);

        if (is_boundary_face)
        {
          for (size_t gsg = 0; gsg < data.gs_size; ++gsg)
            data.cell_transport_view.AddOutflow(
              f, data.gs_gi + gsg, wt * face_mu_values[f] * b[gsg](i) * IntF_shapeI(i));
        }

        double* psi = nullptr;
        if (is_local_face)
          psi = data.fluds.OutgoingPsi(data.cell_local_id, i, as_ss_idx);
        else if (not is_boundary_face)
          psi = data.fluds.NLOutgoingPsi(psi_nonlocal_outgoing, fi, as_ss_idx);
        else if (is_reflecting_boundary_face)
          psi = angle_set.PsiReflected(face.neighbor_id, direction_num, data.cell_local_id, f, fi);

        if (psi != nullptr)
          for (size_t gsg = 0; gsg < data.gs_size; ++gsg)
            psi[gsg] = b[gsg](i);
      }
    }
  }
}

template <unsigned int NumNodes, bool time_dependent>
void CBC_Sweep_FixedN(CBCSweepData& data, AngleSet& angle_set);

} // namespace opensn

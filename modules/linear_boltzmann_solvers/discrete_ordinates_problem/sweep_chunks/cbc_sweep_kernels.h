// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_kernel_utils.h"
#include "framework/data_types/dense_matrix.h"
#include "framework/data_types/vector.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/math/spatial_discretization/finite_element/unit_cell_matrices.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_structs.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_view.h"
#include <algorithm>

namespace opensn
{

/// Data struct holding all state needed by CBC cell-level sweep kernels.
struct CBCSweepData
{
  const SpatialDiscretization& discretization;
  const std::vector<UnitCellMatrices>& unit_cell_matrices;
  std::vector<CellLBSView>& cell_transport_views;
  const std::vector<double>& source_moments;
  const LBSGroupset& groupset;
  const BlockID2XSMap& xs;
  const unsigned int num_moments;
  const unsigned int max_num_cell_dofs;
  const bool save_angular_flux;
  const size_t groupset_angle_group_stride;
  const size_t groupset_group_stride;
  std::vector<double>& destination_phi;
  std::vector<double>& destination_psi;
  bool surface_source_active;
  bool include_rhs_time_term;
  DiscreteOrdinatesProblem* problem; // non-null for time_dependent
  const std::vector<double>* psi_old; // non-null for time_dependent
  unsigned int group_block_size;

  // CBC-specific
  CBC_FLUDS* fluds;
  const Cell* cell;
};

/// Generic (scalar) CBC cell sweep kernel, parameterized by time dependence.
template <bool time_dependent>
inline void
CBC_Sweep_CellKernel_Generic(CBCSweepData& data, AngleSet& angle_set)
{
  const auto& groupset = data.groupset;
  const auto gs_size = groupset.GetNumGroups();
  const auto gs_gi = groupset.first_group;

  const auto& cell = *data.cell;
  const auto cell_local_id = cell.local_id;
  auto& cell_transport_view = data.cell_transport_views[cell_local_id];
  const auto& cell_mapping = data.discretization.GetCellMapping(cell);
  const size_t cell_num_faces = cell.faces.size();
  const size_t cell_num_nodes = cell_mapping.GetNumNodes();

  const auto& face_orientations = angle_set.GetSPDS().GetCellFaceOrientations()[cell_local_id];
  std::vector<double> face_mu_values(cell_num_faces);

  const auto& sigma_t = data.xs.at(cell.block_id)->GetSigmaTotal();

  const auto& unit_mats = data.unit_cell_matrices[cell_local_id];
  const auto& G = unit_mats.intV_shapeI_gradshapeJ;
  const auto& M = unit_mats.intV_shapeI_shapeJ;
  const auto& M_surf = unit_mats.intS_shapeI_shapeJ;

  const auto& m2d_op = groupset.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset.quadrature->GetDiscreteToMomentOperator();

  DenseMatrix<double> Amat(data.max_num_cell_dofs, data.max_num_cell_dofs);
  DenseMatrix<double> Atemp(data.max_num_cell_dofs, data.max_num_cell_dofs);
  std::vector<Vector<double>> b(gs_size, Vector<double>(data.max_num_cell_dofs));
  std::vector<double> source(data.max_num_cell_dofs);

  std::vector<double> tau_gsg;
  if constexpr (time_dependent)
  {
    const auto& inv_velg = data.xs.at(cell.block_id)->GetInverseVelocity();
    const double theta = data.problem->GetTheta();
    const double inv_theta = 1.0 / theta;
    const double dt = data.problem->GetTimeStep();
    const double inv_dt = 1.0 / dt;
    tau_gsg.assign(gs_size, 0.0);
    for (size_t gsg = 0; gsg < gs_size; ++gsg)
      tau_gsg[gsg] = inv_velg[gs_gi + gsg] * inv_theta * inv_dt;
  }

  const double* psi_old_cell =
    (time_dependent and data.psi_old)
      ? &(*data.psi_old)[data.discretization.MapDOFLocal(cell, 0, groupset.psi_uk_man_, 0, 0)]
      : nullptr;

  const std::vector<std::uint32_t>& as_angle_indices = angle_set.GetAngleIndices();
  const size_t num_angles_in_as = as_angle_indices.size();
  const size_t group_stride = angle_set.GetNumGroups();
  const size_t group_angle_stride = group_stride * num_angles_in_as;

  for (size_t as_ss_idx = 0; as_ss_idx < num_angles_in_as; ++as_ss_idx)
  {
    const auto direction_num = as_angle_indices[as_ss_idx];
    const auto omega = groupset.quadrature->omegas[direction_num];
    const auto wt = groupset.quadrature->weights[direction_num];

    for (size_t gsg = 0; gsg < gs_size; ++gsg)
      for (size_t i = 0; i < cell_num_nodes; ++i)
        b[gsg](i) = 0.0;

    for (size_t i = 0; i < cell_num_nodes; ++i)
      for (size_t j = 0; j < cell_num_nodes; ++j)
        Amat(i, j) = omega.Dot(G(i, j));

    for (size_t f = 0; f < cell_num_faces; ++f)
      face_mu_values[f] = omega.Dot(cell.faces[f].normal);

    // Incoming surface integrals
    for (size_t f = 0; f < cell_num_faces; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& face = cell.faces[f];
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const auto* face_nodal_mapping =
        &data.fluds->GetCommonData().GetFaceNodalMapping(cell_local_id, f);

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
            psi = data.fluds->UpwindPsi(*cell_transport_view.FaceNeighbor(f),
                                        face_nodal_mapping->cell_node_mapping_[fj],
                                        as_ss_idx);
          else if (not is_boundary_face)
            psi = data.fluds->NLUpwindPsi(
              cell.global_id, f, face_nodal_mapping->face_node_mapping_[fj], as_ss_idx);
          else
            psi = angle_set.PsiBoundary(
              face.neighbor_id, direction_num, cell_local_id, f, fj, gs_gi, data.surface_source_active);

          if (psi != nullptr)
            for (size_t gsg = 0; gsg < gs_size; ++gsg)
              b[gsg](i) += psi[gsg] * mu_Nij;
        }
      }
    }

    // Mass assembly and solve
    for (unsigned int gsg = 0; gsg < gs_size; ++gsg)
    {
      double sigma_tg = sigma_t[gs_gi + gsg];
      if constexpr (time_dependent)
        sigma_tg += tau_gsg[gsg];

      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        double temp_src = 0.0;
        for (unsigned int m = 0; m < data.num_moments; ++m)
        {
          const auto ir = cell_transport_view.MapDOF(i, m, gs_gi + gsg);
          temp_src += m2d_op[direction_num][m] * data.source_moments[ir];
        }

        if constexpr (time_dependent)
        {
          if (data.include_rhs_time_term and psi_old_cell)
          {
            const size_t imap =
              i * data.groupset_angle_group_stride + direction_num * data.groupset_group_stride;
            temp_src += tau_gsg[gsg] * psi_old_cell[imap + gsg];
          }
        }

        source[i] = temp_src;
      }

      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        double temp = 0.0;
        for (size_t j = 0; j < cell_num_nodes; ++j)
        {
          const double Mij = M(i, j);
          Atemp(i, j) = Amat(i, j) + Mij * sigma_tg;
          temp += Mij * source[j];
        }
        b[gsg](i) += temp;
      }

      GaussElimination(Atemp, b[gsg], static_cast<int>(cell_num_nodes));
    }

    // Update phi
    for (unsigned int m = 0; m < data.num_moments; ++m)
    {
      const double wn_d2m = d2m_op[direction_num][m];
      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        const auto ir = cell_transport_view.MapDOF(i, m, gs_gi);
        for (size_t gsg = 0; gsg < gs_size; ++gsg)
          data.destination_phi[ir + gsg] += wn_d2m * b[gsg](i);
      }
    }

    // Save angular flux
    if (data.save_angular_flux)
    {
      double* cell_psi =
        &data.destination_psi[data.discretization.MapDOFLocal(cell, 0, groupset.psi_uk_man_, 0, 0)];

      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        const size_t addr_offset =
          i * data.groupset_angle_group_stride + direction_num * data.groupset_group_stride;

        for (size_t gsg = 0; gsg < gs_size; ++gsg)
        {
          const double psi_sol = b[gsg](i);
          if constexpr (time_dependent)
          {
            const double theta = data.problem->GetTheta();
            const double inv_theta = 1.0 / theta;
            const double psi_old_val = psi_old_cell ? psi_old_cell[addr_offset + gsg] : 0.0;
            cell_psi[addr_offset + gsg] = inv_theta * (psi_sol + (theta - 1.0) * psi_old_val);
          }
          else
          {
            cell_psi[addr_offset + gsg] = psi_sol;
          }
        }
      }
    }

    // Outgoing surface operations
    for (size_t f = 0; f < cell_num_faces; ++f)
    {
      if (face_orientations[f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face = cell.faces[f];
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face and angle_set.GetBoundaries()[face.neighbor_id]->IsReflecting());
      const auto& IntF_shapeI = unit_mats.intS_shapeI[f];

      const int locality = cell_transport_view.FaceLocality(f);
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const auto& face_nodal_mapping =
        data.fluds->GetCommonData().GetFaceNodalMapping(cell_local_id, f);
      std::vector<double>* psi_nonlocal_outgoing = nullptr;

      if (not is_boundary_face and not is_local_face)
      {
        auto& async_comm = *angle_set.GetCommunicator();
        const size_t data_size_for_msg = num_face_nodes * group_angle_stride;
        psi_nonlocal_outgoing =
          &async_comm.InitGetDownwindMessageData(locality,
                                                 face.neighbor_id,
                                                 face_nodal_mapping.associated_face_,
                                                 angle_set.GetID(),
                                                 data_size_for_msg);
      }

      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping.MapFaceNode(f, fi);

        if (is_boundary_face)
        {
          for (size_t gsg = 0; gsg < gs_size; ++gsg)
            cell_transport_view.AddOutflow(
              f, gs_gi + gsg, wt * face_mu_values[f] * b[gsg](i) * IntF_shapeI(i));
        }

        double* psi = nullptr;
        if (is_local_face)
          psi = data.fluds->OutgoingPsi(cell, i, as_ss_idx);
        else if (not is_boundary_face)
          psi = data.fluds->NLOutgoingPsi(psi_nonlocal_outgoing, fi, as_ss_idx);
        else if (is_reflecting_boundary_face)
          psi = angle_set.PsiReflected(face.neighbor_id, direction_num, cell_local_id, f, fi);

        if (psi != nullptr)
          for (size_t gsg = 0; gsg < gs_size; ++gsg)
            psi[gsg] = b[gsg](i);
      }
    }
  }
}

// Fixed-N CBC cell kernel - declared here, implemented in cbc_avx_sweep_chunk.cc
template <unsigned int NumNodes, bool time_dependent>
void CBC_Sweep_CellKernel_FixedN(CBCSweepData& data, AngleSet& angle_set);

} // namespace opensn

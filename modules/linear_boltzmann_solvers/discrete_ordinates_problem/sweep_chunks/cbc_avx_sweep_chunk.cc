// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mesh/cell/cell.h"
#include "caliper/cali.h"
#include <algorithm>
#include <array>
#include <vector>

namespace opensn
{

template <unsigned int NumNodes>
void
CBCSweepChunk::Sweep_FixedN(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunk::Sweep_FixedN");

  static_assert(NumNodes >= 2 and NumNodes <= 8);

  const auto& m2d_op = groupset_.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset_.quadrature->GetDiscreteToMomentOperator();

  const auto& face_orientations = angle_set.GetSPDS().GetCellFaceOrientations()[cell_local_id_];

  const double rho = densities_[cell_local_id_];
  const auto& sigma_t = xs_.at(cell_->block_id)->GetSigmaTotal();

  const auto& G = *G_;
  const auto& M = *M_;
  const auto& M_surf = *M_surf_;

  constexpr size_t matrix_size = static_cast<size_t>(NumNodes) * static_cast<size_t>(NumNodes);
  auto idx = [](int i, int j) -> size_t
  { return static_cast<size_t>(i) * static_cast<size_t>(NumNodes) + static_cast<size_t>(j); };

  std::array<double, matrix_size> mass_matrix{};
  PRAGMA_UNROLL
  for (int i = 0; i < static_cast<int>(NumNodes); ++i)
  {
    PRAGMA_UNROLL
    for (int j = 0; j < static_cast<int>(NumNodes); ++j)
      mass_matrix[idx(i, j)] = M(i, j);
  }

  std::vector<std::array<size_t, NumNodes>> moment_dof_map(num_moments_);
  for (unsigned int m = 0; m < num_moments_; ++m)
  {
    PRAGMA_UNROLL
    for (int i = 0; i < static_cast<int>(NumNodes); ++i)
      moment_dof_map[m][i] = cell_transport_view_->MapDOF(i, m, gs_gi_);
  }

  std::array<double, matrix_size> Amat{};

  std::vector<double> b(static_cast<std::size_t>(gs_size_) * NumNodes, 0.0);
  std::vector<double> sigma_block;
  sigma_block.reserve(group_block_size_);

  std::vector<double> face_mu_values(cell_num_faces_);

  const std::vector<std::uint32_t>& as_angle_indices = angle_set.GetAngleIndices();

  for (size_t as_ss_idx = 0; as_ss_idx < num_angles_in_as_; ++as_ss_idx)
  {
    const auto direction_num = as_angle_indices[as_ss_idx];
    const auto omega = groupset_.quadrature->omegas[direction_num];
    const auto wt = groupset_.quadrature->weights[direction_num];

    std::fill(b.begin(), b.end(), 0.0);

    PRAGMA_UNROLL
    for (int i = 0; i < static_cast<int>(NumNodes); ++i)
    {
      PRAGMA_UNROLL
      for (int j = 0; j < static_cast<int>(NumNodes); ++j)
        Amat[idx(i, j)] = omega.Dot(G(i, j));
    }

    for (size_t f = 0; f < cell_num_faces_; ++f)
      face_mu_values[f] = omega.Dot(cell_->faces[f].normal);

    // Surface integrals (incoming faces)
    for (size_t f = 0; f < cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& face = cell_->faces[f];
      const bool is_local_face = cell_transport_view_->IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const auto* face_nodal_mapping =
        &fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f);

      const auto& Ms_f = M_surf[f];
      const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
      const double mu_f = -face_mu_values[f];

      // fj outer loop: fetch psi once per fj
      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
        const int j = cell_mapping_->MapFaceNode(f, fj);

        const double* psi = nullptr;
        if (is_local_face)
          psi = fluds_->UpwindPsi(*cell_transport_view_->FaceNeighbor(f),
                                  face_nodal_mapping->cell_node_mapping_[fj],
                                  as_ss_idx);
        else if (not is_boundary_face)
          psi = fluds_->NLUpwindPsi(
            cell_->global_id, f, face_nodal_mapping->face_node_mapping_[fj], as_ss_idx);
        else
          psi = angle_set.PsiBoundary(
            face.neighbor_id, direction_num, cell_local_id_, f, fj, gs_gi_, surface_source_active_);

        for (size_t fi = 0; fi < num_face_nodes; ++fi)
        {
          const int i = cell_mapping_->MapFaceNode(f, fi);
          const double mu_Nij = mu_f * Ms_f(i, j);
          Amat[idx(i, j)] += mu_Nij;

          if (not psi)
            continue;

          for (size_t gsg = 0; gsg < gs_size_; ++gsg)
            b[gsg * NumNodes + i] += psi[gsg] * mu_Nij;
        }
      }
    }

    const double* __restrict m2d_row = m2d_op[direction_num].data();
    const double* __restrict d2m_row = d2m_op[direction_num].data();

    // Group blocking with source assembly and solve
    for (unsigned int g0 = 0; g0 < gs_size_; g0 += group_block_size_)
    {
      const auto g1 = std::min(g0 + group_block_size_, static_cast<unsigned int>(gs_size_));
      const auto block_len = g1 - g0;
      sigma_block.resize(block_len);

      for (unsigned int gsg = g0; gsg < g1; ++gsg)
      {
        const size_t rel = gsg - g0;
        sigma_block[rel] = rho * sigma_t[gs_gi_ + gsg];

        double* __restrict bg = &b[static_cast<std::size_t>(gsg) * NumNodes];
        for (unsigned int m = 0; m < num_moments_; ++m)
        {
          const double w = m2d_row[m];
          std::array<double, NumNodes> nodal_source{};
          for (int i = 0; i < static_cast<int>(NumNodes); ++i)
            nodal_source[i] = w * source_moments_[moment_dof_map[m][i] + gsg];

          for (int i = 0; i < static_cast<int>(NumNodes); ++i)
          {
            double value = 0.0;
            const double* row = &mass_matrix[idx(i, 0)];
            PRAGMA_UNROLL
            for (int j = 0; j < static_cast<int>(NumNodes); ++j)
              value += row[j] * nodal_source[j];
            bg[i] += value;
          }
        }
      }

      // SIMD batch solve
      size_t k = 0;

#if __AVX512F__
      for (; k + simd_width <= block_len; k += simd_width)
        detail::SimdBatchSolve<detail::AVX512Ops, NumNodes>(
          Amat.data(), mass_matrix.data(), &sigma_block[k], &b[(g0 + k) * NumNodes]);
#elif __AVX2__
      for (; k + simd_width <= block_len; k += simd_width)
        detail::SimdBatchSolve<detail::AVX2Ops, NumNodes>(
          Amat.data(), mass_matrix.data(), &sigma_block[k], &b[(g0 + k) * NumNodes]);
#endif

      // Scalar fallback for remaining groups
      for (; k < block_len; ++k)
      {
        const size_t gsg = g0 + k;
        const double sigma_tg = sigma_block[k];

        std::array<double, matrix_size> A{};
        PRAGMA_UNROLL
        for (int i = 0; i < static_cast<int>(NumNodes); ++i)
        {
          PRAGMA_UNROLL
          for (int j = 0; j < static_cast<int>(NumNodes); ++j)
            A[idx(i, j)] = Amat[idx(i, j)] + sigma_tg * mass_matrix[idx(i, j)];
        }

        double* __restrict bg = &b[gsg * NumNodes];

        // Forward elimination
        for (int pivot = 0; pivot < static_cast<int>(NumNodes); ++pivot)
        {
          const double inv = 1.0 / A[idx(pivot, pivot)];
          for (int row = pivot + 1; row < static_cast<int>(NumNodes); ++row)
          {
            const double factor = A[idx(row, pivot)] * inv;
            bg[row] -= factor * bg[pivot];
            PRAGMA_UNROLL
            for (int col = pivot + 1; col < static_cast<int>(NumNodes); ++col)
              A[idx(row, col)] -= factor * A[idx(pivot, col)];
          }
        }

        // Back substitution
        for (int pivot = static_cast<int>(NumNodes) - 1; pivot >= 0; --pivot)
        {
          PRAGMA_UNROLL
          for (int col = pivot + 1; col < static_cast<int>(NumNodes); ++col)
            bg[pivot] -= A[idx(pivot, col)] * bg[col];
          bg[pivot] /= A[idx(pivot, pivot)];
        }
      }

      // Update phi
      for (size_t gsg = g0; gsg < g1; ++gsg)
      {
        const double* __restrict bg = &b[gsg * NumNodes];
        for (unsigned int m = 0; m < num_moments_; ++m)
        {
          const double w = d2m_row[m];
          PRAGMA_UNROLL
          for (int i = 0; i < static_cast<int>(NumNodes); ++i)
          {
            const size_t dof = cell_transport_view_->MapDOF(i, m, gs_gi_);
            destination_phi_[dof + gsg] += w * bg[i];
          }
        }
      }
    }

    // Save angular fluxes
    if (save_angular_flux_)
    {
      double* cell_psi =
        &destination_psi_[discretization_.MapDOFLocal(*cell_, 0, groupset_.psi_uk_man_, 0, 0)];
      PRAGMA_UNROLL
      for (int i = 0; i < static_cast<int>(NumNodes); ++i)
      {
        const size_t imap =
          i * groupset_angle_group_stride_ + direction_num * groupset_group_stride_;
        for (size_t gsg = 0; gsg < gs_size_; ++gsg)
          cell_psi[imap + gsg] = b[gsg * NumNodes + i];
      }
    }

    // Outgoing faces
    for (size_t f = 0; f < cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face = cell_->faces[f];
      const bool is_local_face = cell_transport_view_->IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face and angle_set.GetBoundaries()[face.neighbor_id]->IsReflecting());
      const auto& IntF_shapeI = (*IntS_shapeI_)[f];

      const int locality = cell_transport_view_->FaceLocality(f);
      const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
      const auto& face_nodal_mapping =
        fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f);
      std::vector<double>* psi_nonlocal_outgoing = nullptr;

      if (not is_boundary_face and not is_local_face)
      {
        auto& async_comm = *angle_set.GetCommunicator();
        const size_t data_size_for_msg = num_face_nodes * group_angle_stride_;
        psi_nonlocal_outgoing =
          &async_comm.InitGetDownwindMessageData(locality,
                                                 face.neighbor_id,
                                                 face_nodal_mapping.associated_face_,
                                                 angle_set.GetID(),
                                                 data_size_for_msg);
      }

      const double mu_wt_f = wt * face_mu_values[f];

      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping_->MapFaceNode(f, fi);

        if (is_boundary_face)
        {
          const double flux_i = mu_wt_f * IntF_shapeI(i);
          for (size_t gsg = 0; gsg < gs_size_; ++gsg)
            cell_transport_view_->AddOutflow(f, gs_gi_ + gsg, flux_i * b[gsg * NumNodes + i]);
        }

        double* psi = nullptr;
        if (is_local_face)
          psi = fluds_->OutgoingPsi(*cell_, i, as_ss_idx);
        else if (not is_boundary_face)
          psi = fluds_->NLOutgoingPsi(psi_nonlocal_outgoing, fi, as_ss_idx);
        else if (is_reflecting_boundary_face)
          psi = angle_set.PsiReflected(face.neighbor_id, direction_num, cell_local_id_, f, fi);

        if (psi != nullptr)
        {
          for (size_t gsg = 0; gsg < gs_size_; ++gsg)
            psi[gsg] = b[gsg * NumNodes + i];
        }
      }
    }
  }
}

template void CBCSweepChunk::Sweep_FixedN<2>(AngleSet&);
template void CBCSweepChunk::Sweep_FixedN<3>(AngleSet&);
template void CBCSweepChunk::Sweep_FixedN<4>(AngleSet&);
template void CBCSweepChunk::Sweep_FixedN<5>(AngleSet&);
template void CBCSweepChunk::Sweep_FixedN<6>(AngleSet&);
template void CBCSweepChunk::Sweep_FixedN<7>(AngleSet&);
template void CBCSweepChunk::Sweep_FixedN<8>(AngleSet&);

} // namespace opensn

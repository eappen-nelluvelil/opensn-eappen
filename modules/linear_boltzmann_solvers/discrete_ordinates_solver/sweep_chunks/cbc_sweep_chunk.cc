// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/groupset/lbs_groupset.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mesh/cell/cell.h"
#include "framework/logging/log.h"
#include "caliper/cali.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/communicators/cbc_async_comm.h" // For dynamic_cast

namespace opensn
{

CbcSweepChunk::CbcSweepChunk(std::vector<double>& destination_phi,
                             std::vector<double>& destination_psi,
                             const std::shared_ptr<MeshContinuum> grid,
                             const SpatialDiscretization& discretization,
                             const std::vector<UnitCellMatrices>& unit_cell_matrices,
                             std::vector<CellLBSView>& cell_transport_views,
                             const std::vector<double>& densities,
                             const std::vector<double>& source_moments,
                             const LBSGroupset& groupset,
                             const std::map<int, std::shared_ptr<MultiGroupXS>>& xs,
                             int num_moments,
                             int max_num_cell_dofs)
  : SweepChunk(destination_phi,
               destination_psi,
               grid,
               discretization,
               unit_cell_matrices,
               cell_transport_views,
               densities,
               source_moments,
               groupset,
               xs,
               num_moments,
               max_num_cell_dofs),
    fluds_(nullptr),
    gs_size_(0),
    gs_gi_(0),
    num_angles_in_set_(0),
    compact_angle_stride_(0),
    compact_node_stride_(0),
    surface_source_active_(false),
    cell_(nullptr),
    cell_local_id_(0),
    cell_mapping_(nullptr),
    cell_transport_view_(nullptr),
    cell_num_faces_(0),
    cell_num_nodes_(0)
{
}

void
CbcSweepChunk::SetAngleSet(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CbcSweepChunk::SetAngleSet");

  fluds_ = &dynamic_cast<CBC_FLUDS&>(angle_set.GetFLUDS());

  gs_size_ = groupset_.groups.size(); // Number of groups in LBSGroupset
  gs_gi_ = groupset_.groups.front().id;

  surface_source_active_ = IsSurfaceSourceActive();

  // --- NEW:
  // Calculate and store strides for the compact layout
  num_angles_in_set_ = angle_set.GetNumAngles();
  compact_angle_stride_ = gs_size_;                     // Stride = number of groups
  compact_node_stride_ = num_angles_in_set_ * gs_size_; // Stride = angles_in_set * groups_per_angle
}

void
CbcSweepChunk::SetCell(const Cell* cell_ptr, AngleSet& angle_set)
{
  cell_ = cell_ptr;
  cell_local_id_ = cell_ptr->local_id;
  cell_mapping_ = &discretization_.GetCellMapping(*cell_);
  cell_transport_view_ = &cell_transport_views_[cell_->local_id];
  cell_num_faces_ = cell_->faces.size();
  cell_num_nodes_ = cell_mapping_->GetNumNodes();

  // Get cell matrices
  G_ = unit_cell_matrices_[cell_local_id_].intV_shapeI_gradshapeJ;
  M_ = unit_cell_matrices_[cell_local_id_].intV_shapeI_shapeJ;
  M_surf_ = unit_cell_matrices_[cell_local_id_].intS_shapeI_shapeJ;
  IntS_shapeI_ = unit_cell_matrices_[cell_local_id_].intS_shapeI;
}

/*
void
CbcSweepChunk::Sweep(AngleSet& angle_set)
{
  const auto& m2d_op = groupset_.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset_.quadrature->GetDiscreteToMomentOperator();

  DenseMatrix<double> Amat(max_num_cell_dofs_, max_num_cell_dofs_);
  DenseMatrix<double> Atemp(max_num_cell_dofs_, max_num_cell_dofs_);
  std::vector<Vector<double>> b(groupset_.groups.size(), Vector<double>(max_num_cell_dofs_));
  std::vector<double> source(max_num_cell_dofs_);

  const auto& face_orientations = angle_set.GetSPDS().GetCellFaceOrientations()[cell_local_id_];
  std::vector<double> face_mu_values(cell_num_faces_);

  const auto& rho = densities_[cell_local_id_];
  const auto& sigma_t = xs_.at(cell_->block_id)->GetSigmaTotal();

  // as = angle set
  // ss = subset
  const std::vector<size_t>& as_angle_indices = angle_set.GetAngleIndices();
  for (size_t as_ss_idx = 0; as_ss_idx < as_angle_indices.size(); ++as_ss_idx)
  {
    auto direction_num = as_angle_indices[as_ss_idx];
    auto omega = groupset_.quadrature->omegas[direction_num];
    auto wt = groupset_.quadrature->weights[direction_num];

    // Reset right-hand side
    for (int gsg = 0; gsg < gs_size_; ++gsg)
      for (int i = 0; i < cell_num_nodes_; ++i)
        b[gsg](i) = 0.0;

    for (int i = 0; i < cell_num_nodes_; ++i)
      for (int j = 0; j < cell_num_nodes_; ++j)
        Amat(i, j) = omega.Dot(G_(i, j));

    // Update face orientations
    for (int f = 0; f < cell_num_faces_; ++f)
      face_mu_values[f] = omega.Dot(cell_->faces[f].normal);

    // Surface integrals
    for (int f = 0; f < cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& face = cell_->faces[f];
      const bool is_local_face = cell_transport_view_->IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      auto face_nodal_mapping = &fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f);

      const std::vector<double>* psi_upwnd_data_block = nullptr;

      if ((not is_local_face) and (not is_boundary_face))
      {
        psi_upwnd_data_block = &fluds_->GetNonLocalUpwindData(cell_->global_id, f);
      }

      // IntSf_mu_psi_Mij_dA
      const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
      for (int fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping_->MapFaceNode(f, fi);

        for (int fj = 0; fj < num_face_nodes; ++fj)
        {
          const int j = cell_mapping_->MapFaceNode(f, fj);

          const double mu_Nij = -face_mu_values[f] * M_surf_[f](i, j);
          Amat(i, j) += mu_Nij;

          const double* psi = nullptr;
          if (is_local_face)
          {
            const unsigned int adj_cell_node = face_nodal_mapping->cell_node_mapping_[fj];
            const unsigned int adj_cell_node_offset =
              adj_cell_node * groupset_angle_group_stride_ + direction_num * groupset_group_stride_;
            psi = fluds_->GetLocalUpwindPsi(*cell_transport_view_->FaceNeighbor(f),
                                            adj_cell_node_offset);
          }
          else if (not is_boundary_face)
          {
            assert(psi_upwnd_data_block);
            const unsigned int adj_face_node = face_nodal_mapping->face_node_mapping_[fj];
            psi = fluds_->GetNonLocalUpwindPsi(*psi_upwnd_data_block, adj_face_node, as_ss_idx);
          }
          else
            psi = angle_set.PsiBoundary(face.neighbor_id,
                                        direction_num,
                                        cell_local_id_,
                                        f,
                                        fj,
                                        gs_gi_,
                                        surface_source_active_);

          if (not psi)
            continue;

          for (int gsg = 0; gsg < gs_size_; ++gsg)
            b[gsg](i) += psi[gsg] * mu_Nij;
        } // for face node j
      } // for face node i
    } // for f

    // Looping over groups, assembling mass terms
    for (int gsg = 0; gsg < gs_size_; ++gsg)
    {
      double sigma_tg = rho * sigma_t[gs_gi_ + gsg];

      // Contribute source moments q = M_n^T * q_moms
      for (int i = 0; i < cell_num_nodes_; ++i)
      {
        double temp_src = 0.0;
        for (int m = 0; m < num_moments_; ++m)
        {
          const size_t ir = cell_transport_view_->MapDOF(i, m, static_cast<int>(gs_gi_ + gsg));
          temp_src += m2d_op[m][direction_num] * source_moments_[ir];
        }
        source[i] = temp_src;
      }

      // Mass matrix and source
      // Atemp = Amat + sigma_tgr * M
      // b += M * q
      for (int i = 0; i < cell_num_nodes_; ++i)
      {
        double temp = 0.0;
        for (int j = 0; j < cell_num_nodes_; ++j)
        {
          const double Mij = M_(i, j);
          Atemp(i, j) = Amat(i, j) + Mij * sigma_tg;
          temp += Mij * source[j];
        }
        b[gsg](i) += temp;
      }

      // Solve system
      GaussElimination(Atemp, b[gsg], static_cast<int>(cell_num_nodes_));
    } // for gsg

    // Update phi
    auto& output_phi = GetDestinationPhi();
    for (int m = 0; m < num_moments_; ++m)
    {
      const double wn_d2m = d2m_op[m][direction_num];
      for (int i = 0; i < cell_num_nodes_; ++i)
      {
        const size_t ir = cell_transport_view_->MapDOF(i, m, gs_gi_);
        for (int gsg = 0; gsg < gs_size_; ++gsg)
          output_phi[ir + gsg] += wn_d2m * b[gsg](i);
      }
    }

    // Save angular flux during sweep
    if (save_angular_flux_)
    {
      auto& output_psi = GetDestinationPsi();
      double* cell_psi_data =
        &output_psi[discretization_.MapDOFLocal(*cell_, 0, groupset_.psi_uk_man_, 0, 0)];

      for (size_t i = 0; i < cell_num_nodes_; ++i)
      {
        const size_t imap =
          i * groupset_angle_group_stride_ + direction_num * groupset_group_stride_;
        for (int gsg = 0; gsg < gs_size_; ++gsg)
          cell_psi_data[imap + gsg] = b[gsg](i);
      }
    }

    // Perform outgoing surface operations
    for (int f = 0; f < cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face = cell_->faces[f];
      const bool is_local_face = cell_transport_view_->IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face and angle_set.GetBoundaries()[face.neighbor_id]->IsReflecting());
      const auto& IntF_shapeI = IntS_shapeI_[f];

      const int locality = cell_transport_view_->FaceLocality(f);
      const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
      auto& face_nodal_mapping = fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f);
      std::vector<double>* psi_dnwnd_data = nullptr;
      if (not is_boundary_face and not is_local_face)
      {
        auto& async_comm = *angle_set.GetCommunicator();
        size_t data_size = num_face_nodes * group_angle_stride_;
        psi_dnwnd_data = &async_comm.InitGetDownwindMessageData(locality,
                                                                face.neighbor_id,
                                                                face_nodal_mapping.associated_face_,
                                                                angle_set.GetID(),
                                                                data_size);
      }

      for (int fi = 0; fi < num_face_nodes; ++fi)
      {
        // Given the face index and the face node index, get the index of the
        // cell node that the face node corresponds to
        const int i = cell_mapping_->MapFaceNode(f, fi);

        if (is_boundary_face)
        {
          for (int gsg = 0; gsg < gs_size_; ++gsg)
            cell_transport_view_->AddOutflow(
              f, gs_gi_ + gsg, wt * face_mu_values[f] * b[gsg](i) * IntF_shapeI(i));
        }

        double* psi = nullptr;

        // Map the cell node to the appropriate downwind node if at a local face
        const unsigned int cell_node_offset =
          i * groupset_angle_group_stride_ + direction_num * groupset_group_stride_;
        const unsigned int i_map = is_local_face ? cell_node_offset : 0;

        if (is_local_face)
        {
          psi = fluds_->GetLocalDownwindPsi(*cell_);
        }
        else if (not is_boundary_face)
        {
          assert(psi_dnwnd_data);
          const size_t addr_offset = fi * group_angle_stride_ + as_ss_idx * group_stride_;
          psi = &(*psi_dnwnd_data)[addr_offset];
        }
        else if (is_reflecting_boundary_face)
          psi = angle_set.PsiReflected(face.neighbor_id, direction_num, cell_local_id_, f, fi);

        if (psi)
        {
          if (not is_boundary_face or is_reflecting_boundary_face)
          {
            for (int gsg = 0; gsg < gs_size_; ++gsg)
              psi[gsg + i_map] = b[gsg](i);
          }
        }
      } // for fi
    } // for face
  } // for angleset/subset
}
*/

void
CbcSweepChunk::Sweep(AngleSet& angle_set)
{
  const auto& m2d_op = groupset_.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset_.quadrature->GetDiscreteToMomentOperator();

  DenseMatrix<double> Amat(cell_num_nodes_, cell_num_nodes_); // Use actual cell_num_nodes_
  DenseMatrix<double> Atemp(cell_num_nodes_, cell_num_nodes_);
  std::vector<Vector<double>> b(gs_size_, Vector<double>(cell_num_nodes_));
  std::vector<double> source(cell_num_nodes_);

  const auto& face_orientations = angle_set.GetSPDS().GetCellFaceOrientations()[cell_local_id_];
  std::vector<double> face_mu_values(cell_num_faces_);

  const auto& rho = densities_[cell_local_id_];
  const auto& sigma_t = xs_.at(cell_->block_id)->GetSigmaTotal();

  const std::vector<size_t>& as_angle_indices = angle_set.GetAngleIndices();

  // Loop over angles in this AngleSet using LOCAL index as_ss_idx
  for (size_t as_ss_idx = 0; as_ss_idx < num_angles_in_set_; ++as_ss_idx)
  {
    auto direction_num = as_angle_indices[as_ss_idx]; // Global angle index
    auto omega = groupset_.quadrature->omegas[direction_num];
    auto wt = groupset_.quadrature->weights[direction_num];

    for (int gsg = 0; gsg < gs_size_; ++gsg)
      for (size_t i = 0; i < cell_num_nodes_; ++i)
        b[gsg](i) = 0.0;

    for (size_t i = 0; i < cell_num_nodes_; ++i)
      for (size_t j = 0; j < cell_num_nodes_; ++j)
        Amat(i, j) = omega.Dot(G_(i, j));

    for (int f = 0; f < cell_num_faces_; ++f)
      face_mu_values[f] = omega.Dot(cell_->faces[f].normal);

    // Surface integrals (Upstream contributions)
    for (int f = 0; f < cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& face = cell_->faces[f];
      const bool is_local_face = cell_transport_view_->IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      auto face_nodal_mapping = &fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f);

      const std::vector<double>* psi_nonlocal_upwnd_data_block = nullptr;
      if ((not is_local_face) and (not is_boundary_face))
      {
        psi_nonlocal_upwnd_data_block = &fluds_->GetNonLocalUpwindData(cell_->global_id, f);
      }

      const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
      for (int fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping_->MapFaceNode(f, fi);

        for (int fj = 0; fj < num_face_nodes; ++fj)
        {
          const int j = cell_mapping_->MapFaceNode(f, fj);
          const double mu_Nij = -face_mu_values[f] * M_surf_[f](i, j);
          Amat(i, j) += mu_Nij;

          const double* psi_upwind_groups_ptr = nullptr;

          if (is_local_face)
          {
            // *** CORRECTED Local Upwind Read ***
            const Cell* upwind_cell = cell_transport_view_->FaceNeighbor(f);
            // Node index WITHIN the UPWIND cell corresponding to face node fj
            const unsigned int adj_cell_node_idx = face_nodal_mapping->cell_node_mapping_[fj];

            // Get base pointer for the upwind cell's data block in the compact buffer
            const double* psi_upwind_cell_base_ptr = fluds_->GetLocalUpwindPsi(*upwind_cell);

            // Calculate offset RELATIVE to the cell's base pointer
            // Offset = (node_index_within_cell * node_stride) + (local_angle_index * angle_stride)
            const size_t offset_in_cell_block =
              adj_cell_node_idx * compact_node_stride_ + as_ss_idx * compact_angle_stride_;
            psi_upwind_groups_ptr = psi_upwind_cell_base_ptr + offset_in_cell_block;
          }
          else if (not is_boundary_face)
          {
            assert(psi_nonlocal_upwnd_data_block != nullptr);
            const unsigned int adj_face_node_idx = face_nodal_mapping->face_node_mapping_[fj];
            psi_upwind_groups_ptr = fluds_->GetNonLocalUpwindPsi(
              *psi_nonlocal_upwnd_data_block, adj_face_node_idx, as_ss_idx);
          }
          else
          {
            psi_upwind_groups_ptr = angle_set.PsiBoundary(face.neighbor_id,
                                                          direction_num,
                                                          cell_local_id_,
                                                          f,
                                                          fj,
                                                          gs_gi_,
                                                          surface_source_active_);
          }

          if (psi_upwind_groups_ptr != nullptr)
          {
            for (int gsg = 0; gsg < gs_size_; ++gsg)
            {
              b[gsg](i) += psi_upwind_groups_ptr[gsg] * mu_Nij;
            }
          }
        } // for fj
      } // for fi
    } // for face f

    // Volumetric source and solve loop (no changes needed here)
    for (int gsg = 0; gsg < gs_size_; ++gsg)
    {
      double sigma_tg = rho * sigma_t[gs_gi_ + gsg];
      for (size_t i = 0; i < cell_num_nodes_; ++i)
      {
        double temp_src = 0.0;
        for (int m = 0; m < num_moments_; ++m)
        {
          const size_t ir = cell_transport_view_->MapDOF(i, m, static_cast<int>(gs_gi_ + gsg));
          temp_src += m2d_op[m][direction_num] * source_moments_[ir];
        }
        source[i] = temp_src;
      }

      for (size_t i = 0; i < cell_num_nodes_; ++i)
      {
        double temp_M_source = 0.0;
        for (size_t j = 0; j < cell_num_nodes_; ++j)
        {
          const double Mij = M_(i, j);
          Atemp(i, j) = Amat(i, j) + Mij * sigma_tg;
          temp_M_source += Mij * source[j];
        }
        b[gsg](i) += temp_M_source;
      }
      GaussElimination(Atemp, b[gsg], static_cast<int>(cell_num_nodes_));
    }

    // Update phi moments (no changes needed here)
    auto& output_phi = GetDestinationPhi();
    for (int m = 0; m < num_moments_; ++m)
    {
      const double wn_d2m = d2m_op[m][direction_num];
      for (size_t i = 0; i < cell_num_nodes_; ++i)
      {
        const size_t ir = cell_transport_view_->MapDOF(i, m, gs_gi_);
        for (int gsg = 0; gsg < gs_size_; ++gsg)
          output_phi[ir + gsg] += wn_d2m * b[gsg](i);
      }
    }

    // Perform outgoing surface operations
    for (int f = 0; f < cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face = cell_->faces[f];
      const bool is_local_face = cell_transport_view_->IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face and angle_set.GetBoundaries().count(face.neighbor_id) > 0 and
         angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting());

      // Note: The outflow calculation needed correction in previous step, ensure it's right:
      const auto& IntFi_shapeI_vec = IntS_shapeI_[f]; // Get the vector for face f

      std::vector<double>* psi_nonlocal_dnwnd_data_block_ptr = nullptr;
      if (not is_boundary_face and not is_local_face)
      {
        const int locality = cell_transport_view_->FaceLocality(f);
        auto& face_nodal_mapping = fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f);
        auto& async_comm = *angle_set.GetCommunicator();
        size_t num_face_nodes = face_nodal_mapping.face_node_mapping_.size(); // Use mapping size
        size_t data_size_for_msg =
          num_face_nodes * gs_size_; // Size for all nodes, all groups for ONE angle.
        psi_nonlocal_dnwnd_data_block_ptr =
          &async_comm.InitGetDownwindMessageData(locality,
                                                 face.neighbor_id,
                                                 face_nodal_mapping.associated_face_,
                                                 angle_set.GetID(),
                                                 data_size_for_msg);
      }

      const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
      for (int fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping_->MapFaceNode(f, fi); // Node index in current cell

        // Outflow for balance (Corrected IntFi_shapeI access)
        if (is_boundary_face and not is_reflecting_boundary_face)
        {
          const double IntFi_shapeI_val = IntFi_shapeI_vec(i); // Access value for node i
          for (int gsg = 0; gsg < gs_size_; ++gsg)
            cell_transport_view_->AddOutflow(
              f, gs_gi_ + gsg, wt * face_mu_values[f] * b[gsg](i) * IntFi_shapeI_val);
        }

        double* psi_downwind_groups_ptr = nullptr;

        if (is_local_face)
        {
          // *** CORRECTED Local Downwind Write ***
          // Get base pointer for the current cell's data block in the compact buffer
          double* psi_downwind_cell_base_ptr = fluds_->GetLocalDownwindPsi(*cell_);

          // Calculate offset RELATIVE to the cell's base pointer
          // Offset = (node_index_within_cell * node_stride) + (local_angle_index * angle_stride)
          const size_t offset_in_cell_block =
            i * compact_node_stride_ + as_ss_idx * compact_angle_stride_;
          psi_downwind_groups_ptr = psi_downwind_cell_base_ptr + offset_in_cell_block;
        }
        else if (not is_boundary_face)
        {
          // *** CORRECTED Non-Local Downwind Write ***
          assert(psi_nonlocal_dnwnd_data_block_ptr != nullptr);
          // Offset within the buffer obtained from InitGet...
          // Buffer structure: [face_node_0_group_0, ..., face_node_0_group_N-1,
          // face_node_1_group_0, ...]
          const size_t nonlocal_offset =
            fi * gs_size_; // Offset = face_node_index * groups_per_node
          psi_downwind_groups_ptr = &((*psi_nonlocal_dnwnd_data_block_ptr)[nonlocal_offset]);
        }
        else if (is_reflecting_boundary_face)
        {
          psi_downwind_groups_ptr =
            angle_set.PsiReflected(face.neighbor_id, direction_num, cell_local_id_, f, fi);
        }

        if (psi_downwind_groups_ptr != nullptr)
        {
          for (int gsg = 0; gsg < gs_size_; ++gsg)
          {
            psi_downwind_groups_ptr[gsg] = b[gsg](i);
          }
        }
      } // for fi
    } // for outgoing face f
  } // for as_ss_idx
} // Sweep function end

} // namespace opensn

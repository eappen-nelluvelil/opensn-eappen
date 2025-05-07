// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/groupset/lbs_groupset.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mesh/cell/cell.h"
#include "framework/logging/log.h"
#include "caliper/cali.h"

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/angle_set/cbc_angle_set.h"

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
    group_stride_(0),
    group_angle_stride_(0),
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

  gs_size_ = groupset_.groups.size();
  gs_gi_ = groupset_.groups.front().id;

  surface_source_active_ = IsSurfaceSourceActive();
  group_stride_ = angle_set.GetNumGroups();
  // This is size of the local_psi_data vector in CBC FLUDS
  group_angle_stride_ = angle_set.GetNumGroups() * angle_set.GetNumAngles();
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

/*
void
CbcSweepChunk::Sweep(AngleSet& generic_angle_set) // Renamed param for clarity
{
    // Cast to CBC_AngleSet to access GetLocalAngleIndex
    // This assumes that when sweep_type_ is CBC, angle_set will always be a CBC_AngleSet.
    // A dynamic_cast could be used for safety if there's any doubt, but static_cast
    // is fine if the type is guaranteed by the calling context (SweepScheduler).
    const auto& current_cbc_angle_set = static_cast<const CBC_AngleSet&>(generic_angle_set);

    // --- SNIFF_TEST LOGGING: Start of sweep for this cell ---
    log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << " (local " << cell_local_id_
    << ") START SWEEP. AngleSet ID: " << current_cbc_angle_set.GetID();

    // --- Pre-loop initializations (already members or set by SetAngleSet/SetCell) ---
    // cell_ (current cell pointer)
    // cell_local_id_
    // cell_mapping_
    // cell_transport_view_
    // cell_num_faces_
    // cell_num_nodes_
    // G_ (IntV_shapeI_gradshapeJ for current cell)
    // M_ (IntV_shapeI_shapeJ for current cell)
    // M_surf_ (intS_shapeI_shapeJ for current cell's faces)
    // IntS_shapeI_ (intS_shapeI for current cell's faces)
    // fluds_ (pointer to the CBC_FLUDS instance for current_cbc_angle_set)
    // gs_size_ (number of groups in groupset)
    // gs_gi_ (global starting index of groups in groupset)
    // surface_source_active_ (flag from sweep scheduler)
    // save_angular_flux_ (from LBSProblem options, likely a member of SweepChunk or accessible)
    // num_moments_
    // max_num_cell_dofs_ (used for Amat/Atemp sizing)
    // groupset_ (reference to the LBSGroupset)
    // discretization_ (reference to SpatialDiscretization)
    // m2d_op, d2m_op from groupset_.quadrature

    // --- Allocate local matrices/vectors for the cell solve ---
    DenseMatrix<double> Amat(max_num_cell_dofs_, max_num_cell_dofs_);
    DenseMatrix<double> Atemp(max_num_cell_dofs_, max_num_cell_dofs_); // To hold Amat + sigma_t * M
    std::vector<Vector<double>> b_rhs(gs_size_, Vector<double>(max_num_cell_dofs_)); // RHS,
b_rhs[gsg](node_i) std::vector<double> cell_source_for_angle(max_num_cell_dofs_); // q_moms
projected for current angle

    const auto& face_orientations =
current_cbc_angle_set.GetSPDS().GetCellFaceOrientations()[cell_local_id_]; std::vector<double>
face_mu_values(cell_num_faces_);

    const auto& rho_density = densities_[cell_local_id_]; // Assuming densities_ is cell-local array
    const auto& sigma_t_xs = xs_.at(cell_->block_id)->GetSigmaTotal(); // Total cross-section vector
for material

    // --- Loop over angles in the current AngleSet ---
    const std::vector<size_t>& global_angle_indices_in_set =
current_cbc_angle_set.GetAngleIndices();

    for (size_t ang_loop_idx = 0; ang_loop_idx < global_angle_indices_in_set.size(); ++ang_loop_idx)
    {
        const size_t direction_num = global_angle_indices_in_set[ang_loop_idx]; // GLOBAL angle
index const unsigned int angle_idx_as = current_cbc_angle_set.GetLocalAngleIndex(direction_num); //
LOCAL index within AngleSet

        const Vector3& omega = groupset_.quadrature->omegas[direction_num];
        const double wt = groupset_.quadrature->weights[direction_num];

        // --- SNIFF_TEST LOGGING: Current Angle ---
        log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id
                           << "  Angle_gidx=" << direction_num << " (AS_idx=" << angle_idx_as
                           << ") Omega=" << omega.PrintStr() << " Weight=" << wt;

        // 1. Reset RHS for current angle
        for (int gsg = 0; gsg < gs_size_; ++gsg) {
            // b_rhs[gsg].Set(0.0); // If Vector has a Set method
            for (unsigned int i = 0; i < cell_num_nodes_; ++i) b_rhs[gsg](i) = 0.0;
        }

        // 2. Assemble Streaming part of Amat (Omega . Grad N_i, Grad N_j)
        //    This is done once per angle for all groups.
        for (unsigned int i = 0; i < cell_num_nodes_; ++i) {
            for (unsigned int j = 0; j < cell_num_nodes_; ++j) {
                Amat(i, j) = omega.Dot(G_(i, j)); // G_ is IntV_shapeI_gradshapeJ
            }
        }

        // 3. Calculate mu for each face for this angle
        for (unsigned int f = 0; f < cell_num_faces_; ++f) {
            face_mu_values[f] = omega.Dot(cell_->faces[f].normal);
        }

        // 4. Surface Integrals (contribute to Amat diagonal and RHS b_rhs)
        for (unsigned int f = 0; f < cell_num_faces_; ++f) {
            if (face_orientations[f] != FaceOrientation::INCOMING) {
                continue;
            }

            const auto& face = cell_->faces[f];
            const bool is_local_face = cell_transport_view_->IsFaceLocal(f);
            const bool is_boundary_face = !face.has_neighbor;
            const auto* face_nodal_mapping =
              &fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f);

            // --- SNIFF_TEST LOGGING: Incoming Face ---
            log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" <<
direction_num
                               << "    IncFace=" << f << " Normal=" << face.normal.PrintStr()
                               << " Mu=" << std::fixed << std::setprecision(6) << face_mu_values[f];

            const std::vector<double>* psi_upwnd_data_block_from_mpi = nullptr;
            if (!is_local_face && !is_boundary_face) { // Data from MPI neighbor
                psi_upwnd_data_block_from_mpi = &fluds_->GetNonLocalUpwindData(cell_->global_id, f);
            }

            const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
            for (unsigned int fi = 0; fi < num_face_nodes; ++fi) { // Current face node on current
cell's face const int i_local_cell = cell_mapping_->MapFaceNode(f, fi); // Map face node fi to cell
node i_local_cell

                for (unsigned int fj = 0; fj < num_face_nodes; ++fj) { // Upstream face node on
adjacent cell/boundary const int j_local_cell = cell_mapping_->MapFaceNode(f, fj); // Map face node
fj to cell node j_local_cell (for Amat)

                    const double mu_Nij_coeff = -face_mu_values[f] * M_surf_[f](i_local_cell,
j_local_cell); Amat(i_local_cell, j_local_cell) += mu_Nij_coeff;

                    const double* psi_upwind_groups_ptr = nullptr;

                    // --- SNIFF_TEST LOGGING
                    std::string upwind_source_info = "NONE";

                    if (is_local_face) {
                      const unsigned int upwind_node_idx_in_neighbor_cell =
face_nodal_mapping->cell_node_mapping_[fj]; const Cell* upwind_cell_ptr =
cell_transport_view_->FaceNeighbor(f); upwind_source_info = "LOCAL_UPWIND from cell " +
(upwind_cell_ptr ? std::to_string(upwind_cell_ptr->global_id) : "NULL") +
                                            ", node_in_neigh=" +
std::to_string(upwind_node_idx_in_neighbor_cell) +
                                            ", reading for angle_as=" +
std::to_string(angle_idx_as); psi_upwind_groups_ptr = fluds_->GetLocalUpwindPsi_Compact(
                          *upwind_cell_ptr,
                          upwind_node_idx_in_neighbor_cell,
                          angle_idx_as
                      );
                    } else if (!is_boundary_face) { // From MPI neighbor
                      assert(psi_upwnd_data_block_from_mpi);
                      const unsigned int face_node_idx_in_mpi_buffer =
face_nodal_mapping->face_node_mapping_[fj]; upwind_source_info = "MPI_UPWIND, mpi_face_node_idx=" +
std::to_string(face_node_idx_in_mpi_buffer) +
                                           ", reading for angle_as=" + std::to_string(angle_idx_as);
                      psi_upwind_groups_ptr = fluds_->GetNonLocalUpwindPsi(
                          *psi_upwnd_data_block_from_mpi,
                          face_node_idx_in_mpi_buffer,
                          angle_idx_as
                      );
                    } else { // From prescribed boundary or reflecting boundary
                      upwind_source_info = "BOUNDARY, bnd_id=" + std::to_string(face.neighbor_id) +
                      ", face_node_on_bnd=" + std::to_string(fj);
                      psi_upwind_groups_ptr = generic_angle_set.PsiBoundary(
                          face.neighbor_id, direction_num, cell_local_id_, f, fj, gs_gi_,
surface_source_active_
                      );
                    }

                    // Log Upwind Source Info
                    log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx="
<< direction_num
                    << "    IncFace=" << f << " Node_on_face_fi=" << fi << " (maps to cell_node=" <<
i_local_cell << ")"
                    << "      UpwindSource: " << upwind_source_info;

                    if (psi_upwind_groups_ptr) {
                        for (int gsg = 0; gsg < gs_size_; ++gsg) {
                          b_rhs[gsg](i_local_cell) += psi_upwind_groups_ptr[gsg] * mu_Nij_coeff;
                          log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "
Angle_gidx=" << direction_num
                                               << "      UpwindPsi[gsg=" << gsg << "]=" <<
std::scientific << std::setprecision(6)
                                               << psi_upwind_groups_ptr[gsg] << "
Contrib_to_RHS_node_" << i_local_cell
                                               << "=" << (psi_upwind_groups_ptr[gsg] *
mu_Nij_coeff);
                        }
                    }
                    else
                    {
                      log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "
Angle_gidx=" << direction_num
                                            << "      UpwindPsi_ptr is NULL";
                    }
                } // for fj (upwind contributing node)
            } // for fi (current node on face)
        } // for f (face)

        // 5. Volumetric Source Term (projected onto current angle) and Total Collision Term
        //    Solve linear system for each group
        for (int gsg = 0; gsg < gs_size_; ++gsg) {
            const double sigma_t_val = rho_density * sigma_t_xs[gs_gi_ + gsg];

            // Project volumetric source moments onto current angle for this group
            for (unsigned int i_node = 0; i_node < cell_num_nodes_; ++i_node) {
              double q_angle_i = 0.0;

              log.Log0Verbose1() << "SNIFF_TEST_QCALC: Cell " << cell_->global_id
                                 << " Ang=" << direction_num << " Node=" << i_node
                                 << " START q_angle_i calc. Initial q_angle_i=" << q_angle_i;

              for (int m = 0; m < num_moments_; ++m) {
                const size_t dof_map_phi = cell_transport_view_->MapDOF(i_node, m, gs_gi_ + gsg);
                const double current_source_moment = source_moments_[dof_map_phi];
                const double current_m2d_op =
groupset_.quadrature->GetMomentToDiscreteOperator()[m][direction_num];

                log.Log0Verbose1()
                  << "SNIFF_TEST_QCALC: Cell " << cell_->global_id << " Ang=" << direction_num
                  << " Node=" << i_node << " Mom=" << m << "  Reading src_mom[" << dof_map_phi
                  << "]=" << current_source_moment << ", m2d=" << current_m2d_op;

                double term = current_m2d_op * current_source_moment;

                log.Log0Verbose1() << "SNIFF_TEST_QCALC: Cell " << cell_->global_id << " Ang=" <<
direction_num
                                       << " Node=" << i_node << " Mom=" << m << "  Calculated term="
<< term;


                // // *** ENHANCED LOGGING: Log source moment value being used ***
                // log.Log0Verbose1() << "SNIFF_TEST_SOURCE: Cell " << cell_->global_id << "
                // Angle_gidx=" << direction_num
                // << " Grp=" << gsg << " Node=" << i_node << " Moment=" << m
                // << "  Reading source_moments_[" << dof_map_phi << "]="
                // << std::scientific << std::setprecision(6) << source_moments_[dof_map_phi];

                q_angle_i += term;

                log.Log0Verbose1() << "SNIFF_TEST_QCALC: Cell " << cell_->global_id << " Ang=" <<
direction_num
                                       << " Node=" << i_node << " Mom=" << m << "  Updated
q_angle_i=" << q_angle_i;
              }
              cell_source_for_angle[i_node] = q_angle_i;

              // Log the final result immediately after assignment
              log.Log0Verbose1() << "SNIFF_TEST_QCALC: Cell " << cell_->global_id << " Ang=" <<
direction_num
              << " Node=" << i_node << " FINAL cell_source_for_angle=" <<
cell_source_for_angle[i_node];
            }

             // Now log the value again right before the M_q_contrib loop starts, using the array
             if (gs_size_ == 1) {
              for (unsigned int i_log = 0; i_log < cell_num_nodes_; ++i_log) {
                   log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx="
<< direction_num << "  Grp=" << gsg
                                   << "    Node=" << i_log << "
VolumetricSourceAngle_Before_MqLoop[" << i_log << "]=" << cell_source_for_angle[i_log];
               }
           }

            // Assemble Atemp = Amat + M * sigma_t for current group
            // Add M * q_angle to b_rhs
            for (unsigned int i_row = 0; i_row < cell_num_nodes_; ++i_row) {
                double M_q_contrib = 0.0;
                for (unsigned int j_col = 0; j_col < cell_num_nodes_; ++j_col) {
                    const double Mij = M_(i_row, j_col); // M_ is IntV_shapeI_shapeJ
                    Atemp(i_row, j_col) = Amat(i_row, j_col) + Mij * sigma_t_val;
                    M_q_contrib += Mij * cell_source_for_angle[j_col];
                }
                b_rhs[gsg](i_row) += M_q_contrib;

                // Log M_q_contrib and TotalRHS
                if (gs_size_ == 1) {
                  // Removed the potentially confusing VolumetricSourceAngle[i_row] log here
                  log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx="
<< direction_num << "  Grp=" << gsg
                                  << "    Node=" << i_row << " M_q_contrib=" << M_q_contrib << "
TotalRHS_before_solve=" << b_rhs[gsg](i_row);
                }
            }

            // Solve Atemp * psi_solution = b_rhs[gsg]
            // Solution psi_solution overwrites b_rhs[gsg]
            GaussElimination(Atemp, b_rhs[gsg], static_cast<int>(cell_num_nodes_));

            // --- SNIFF_TEST LOGGING: Solved Psi ---
            if (gs_size_ == 1) { // Only print for 1-group case
              for (unsigned int i_sol = 0; i_sol < cell_num_nodes_; ++i_sol) {
                  log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx="
<< direction_num << "  Grp=" << gsg
                                  << "    Node=" << i_sol << " SOLVED_PSI=" << std::scientific <<
std::setprecision(6) << b_rhs[gsg](i_sol);
              }
          }
        } // for gsg (group)

        // 6. Update Scalar Flux Moments (phi) - Unchanged
        auto& phi_destination = GetDestinationPhi(); // LBSProblem::phi_new_local_
        for (int m = 0; m < num_moments_; ++m) {
            const double wt_d2m_op_val = wt *
groupset_.quadrature->GetDiscreteToMomentOperator()[m][direction_num]; if (std::fabs(wt_d2m_op_val)
< 1.0e-16) continue; // Skip if contribution is zero

            for (unsigned int i_node = 0; i_node < cell_num_nodes_; ++i_node) {
                const size_t dof_map_phi_base = cell_transport_view_->MapDOF(i_node, m, gs_gi_);
                for (int gsg = 0; gsg < gs_size_; ++gsg)
                {
                  // *** ENHANCED LOGGING: Log phi update components ***
                  log.Log0Verbose1()
                    << "SNIFF_TEST_PHI: Cell " << cell_->global_id
                    << " Angle_gidx=" << direction_num << " Mom=" << m << " Grp=" << gsg
                    << " Node=" << i_node << "  Updating phi_destination["
                    << (dof_map_phi_base + gsg) << "]"
                    << " += " << std::scientific << std::setprecision(6) << wt_d2m_op_val << " * "
                    << b_rhs[gsg](i_node) << " (value=" << (wt_d2m_op_val * b_rhs[gsg](i_node))
                    << ")";

                    phi_destination[dof_map_phi_base + gsg] += wt_d2m_op_val * b_rhs[gsg](i_node);
                }
            }
        }

        // 7. Save Angular Flux to LBSProblem main storage (if enabled) - Unchanged
        if (save_angular_flux_) {
            auto& psi_destination_main = GetDestinationPsi(); // LBSProblem::psi_new_local_[gs_id]
            // Get base pointer for the current cell's data IN THE MAIN LBSProblem STORAGE
            // This MapDOFLocal uses the groupset's full psi_uk_man_
            double* cell_psi_base_in_main_storage =
                &psi_destination_main[discretization_.MapDOFLocal(*cell_, 0, groupset_.psi_uk_man_,
0, 0)];

            for (unsigned int i_node = 0; i_node < cell_num_nodes_; ++i_node) {
                // Indexing for the main storage (uses global direction_num)
                const size_t node_block_offset_in_main_storage =
                    i_node * groupset_angle_group_stride_ +      // Offset to node i_node's psi data
block direction_num * groupset_group_stride_;     // Offset to global direction_num's data for (int
gsg = 0; gsg < gs_size_; ++gsg) { cell_psi_base_in_main_storage[node_block_offset_in_main_storage +
gsg] = b_rhs[gsg](i_node);
                }
            }
        }

        // 8. Perform Outgoing Surface Operations
        for (unsigned int f = 0; f < cell_num_faces_; ++f) {
            if (face_orientations[f] != FaceOrientation::OUTGOING) {
                continue;
            }

            const auto& face = cell_->faces[f];
            const bool is_local_outgoing = cell_transport_view_->IsFaceLocal(f);
            const bool is_boundary_outgoing = !face.has_neighbor;
            const bool is_reflecting_boundary_outgoing =
              (is_boundary_outgoing &&
               generic_angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting());

            // --- SNIFF_TEST LOGGING: Outgoing Face ---
            log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" <<
direction_num
                               << "    OutFace=" << f << " Normal=" << face.normal.PrintStr()
                               << " Mu=" << std::fixed << std::setprecision(6) << face_mu_values[f];

            // Outflow Tally for non-reflecting physical boundaries
            if (is_boundary_outgoing && !is_reflecting_boundary_outgoing) {
                const auto& IntF_shapeI_face_f = IntS_shapeI_[f]; // For current face f
                const size_t num_face_nodes_for_tally = cell_mapping_->GetNumFaceNodes(f);
                for (unsigned int fi_tally = 0; fi_tally < num_face_nodes_for_tally; ++fi_tally) {
                    const int i_node_tally = cell_mapping_->MapFaceNode(f, fi_tally);
                    for (int gsg = 0; gsg < gs_size_; ++gsg) {
                        cell_transport_view_->AddOutflow(
                            f, gs_gi_ + gsg,
                            wt * face_mu_values[f] * b_rhs[gsg](i_node_tally) *
IntF_shapeI_face_f(fi_tally)
                        );
                    }
                }
            }

            // Prepare data for downwind cells/boundaries
            std::vector<double>* psi_dnwnd_mpi_buffer = nullptr;
            if (!is_local_outgoing && !is_boundary_outgoing) { // Outgoing to MPI neighbor
                auto& async_comm = *generic_angle_set.GetCommunicator();
                // group_angle_stride_ is N_as * G_gs for CbcSweepChunk
                size_t data_size_for_mpi_face = cell_mapping_->GetNumFaceNodes(f) *
group_angle_stride_; psi_dnwnd_mpi_buffer = &async_comm.InitGetDownwindMessageData(
                    cell_transport_view_->FaceLocality(f),
                    face.neighbor_id,
                    fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f).associated_face_,
                    generic_angle_set.GetID(),
                    data_size_for_mpi_face
                );
            }

            const size_t num_face_nodes_on_f = cell_mapping_->GetNumFaceNodes(f);
            for (unsigned int fi_on_face = 0; fi_on_face < num_face_nodes_on_f; ++fi_on_face) {
                const int i_node_in_cell = cell_mapping_->MapFaceNode(f, fi_on_face); // Cell node
index

                if (is_local_outgoing) { // Write to compact CBC_FLUDS::local_psi_data_
                    double* psi_target_in_compact_fluds = fluds_->GetLocalDownwindPsi_Compact(
                        *cell_, i_node_in_cell, angle_idx_as
                    );
                    if (psi_target_in_compact_fluds) {
                        for (int gsg = 0; gsg < gs_size_; ++gsg) {
                          psi_target_in_compact_fluds[gsg] = b_rhs[gsg](i_node_in_cell);

                          // --- SNIFF_TEST LOGGING: Writing to Compact FLUDS ---
                          size_t expected_idx = fluds_->MapDOFCompactLocal(*cell_, i_node_in_cell,
angle_idx_as, gsg); log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx="
<< direction_num
                                             << "    OutFace=" << f << " Node_on_face_fi=" <<
fi_on_face << " (cell_node=" << i_node_in_cell << ")"
                                             << "      WRITE_TO_COMPACT_FLUDS: AngleAS=" <<
angle_idx_as << " Grp=" << gsg
                                             << " TargetIdx=" << expected_idx
                                             << " Value=" << std::scientific << std::setprecision(6)
<< psi_target_in_compact_fluds[gsg];
                        }
                    }
                } else if (!is_boundary_outgoing) { // Write to MPI buffer
                    assert(psi_dnwnd_mpi_buffer);
                    // MPI buffer layout: FaceNode -> Angle_AS -> Group
                    // angle_set.GetNumAngles() is N_as
                    // gs_size_ is G_gs
                    const size_t mpi_buffer_offset =
                        fi_on_face * (generic_angle_set.GetNumAngles() * gs_size_) + // Stride to
current face node angle_idx_as * gs_size_;                                  // Stride to current
angle_as

                    double* psi_target_in_mpi_buffer_ptr =
&((*psi_dnwnd_mpi_buffer)[mpi_buffer_offset]); for (int gsg = 0; gsg < gs_size_; ++gsg) {
                        psi_target_in_mpi_buffer_ptr[gsg] = b_rhs[gsg](i_node_in_cell);
                    }
                } else if (is_reflecting_boundary_outgoing) { // Write to ReflectingBoundary storage
                    // PsiReflected uses global direction_num
                    double* psi_target_in_reflecting_boundary = generic_angle_set.PsiReflected(
                        face.neighbor_id, direction_num, cell_local_id_, f, fi_on_face
                    );
                    if (psi_target_in_reflecting_boundary) {
                        for (int gsg = 0; gsg < gs_size_; ++gsg) {
                            psi_target_in_reflecting_boundary[gsg] = b_rhs[gsg](i_node_in_cell);
                        }
                    }
                }
            } // for fi_on_face (node on outgoing face)
        } // for f (face) - End Outgoing Surface Ops

    } // for ang_loop_idx (angle in AngleSet)

    // --- SNIFF_TEST LOGGING: End of sweep for this cell ---
    log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << " (local " << cell_local_id_
    << ") END SWEEP. AngleSet ID: " << current_cbc_angle_set.GetID();
} // End CbcSweepChunk::Sweep
*/

void
CbcSweepChunk::Sweep(AngleSet& generic_angle_set) // Parameter name kept generic
{
  // Cast to CBC_AngleSet to access GetLocalAngleIndex
  const auto& current_cbc_angle_set = static_cast<const CBC_AngleSet&>(generic_angle_set);

  // SNIFF_TEST Logging (optional, but useful for debugging)
  log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << " (local " << cell_local_id_
                     << ") START SWEEP. AngleSet ID: " << current_cbc_angle_set.GetID();


  const auto& m2d_op = groupset_.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset_.quadrature->GetDiscreteToMomentOperator();

  DenseMatrix<double> Amat(max_num_cell_dofs_, max_num_cell_dofs_);
  DenseMatrix<double> Atemp(max_num_cell_dofs_, max_num_cell_dofs_);
  // b_rhs is a better name than b, as it's the Right Hand Side
  std::vector<Vector<double>> b_rhs(gs_size_, Vector<double>(max_num_cell_dofs_));
  // source_nodal_q_angle is a better name for this temporary variable
  std::vector<double> source_nodal_q_angle(max_num_cell_dofs_);

  const auto& face_orientations = current_cbc_angle_set.GetSPDS().GetCellFaceOrientations()[cell_local_id_];
  std::vector<double> face_mu_values(cell_num_faces_);

  const auto& rho = densities_[cell_local_id_]; // Assuming densities_ is correctly cell-local
  const auto& sigma_t_vector = xs_.at(cell_->block_id)->GetSigmaTotal(); // Vector of sigma_t per group

  const std::vector<size_t>& as_angle_indices = current_cbc_angle_set.GetAngleIndices(); // Global angle indices in this set

  for (size_t ang_set_loop_idx = 0; ang_set_loop_idx < as_angle_indices.size(); ++ang_set_loop_idx)
  {
    const size_t direction_num = as_angle_indices[ang_set_loop_idx]; // GLOBAL angle index
    // *** NEW: Get local angle index for compact FLUDS access ***
    const unsigned int angle_idx_as = current_cbc_angle_set.GetLocalAngleIndex(direction_num);

    const Vector3& omega = groupset_.quadrature->omegas[direction_num];
    const double wt = groupset_.quadrature->weights[direction_num];

    // SNIFF_TEST Logging
    log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id
                       << "  Angle_gidx=" << direction_num << " (AS_idx=" << angle_idx_as
                       << ") Omega=" << omega.PrintStr() << " Weight=" << wt;

    // Reset right-hand side for current angle
    for (int gsg = 0; gsg < gs_size_; ++gsg)
      for (unsigned int i = 0; i < cell_num_nodes_; ++i) // Use unsigned int for loop consistency
        b_rhs[gsg](i) = 0.0;

    // Assemble Streaming part of Amat
    for (unsigned int i = 0; i < cell_num_nodes_; ++i)
      for (unsigned int j = 0; j < cell_num_nodes_; ++j)
        Amat(i, j) = omega.Dot(G_(i, j));

    // Calculate mu for each face
    for (unsigned int f = 0; f < cell_num_faces_; ++f)
      face_mu_values[f] = omega.Dot(cell_->faces[f].normal);

    // Surface integrals
    for (unsigned int f = 0; f < cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& face = cell_->faces[f];
      const bool is_local_inc_face = cell_transport_view_->IsFaceLocal(f); // Is upwind cell local?
      const bool is_boundary_inc_face = !face.has_neighbor;
      const auto* face_nodal_mapping = &fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f);

      // SNIFF_TEST Logging
      log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" << direction_num
                         << "    IncFace=" << f << " Normal=" << face.normal.PrintStr()
                         << " Mu=" << std::fixed << std::setprecision(6) << face_mu_values[f];

      const std::vector<double>* psi_upwnd_data_block_mpi = nullptr;
      if (!is_local_inc_face && !is_boundary_inc_face) // From MPI neighbor
      {
        psi_upwnd_data_block_mpi = &fluds_->GetNonLocalUpwindData(cell_->global_id, f);
      }

      const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
      for (unsigned int fi = 0; fi < num_face_nodes; ++fi) // Current face node on current cell's face
      {
        const int i_curr_cell_node = cell_mapping_->MapFaceNode(f, fi); // Node in current cell

        for (unsigned int fj = 0; fj < num_face_nodes; ++fj) // Upstream face node on adjacent entity
        {
          const int j_amat_idx = cell_mapping_->MapFaceNode(f, fj); // Node in current cell for Amat
          const double mu_Nij_coeff = -face_mu_values[f] * M_surf_[f](i_curr_cell_node, j_amat_idx);
          Amat(i_curr_cell_node, j_amat_idx) += mu_Nij_coeff;

          const double* psi_upwind_src_ptr = nullptr;
          std::string upwind_source_info_log = "NONE";

          if (is_local_inc_face)
          {
            const unsigned int upwind_node_in_neighbor = face_nodal_mapping->cell_node_mapping_[fj];
            const Cell* upwind_cell_ptr = cell_transport_view_->FaceNeighbor(f);
            upwind_source_info_log = "LOCAL_UPWIND from cell " + (upwind_cell_ptr ? std::to_string(upwind_cell_ptr->global_id) : "NULL_PTR") +
                                     ", node_in_neigh=" + std::to_string(upwind_node_in_neighbor) +
                                     ", reading for angle_as=" + std::to_string(angle_idx_as);
            // *** MODIFIED: Use compact accessor ***
            psi_upwind_src_ptr = fluds_->GetLocalUpwindPsi_Compact(
                *upwind_cell_ptr,
                upwind_node_in_neighbor,
                angle_idx_as // Pass local angle index
            );
          }
          else if (!is_boundary_inc_face) // From MPI
          {
            assert(psi_upwnd_data_block_mpi != nullptr);
            const unsigned int upwind_face_node_in_mpi_buffer = face_nodal_mapping->face_node_mapping_[fj];
            upwind_source_info_log = "MPI_UPWIND, mpi_face_node_idx=" + std::to_string(upwind_face_node_in_mpi_buffer) +
                                     ", reading for angle_as=" + std::to_string(angle_idx_as);
            // *** MODIFIED: Pass local angle index (angle_idx_as) ***
            // The original as_ss_idx was the loop variable ang_set_loop_idx, which is equivalent to angle_idx_as here.
            psi_upwind_src_ptr = fluds_->GetNonLocalUpwindPsi(
                *psi_upwnd_data_block_mpi,
                upwind_face_node_in_mpi_buffer,
                angle_idx_as // Pass local angle index
            );
          }
          else // Boundary face
          {
            upwind_source_info_log = "BOUNDARY, bnd_id=" + std::to_string(face.neighbor_id) +
                                     ", face_node_on_bnd=" + std::to_string(fj);
            psi_upwind_src_ptr = generic_angle_set.PsiBoundary(
                face.neighbor_id,
                direction_num, // Global angle index for PsiBoundary
                cell_local_id_, f, fj, gs_gi_, surface_source_active_
            );
          }

          log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" << direction_num
                             << "    IncFace=" << f << " Node_on_face_fi=" << fi << " (cell_node=" << i_curr_cell_node << ")"
                             << "      UpwindSource: " << upwind_source_info_log;

          if (psi_upwind_src_ptr != nullptr)
          {
            for (int gsg = 0; gsg < gs_size_; ++gsg)
            {
              b_rhs[gsg](i_curr_cell_node) += psi_upwind_src_ptr[gsg] * mu_Nij_coeff;
              log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" << direction_num
                                 << "      UpwindPsi[gsg=" << gsg << "]=" << std::scientific << std::setprecision(6)
                                 << psi_upwind_src_ptr[gsg] << " Contrib_to_RHS_node_" << i_curr_cell_node
                                 << "=" << (psi_upwind_src_ptr[gsg] * mu_Nij_coeff);
            }
          } else {
            log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" << direction_num
                               << "      UpwindPsi_ptr is NULL";
          }
        } // for fj
      } // for fi
    } // for f

    // Looping over groups, assembling mass terms and solving
    for (int gsg = 0; gsg < gs_size_; ++gsg)
    {
      double sigma_t_val = rho * sigma_t_vector[gs_gi_ + gsg];

      // Calculate angular source q_angle for all nodes for this group/angle
      for (unsigned int i_node = 0; i_node < cell_num_nodes_; ++i_node)
      {
        double q_angle_i_node = 0.0;
        // SNIFF_TEST_QCALC Logging (as per your previous detailed version)
        log.Log0Verbose1() << "SNIFF_TEST_QCALC: Cell " << cell_->global_id << " Ang=" << direction_num
                           << " Node=" << i_node << " Grp=" << gsg << " START q_angle_i calc. Initial=" << q_angle_i_node;
        for (int m = 0; m < num_moments_; ++m)
        {
          const size_t src_map_idx = cell_transport_view_->MapDOF(i_node, m, gs_gi_ + gsg);
          const double current_source_moment_val = source_moments_[src_map_idx];
          const double current_m2d_op_val = m2d_op[m][direction_num];
          double term_val = current_m2d_op_val * current_source_moment_val;
          q_angle_i_node += term_val;
          log.Log0Verbose1() << "SNIFF_TEST_QCALC: Cell " << cell_->global_id << " Ang=" << direction_num
                             << " Node=" << i_node << " Grp=" << gsg << " Mom=" << m
                             << "  src_mom[" << src_map_idx << "]=" << current_source_moment_val
                             << ", m2d=" << current_m2d_op_val << ", term=" << term_val << ", upd_q_angle_i=" << q_angle_i_node;
        }
        source_nodal_q_angle[i_node] = q_angle_i_node;
        log.Log0Verbose1() << "SNIFF_TEST_QCALC: Cell " << cell_->global_id << " Ang=" << direction_num
                           << " Node=" << i_node << " Grp=" << gsg << " FINAL cell_source_for_angle=" << source_nodal_q_angle[i_node];

      }

      if (gs_size_ == 1) { // Log only for 1-group case to reduce spam
          for (unsigned int i_log = 0; i_log < cell_num_nodes_; ++i_log) {
               log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" << direction_num << "  Grp=" << gsg
                               << "    Node=" << i_log << " VolumetricSourceAngle_Before_MqLoop[" << i_log << "]=" << source_nodal_q_angle[i_log];
           }
       }

      // Mass matrix and source
      for (unsigned int i_row = 0; i_row < cell_num_nodes_; ++i_row)
      {
        double M_q_contrib = 0.0;
        for (unsigned int j_col = 0; j_col < cell_num_nodes_; ++j_col)
        {
          const double Mij = M_(i_row, j_col);
          Atemp(i_row, j_col) = Amat(i_row, j_col) + Mij * sigma_t_val;
          M_q_contrib += Mij * source_nodal_q_angle[j_col]; // Use j_col
        }
        b_rhs[gsg](i_row) += M_q_contrib;
        if (gs_size_ == 1) {
            log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" << direction_num << "  Grp=" << gsg
                            << "    Node=" << i_row << " M_q_contrib=" << M_q_contrib << " TotalRHS_before_solve=" << b_rhs[gsg](i_row);
        }
      }

      GaussElimination(Atemp, b_rhs[gsg], static_cast<int>(cell_num_nodes_));

      if (gs_size_ == 1) {
          for (unsigned int i_sol = 0; i_sol < cell_num_nodes_; ++i_sol) {
              log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" << direction_num << "  Grp=" << gsg
                              << "    Node=" << i_sol << " SOLVED_PSI=" << std::scientific << std::setprecision(6) << b_rhs[gsg](i_sol);
          }
      }
    } // for gsg

    // Update phi
    auto& output_phi = GetDestinationPhi();
    for (int m = 0; m < num_moments_; ++m)
    {
      const double wn_d2m = wt * d2m_op[m][direction_num]; // wt was from current global angle
      for (unsigned int i_node = 0; i_node < cell_num_nodes_; ++i_node)
      {
        const size_t ir = cell_transport_view_->MapDOF(i_node, m, gs_gi_);
        for (int gsg = 0; gsg < gs_size_; ++gsg)
        {
          log.Log0Verbose1() << "SNIFF_TEST_PHI: Cell " << cell_->global_id << " Angle_gidx=" << direction_num
                             << " Mom=" << m << " Grp=" << gsg << " Node=" << i_node
                             << "  Updating phi_destination[" << (ir + gsg) << "]"
                             << " += " << std::scientific << std::setprecision(6) << wn_d2m
                             << " * " << b_rhs[gsg](i_node) << " (value=" << (wn_d2m * b_rhs[gsg](i_node)) << ")";
          output_phi[ir + gsg] += wn_d2m * b_rhs[gsg](i_node);
        }
      }
    }

    // Save angular flux to main problem storage (if enabled)
    if (save_angular_flux_)
    {
      auto& output_psi_main_storage = GetDestinationPsi();
      // Base pointer for (current_cell, node 0, GLOBAL_angle 0, group 0) in main storage
      double* cell_psi_base_main =
        &output_psi_main_storage[discretization_.MapDOFLocal(*cell_, 0, groupset_.psi_uk_man_, 0, 0)];

      for (unsigned int i_node = 0; i_node < cell_num_nodes_; ++i_node)
      {
        // Offset from cell_psi_base_main to (current_node, current_GLOBAL_angle, group 0)
        const size_t imap_offset_main_storage =
          i_node * groupset_angle_group_stride_ + // Stride based on TOTAL angles in quad
          direction_num * groupset_group_stride_;  // Stride based on groupset groups
        for (int gsg = 0; gsg < gs_size_; ++gsg)
          cell_psi_base_main[imap_offset_main_storage + gsg] = b_rhs[gsg](i_node);
      }
    }

    // Perform outgoing surface operations
    for (unsigned int f = 0; f < cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::OUTGOING)
        continue;

      const auto& face = cell_->faces[f];
      const bool is_local_out_face = cell_transport_view_->IsFaceLocal(f);
      const bool is_boundary_out_face = !face.has_neighbor;
      const bool is_reflecting_boundary_out_face =
        (is_boundary_out_face && generic_angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting());
      const auto& IntF_shapeI_face_f = IntS_shapeI_[f]; // Integral of shape func over face f

      log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" << direction_num
                         << "    OutFace=" << f << " Normal=" << face.normal.PrintStr()
                         << " Mu=" << std::fixed << std::setprecision(6) << face_mu_values[f];


      std::vector<double>* psi_dnwnd_mpi_buffer = nullptr;
      if (!is_boundary_out_face && !is_local_out_face) // Outgoing to MPI
      {
        auto& async_comm = *generic_angle_set.GetCommunicator();
        // group_angle_stride_ in CbcSweepChunk is num_groups_in_AS * num_groups_in_GS
        // this is correct for packing MPI data which is for the current AngleSet only
        size_t data_size_for_mpi = cell_mapping_->GetNumFaceNodes(f) * group_angle_stride_;
        psi_dnwnd_mpi_buffer = &async_comm.InitGetDownwindMessageData(
            cell_transport_view_->FaceLocality(f),
            face.neighbor_id, // Global ID of ghost cell
            fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f).associated_face_,
            generic_angle_set.GetID(),
            data_size_for_mpi
        );
      }

      const size_t num_face_nodes_on_f = cell_mapping_->GetNumFaceNodes(f);
      for (unsigned int fi_on_face = 0; fi_on_face < num_face_nodes_on_f; ++fi_on_face)
      {
        const int i_node_in_cell = cell_mapping_->MapFaceNode(f, fi_on_face);

        // Outflow Tally (always uses global angle, done for non-reflecting boundaries)
        if (is_boundary_out_face && !is_reflecting_boundary_out_face)
        {
          for (int gsg = 0; gsg < gs_size_; ++gsg)
            cell_transport_view_->AddOutflow(
              f, gs_gi_ + gsg,
              wt * face_mu_values[f] * b_rhs[gsg](i_node_in_cell) * IntF_shapeI_face_f(fi_on_face) // fi_on_face is correct index for IntF_shapeI
            );
        }

        if (is_local_out_face) // Write to compact FLUDS
        {
          // *** MODIFIED: Use compact accessor ***
          double* psi_target_compact = fluds_->GetLocalDownwindPsi_Compact(
              *cell_, i_node_in_cell, angle_idx_as // Use local angle index
          );
          if (psi_target_compact != nullptr)
          {
            for (int gsg = 0; gsg < gs_size_; ++gsg)
            {
              psi_target_compact[gsg] = b_rhs[gsg](i_node_in_cell);
              // SNIFF_TEST Logging
              size_t expected_idx = fluds_->MapDOFCompactLocal(*cell_, i_node_in_cell, angle_idx_as, gsg);
              log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << "  Angle_gidx=" << direction_num
                                 << "    OutFace=" << f << " Node_on_face_fi=" << fi_on_face << " (cell_node=" << i_node_in_cell << ")"
                                 << "      WRITE_TO_COMPACT_FLUDS: AngleAS=" << angle_idx_as << " Grp=" << gsg
                                 << " TargetIdx=" << expected_idx
                                 << " Value=" << std::scientific << std::setprecision(6) << psi_target_compact[gsg];
            }
          }
        }
        else if (!is_boundary_out_face) // Write to MPI buffer
        {
          assert(psi_dnwnd_mpi_buffer != nullptr);
          // MPI buffer layout: FaceNode -> Angle_AS -> Group
          // ang_set_loop_idx is equivalent to angle_idx_as for this MPI write structure
          const size_t mpi_buffer_offset =
              fi_on_face * group_angle_stride_ + // Stride to current face node (uses compact N_as*N_gs)
              angle_idx_as * group_stride_;     // Stride to current angle_as (uses compact N_gs)

          double* psi_target_in_mpi_ptr = &((*psi_dnwnd_mpi_buffer)[mpi_buffer_offset]);
          for (int gsg = 0; gsg < gs_size_; ++gsg) {
            psi_target_in_mpi_ptr[gsg] = b_rhs[gsg](i_node_in_cell);
          }
        }
        else if (is_reflecting_boundary_out_face) // Write to ReflectingBoundary storage
        {
          // PsiReflected uses global direction_num
          double* psi_target_refl_bnd = generic_angle_set.PsiReflected(
              face.neighbor_id, direction_num, cell_local_id_, f, fi_on_face);
          if (psi_target_refl_bnd != nullptr) {
            for (int gsg = 0; gsg < gs_size_; ++gsg) {
              psi_target_refl_bnd[gsg] = b_rhs[gsg](i_node_in_cell);
            }
          }
        }
      } // for fi_on_face
    } // for f (outgoing)
  } // for ang_set_loop_idx (angle in AngleSet)

  log.Log0Verbose1() << "SNIFF_TEST: Cell " << cell_->global_id << " (local " << cell_local_id_
                     << ") END SWEEP. AngleSet ID: " << current_cbc_angle_set.GetID();
} // End CbcSweepChunk::Sweep

} // namespace opensn

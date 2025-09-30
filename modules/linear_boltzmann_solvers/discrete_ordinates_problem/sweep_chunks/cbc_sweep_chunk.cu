#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/memory_pinner.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/outflow_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/quadrature_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/storage.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/view/mesh_view.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/view/quadrature_view.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/view/xs_view.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"
#include "caliper/cali.h"
#include "caribou/caribou.h"

namespace crb = caribou;

namespace opensn
{

// __device__ inline void
// DeviceGaussElimination(std::array<double, matrix_size>& sweep_matrix,
//                        std::array<double, max_dof>& psi,
//                        const std::uint32_t& cell_num_nodes)
// {
//   // Forward elimination
//   double* A_i = sweep_matrix.data();
//   for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
//   {
//     double inv_diag = 1.0 / A_i[i];

//     // Normalize pivot row
//     for (std::uint32_t j = i; j < cell_num_nodes; ++j)
//       A_i[j] *= inv_diag;
//     psi[i] *= inv_diag;

//     // Eliminate rows belows
//     double* A_k = A_i + max_dof;
//     for (std::uint32_t k = i + 1; k < cell_num_nodes; ++k)
//     {
//       double factor = -A_k[i];
//       for (std::uint32_t j = i; j < cell_num_nodes; ++j)
//         A_k[j] += factor * A_i[j];
//       psi[k] += factor * psi[i];
//       A_k += max_dof;
//     }
//     A_i += max_dof;
//   }

//   // Back substitution
//   for (std::int32_t j = cell_num_nodes - 2; j >= 0; --j)
//   {
//     double* A_j = sweep_matrix.data() + j * max_dof;
//     for (std::int32_t i = j + 1; i < cell_num_nodes; ++i)
//       psi[j] -= A_j[i] * psi[i];
//   }
// }

// __global__ void
// SolveKernel(std::array<double, matrix_size>& sweep_matrix,
//             std::array<double, max_dof>& psi,
//             const std::uint32_t& cell_num_nodes)
// {
//   DeviceGaussElimination(sweep_matrix, psi, cell_num_nodes);
// }

// __device__ inline void
// DeviceGaussElimination(double* sweep_matrix,
//                        double* psi,
//                        const std::uint32_t cell_num_nodes)
// {
//   // Forward elimination
//   double* A_i = sweep_matrix;
//   for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
//   {
//     double inv_diag = 1.0 / A_i[i];

//     // Normalize pivot row
//     for (std::uint32_t j = i; j < cell_num_nodes; ++j)
//       A_i[j] *= inv_diag;
//     psi[i] *= inv_diag;

//     // Eliminate rows belows
//     double* A_k = A_i + cell_num_nodes;
//     for (std::uint32_t k = i + 1; k < cell_num_nodes; ++k)
//     {
//       double factor = -A_k[i];
//       for (std::uint32_t j = i; j < cell_num_nodes; ++j)
//         A_k[j] += factor * A_i[j];
//       psi[k] += factor * psi[i];
//       A_k += cell_num_nodes;
//     }
//     A_i += cell_num_nodes;
//   }

//   // Back substitution
//   for (std::int32_t j = cell_num_nodes - 2; j >= 0; --j)
//   {
//     double* A_j = sweep_matrix + j * cell_num_nodes;
//     for (std::int32_t i = j + 1; i < cell_num_nodes; ++i)
//       psi[j] -= A_j[i] * psi[i];
//   }
// }

/*
__device__ inline void
DeviceGaussElimination(double* sweep_matrix,
                       double* psi,
                       const std::uint32_t cell_num_nodes)
{
  // Forward elimination
  double* A_i = sweep_matrix;
  for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
  {
    double inv_diag = 1.0 / A_i[i];

    // Normalize pivot row
    for (std::uint32_t j = i; j < cell_num_nodes; ++j)
      A_i[j] *= inv_diag;
    psi[i] *= inv_diag;

    // Eliminate rows belows
    double* A_k = A_i + cell_num_nodes;
    for (std::uint32_t k = i + 1; k < cell_num_nodes; ++k)
    {
      double factor = -A_k[i];
      for (std::uint32_t j = i; j < cell_num_nodes; ++j)
        A_k[j] += factor * A_i[j];
      psi[k] += factor * psi[i];
      A_k += cell_num_nodes;
    }
    A_i += cell_num_nodes;
  }

  // Back substitution
  for (std::int32_t j = cell_num_nodes - 2; j >= 0; --j)
  {
    double* A_j = sweep_matrix + j * cell_num_nodes;
    for (std::int32_t i = j + 1; i < cell_num_nodes; ++i)
      psi[j] -= A_j[i] * psi[i];
  }
}
*/

__device__ inline void
DeviceGaussElimination(double* A,
                       double* b,
                       const std::uint32_t n)
{
  // Forward elimination
  for (std::uint32_t i = 0; i < n - 1; ++i)
  {
    double bi = b[i];
    
    double pivot = A[i * n + i];

    double factor = 1.0 / pivot;

    for (std::uint32_t j = i + 1; j < n; ++j)
    {
      double val = A[j * n + i] * factor;
      
      b[j] -= val * bi;
      
      for (std::uint32_t k = i + 1; k < n; ++k)
      {
        A[j * n + k] -= val * A[i * n + k];
      }
    }
  }

  // Back substitution
  for (std::int32_t i = n - 1; i >= 0; --i)
  {
    double bi = b[i];
    
    for (std::uint32_t j = i + 1; j < n; ++j)
    {
      bi -= A[i * n + j] * b[j];
    }

    double pivot = A[i * n + i];

    b[i] = bi / pivot;
  }
}

__global__ void
SolveKernel(double* sweep_matrix,
            double* psi,
            const std::uint32_t cell_num_nodes)
{
  DeviceGaussElimination(sweep_matrix, psi, cell_num_nodes);
}

void
CBCSweepChunk::GPUSweep(AngleSet& angle_set)
{
	CALI_CXX_MARK_SCOPE("CBCSweepChunk::GPUSweep");

	const auto& m2d_op = groupset_.quadrature->GetMomentToDiscreteOperator();	// type is std::vector<std::vector<double>> const&
	const auto& d2m_op = groupset_.quadrature->GetDiscreteToMomentOperator(); // type is std::vector<std::vector<double>> const&

	DenseMatrix<double> Amat(max_num_cell_dofs_, max_num_cell_dofs_);
  DenseMatrix<double> Atemp(max_num_cell_dofs_, max_num_cell_dofs_);  
  std::vector<Vector<double>> b(gs_size_, Vector<double>(max_num_cell_dofs_));
  std::vector<double> source(max_num_cell_dofs_);

	// type is const std::vector<std::vector<FaceOrientation>>&
  const auto& face_orientations = angle_set.GetSPDS().GetCellFaceOrientations()[cell_local_id_];
  std::vector<double> face_mu_values(cell_num_faces_);

	// type is double
  const auto& rho = densities_[cell_local_id_];

	// type is const std::vector<double>&
  const auto& sigma_t = xs_.at(cell_->block_id)->GetSigmaTotal();

	// as = angle set
  // ss = subset
	const std::vector<std::uint32_t>& as_angle_indices = angle_set.GetAngleIndices();

  // Allocate pinned host memory for a single system
  // Storage<double> A_storage(max_num_cell_dofs_ * max_num_cell_dofs_);
  // Storage<double> b_storage(max_num_cell_dofs_);

  // crb::HostVector<double> Atemp_host(max_num_cell_dofs_ * max_num_cell_dofs_);
  // crb::HostVector<double> b_gsg_host(max_num_cell_dofs_);

  // crb::DeviceMemory<double> A_device(max_num_cell_dofs_ * max_num_cell_dofs_);
  // crb::DeviceMemory<double> b_device(max_num_cell_dofs_);

  double *Atemp_host, *b_gsg_host;
  cudaMallocManaged(&Atemp_host, max_num_cell_dofs_ * max_num_cell_dofs_ * sizeof(double));
  cudaMallocManaged(&b_gsg_host, max_num_cell_dofs_ * sizeof(double));

  for (size_t as_ss_idx = 0; as_ss_idx < num_angles_in_as_; ++as_ss_idx)
  {
    auto direction_num = as_angle_indices[as_ss_idx];
    auto omega = groupset_.quadrature->omegas[direction_num];
    auto wt = groupset_.quadrature->weights[direction_num];

    // Reset right-hand side
    for (size_t gsg = 0; gsg < gs_size_; ++gsg)
      for (size_t i = 0; i < cell_num_nodes_; ++i)
        b[gsg](i) = 0.0;

    for (size_t i = 0; i < cell_num_nodes_; ++i)
      for (size_t j = 0; j < cell_num_nodes_; ++j)
        Amat(i, j) = omega.Dot(G_(i, j));

    // Update face orientations
    for (size_t f = 0; f < cell_num_faces_; ++f)
      face_mu_values[f] = omega.Dot(cell_->faces[f].normal);

    // Surface integrals
    for (size_t f = 0; f < cell_num_faces_; ++f)
    {
      if (face_orientations[f] != FaceOrientation::INCOMING)
        continue;

      const auto& face = cell_->faces[f];
      const bool is_local_face = cell_transport_view_->IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const auto* face_nodal_mapping =
        &fluds_->GetCommonData().GetFaceNodalMapping(cell_local_id_, f);

      // For remote faces, get the pre-received data block
      const std::vector<double>* psi_nonlocal_upwnd_data_block = nullptr;
      if ((not is_local_face) and (not is_boundary_face))
      {
        psi_nonlocal_upwnd_data_block = &fluds_->GetNonLocalUpwindData(cell_->global_id, f);
      }

      // IntSf_mu_psi_Mij_dA
      const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping_->MapFaceNode(f, fi);

        for (size_t fj = 0; fj < num_face_nodes; ++fj)
        {
          const int j = cell_mapping_->MapFaceNode(f, fj);

          const double mu_Nij = -face_mu_values[f] * M_surf_[f](i, j);
          Amat(i, j) += mu_Nij;

          const double* psi_upwind_groups_ptr = nullptr;

          if (is_local_face)
          {
            const Cell* upwind_cell = cell_transport_view_->FaceNeighbor(f);
            const unsigned int adj_cell_node = face_nodal_mapping->cell_node_mapping_[fj];

            // Get base pointer to the start of upwind_cell's data block in local_psi_data_
            const double* psi_local_upwind_cell_base_ptr = fluds_->GetCellBlock(upwind_cell->local_id);

            // Calculate relative offset to the specific node and angle within that block
            const size_t addr_offset =
              adj_cell_node * group_angle_stride_ + as_ss_idx * group_stride_;

            psi_upwind_groups_ptr = psi_local_upwind_cell_base_ptr + addr_offset;
          }
          else if (not is_boundary_face)
          {
            assert(psi_nonlocal_upwnd_data_block != nullptr);
            const unsigned int adj_face_node_idx = face_nodal_mapping->face_node_mapping_[fj];
            psi_upwind_groups_ptr = fluds_->GetNonLocalUpwindPsi(
              *psi_nonlocal_upwnd_data_block, adj_face_node_idx, as_ss_idx);
          }
          else
            psi_upwind_groups_ptr = angle_set.PsiBoundary(face.neighbor_id,
                                                          direction_num,
                                                          cell_local_id_,
                                                          f,
                                                          fj,
                                                          gs_gi_,
                                                          surface_source_active_);

          if (psi_upwind_groups_ptr != nullptr)
            for (size_t gsg = 0; gsg < gs_size_; ++gsg)
              b[gsg](i) += psi_upwind_groups_ptr[gsg] * mu_Nij;
        } // for face node j
      } // for face node i
    } // for f

    // Looping over groups, assembling mass terms
    for (size_t gsg = 0; gsg < gs_size_; ++gsg)
    {
      double sigma_tg = rho * sigma_t[gs_gi_ + gsg];

      // Contribute source moments q = M_n^T * q_moms
      for (size_t i = 0; i < cell_num_nodes_; ++i)
      {
        double temp_src = 0.0;
        for (int m = 0; m < num_moments_; ++m)
        {
          const size_t ir = cell_transport_view_->MapDOF(i, m, gs_gi_ + gsg);
          temp_src += m2d_op[m][direction_num] * source_moments_[ir];
        }
        source[i] = temp_src;
      }

      // Mass matrix and source
      // Atemp = Amat + sigma_tgr * M
      // b += M * q
      for (size_t i = 0; i < cell_num_nodes_; ++i)
      {
        double temp = 0.0;
        for (size_t j = 0; j < cell_num_nodes_; ++j)
        {
          const double Mij = M_(i, j);
          Atemp(i, j) = Amat(i, j) + Mij * sigma_tg;
          temp += Mij * source[j];
        }
        b[gsg](i) += temp;
      }

      for (size_t i = 0; i < cell_num_nodes_; ++i)
      {
        b_gsg_host[i] = b[gsg](i);
        for (size_t j = 0; j < cell_num_nodes_; ++j)
        {
          Atemp_host[i * cell_num_nodes_ + j] = Atemp(i, j);
        }
      }

      // crb::copy(A_device, Atemp_host, max_num_cell_dofs_ * max_num_cell_dofs_);
      // crb::copy(b_device, b_gsg_host, max_num_cell_dofs_);

      // auto& A_data = A_storage.GetHostVector();
      // auto& b_data = b_storage.GetHostVector();

      // for (int i = 0; i < cell_num_nodes_; ++i)
      // {
      //   b_data[i] = b_gsg_host[i];
      //   for (int j = 0; j < cell_num_nodes_; ++j)
      //     A_data[i * cell_num_nodes_ + j] = Atemp_host[i * cell_num_nodes_ + j];
      // }

      // A_storage.Copy(Atemp_host.begin(), Atemp_host.end());
      // b_storage.Copy(b_gsg_host.begin(), b_gsg_host.end());

      // Launch kernel to solve the system
      SolveKernel<<<1, 1>>>(Atemp_host, 
                            b_gsg_host, 
                            static_cast<int>(cell_num_nodes_));

      cudaDeviceSynchronize();

      // Copy solution back to host
      // b_storage.CopyFromDevice();

      // Copy solution to b[gsg]
      // const auto& b_host = b_storage.GetHostVector();
      for (size_t i = 0; i < cell_num_nodes_; ++i)
        b[gsg](i) = b_gsg_host[i];

      // GaussElimination(Atemp, b[gsg], cell_num_nodes_);

      // Check that b_gsg_host and b[gsg] are the same
      // for (size_t i = 0; i < cell_num_nodes_; ++i)
      //   if (std::abs(b[gsg](i) - b_gsg_host[i]) > 1e-6)
      //     opensn::log.Log() << "Mismatch in GPU and CPU solutions: " 
      //                       << b[gsg](i) << " vs " << b_gsg_host[i] << "\n";
    } // for gsg

    // Ensure that if GPUs are used, get the kernel running in the constructor
    // IDEA: CellSweepKernel<<<1, gs_size>>>(args);

    // IDEA: Take care of updating phi on the device as well within the CellSweepKernel
    // Update phi
    for (int m = 0; m < num_moments_; ++m)
    {
      const double wn_d2m = d2m_op[m][direction_num];
      for (size_t i = 0; i < cell_num_nodes_; ++i)
      {
        const size_t ir = cell_transport_view_->MapDOF(i, m, gs_gi_);
        for (size_t gsg = 0; gsg < gs_size_; ++gsg)
          destination_phi_[ir + gsg] += wn_d2m * b[gsg](i);
      }
    }

    // IDEA: Deal with saving angular fluxes at a later point
    // If requested, save angular fluxes during sweep
    if (save_angular_flux_)
    {
      double* cell_psi_data_base_ptr =
        &destination_psi_[discretization_.MapDOFLocal(*cell_, 0, groupset_.psi_uk_man_, 0, 0)];

      for (size_t i = 0; i < cell_num_nodes_; ++i)
      {
        const size_t addr_offset =
          i * groupset_angle_group_stride_ + direction_num * groupset_group_stride_;

        for (size_t gsg = 0; gsg < gs_size_; ++gsg)
          cell_psi_data_base_ptr[addr_offset + gsg] = b[gsg](i);
      }
    }

    // Perform outgoing surface operations
    for (size_t f = 0; f < cell_num_faces_; ++f)
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
      std::vector<double>* psi_nonlocal_dnwnd_data_block_ptr = nullptr;

      if (not is_boundary_face and not is_local_face)
      {
        auto& async_comm = *angle_set.GetCommunicator();
        const size_t data_size_for_msg = num_face_nodes * group_angle_stride_;
        psi_nonlocal_dnwnd_data_block_ptr =
          &async_comm.InitGetDownwindMessageData(locality,
                                                 face.neighbor_id,
                                                 face_nodal_mapping.associated_face_,
                                                 angle_set.GetID(),
                                                 data_size_for_msg);
      }

      for (size_t fi = 0; fi < num_face_nodes; ++fi)
      {
        const int i = cell_mapping_->MapFaceNode(f, fi);

        // Tally outflow for particle balance
        if (is_boundary_face)
        {
          for (size_t gsg = 0; gsg < gs_size_; ++gsg)
            cell_transport_view_->AddOutflow(
              f, gs_gi_ + gsg, wt * face_mu_values[f] * b[gsg](i) * IntF_shapeI(i));
        }

        double* psi_downwind_groups_ptr = nullptr;

        if (is_local_face)
        {
          // Local downwind write
          double* psi_downwind_cell_base_ptr = fluds_->GetCellBlock(cell_->local_id);
          const size_t addr_offset = i * group_angle_stride_ + as_ss_idx * group_stride_;
          psi_downwind_groups_ptr = psi_downwind_cell_base_ptr + addr_offset;
        }
        else if (not is_boundary_face)
        {
          // Remote downwind write
          assert(psi_nonlocal_dnwnd_data_block_ptr != nullptr);
          const size_t addr_offset = fi * group_angle_stride_ + as_ss_idx * group_stride_;
          psi_downwind_groups_ptr = &(*psi_nonlocal_dnwnd_data_block_ptr)[addr_offset];
        }
        else if (is_reflecting_boundary_face)
          psi_downwind_groups_ptr =
            angle_set.PsiReflected(face.neighbor_id, direction_num, cell_local_id_, f, fi);

        // Write the solved angular flux to the determined location
        if (psi_downwind_groups_ptr != nullptr)
          for (size_t gsg = 0; gsg < gs_size_; ++gsg)
            psi_downwind_groups_ptr[gsg] = b[gsg](i);
      } // for fi
    } // for face
  } // for angleset/subset

  cudaFree(Atemp_host);
  cudaFree(b_gsg_host);
}
	
} // namespace opensn
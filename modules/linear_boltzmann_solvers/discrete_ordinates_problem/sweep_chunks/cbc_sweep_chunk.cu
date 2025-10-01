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

inline constexpr std::uint32_t cbc_matrix_size = cbc_max_dof * cbc_max_dof;

struct CBCSweepKernelArgs
{
  // Mesh and quadrature
  const char* mesh_data;
  const char* quad_data;

  // Source moments and phi
  const double* src_moment;
  double* phi;

  // Angle set
  const std::uint32_t* directions;
  std::uint32_t angleset_size;

  // Group set
  std::uint32_t num_groups;
  std::uint32_t groupset_start;
  std::uint32_t groupset_size;

  // Cell data
  std::uint32_t cell_local_id;
  std::uint32_t cell_num_faces;
  std::uint32_t max_num_cell_dofs;

  // Upwind psi
  const double* upwind_psi;
  std::uint32_t stride_as;
  std::uint32_t stride_f;
  std::uint32_t stride_fj;

  // Batch
  std::uint32_t batch_size;
};

__device__ inline void
ComputeGMS(std::array<double, cbc_matrix_size>& sweep_matrix,
           std::array<double, cbc_max_dof>& psi,
           const std::uint32_t& cell_num_nodes,
           DirectionView& direction,
           CellView& cell,
           const double* src_moment,
           const std::uint32_t& groupset_start,
           const std::uint32_t& group_idx,
           const std::uint32_t& num_groups,
           const std::uint32_t& num_moments)
{
  // Get sigma_t
  double sigma_t = cell.total_xs[groupset_start + group_idx];

  // Compute source term
  std::array<double, cbc_max_dof> s;
  s.fill(0.0);
  src_moment += cell.phi_address + groupset_start + group_idx;

  for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
  {
    double src_per_moment = 0.0;
    for (std::uint32_t m = 0; m < num_moments; ++m)
    {
      src_per_moment += direction.m2d[m] * (*src_moment);
      src_moment += num_groups;
    }
    s[i] = src_per_moment;
  }

  // Add source, transfer, and mass contributions
  double* A = sweep_matrix.data();
  const std::array<double, 4>* GM_data = 
    reinterpret_cast<const std::array<double, 4>*>(cell.GM_data);
  for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
  {
    for (std::uint32_t j = 0; j < cell_num_nodes; ++j)
    {
      std::array<double, 4> GM = *(GM_data++);
      
      // Compute A += G * omega + M * sigma_t
      A[j] += direction.omega[0] * GM[0] + direction.omega[1] * GM[1] + direction.omega[2] * GM[2] +
              sigma_t * GM[3];

      // Compute psi += M * s
      psi[i] += GM[3] * s[j];
    }
    A += cbc_max_dof;
  }
}

__device__ inline void
ComputeSurfaceIntegral(std::array<double, cbc_matrix_size>& sweep_matrix,
                       std::array<double, cbc_max_dof>& psi,
                       const CBCSweepKernelArgs& args,
                       const std::uint32_t& as_ss_idx,
                       const std::uint32_t& gsg,
                       DirectionView& direction,
                       CellView& cell)
{
  // Loop over each face
  for (std::uint32_t f = 0; f < cell.num_faces; ++f)
  {
    FaceView face;
    cell.GetFaceView(face, f);

    // Compute mu = omega . n
    const double mu = direction.omega[0] * face.normal[0] + 
                      direction.omega[1] * face.normal[1] + 
                      direction.omega[2] * face.normal[2];

    // Skip if not incoming face
    if (mu >= 0.0)
      continue;
    
    // Compute surface integral
    for (std::uint32_t fi = 0; fi < face.num_face_nodes; ++fi)
    {
      std::uint32_t i = face.cell_mapping_data[fi];
      double* Ai = sweep_matrix.data() + i * cbc_max_dof;
      for (std::uint32_t fj = 0; fj < face.num_face_nodes; ++fj)
      {
        std::uint32_t j = face.cell_mapping_data[fj];
        double mu_Nij = -mu * face.M_surf_data[fi * face.num_face_nodes + fj];
        Ai[j] += mu_Nij;
        
        const size_t offset = as_ss_idx * args.stride_as + 
                              f * args.stride_f + 
                              fj * args.stride_fj + 
                              gsg;
        const double upwind_psi_val = args.upwind_psi[offset];

        psi[i] += upwind_psi_val * mu_Nij;
      }
    }
  }
}

__device__ inline void
DeviceGaussianElimination(std::array<double, cbc_matrix_size>& sweep_matrix,
                          std::array<double, cbc_max_dof>& psi,
                          const std::uint32_t& cell_num_nodes)
{
  // Forward elimination
  double* A_i = sweep_matrix.data();
  for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
  {
    double inv_diag = 1.0 / A_i[i];

    // Normalize the pivot row
    for (std::uint32_t j = i; j < cell_num_nodes; ++j)
      A_i[j] *= inv_diag;

    psi[i] *= inv_diag;

    // Eliminate rows below
    double* A_k = A_i + cbc_max_dof;

    for (std::uint32_t k = i + 1; k < cell_num_nodes; ++k)
    {
      double factor = -A_k[i];
      for (std::uint32_t j = i; j < cell_num_nodes; ++j)
        A_k[j] += factor * A_i[j];
      psi[k] += factor * psi[i];
      A_k += cbc_max_dof;
    }
    A_i += cbc_max_dof;
  }

  // Back-substitution - row-wise access
  for (std::int32_t j = (cell_num_nodes - 2); j >= 0; --j)
  {
    double* A_j = sweep_matrix.data() + j * cbc_max_dof;
    for (std::int32_t i = (j + 1); i < cell_num_nodes; ++i)
      psi[j] -= A_j[i] * psi[i];
  }
}

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
        A[j * n + k] -= val * A[i * n + k];
    }
  }

  // Back substitution
  for (std::int32_t i = n - 1; i >= 0; --i)
  {
    double bi = b[i];
    for (std::uint32_t j = i + 1; j < n; ++j)
      bi -= A[i * n + j] * b[j];
    double pivot = A[i * n + i];
    b[i] = bi / pivot;
  }
}

__device__ void
DeviceComputePhi(const std::array<double, cbc_max_dof>& psi,
           const std::uint32_t& cell_num_nodes,
           DirectionView& direction,
           CellView& cell,
           double* phi,
           const std::uint32_t& groupset_start,
           const std::uint32_t& group_idx,
           const std::uint32_t& num_groups,
           const std::uint32_t& num_moments)
{
  phi += cell.phi_address + groupset_start + group_idx;
  for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
  {
    for (std::uint32_t m = 0; m < num_moments; ++m)
    {
      crb::atomic_add(phi, direction.d2m[m] * psi[i]);
      phi += num_groups;
    }
  }
}

__device__ inline void
DeviceRecordPsi(const std::array<double, cbc_max_dof>& psi,
                const std::uint32_t& cell_num_nodes,
                double* psi_device,
                const std::uint32_t& as_ss_idx,
                const std::uint32_t& gsg,
                const CBCSweepKernelArgs& args)
{
  // Compute stride into flat buffer
  const size_t offset = (as_ss_idx * args.groupset_size + gsg) * args.max_num_cell_dofs;
  double* dest_psi = psi_device + offset;

  // Record the flux
  for (std::uint32_t i = 0; i < args.max_num_cell_dofs; ++i)
    dest_psi[i] = 0.0;

  for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
    dest_psi[i] = psi[i];
}

__global__ void
SolveKernel(double* sweep_matrix,
            double* psi,
            const std::uint32_t cell_num_nodes)
{
  DeviceGaussElimination(sweep_matrix, psi, cell_num_nodes);
}

__global__ void
CBCSweepKernel(CBCSweepKernelArgs args, double* psi_device)
{
  std::uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= args.batch_size)
    return;
  
  const std::uint32_t as_ss_idx = thread_idx / args.groupset_size;
  const std::uint32_t gsg = thread_idx % args.groupset_size;

  // Get corresponding cell
  CellView cell;
  MeshView(args.mesh_data).GetCellView(cell, args.cell_local_id);

  // Get corresponding direction and number of moments
  std::uint32_t num_moments;
  std::uint32_t direction_num = args.directions[as_ss_idx];
  DirectionView direction;
  {
    QuadratureView quadrature(args.quad_data);
    num_moments = quadrature.num_moments;
    quadrature.GetDirectionView(direction, direction_num);
  }

  // Initialize psi
  std::array<double, cbc_max_dof> psi;
  psi.fill(0.0);
  
  // Gaussian elimination
  {
    // Initialize sweep matrix
    std::array<double, cbc_matrix_size> sweep_matrix;
    sweep_matrix.fill(0.0);

    // Prepare the linear system
    ComputeGMS(sweep_matrix,
               psi,
               cell.num_nodes,
               direction,
               cell,
               args.src_moment,
               args.groupset_start,
               gsg,
               args.num_groups,
               num_moments);

    ComputeSurfaceIntegral(sweep_matrix,
                           psi,
                           args,
                           as_ss_idx,
                           gsg,
                           direction,
                           cell);

    // Solve
    DeviceGaussianElimination(sweep_matrix, psi, cell.num_nodes);
  }
  // Update scalar flux
  DeviceComputePhi(psi,
                   cell.num_nodes,
                   direction,
                   cell,
                   args.phi,
                   args.groupset_start,
                   gsg,
                   args.num_groups,
                   num_moments);

  // Post-processing
  if (psi_device)
    DeviceRecordPsi(psi, cell.num_nodes, psi_device, as_ss_idx, gsg, args);
}

void
CBCSweepChunk::GPUSweep(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunk::GPUSweep");

  // These operators are handled by QuadratureCarrier
	const auto& m2d_op = groupset_.quadrature->GetMomentToDiscreteOperator();	// type is std::vector<std::vector<double>> const&
	const auto& d2m_op = groupset_.quadrature->GetDiscreteToMomentOperator(); // type is std::vector<std::vector<double>> const&

	DenseMatrix<double> Amat(max_num_cell_dofs_, max_num_cell_dofs_);
  DenseMatrix<double> Atemp(max_num_cell_dofs_, max_num_cell_dofs_);  
  std::vector<Vector<double>> b(gs_size_, Vector<double>(max_num_cell_dofs_));

  // Use MemoryPinner 
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

  // ------------------------------------------------------------------------------
  CBCSweepKernelArgs args;

  // Get mesh and quadrature data
  MeshCarrier* mesh = reinterpret_cast<MeshCarrier*>(problem_.GetCarrier(2));
  args.mesh_data = mesh->GetDevicePtr();

  QuadratureCarrier* quadrature = reinterpret_cast<QuadratureCarrier*>(groupset_.quad_carrier);
  args.quad_data = quadrature->GetDevicePtr();

  // Copy source moment and destination phi data to GPU
  MemoryPinner<double>* src = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(0));
  src->CopyToDevice();
  args.src_moment = src->GetDevicePtr();

  MemoryPinner<double>* phi = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(1));
  phi->CopyToDevice();
  args.phi = phi->GetDevicePtr();

  // Copy angleset data to GPU
  MemoryPinner<std::uint32_t>* directions = reinterpret_cast<MemoryPinner<std::uint32_t>*>(angle_set.GetMemoryPin());
  args.directions = directions->GetDevicePtr();
  std::size_t angleset_size = angle_set.GetNumAngles();
  args.angleset_size = angleset_size;

  // Copy groupset data to GPU
  std::size_t groupset_size = groupset_.groups.size();
  int groupset_start = groupset_.groups.front().id;
  std::size_t num_groups = problem_.GetGroups().size();
  args.groupset_start = groupset_start;
  args.groupset_size = groupset_size;
  args.num_groups = num_groups;

  // TODO: For the time being, worry about saving angular fluxes later
  // Allocate data for angular fluxes if required
  // Storage<double> psi_storage;
  // double* psi_device = nullptr;
  // if (save_angular_flux_)
  // {
  //   psi_storage = Storage<double>(angleset_size * groupset_size * max_num_cell_dofs_);
  //   psi_device = psi_storage.GetDevicePtr();
  // }

  Storage<double> psi_storage(angleset_size * groupset_size * max_num_cell_dofs_);
  double* psi_device = psi_storage.GetDevicePtr();

  args.cell_local_id = cell_local_id_;
  args.cell_num_faces = cell_num_faces_;
  args.max_num_cell_dofs = max_num_cell_dofs_;

  std::uint32_t batch_size = angleset_size * groupset_size;
  args.batch_size = batch_size;

  // Prepare upwind psi values
  size_t max_num_face_nodes = 0;
  for (size_t f = 0; f < cell_num_faces_; ++f)
  {
    const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);
    max_num_face_nodes = std::max(max_num_face_nodes, num_face_nodes);
  }

  const auto upwind_psi_size = angleset_size * cell_num_faces_ * max_num_face_nodes * gs_size_;
  crb::HostVector<double> upwind_psi_host(upwind_psi_size);
  crb::DeviceMemory<double> upwind_psi_device(upwind_psi_size);

  const size_t stride_g = 1;
  const size_t stride_fj = gs_size_ * stride_g;
  const size_t stride_f  = max_num_face_nodes * stride_fj;
  const size_t stride_as = cell_num_faces_ * stride_f;

  for (size_t as_ss_idx = 0; as_ss_idx < num_angles_in_as_; ++as_ss_idx)
  {
    auto direction_num = as_angle_indices[as_ss_idx];

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

      const size_t num_face_nodes = cell_mapping_->GetNumFaceNodes(f);

      for (size_t fj = 0; fj < num_face_nodes; ++fj)
      {
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

        // Calculate desination address in host upwind psi buffer
        const size_t dest_offset = as_ss_idx * stride_as + f * stride_f + fj * stride_fj;
        double* dest_ptr = &upwind_psi_host[dest_offset];

        if (psi_upwind_groups_ptr != nullptr)
            std::copy(psi_upwind_groups_ptr, psi_upwind_groups_ptr + gs_size_, dest_ptr);
      } // for face node j
    } // for f
  }

  crb::copy(upwind_psi_device, upwind_psi_host, upwind_psi_size);
  args.upwind_psi = upwind_psi_device.get();
  args.stride_as = stride_as;
  args.stride_f  = stride_f;
  args.stride_fj = stride_fj;

  // Launch the sweep kernel
  std::uint32_t threads_per_block = 128;
  std::uint32_t num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
  CBCSweepKernel<<<num_blocks, threads_per_block>>>(args, psi_device);

  // Copy phi back to CPU
  phi->CopyFromDevice();

  // Retrieve outflow 
  // OutflowCarrier* outflow = reinterpret_cast<OutflowCarrier*>(problem_.GetCarrier(1));
  // outflow->AccumulateBack(cell_transport_views_);
  // outflow->Reset();

  if (psi_device)
  {
    psi_storage.CopyFromDevice();
    crb::HostVector<double>& psi_host = psi_storage.GetHostVector();

    for (size_t as_ss_idx = 0; as_ss_idx < num_angles_in_as_; ++as_ss_idx)
    {
      auto direction_num = as_angle_indices[as_ss_idx];
      auto omega = groupset_.quadrature->omegas[direction_num];
      auto wt = groupset_.quadrature->weights[direction_num];

      // Update face orientations
      for (size_t f = 0; f < cell_num_faces_; ++f)
        face_mu_values[f] = omega.Dot(cell_->faces[f].normal);

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
          double* psi_downwind_groups_ptr = nullptr;

          if (is_local_face)
          {
            double* psi_downwind_cell_base_ptr = fluds_->GetCellBlock(cell_->local_id);
            const size_t addr_offset = i * group_angle_stride_ + as_ss_idx * group_stride_;
            psi_downwind_groups_ptr = psi_downwind_cell_base_ptr + addr_offset;
          }
          else if (not is_boundary_face)
          {
            assert(psi_nonlocal_dnwnd_data_block_ptr != nullptr);
            const size_t addr_offset = fi * group_angle_stride_ + as_ss_idx * group_stride_;
            psi_downwind_groups_ptr = &(*psi_nonlocal_dnwnd_data_block_ptr)[addr_offset];
          }
          else if (is_reflecting_boundary_face)
            psi_downwind_groups_ptr =
              angle_set.PsiReflected(face.neighbor_id, direction_num, cell_local_id_, f, fi);

          // Write the solved angular flux to the determined location
          if (psi_downwind_groups_ptr != nullptr)
          {
            for (size_t gsg = 0; gsg < gs_size_; ++gsg)
            {
              const size_t offset = (as_ss_idx * gs_size_ + gsg) * max_num_cell_dofs_;
              psi_downwind_groups_ptr[gsg] = psi_host[offset + i];
            }
          }

          // Tally outflow for particle balance on non-reflecting boundaries
          if (is_boundary_face && (not is_reflecting_boundary_face))
          {
            for (size_t gsg = 0; gsg < gs_size_; ++gsg)
            {
              const size_t offset = (as_ss_idx * gs_size_ + gsg) * max_num_cell_dofs_;
              cell_transport_view_->AddOutflow(f,
                                               gs_gi_ + gsg,
                                               wt * face_mu_values[f] * psi_host[offset + i] *
                                               IntF_shapeI(i));
            }
          }
        }
      }
    }
  }
  // ------------------------------------------------------------------------------
}

/*
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

      // Launch kernel to solve the system
      SolveKernel<<<1, 1>>>(Atemp_host, 
                            b_gsg_host, 
                            static_cast<int>(cell_num_nodes_));

      cudaDeviceSynchronize();

      // Copy solution to b[gsg]
      for (size_t i = 0; i < cell_num_nodes_; ++i)
        b[gsg](i) = b_gsg_host[i];
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
*/
} // namespace opensn
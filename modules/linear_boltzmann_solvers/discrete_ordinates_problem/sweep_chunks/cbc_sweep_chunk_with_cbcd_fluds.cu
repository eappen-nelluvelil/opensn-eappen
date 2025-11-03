#include "cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
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

/// @brief Index for each thread.
struct Index
{
  /// @brief Constructor
  __device__ inline Index() {}

  __device__ inline void Compute(std::uint32_t thread_idx,
                                 const std::uint32_t& angleset_size,
                                 const std::uint32_t& groupset_size)
  {
    const auto angle_group_stride = angleset_size * groupset_size;
    cell_idx = thread_idx / angle_group_stride;
    angle_idx = (thread_idx % angle_group_stride) / groupset_size;
    group_idx = (thread_idx % angle_group_stride) % groupset_size;
  }

  /// @brief Index of the cell associated to the current thread in the current level vector.
  std::uint32_t cell_idx;
  /// @brief Index of the angle associated to the current thread in the current angleset.
  std::uint32_t angle_idx;
  /// @brief Index of the group associated to the current thread in the current groupset.
  std::uint32_t group_idx;
};

struct CBCSweepKernelArgs_WITH_CBCD_FLUDS
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

  const uint64_t* cell_local_ids;
  const int* cell_face_offset_map;
  std::uint32_t num_cells;

  // Local angular flux buffer
  double* local_psi_data;
  const size_t* cell_dof_map;
  size_t num_angles_in_as;
  size_t num_groups_and_angles;
  const uint64_t* face_neighbor_local_ids;
  const unsigned int* face_neighbor_cell_node_map;

  // Boundary angular flux buffers
  const double* boundary_psi_data;
  const int* boundary_psi_map;
  const int* cell_to_local_face_offset_map;

  const int* incoming_face_category_map;
  const int* outgoing_face_category_map;

  // Upwind/downwind psi buffers
  const double* upwind_psi_data;
  const int* upwind_psi_offsets;
  size_t upwind_face_stride;

  double* downwind_psi_data;
  const int* downwind_psi_offsets;
  size_t downwind_face_stride;

  // Batch
  std::uint32_t batch_size;
  bool save_angular_flux;
  double* destination_psi;
  size_t groupset_angle_group_stride;
  size_t groupset_group_stride;
};

__device__ inline void
  ComputeGMS_WITH_CBCD_FLUDS(std::array<double, cbc_matrix_size>& sweep_matrix,
                           std::array<double, cbc_max_dof>& psi,
                           const std::uint32_t& cell_num_nodes,
                           DirectionView& direction,
                           CellView& cell,
                           const double* __restrict src_moment,
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
  ComputeSurfaceIntegral_WITH_CBCD_FLUDS(std::array<double, cbc_matrix_size>& sweep_matrix,
                                       std::array<double, cbc_max_dof>& psi,
                                       const CBCSweepKernelArgs_WITH_CBCD_FLUDS& args,
                                       const std::uint32_t cell_idx,
                                       const std::uint32_t angle_idx,
                                       const std::uint32_t group_idx,
                                       DirectionView& direction,
                                       CellView& cell)
{
  const size_t cell_local_id = args.cell_local_ids[cell_idx];
  const size_t cell_face_start_idx = args.cell_to_local_face_offset_map[cell_local_id];
  const int face_offset = args.cell_face_offset_map[cell_idx];

  // Loop over each face
  for (std::uint32_t f = 0; f < cell.num_faces; ++f)
  {
    FaceView face;
    cell.GetFaceView(face, f);

    // Compute mu = omega . n
    const double mu = direction.omega[0] * face.normal[0] + direction.omega[1] * face.normal[1] +
                      direction.omega[2] * face.normal[2];

    // Skip if not incoming face
    if (mu >= 0.0)
      continue;

    // const int offset = args.upwind_psi_offsets[current_face_offset];
    const int incoming_face_category = args.incoming_face_category_map[cell_face_start_idx + f];

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

        // Differentiate between non-local/boundary upwind fluxes
        // If the offset is negative, the face is a local face or boundary face,
        // and we compute the index into the on-device local psi buffer or incoming
        // boundary psi buffer
        // Otherwise, the face is a non-local or boundary face, and we index into
        // upwind_psi_data buffer

        const double* psi_in_ptr;
        if (incoming_face_category >= 0) // Non-local face
        {
          const int current_face_offset = face_offset + f;
          const int offset = args.upwind_psi_offsets[current_face_offset];
          const size_t upwind_offset =
            offset + fj * args.upwind_face_stride + angle_idx * args.groupset_size + group_idx;
          psi_in_ptr = &args.upwind_psi_data[upwind_offset];
        }
        else if (incoming_face_category == -1) // Local face
        {
          const int local_face_offset = args.cell_to_local_face_offset_map[cell_local_id] + f;
          const size_t nbr_cell_local_id = args.face_neighbor_local_ids[local_face_offset];
          const size_t nbr_cell_data_start_idx = args.cell_dof_map[nbr_cell_local_id];
          const size_t nbr_node_map_offset = local_face_offset * cbc_max_face_dofs;
          const size_t adj_cell_node = args.face_neighbor_cell_node_map[nbr_node_map_offset + fj];
          const size_t addr_offset =
            adj_cell_node * args.num_groups_and_angles + angle_idx * args.groupset_size + group_idx;
          const size_t nbr_cell_data_idx = nbr_cell_data_start_idx + addr_offset;
          psi_in_ptr = &args.local_psi_data[nbr_cell_data_idx];
        }
        else if (incoming_face_category == -2)  // Boundary face
        {
          const int local_face_offset = args.cell_to_local_face_offset_map[cell_local_id];
          const size_t boundary_psi_map_idx =
            local_face_offset + f; // Map from cell face to boundary psi index
          const int boundary_psi_data_idx = args.boundary_psi_map[boundary_psi_map_idx];
          psi_in_ptr = &args.boundary_psi_data[boundary_psi_data_idx +
                                              fj * args.num_groups_and_angles +
                                              angle_idx * args.groupset_size + group_idx];
        }

        psi[i] += (*psi_in_ptr) * mu_Nij;
      }
    }
  }
}

__device__ inline void
  DeviceGaussianElimination_WITH_CBCD_FLUDS(std::array<double, cbc_matrix_size>& sweep_matrix,
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
DeviceRecordDownwindPsiAndOutflow_WITH_CBCD_FLUDS(const std::array<double, cbc_max_dof>& psi,
                                                  CellView& cell,
                                                  const CBCSweepKernelArgs_WITH_CBCD_FLUDS& args,
                                                  const std::uint32_t cell_idx,
                                                  const std::uint32_t angle_idx,
                                                  const std::uint32_t group_idx,
                                                  DirectionView& direction)
{
  const size_t cur_cell_local_id = args.cell_local_ids[cell_idx];
  const size_t cell_face_start_idx = args.cell_to_local_face_offset_map[cur_cell_local_id];
  const size_t cur_cell_data_start_idx = args.cell_dof_map[cur_cell_local_id];

  const int face_offset_base = args.cell_face_offset_map[cell_idx];

  for (std::uint32_t f = 0; f < cell.num_faces; ++f)
  {
    FaceView face;
    cell.GetFaceView(face, f);
    double mu = direction.omega[0] * face.normal[0] + direction.omega[1] * face.normal[1] +
                direction.omega[2] * face.normal[2];

    if (mu <= 0.0)
      continue;

    const int outgoing_face_category = args.outgoing_face_category_map[cell_face_start_idx + f];

    // Outgoing face

    // Differentiate between local and non-local/reflecting boundary downwind outgoing fluxes
    // If the offset is negative, the face is a local face, and we write the fluxes
    // directly to the on-device local_psi_data buffer
    // Otherwise, the face is a non-local or reflecting boundary face, and we write to the
    // downwind_psi_data buffer for a device-to-host transfer

    for (std::uint32_t fi = 0; fi < face.num_face_nodes; ++fi)
    {
      const int i = face.cell_mapping_data[fi];
      // if (buffer_offset >= 0)
      if (outgoing_face_category >= 0) // Non-local or reflecting boundary face
      {
        const int buffer_offset = args.downwind_psi_offsets[face_offset_base + f];
        const size_t downwind_offset = buffer_offset + fi * args.downwind_face_stride +
                                       angle_idx * args.groupset_size + group_idx;
        args.downwind_psi_data[downwind_offset] = psi[i];
      }
      else
      {
        const size_t addr_offset =
          i * args.num_groups_and_angles + angle_idx * args.groupset_size + group_idx;
        const size_t cur_cell_data_idx = cur_cell_data_start_idx + addr_offset;
        args.local_psi_data[cur_cell_data_idx] = psi[i];
      }

      // Tally outflow for boundary faces
      if (face.outflow != nullptr)
      {
        double outflow = direction.weight * mu * face.IntS_shapeI_data[fi] * psi[i];
        crb::atomic_add(face.outflow + group_idx, outflow);
      }
    }
  }
}

__device__ void
DeviceComputePhi_WITH_CBCD_FLUDS(const std::array<double, cbc_max_dof>& psi,
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

__global__ void
CBCSweepKernel_WITH_CBCD_FLUDS(CBCSweepKernelArgs_WITH_CBCD_FLUDS args)
{
  // Compute index (cell, angle, group) from thread flattened index
  Index idx;
  {
    std::uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= args.batch_size)
      return;
    idx.Compute(thread_idx, args.angleset_size, args.groupset_size);
  }

  // Get corresponding cell
  const uint64_t cell_local_id = args.cell_local_ids[idx.cell_idx];
  CellView cell;
  MeshView(args.mesh_data).GetCellView(cell, cell_local_id);

  // Get corresponding direction and number of moments
  std::uint32_t num_moments;
  std::uint32_t direction_num = args.directions[idx.angle_idx];
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
    ComputeGMS_WITH_CBCD_FLUDS(sweep_matrix,
                               psi,
                               cell.num_nodes,
                               direction,
                               cell,
                               args.src_moment,
                               args.groupset_start,
                               idx.group_idx,
                               args.num_groups,
                               num_moments);

    ComputeSurfaceIntegral_WITH_CBCD_FLUDS(
      sweep_matrix, psi, args, idx.cell_idx, idx.angle_idx, idx.group_idx, direction, cell);

    // Solve
    DeviceGaussianElimination_WITH_CBCD_FLUDS(sweep_matrix, psi, cell.num_nodes);
  }
  // Update scalar flux
  DeviceComputePhi_WITH_CBCD_FLUDS(psi,
                                   cell.num_nodes,
                                   direction,
                                   cell,
                                   args.phi,
                                   args.groupset_start,
                                   idx.group_idx,
                                   args.num_groups,
                                   num_moments);

  DeviceRecordDownwindPsiAndOutflow_WITH_CBCD_FLUDS(
    psi, cell, args, idx.cell_idx, idx.angle_idx, idx.group_idx, direction);
}

void
CBCSweepChunk::GPUSweep_With_CBCD_FLUDS(AngleSet& angle_set)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunk::GPUSweep_With_CBCD_FLUDS");

  // Determine sizes for host and device vectors
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  auto& cbc_fluds = dynamic_cast<CBC_FLUDS&>(*fluds_);
  auto& cbcd_fluds = *static_cast<CBCD_FLUDS*>(cbc_fluds.Get_CBCD_FLUDS_Ptr());
  const auto& as_angle_indices = cbc_angle_set.GetAngleIndices();
  const auto num_angles_in_as = as_angle_indices.size();
  const auto gs_size = groupset_.groups.size();
  const auto gs_gi = groupset_.groups.front().id;

  // Set up kernel arguments
  CBCSweepKernelArgs_WITH_CBCD_FLUDS args;
  
  MeshCarrier* mesh = reinterpret_cast<MeshCarrier*>(problem_.GetCarrier(2));
  args.mesh_data = mesh->GetDevicePtr();

  QuadratureCarrier* quadrature = reinterpret_cast<QuadratureCarrier*>(groupset_.quad_carrier);
  args.quad_data = quadrature->GetDevicePtr();

  // Copy source moment and destination phi data to device
  MemoryPinner<double>* src = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(0));
  src->CopyToDevice();
  args.src_moment = src->GetDevicePtr();

  MemoryPinner<double>* phi = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(1));
  phi->CopyToDevice();
  args.phi = phi->GetDevicePtr();

  // Allocate data for saving angular fluxes if needed
  // Storage<double> psi_storage;
  // double* psi_device_ptr = nullptr;

  // if (save_angular_flux_)
  // {
  //   auto* psi_pinner = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(2));
  //   psi_pinner->CopyToDevice();
  //   args.destination_psi = psi_pinner->GetDevicePtr();
  // }
  // else
  //   args.destination_psi = nullptr;

  args.destination_psi = nullptr;

  // Copy angleset data to device
  MemoryPinner<std::uint32_t>* directions =
    reinterpret_cast<MemoryPinner<std::uint32_t>*>(angle_set.GetMemoryPin());
  args.directions = directions->GetDevicePtr();
  args.angleset_size = num_angles_in_as;
  args.groupset_size = gs_size;
  args.groupset_start = gs_gi;
  args.num_groups = problem_.GetGroups().size();
  args.groupset_angle_group_stride = groupset_angle_group_stride_;
  args.groupset_group_stride = groupset_group_stride_;
  args.save_angular_flux = save_angular_flux_;

  args.local_psi_data = cbcd_fluds.GetDevicePtr();
  args.cell_dof_map = cbcd_fluds.GetCellDOFMapDevicePtr();
  args.num_angles_in_as = num_angles_in_as;
  args.num_groups_and_angles = num_angles_in_as * gs_size;

  args.face_neighbor_cell_node_map = cbcd_fluds.face_neighbor_cell_node_map_storage_.GetDevicePtr();
  args.face_neighbor_local_ids = cbcd_fluds.face_neighbor_local_ids_storage_.GetDevicePtr();

  // Boundary angular flux buffers
  args.boundary_psi_data = cbcd_fluds.GetBoundaryPsiDevicePtr();
  args.boundary_psi_map = cbcd_fluds.GetBoundaryPsiMapDevicePtr();
  args.cell_to_local_face_offset_map = cbcd_fluds.cell_to_local_face_offset_storage_.GetDevicePtr();

  // Incoming/outgoing face category maps
  args.incoming_face_category_map = cbcd_fluds.incoming_face_category_map_storage_.GetDevicePtr();
  args.outgoing_face_category_map = cbcd_fluds.outgoing_face_category_map_storage_.GetDevicePtr();

  // Determine sizes of buffers
  std::vector<uint64_t> cell_local_ids(tasks_to_execute_.size());

  size_t cells_with_incoming_local_and_boundary_faces = 0;
  size_t cells_with_incoming_non_local_faces = 0;

  size_t total_faces = 0;
  size_t total_upwind_buffer_size = 0;
  size_t total_downwind_buffer_size = 0;

  for (int idx = 0; idx < tasks_to_execute_.size(); ++idx)
  {
    const auto& cell = *tasks_to_execute_[idx]->cell_ptr;
    cell_local_ids[idx] = cell.local_id;
    total_faces += cell.faces.size();
    const auto& face_orientations = cbc_angle_set.GetSPDS().GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = discretization_.GetCellMapping(cell);
    auto& cell_transport_view = cell_transport_views_[cell.local_id];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_angles_in_as * gs_size;
      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not cell.faces[f].has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face) and (cbc_angle_set.GetBoundaries().at(cell.faces[f].neighbor_id)->IsReflecting());

      // Size upwind/downwind buffers for only non-local faces
      if ((face_orientations[f] == FaceOrientation::INCOMING))
      {
        if ((not is_local_face) and (not is_boundary_face))
        {
          ++cells_with_incoming_non_local_faces;
          total_upwind_buffer_size += face_data_size;
        }
        else
          ++cells_with_incoming_local_and_boundary_faces;
      }
      else if (face_orientations[f] == FaceOrientation::OUTGOING)
      {
        if (((not is_local_face) and (not is_boundary_face)) or 
            (is_reflecting_boundary_face))
          total_downwind_buffer_size += face_data_size;
      }
    }
  }

  // Prepare angular flux buffers for H2D transfer
  std::vector<double> upwind_psi_buffer(total_upwind_buffer_size);
  std::vector<int> upwind_psi_offsets(total_faces, -1);

  std::vector<double> downwind_psi_buffer(total_downwind_buffer_size);
  std::vector<int> downwind_psi_offsets(total_faces, -1);

  std::vector<int> cell_face_offset_map(tasks_to_execute_.size());

  size_t face_offset_stride = 0;
  size_t upwind_buffer_offset = 0;
  size_t downwind_buffer_offset = 0;

  for (int idx = 0; idx < tasks_to_execute_.size(); ++idx)
  {
    const auto& cell = *tasks_to_execute_[idx]->cell_ptr;
    cell_face_offset_map[idx] = face_offset_stride;

    const auto& cell_mapping = discretization_.GetCellMapping(cell);
    const auto& face_orientations = angle_set.GetSPDS().GetCellFaceOrientations()[cell.local_id];
    const auto& cell_transport_view = cell_transport_views_[cell.local_id];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t face_data_size = num_face_nodes * num_angles_in_as * gs_size;
      const auto* face_nodal_mapping =
        &fluds_->GetCommonData().GetFaceNodalMapping(cell.local_id, f);

      const bool is_local_face = cell_transport_view.IsFaceLocal(f);
      const bool is_boundary_face = not face.has_neighbor;
      const bool is_reflecting_boundary_face =
        (is_boundary_face) and (cbc_angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting());

      const size_t current_face_offset = face_offset_stride + f;

      // Upwind data packing
      if (face_orientations[f] == FaceOrientation::INCOMING)
      {
        if (is_local_face)
          upwind_psi_offsets[current_face_offset] = -1;
        else if (not is_boundary_face)
        {
          upwind_psi_offsets[current_face_offset] = upwind_buffer_offset;

          for (size_t fj = 0; fj < num_face_nodes; ++fj)
          {
            const double* psi_in = nullptr;
            const unsigned int adj_face_node = face_nodal_mapping->face_node_mapping_[fj];

            for (size_t as_ss_idx = 0; as_ss_idx < num_angles_in_as; ++as_ss_idx)
            {
              psi_in = cbc_fluds.NLUpwindPsi(cell.global_id, f, adj_face_node, as_ss_idx);
              const size_t offset = fj * (num_angles_in_as * gs_size) + as_ss_idx * (gs_size);
              double* buffer_ptr = &upwind_psi_buffer[upwind_buffer_offset + offset];

              if (psi_in)
                std::copy(psi_in, psi_in + gs_size, buffer_ptr);
            }
          }
          upwind_buffer_offset += face_data_size;
        }
        else
          upwind_psi_offsets[current_face_offset] = -2;
      }
      // Downwind data packing
      else if (face_orientations[f] == FaceOrientation::OUTGOING)
      {
        if (is_local_face)
          downwind_psi_offsets[current_face_offset] = -1;
        else if ((not is_boundary_face) or (is_reflecting_boundary_face))
        {
          downwind_psi_offsets[current_face_offset] = downwind_buffer_offset;
          downwind_buffer_offset += face_data_size;
        }
      }
    }
    face_offset_stride += cell.faces.size();
  }

  cbcd_fluds.cell_id_storage_.Copy(cell_local_ids.begin(), cell_local_ids.end());
  args.cell_local_ids = cbcd_fluds.cell_id_storage_.GetDevicePtr();
  args.num_cells = cell_local_ids.size();

  cbcd_fluds.cell_face_offset_storage_.Copy(cell_face_offset_map.begin(), cell_face_offset_map.end());
  args.cell_face_offset_map = cbcd_fluds.cell_face_offset_storage_.GetDevicePtr();

  // Transfer non-local incoming upwind buffer to device
  cbcd_fluds.non_local_upwind_psi_buffer_storage_.Copy(
    upwind_psi_buffer.begin(), upwind_psi_buffer.end());
  args.upwind_psi_data = cbcd_fluds.non_local_upwind_psi_buffer_storage_.GetDevicePtr();
  args.upwind_face_stride = num_angles_in_as * gs_size;

  Storage<int> upwind_offset_storage(upwind_psi_offsets.size());
  upwind_offset_storage.Copy(upwind_psi_offsets.begin(), upwind_psi_offsets.end());
  args.upwind_psi_offsets = upwind_offset_storage.GetDevicePtr();

  // Transfer non-local and reflecting outgoing downwind buffer to device
  cbcd_fluds.non_local_and_reflecting_psi_buffer_storage_.Copy(
    downwind_psi_buffer.begin(), downwind_psi_buffer.end());
  args.downwind_psi_data = cbcd_fluds.non_local_and_reflecting_psi_buffer_storage_.GetDevicePtr();
  args.downwind_face_stride = num_angles_in_as * gs_size;

  Storage<int> downwind_offset_storage(downwind_psi_offsets.size());
  downwind_offset_storage.Copy(downwind_psi_offsets.begin(), downwind_psi_offsets.end());
  args.downwind_psi_offsets = downwind_offset_storage.GetDevicePtr();

  args.batch_size = args.num_cells * args.angleset_size * args.groupset_size;
  std::uint32_t threads_per_block = 128;
  std::uint32_t num_blocks = (args.batch_size + threads_per_block - 1) / threads_per_block;

  CBCSweepKernel_WITH_CBCD_FLUDS<<<num_blocks, threads_per_block>>>(args);

  // Post-kernel processing
  phi->CopyFromDevice();
  if (save_angular_flux_)
    reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(2))->CopyFromDevice();

  // Retrieve outflow
  OutflowCarrier* outflow = reinterpret_cast<OutflowCarrier*>(problem_.GetCarrier(1));
  outflow->AccumulateBack(cell_transport_views_);
  outflow->Reset();

  // Copy downwind angular flux data from device to host
  cbcd_fluds.non_local_and_reflecting_psi_buffer_storage_.CopyFromDevice();
  const auto& downwind_results = cbcd_fluds.non_local_and_reflecting_psi_buffer_storage_.GetHostVector();

  for (size_t i = 0; i < tasks_to_execute_.size(); ++i)
  {
    auto* task = tasks_to_execute_[i];
    const auto& cell = *task->cell_ptr;
    const auto& face_orientations = angle_set.GetSPDS().GetCellFaceOrientations()[cell.local_id];
    const auto& cell_mapping = discretization_.GetCellMapping(cell);
    const auto& cell_transport_view = cell_transport_views_[cell.local_id];

    const int face_offset_base = cell_face_offset_map[i];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      if (face_orientations[f] == FaceOrientation::OUTGOING)
      {
        const auto& face = cell.faces[f];
        const bool is_local_face = cell_transport_view.IsFaceLocal(f);
        const bool is_boundary_face = not face.has_neighbor;
        const bool is_reflecting_boundary_face =
          (is_boundary_face) and (angle_set.GetBoundaries().at(face.neighbor_id)->IsReflecting());
        const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
        const size_t face_data_size = num_face_nodes * num_angles_in_as * gs_size;
        const int buffer_offset = downwind_psi_offsets[face_offset_base + f];

        if (buffer_offset < 0)
          continue;

        if ((not is_local_face) and (not is_boundary_face))
        {
          const int locality = cell_transport_view.FaceLocality(f);
          auto& async_comm = *cbc_angle_set.GetCommunicator();
          const auto* face_nodal_mapping =
            &fluds_->GetCommonData().GetFaceNodalMapping(cell.local_id, f);
          std::vector<double>& psi_nonlocal_outgoing =
            async_comm.InitGetDownwindMessageData(locality,
                                                  face.neighbor_id,
                                                  face_nodal_mapping->associated_face_,
                                                  angle_set.GetID(),
                                                  face_data_size);

          std::copy(&downwind_results[buffer_offset],
                    &downwind_results[buffer_offset + face_data_size],
                    psi_nonlocal_outgoing.begin());
        }
        else if (is_reflecting_boundary_face)
        {
          for (size_t fi = 0; fi < num_face_nodes; ++fi)
          {
            for (size_t as_ss_idx = 0; as_ss_idx < num_angles_in_as; ++as_ss_idx)
            {
              const auto direction_num = as_angle_indices[as_ss_idx];
              const size_t offset = fi * (num_angles_in_as * gs_size) + as_ss_idx * (gs_size);
              const double* result_ptr = &downwind_results[buffer_offset + offset];
              double* psi_out =
                cbc_angle_set.PsiReflected(face.neighbor_id, direction_num, cell.local_id, f, fi);

              if (psi_out)
                std::copy(result_ptr, result_ptr + gs_size, psi_out);
            }
          }
        }
      }
    }
  }
}
} // namespace opensn
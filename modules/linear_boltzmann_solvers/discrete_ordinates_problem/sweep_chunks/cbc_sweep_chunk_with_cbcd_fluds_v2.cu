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

#include <sys/types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

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
  std::uint32_t groupset_start;
  std::uint32_t groupset_size;

  const uint64_t* cell_local_ids;
  std::uint32_t num_cells;

  // Psi data buffer and auxiliary vector maps
  const uint64_t* cell_to_face_offset_map;
  const uint64_t* cell_face_node_angle_group_offsets_map;
  double* cell_psi_data;

  // Batch
  std::uint32_t batch_size;
  bool save_angular_flux;
  double* destination_psi;
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
                       const uint64_t cell_local_id,
                       const std::uint32_t cell_idx,
                       const std::uint32_t angle_idx,
                       const std::uint32_t group_idx,
                       DirectionView& direction,
                       CellView& cell)
{
  const uint64_t cell_face_offset = args.cell_to_face_offset_map[cell_local_id];
  size_t face_offset_stride = 0;

  // Loop over each face
  for (std::uint32_t f = 0; f < cell.num_faces; ++f)
  {
    // std::printf("Face f %u\n", f);

    FaceView face;
    cell.GetFaceView(face, f);
    const uint64_t face_data_size = face.num_face_nodes * args.angleset_size * args.groupset_size;

    // Compute mu = omega . n
    const double mu = direction.omega[0] * face.normal[0] + direction.omega[1] * face.normal[1] +
                      direction.omega[2] * face.normal[2];

    // std::printf("Mu for Face %u: %f\n", f, mu);

    // Get first bit
    // First bit == 0 -> incoming face
    // First bit == 1 -> outgoing face
    uint64_t face_map_val =
      args.cell_face_node_angle_group_offsets_map[cell_face_offset + face_offset_stride];

    // bool is_incoming_face = !((face_map_val >> 63) & 1);

    // Get first two bits
    // First two bits == 01 -> incoming face
    // First two bits == 10 -> outgoing face
    bool is_incoming_face = ((face_map_val >> 62) & 0x3) == 0x1;

    // Get second two bits
    // Second two bits == 00 -> local face
    // Second two bits == 10 -> non-local face
    // Second two bits == 11 -> boundary face
    // bool is_incoming_local_face = ((face_map_val >> 60) & 0x3) == 0x1;
    // bool is_incoming_nonlocal_face = ((face_map_val >> 60) & 0x3) == 0x2;
    // bool is_incoming_boundary_face = ((face_map_val >> 60) & 0x3) == 0x3;

    if (!is_incoming_face)
    {
      face_offset_stride += face_data_size;
      continue;
    }

    // Skip if not incoming local, non-local, or boundary face
    // if (!is_incoming_face or !(is_incoming_local_face || is_incoming_nonlocal_face ||
    // is_incoming_boundary_face))
    // {
    //   face_offset_stride += face_data_size;
    //   continue;
    // }

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

        const double* psi_in = nullptr;

        const uint64_t map_idx = cell_face_offset +
                                 face_offset_stride +
                                 (fj * args.angleset_size * args.groupset_size) +
                                 (angle_idx * args.groupset_size) + group_idx;

        const uint64_t encoded_device_buffer_index =
          args.cell_face_node_angle_group_offsets_map[map_idx];

        // Get the second bit of encoded_device_buffer_index
        // Second bit == 0 -> local face
        // Second bit == 1 -> non-local or boundary face
        // bool is_local_face = !((encoded_device_buffer_index >> 62) & 1);

        // const uint64_t true_device_buffer_index =
        //   encoded_device_buffer_index & 0x3FFFFFFFFFFFFFFF;

        // Get remaining 60 bits of encoded_device_buffer_index
        const uint64_t true_device_buffer_index = encoded_device_buffer_index & 0x0FFFFFFFFFFFFFFF;

        psi_in = &args.cell_psi_data[true_device_buffer_index];

        psi[i] += (*psi_in) * mu_Nij;

        // std::printf("(INCOMING) Cell %d Face %d FaceNode i %d FaceNode j %d Angle %d Group %d Mu %f Mu_Nij %f "
        //             "Psi_in %f\n",
        //             (int)cell_local_id, (int)f, (int)i, (int)j, (int)args.directions[angle_idx], (int)group_idx, mu, mu_Nij, *psi_in);
        
        // std::printf("face node i = %d, face node j = %d, angle = %d, group = %d\n",
        //             i,
        //             j,
        //             angle_idx,
        //             group_idx);

        // std::printf("mu = %f\n", mu);
      }
    }
    face_offset_stride += face_data_size;
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
DeviceRecordDownwindPsiAndOutflow(const std::array<double, cbc_max_dof>& psi,
                                  CellView& cell,
                                  const CBCSweepKernelArgs& args,
                                  const uint64_t cell_local_id,
                                  const std::uint32_t cell_idx,
                                  const std::uint32_t angle_idx,
                                  const std::uint32_t group_idx,
                                  DirectionView& direction)
{
  const uint64_t cell_face_offset = args.cell_to_face_offset_map[cell_local_id];
  size_t face_offset_stride = 0;

  for (std::uint32_t f = 0; f < cell.num_faces; ++f)
  {
    FaceView face;
    cell.GetFaceView(face, f);
    const uint64_t face_data_size = face.num_face_nodes * args.angleset_size * args.groupset_size;
    double mu = direction.omega[0] * face.normal[0] + direction.omega[1] * face.normal[1] +
                direction.omega[2] * face.normal[2];

    // std::printf("mu: %f\n", mu);

    // Get first bit
    // First bit == 0 -> incoming face 
    // First bit == 1 -> outgoing face
    uint64_t face_map_val = args.cell_face_node_angle_group_offsets_map[cell_face_offset + face_offset_stride];
    // bool is_outgoing_face = (face_map_val >> 63) & 1;

    // Get first two bits
    // First two bits == 01 -> incoming face
    // First two bits == 10 -> outgoing face
    bool is_outgoing_face = ((face_map_val >> 62) & 0x3) == 0x2;

    // Get second two bits
    // Second two bits == 01 -> local face
    // Second two bits == 10 -> non-local face
    // Second two bits == 11 -> reflecting boundary face
    // Second two bits == 00 -> boundary face
    // bool is_outgoing_local_face = ((face_map_val >> 60) & 0x3) == 0x1;
    // bool is_outgoing_nonlocal_face = ((face_map_val >> 60) & 0x3) == 0x2;
    // bool is_outgoing_reflecting_boundary_face = ((face_map_val >> 60) & 0x3) == 0x3;
    // bool is_outgoing_boundary_face = ((face_map_val >> 60) & 0x3) == 0x0;

    if (!is_outgoing_face)
    {
      face_offset_stride += face_data_size;
      continue;
    }

    // if (!is_outgoing_face or !(is_outgoing_local_face || is_outgoing_nonlocal_face ||
    // is_outgoing_boundary_face))
    // {
    //   face_offset_stride += face_data_size;
    //   continue;
    // }

    // NOTE: We need to record outgoing psi for local, non-local, and reflecting boundary faces
    // Outgoing non-reflecting boundary angular fluxes are still written
    // to the device buffer, which is unnecessary
    // Need to figure out how to skip writing angular fluxes for outgoing non-reflecting
    // boundary faces to save memory bandwidth
    // However, outflow still needs to be tallied for outgoing boundary faces,
    // regardless if they're non-reflecting or reflecting
    for (std::uint32_t fi = 0; fi < face.num_face_nodes; ++fi)
    {
      const int i = face.cell_mapping_data[fi];

      const uint64_t map_idx = cell_face_offset +
                              face_offset_stride +
                              (fi * args.angleset_size * args.groupset_size) +
                              (angle_idx * args.groupset_size) + group_idx;

      const uint64_t encoded_device_buffer_index =
        args.cell_face_node_angle_group_offsets_map[map_idx];

      // const uint64_t true_device_buffer_index = encoded_device_buffer_index & 0x3FFFFFFFFFFFFFFF;

      // Get remaining 60 bits of encoded_device_buffer_index
      const uint64_t true_device_buffer_index = encoded_device_buffer_index & 0x0FFFFFFFFFFFFFFF;

      args.cell_psi_data[true_device_buffer_index] = psi[i];

      // std::printf("(OUTGOING) Cell %d Face %d FaceNode i %d Angle %d Group %d Mu %f"
      //             " Psi_out %f\n",
      //             (int)cell_local_id,
      //             (int)f,
      //             (int)i,
      //             (int)args.directions[angle_idx],
      //             (int)group_idx,
      //             mu,
      //             args.cell_psi_data[true_device_buffer_index]);

      // Tally outflow for boundary faces
      if (face.outflow != nullptr)
      {
        double outflow = direction.weight * mu * face.IntS_shapeI_data[fi] * psi[i];
        crb::atomic_add(face.outflow + group_idx, outflow);
      }
    }

    face_offset_stride += face_data_size;
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

__global__ void
CBCSweepKernel(CBCSweepKernelArgs args)
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
    ComputeGMS(sweep_matrix,
               psi,
               cell.num_nodes,
               direction,
               cell,
               args.src_moment,
               args.groupset_start,
               idx.group_idx,
               args.groupset_size,
               num_moments);

    ComputeSurfaceIntegral(sweep_matrix,
                           psi,
                           args,
                           cell_local_id,
                           idx.cell_idx,
                           idx.angle_idx,
                           idx.group_idx,
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
                   idx.group_idx,
                   args.groupset_size,
                   num_moments);

  DeviceRecordDownwindPsiAndOutflow(
    psi, cell, args, cell_local_id, idx.cell_idx, idx.angle_idx, idx.group_idx, direction);
}

void
CBCSweepChunk::CopyPhiAndSrcToDevice()
{
  // Copy source moment and destination phi data to device
  MemoryPinner<double>* src = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(0));
  src->CopyToDevice();

  MemoryPinner<double>* phi = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(1));
  phi->CopyToDevice();
}

void
CBCSweepChunk::CopyOutflowAndPhiFromDevice()
{
  // Post-kernel processing
  MemoryPinner<double>* phi = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(1));
  phi->CopyFromDevice();

  // Retrieve outflow from device
  OutflowCarrier* outflow = reinterpret_cast<OutflowCarrier*>(problem_.GetCarrier(1));
  outflow->AccumulateBack(cell_transport_views_);
  outflow->Reset();

  // if (save_angular_flux_)
  //   reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(2))->CopyFromDevice();
}

void
CBCSweepChunk::GPUSweep(AngleSet& angle_set)
{
  // CALI_CXX_MARK_SCOPE("CBCSweepChunk::GPUSweep");

  opensn::log.Log() << "Here in GPUSweep";

  // Determine sizes for host and device vectors
  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  auto& cbc_fluds = dynamic_cast<CBC_FLUDS&>(*fluds_);
  auto& cbcd_fluds = *static_cast<CBCD_FLUDS*>(cbc_fluds.Get_CBCD_FLUDS_Ptr());
  const auto& as_angle_indices = cbc_angle_set.GetAngleIndices();
  const auto num_angles_in_as = as_angle_indices.size();
  const auto gs_size = cbc_angle_set.GetNumGroups();

  // opensn::log.Log() << "gs_size = " << gs_size << "\n";

  const auto gs_gi = groupset_.groups.front().id;

  // Determine sizes of buffers
  std::vector<uint64_t> cell_local_ids(tasks_to_execute_.size());

  for (int idx = 0; idx < tasks_to_execute_.size(); ++idx)
  {
    const auto& cell = *tasks_to_execute_[idx]->cell_ptr;
    cell_local_ids[idx] = cell.local_id;
  }

  MeshCarrier* mesh = reinterpret_cast<MeshCarrier*>(problem_.GetCarrier(2));
  QuadratureCarrier* quadrature = reinterpret_cast<QuadratureCarrier*>(groupset_.quad_carrier);

  // Copy source moment and destination phi data to device
  MemoryPinner<double>* src = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(0));

  MemoryPinner<double>* phi = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(1));

  // Copy angleset data to device
  MemoryPinner<std::uint32_t>* directions =
    reinterpret_cast<MemoryPinner<std::uint32_t>*>(angle_set.GetMemoryPin());

  // Prior to executing the sweep kernel, need to retrieve
  // incoming non-local cell angular fluxes and write them to the appropriate
  // device buffer locations
  cbc_fluds.GetNonlocalPsiData(*this, cbc_angle_set, tasks_to_execute_);

  // Set up kernel arguments
  CBCSweepKernelArgs args;

  args.mesh_data = mesh->GetDevicePtr();
  args.quad_data = quadrature->GetDevicePtr();

  args.src_moment = src->GetDevicePtr();
  args.phi = phi->GetDevicePtr();

  args.directions = directions->GetDevicePtr();
  args.angleset_size = num_angles_in_as;

  args.groupset_start = gs_gi;
  args.groupset_size = gs_size;

  cbcd_fluds.cell_local_ids_storage_.Copy(cell_local_ids.begin(), cell_local_ids.end());
  args.cell_local_ids = cbcd_fluds.cell_local_ids_storage_.GetDevicePtr();
  args.num_cells = tasks_to_execute_.size();

  args.cell_to_face_offset_map = cbcd_fluds.cell_to_face_offset_map_storage_.GetDevicePtr();
  args.cell_face_node_angle_group_offsets_map =
    cbcd_fluds.cell_face_node_angle_group_offsets_map_storage_.GetDevicePtr();
  args.cell_psi_data = cbcd_fluds.cell_psi_data_buffer_storage_.GetDevicePtr();

  args.batch_size = args.num_cells * args.angleset_size * args.groupset_size;
  args.save_angular_flux = false;
  args.destination_psi = nullptr;

  const std::uint32_t threads_per_block = 256;
  const std::uint32_t num_blocks = (args.batch_size + threads_per_block - 1) / threads_per_block;

  // Invoke kernel asynchronously with the current angle set's stream
  CBCSweepKernel<<<num_blocks, threads_per_block>>>(args);

  // After executing the sweep kernel, send outgoing non-local and reflecting
  // boundary angular flux to appropriate locations
  cbc_fluds.SetNonlocalAndReflectingBoundaryPsiData(*this, cbc_angle_set, tasks_to_execute_);
}

void
CBCSweepChunk::GPUSweepAsync(AngleSet& angle_set, std::vector<Task*>& tasks_to_execute)
{
  CALI_CXX_MARK_SCOPE("CBCSweepChunk::GPUSweepAsync");

  auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(angle_set);
  auto& cbc_fluds = dynamic_cast<CBC_FLUDS&>(cbc_angle_set.GetFLUDS());

  auto& cbcd_fluds = *static_cast<CBCD_FLUDS*>(cbc_fluds.Get_CBCD_FLUDS_Ptr());

  const auto& as_angle_indices = cbc_angle_set.GetAngleIndices();
  const auto num_angles_in_as = as_angle_indices.size();
  const auto gs_size = cbc_angle_set.GetNumGroups();
  const auto gs_gi = groupset_.groups.front().id;

  std::vector<std::uint64_t> cell_local_ids(tasks_to_execute.size());
  for (int idx = 0; idx < tasks_to_execute.size(); ++idx)
  {
    const auto& cell = *tasks_to_execute[idx]->cell_ptr;
    cell_local_ids[idx] = cell.local_id;
  }

  MeshCarrier* mesh = reinterpret_cast<MeshCarrier*>(problem_.GetCarrier(2));
  QuadratureCarrier* quadrature = reinterpret_cast<QuadratureCarrier*>(groupset_.quad_carrier);
  MemoryPinner<double>* src = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(0));
  MemoryPinner<double>* phi = reinterpret_cast<MemoryPinner<double>*>(problem_.GetPinner(1));
  MemoryPinner<std::uint32_t>* directions =
    reinterpret_cast<MemoryPinner<std::uint32_t>*>(angle_set.GetMemoryPin());

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cbc_angle_set.stream_ptr);
  // cudaStreamSynchronize(stream);

  // std::copy(cell_local_ids.begin(),
  //           cell_local_ids.end(),
  //           cbcd_fluds.cell_local_ids_storage_.GetHostVector().begin());

  // cudaMemcpyAsync(cbcd_fluds.cell_local_ids_storage_.GetDevicePtr(),
  //                     cbcd_fluds.cell_local_ids_storage_.GetHostVector().data(),
  //                     sizeof(uint64_t) * cell_local_ids.size(),
  //                     cudaMemcpyHostToDevice,
  //                     stream);

  cbcd_fluds.cell_local_ids_storage_.Copy(cell_local_ids.begin(), cell_local_ids.end());

  // cbc_fluds.GetNonlocalPsiDataAsync(*this, cbc_angle_set, tasks_to_execute);
  cbc_fluds.GetNonlocalPsiData(*this, cbc_angle_set, tasks_to_execute);

  CBCSweepKernelArgs args;

  args.mesh_data = mesh->GetDevicePtr();
  args.quad_data = quadrature->GetDevicePtr();
  args.src_moment = src->GetDevicePtr();
  args.phi = phi->GetDevicePtr();
  args.directions = directions->GetDevicePtr();
  args.angleset_size = num_angles_in_as;
  args.groupset_start = gs_gi;
  args.groupset_size = gs_size;
  args.cell_local_ids = cbcd_fluds.cell_local_ids_storage_.GetDevicePtr();
  args.num_cells = tasks_to_execute.size();
  args.cell_to_face_offset_map = cbcd_fluds.cell_to_face_offset_map_storage_.GetDevicePtr();
  args.cell_face_node_angle_group_offsets_map =
    cbcd_fluds.cell_face_node_angle_group_offsets_map_storage_.GetDevicePtr();
  args.cell_psi_data = cbcd_fluds.cell_psi_data_buffer_storage_.GetDevicePtr();

  args.batch_size = args.num_cells * args.angleset_size * args.groupset_size;
  args.save_angular_flux = false;
  args.destination_psi = nullptr; 

  const std::uint32_t threads_per_block = 128;
  const std::uint32_t num_blocks = (args.batch_size + threads_per_block - 1) / threads_per_block;
  
  CBCSweepKernel<<<num_blocks, threads_per_block, 0, stream>>>(args);
}
} // namespace opensn

// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/ndarray.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/storage.h"
#include "caribou/caribou.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"
#include <vector>
#include <utility>

namespace crb = caribou;

namespace opensn
{

class CBCD_FLUDS
{
public:
  CBCD_FLUDS(CBC_FLUDS& cbc_fluds,
             size_t num_total_faces,
             size_t incoming_boundary_psi_buffer_size,
             const std::vector<uint64_t>& face_neighbor_local_ids,
             const std::vector<unsigned int>& face_neighbor_cell_node_map,
             const std::vector<int>& cell_to_local_face_offset_map,
             const std::vector<int>& boundary_psi_map,
             const std::vector<int>& incoming_face_category_map,
             const std::vector<int>& outgoing_face_category_map,
             size_t non_local_upwind_psi_buffer_size,
             size_t non_local_and_reflecting_psi_buffer_size);

  Storage<uint64_t> cell_id_storage_;
  Storage<int> cell_face_offset_storage_;

  Storage<unsigned int> face_neighbor_cell_node_map_storage_;
  Storage<uint64_t> face_neighbor_local_ids_storage_;

  Storage<int> incoming_face_category_map_storage_;
  Storage<int> outgoing_face_category_map_storage_; 

  /// Device storage for boundary angular fluxes
  Storage<double> boundary_psi_buffer_;

  /// Device storage for mapping into the boundary psi buffer
  Storage<int> boundary_psi_map_storage_;

  /// Device storage for non-local upwind angular fluxes
  Storage<double> non_local_upwind_psi_buffer_storage_;

  /// Device storage for outgoing non-local and reflecting boundary angular fluxes
  Storage<double> non_local_and_reflecting_psi_buffer_storage_;

  /// Device storage for mapping cells to their corresponding faces in the boundary_psi_map_storage_
  /// vector
  Storage<int> cell_to_local_face_offset_storage_;

  /// Get the device memory.
  inline double* GetDevicePtr() { return device_buffer_.get(); }

  /// Get the host memory.
  inline double* GetHostPtr() { return host_buffer_.data(); }

  /// Get device pointer to cell DOF map
  inline const size_t* GetCellDOFMapDevicePtr() { return cell_dof_map_storage_.GetDevicePtr(); }

  /// Get device pointer to boundary psi data
  inline const double* GetBoundaryPsiDevicePtr() { return boundary_psi_buffer_.GetDevicePtr(); }

  /// Get device pointer to boundary psi map
  inline const int* GetBoundaryPsiMapDevicePtr()
  {
    return boundary_psi_map_storage_.GetDevicePtr();
  }

protected:
  /// Contiguous memory on the host (CPU) for angular flux.
  crb::HostVector<double> host_buffer_;
  /// Contiguous memory on the device (GPU) for angular flux.
  crb::DeviceMemory<double> device_buffer_;

  /// Device storage for cell DOF map
  Storage<size_t> cell_dof_map_storage_;
};

} // namespace opensn
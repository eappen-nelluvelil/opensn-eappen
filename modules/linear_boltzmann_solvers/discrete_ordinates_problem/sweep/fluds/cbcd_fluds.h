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
  CBCD_FLUDS(CBC_FLUDS& cbc_fluds);

  Storage<unsigned int> face_neighbor_cell_node_map_storage_;
  Storage<uint64_t> face_neighbor_local_ids_storage_;

  Storage<int> incoming_face_category_map_storage_;
  Storage<int> outgoing_face_category_map_storage_; 

  /// Device storage for boundary angular fluxes
  Storage<double> boundary_psi_buffer_;

  /// Device storage for mapping into the boundary psi buffer
  Storage<int> boundary_psi_map_storage_;

  /// Device storage for mapping cells to their corresponding faces in the boundary_psi_map_storage_
  Storage<int> cell_to_local_face_offset_storage_;

  /// 11/4: Contains for working with streams
  /// TODO: Refactor CBCD_FLUDS at some point

  /// Device storage for cells in stream 1
  Storage<uint64_t> cells_for_stream_1_storage_;

  /// Device storage for cells in stream 2
  Storage<uint64_t> cells_for_stream_2_storage_;
  Storage<int> cell_face_offset_for_stream_2_storage_;

  Storage<double> upwind_psi_buffer_storage_;
  Storage<int> upwind_psi_offsets_storage_;

  Storage<double> downwind_psi_buffer_storage_;
  Storage<int> downwind_psi_offsets_storage_;

  // -----------------------------------------------------------------------------
  // Idea for dealing with surface integrals on device
  Storage<uint64_t> cell_to_local_face_offset_map_gpu_storage_;
  crb::DeviceMemory<double> local_and_nonlocal_psi_buffer_;

  // -----------------------------------------------------------------------------

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
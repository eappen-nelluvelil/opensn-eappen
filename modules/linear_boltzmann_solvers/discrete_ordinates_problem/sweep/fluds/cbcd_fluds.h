// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/ndarray.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/storage.h"
#include "caribou/caribou.h"
#include <vector>
#include <utility>

namespace crb = caribou;

namespace opensn
{

class CBCD_FLUDS
{
public:
	CBCD_FLUDS(CBC_FLUDS& cbc_fluds, const std::vector<double>& boundary_psi, const std::vector<int>& boundary_psi_map);

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

  Storage<uint64_t> cell_id_storage_;
  Storage<int> cell_face_offset_storage_;

protected:
	/// Contiguous memory on the host (CPU) for angular flux.
  crb::HostVector<double> host_buffer_;
  /// Contiguous memory on the device (GPU) for angular flux.
  crb::DeviceMemory<double> device_buffer_;

	/// Device storage for cell DOF map
	Storage<size_t> cell_dof_map_storage_;

	/// Device storage for boundary angular fluxes
	Storage<double> boundary_psi_buffer_;

	/// Device storage for mapping into the boundary psi buffer
  Storage<int> boundary_psi_map_storage_;
};

} // namespace opensn
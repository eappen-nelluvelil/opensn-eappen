// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/ndarray.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "caribou/caribou.h"
#include <vector>
#include <utility>

namespace crb = caribou;

namespace opensn
{

class CBCD_FLUDS
{
public:
	CBCD_FLUDS(CBC_FLUDS& cbc_fluds);

	/// Get the device memory.
  inline double* GetDevicePtr() { return device_buffer_.get(); }

	/// Get the host memory.
	inline double* GetHostPtr() { return host_buffer_.data(); }

protected:
	/// Contiguous memory on the host (CPU) for angular flux.
  crb::HostVector<double> host_buffer_;
  /// Contiguous memory on the device (GPU) for angular flux.
  crb::DeviceMemory<double> device_buffer_;
};

} // namespace opensn
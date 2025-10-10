// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "framework/data_types/ndarray.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "caribou/caribou.h"
#include <map>
#include <vector>
#include <utility>

namespace crb = caribou;

namespace opensn
{

struct GPUBoundaryData
{
	// Host staging
	crb::HostVector<double> host_boundary_psi_;
	crb::HostVector<int64_t> host_boundary_map_;

	// Device memory
	crb::DeviceMemory<double> device_boundary_psi_;
	crb::DeviceMemory<int64_t> device_boundary_map_;

	// Metadata
	size_t max_boundary_id = 0;
	size_t num_angles = 0;
	size_t num_groups = 0;
};

struct GPUReflectingData
{
  // Host staging
  crb::HostVector<double> host_reflected_psi_;
  crb::HostVector<int> host_reflected_angle_map_;
  crb::HostVector<int64_t> host_reflecting_map_;
  
  // Device memory
  crb::DeviceMemory<double> device_reflected_psi_;
  crb::DeviceMemory<int> device_reflected_angle_map_;
  crb::DeviceMemory<int64_t> device_reflecting_map_;
  
  size_t max_reflecting_id = 0;
  size_t num_angles = 0;
  size_t num_groups = 0;

  std::map<uint64_t, bool> host_opposing_reflected_flags_;
};

struct GPUNonLocalData
{
  crb::HostVector<double> host_nonlocal_psi_;
  crb::HostVector<uint64_t> host_cell_global_ids_;
  crb::HostVector<uint32_t> host_face_ids_;
  crb::HostVector<uint32_t> host_face_nodes_;
  crb::HostVector<int64_t> host_psi_offsets_;
  
  crb::DeviceMemory<double> device_nonlocal_psi_;
  crb::DeviceMemory<uint64_t> device_cell_global_ids_;
  crb::DeviceMemory<uint32_t> device_face_ids_;
  crb::DeviceMemory<uint32_t> device_face_nodes_;
  crb::DeviceMemory<int64_t> device_psi_offsets_;
  
  size_t num_entries = 0;
};

CBC_FLUDS::~CBC_FLUDS()
{
  delete gpu_boundary_data_;
  delete gpu_reflecting_data_;
  delete gpu_nonlocal_data_;
}

void
CBC_FLUDS::CreateGPUFluds(LBSProblem& lbs_problem, const LBSGroupset& group_set,
                          AngleSet& angle_set, bool is_surface_source_active)
{
  gpu_boundary_data_ = new GPUBoundaryData();
  gpu_reflecting_data_ = new GPUReflectingData();
  gpu_nonlocal_data_ = new GPUNonLocalData();
}

void
CBC_FLUDS::UpdateGPUNonLocalData(const std::vector<Task*>& tasks)
{
  
}

} // namespace opensn
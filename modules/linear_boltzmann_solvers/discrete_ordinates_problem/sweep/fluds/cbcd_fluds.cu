// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"

namespace opensn
{

CBCD_FLUDS::CBCD_FLUDS(CBC_FLUDS& cbc_fluds, const std::vector<double>& boundary_psi,
					   const std::vector<int>& boundary_psi_map)
{
	device_buffer_ = crb::DeviceMemory<double>(cbc_fluds.GetGPULocalPsiDataSize());

	cbc_fluds.BuildDeviceCellDOFMap();
	cell_dof_map_storage_ = Storage<size_t>(cbc_fluds.GetCellDOFMap().size());
	cell_dof_map_storage_.Copy(cbc_fluds.GetCellDOFMap().begin(),
	                          cbc_fluds.GetCellDOFMap().end());

	boundary_psi_buffer_ = Storage<double>(boundary_psi.size());
	boundary_psi_buffer_.Copy(boundary_psi.begin(), boundary_psi.end());

	boundary_psi_map_storage_ = Storage<int>(boundary_psi_map.size());
	boundary_psi_map_storage_.Copy(boundary_psi_map.begin(), boundary_psi_map.end());

  cell_id_storage_ = Storage<uint64_t>(cbc_fluds.GetNumLocalCells());
  cell_face_offset_storage_ = Storage<int>(cbc_fluds.GetNumLocalCells());
}

void
CBC_FLUDS::Create_CBCD_FLUDS(const std::vector<double>& boundary_psi, const std::vector<int>& boundary_psi_map)
{
	if (not cbcd_fluds_)
	{
		CBCD_FLUDS* cbcd_fluds = new CBCD_FLUDS(*this, boundary_psi, boundary_psi_map);
		cbcd_fluds_ = cbcd_fluds;
	}
}

void
CBC_FLUDS::Destroy_CBCD_FLUDS()
{
	if (cbcd_fluds_)
	{
		CBCD_FLUDS* cbcd_fluds = reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_);
		delete cbcd_fluds;
		cbcd_fluds_ = nullptr;
	}
}



} // namespace opensn
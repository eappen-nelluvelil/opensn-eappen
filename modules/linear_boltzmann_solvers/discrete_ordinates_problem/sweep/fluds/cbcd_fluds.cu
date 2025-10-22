// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"

namespace opensn
{

CBCD_FLUDS::CBCD_FLUDS(CBC_FLUDS& cbc_fluds)
{
	cell_dof_map_storage_ = Storage<size_t>(cbc_fluds.GetCellDOFMap().size());
	cell_dof_map_storage_.Copy(cbc_fluds.GetCellDOFMap().begin(),
	                          cbc_fluds.GetCellDOFMap().end());
	device_buffer_ = crb::DeviceMemory<double>(cbc_fluds.GetGPULocalPsiDataSize());
	// device_buffer_.zero_fill();
}

void
CBC_FLUDS::Create_CBCD_FLUDS()
{
	if (not cbcd_fluds_)
	{
		// cbcd_fluds_ = new CBCD_FLUDS(*this);
		CBCD_FLUDS* cbcd_fluds = new CBCD_FLUDS(*this);
		cbcd_fluds_ = cbcd_fluds;
	}
}

void
CBC_FLUDS::Destroy_CBCD_FLUDS()
{
	if (cbcd_fluds_)
	{
		// delete static_cast<CBCD_FLUDS*>(cbcd_fluds_);
		// cbcd_fluds_ = nullptr;
		CBCD_FLUDS* cbcd_fluds = reinterpret_cast<CBCD_FLUDS*>(cbcd_fluds_);
		delete cbcd_fluds;
		cbcd_fluds_ = nullptr;
	}
}



} // namespace opensn
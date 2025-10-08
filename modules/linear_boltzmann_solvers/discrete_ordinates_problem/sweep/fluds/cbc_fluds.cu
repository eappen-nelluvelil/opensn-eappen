// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "caribou/caribou.h"
#include <unordered_map>

namespace crb = caribou;

namespace opensn
{

void
CBC_FLUDS::InitializeGPUMemory()
{

}

void 
CBC_FLUDS::DestroyGPUMemory()
{

}

void
CBC_FLUDS::CopyCellToDevice(uint64_t cell_local_id)
{

}

void
CBC_FLUDS::CopyCellFromDevice(uint64_t cell_local_id)
{

}

void 
CBC_FLUDS::CopyNLUpwindToDevice(uint64_t cell_global_id, unsigned int face_id)
{
    
}

void CBC_FLUDS::CopyBoundaryDataToDevice()
{

}

} // namespace opensn
// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/runtime.h"
#include "framework/math/math.h"
#include "framework/object_factory.h"
#include "framework/utils/timer.h"
#include "config.h"
#include "caliper/cali.h"
#include "hdf5.h"
#include <iostream>
#include <stdexcept>

namespace opensn
{

namespace
{

bool caliper_started = false;

void
CheckCaliperConfig()
{
  if (cali_mgr.error())
    throw std::runtime_error(std::string("Caliper configuration error: ") + cali_mgr.error_msg());
}

} // namespace

// Global variables
mpi::Communicator mpi_comm;
bool use_caliper = false;
const std::string default_caliper_config =
  "runtime-report(calc.inclusive=true,aggregate_across_ranks=true,max_column_width=120)";
std::string cali_config(default_caliper_config);
cali::ConfigManager cali_mgr;
Timer program_timer;
std::filesystem::path input_path;

int
Initialize()
{
  StartCaliper();
  CALI_MARK_BEGIN(opensn::program.c_str());

  // Disable internal HDF error reporting
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

  return 0;
}

void
Finalize()
{
  // Flush standard streams
  std::cout.flush();
  std::cerr.flush();
  std::clog.flush();

  opensn::mpi_comm.barrier();

  CALI_MARK_END(opensn::program.c_str());
  FlushCaliper();
}

std::string
GetVersionStr()
{
  return PROJECT_VERSION;
}

std::string
GetCaliperPresetConfig(const std::string& preset)
{
  if (preset == "runtime")
    return default_caliper_config;
  if (preset == "mpi")
    return "runtime-report(calc.inclusive=true,aggregate_across_ranks=true,max_column_width=120),"
           "profile.mpi,mpi.message.count,mpi.message.size,mem.highwatermark";
  if (preset == "mpi-report")
    return "mpi-report";
  if (preset == "profile")
    return "runtime-profile(output=opensn-region-profile,output.format=cali,use.mpi=true,"
           "mem.highwatermark=true)";
  if (preset == "cuda")
    return "cuda-activity-report(show_kernels=true,aggregate_across_ranks=true),cuda.memcpy";
  if (preset == "cuda-profile")
    return "cuda-activity-profile(output=opensn-cuda-profile,output.format=cali,use.mpi=true)";
  if (preset == "hip" or preset == "rocm")
    return "rocm-activity-report(show_kernels=true,aggregate_across_ranks=true)";
  if (preset == "hip-profile" or preset == "rocm-profile")
    return "rocm-activity-profile(output=opensn-rocm-profile,output.format=cali,use.mpi=true,"
           "memcpy=true)";

  throw std::invalid_argument("Unknown Caliper preset: " + preset);
}

void
StartCaliper()
{
  if (not use_caliper or caliper_started)
    return;

  cali_mgr.add(cali_config.c_str());
  CheckCaliperConfig();

  const auto input = input_path.empty() ? std::string() : input_path.string();
  cali_set_global_string_byname("opensn.version", GetVersionStr().c_str());
  cali_set_global_string_byname("opensn.input", input.c_str());
  cali_set_global_string_byname("opensn.build_type", OPENSN_BUILD_TYPE);
  cali_set_global_string_byname("opensn.caliper.config", cali_config.c_str());
  cali_set_global_int_byname("opensn.mpi.rank", mpi_comm.rank());
  cali_set_global_int_byname("opensn.mpi.size", mpi_comm.size());

  cali_mgr.start();
  CheckCaliperConfig();
  caliper_started = true;
}

void
FlushCaliper()
{
  if (caliper_started)
    cali_mgr.flush();
}

} // namespace opensn

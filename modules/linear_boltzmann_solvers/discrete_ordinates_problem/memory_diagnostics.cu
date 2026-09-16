// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/utils/memory.h"
#include "caribou/backend.hpp"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "caribou/cuhip/api.hpp"
#endif
#include <ostream>

namespace opensn
{

void
TraceDeviceMemory(std::ostream& out)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  int device = -1;
  std::size_t free = 0;
  std::size_t total = 0;
  const auto device_status = GPU_API(GetDevice)(&device);
  const auto memory_status = GPU_API(MemGetInfo)(&free, &total);
  out << " device=" << device << " device_status=" << static_cast<int>(device_status)
      << " memory_status=" << static_cast<int>(memory_status) << " device_free_bytes=" << free
      << " device_total_bytes=" << total;
#else
  out << " device_memory=unavailable";
#endif
}

} // namespace opensn

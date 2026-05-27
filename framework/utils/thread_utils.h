// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>

namespace opensn
{

/// Summary of CPU-thread resources visible to the current process.
struct ThreadResourceInfo
{
  std::size_t hardware_threads = 1;
  std::size_t affinity_threads = 0;
  std::size_t affinity_cores = 0;
  std::size_t requested_threads = 0;
  std::string requested_source;
  std::size_t available_threads = 1;
};

/// Return the CPU-thread resources available to the current process.
ThreadResourceInfo GetThreadResourceInfo();

/// Format CPU-thread resource metadata.
std::string FormatThreadResourceInfo(const ThreadResourceInfo& info);

} // namespace opensn

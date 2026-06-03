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
  /// Hardware threads reported by the C++ runtime.
  std::size_t hardware_threads = 1;
  /// Threads in the process CPU-affinity mask, or zero when unavailable.
  std::size_t affinity_threads = 0;
  /// Thread count requested by scheduler/runtime environment variables.
  std::size_t requested_threads = 0;
  /// Environment variable that provided `requested_threads`.
  std::string requested_source;
  /// Conservative thread count available to this process.
  std::size_t available_threads = 1;
  /// Number of allocated job nodes, when reported by the scheduler.
  std::size_t job_nodes = 0;
  /// Environment variable that provided `job_nodes`.
  std::string job_nodes_source;
  /// Number of allocated job tasks, when reported by the scheduler.
  std::size_t job_tasks = 0;
  /// Environment variable that provided `job_tasks`.
  std::string job_tasks_source;
};

/// Return conservative CPU-thread resources for the current process.
ThreadResourceInfo GetThreadResourceInfo();

/// Format thread-resource metadata for diagnostics.
std::string FormatThreadResourceInfo(const ThreadResourceInfo& info);

} // namespace opensn

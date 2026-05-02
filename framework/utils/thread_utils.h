// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>

namespace opensn
{

struct ThreadResourceInfo
{
  std::size_t hardware_threads = 1;
  std::size_t affinity_threads = 0;
  std::size_t requested_threads = 0;
  std::string requested_source;
  std::size_t available_threads = 1;
  std::size_t job_nodes = 0;
  std::string job_nodes_source;
  std::size_t job_tasks = 0;
  std::string job_tasks_source;
};

ThreadResourceInfo GetThreadResourceInfo();

std::string FormatThreadResourceInfo(const ThreadResourceInfo& info);

} // namespace opensn

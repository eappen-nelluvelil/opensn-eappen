// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/utils/thread_utils.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace opensn
{

namespace
{

std::size_t
ParsePositiveEnvironmentValue(const char* name)
{
  const char* value = std::getenv(name); // NOLINT(concurrency-mt-unsafe)
  if (value == nullptr or value[0] == '\0')
    return 0;

  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(value, &end, 10);
  if (errno != 0 or end == value or *end != '\0' or parsed == 0 or
      parsed > std::numeric_limits<std::size_t>::max())
    return 0;

  return static_cast<std::size_t>(parsed);
}

std::size_t
GetAffinityThreadCount()
{
#if defined(__linux__)
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) == 0)
    return static_cast<std::size_t>(CPU_COUNT(&cpu_set));
#endif
  return 0;
}

std::size_t
GetAffinityCoreCount()
{
#if defined(__linux__)
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) != 0)
    return 0;

  std::vector<std::pair<int, int>> cores;
  cores.reserve(CPU_COUNT(&cpu_set));
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
  {
    if (not CPU_ISSET(cpu, &cpu_set))
      continue;

    const auto topology = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream package_file(topology + "physical_package_id");
    std::ifstream core_file(topology + "core_id");
    int package = 0;
    int core = 0;
    if (not(package_file >> package) or not(core_file >> core))
      return 0;
    cores.emplace_back(package, core);
  }

  std::sort(cores.begin(), cores.end());
  return static_cast<std::size_t>(std::unique(cores.begin(), cores.end()) - cores.begin());
#else
  return 0;
#endif
}

std::pair<std::size_t, std::string>
GetEnvironmentCount(const std::initializer_list<const char*> names)
{
  for (const char* name : names)
    if (const auto count = ParsePositiveEnvironmentValue(name); count > 0)
      return {count, name};

  return {0, {}};
}

} // namespace

std::size_t
GetPositiveEnvironmentValue(const char* name)
{
  return ParsePositiveEnvironmentValue(name);
}

ThreadResourceInfo
GetThreadResourceInfo()
{
  ThreadResourceInfo info;
  info.hardware_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  info.affinity_threads = GetAffinityThreadCount();
  info.affinity_cores = GetAffinityCoreCount();

  const auto [requested_threads, requested_source] = GetEnvironmentCount({"OPENSN_NUM_THREADS",
                                                                          "SLURM_CPUS_PER_TASK",
                                                                          "OMP_NUM_THREADS",
                                                                          "FLUX_CPUS_PER_TASK",
                                                                          "FLUX_TASK_CPUS"});
  info.requested_threads = requested_threads;
  info.requested_source = requested_source;

  info.available_threads = info.hardware_threads;
  if (info.affinity_cores > 0)
    info.available_threads = std::min(info.available_threads, info.affinity_cores);
  else if (info.affinity_threads > 0)
    info.available_threads = std::min(info.available_threads, info.affinity_threads);
  if (info.requested_threads > 0)
    info.available_threads = std::min(info.available_threads, info.requested_threads);
  info.available_threads = std::max<std::size_t>(1, info.available_threads);

  return info;
}

std::string
FormatThreadResourceInfo(const ThreadResourceInfo& info)
{
  std::ostringstream out;
  out << "hardware_threads=" << info.hardware_threads
      << ", affinity_threads=" << info.affinity_threads
      << ", affinity_cores=" << info.affinity_cores
      << ", requested_threads=" << info.requested_threads;
  if (not info.requested_source.empty())
    out << " from " << info.requested_source;
  out << ", available_threads=" << info.available_threads;

  return out.str();
}

} // namespace opensn

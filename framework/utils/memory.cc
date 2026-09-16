// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/utils/memory.h"
#include "framework/runtime.h"
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace opensn
{

void
TraceMemory(const char* stage, std::uintptr_t object, bool device)
{
  const char* directory = std::getenv("OPENSN_MEMORY_TRACE_DIR");
  if (directory == nullptr or directory[0] == '\0')
    return;

  static std::atomic<std::uint64_t> sequence{0};
  const auto id = sequence.fetch_add(1, std::memory_order_relaxed);
  const auto prefix = std::string(directory) + "/rank-" + std::to_string(mpi_comm.rank()) +
                      "-pid-" + std::to_string(getpid());
  std::ofstream out(prefix + ".log", std::ios::app);
  if (not out)
  {
    std::fprintf(stderr, "Memory trace: cannot open %s.log\n", prefix.c_str());
    return;
  }
  char hostname[256] = {};
  gethostname(hostname, sizeof(hostname) - 1);
  const auto time =
    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  out.precision(17);
  out << "sample=" << id << " time=" << time << " host=" << hostname << " stage=" << stage
      << " object=" << object;
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line))
    if (line.starts_with("VmRSS:") or line.starts_with("VmHWM:") or line.starts_with("RssAnon:") or
        line.starts_with("VmSize:") or line.starts_with("VmLck:") or line.starts_with("VmPin:"))
      out << " | " << line;
#ifdef __OPENSN_WITH_GPU__
  if (device)
    TraceDeviceMemory(out);
#endif
  out << '\n';
  out.flush();
#ifdef __GLIBC__
  const char* allocator = std::getenv("OPENSN_MEMORY_ALLOCATOR");
  if (allocator != nullptr and std::string(allocator) == "1")
  {
    const auto filename = prefix + "-" + std::to_string(id) + ".malloc.xml";
    if (auto* file = std::fopen(filename.c_str(), "w"))
    {
      if (malloc_info(0, file) != 0)
        std::fprintf(stderr, "Memory trace: malloc_info failed for %s\n", filename.c_str());
      std::fclose(file);
    }
    else
      std::fprintf(stderr, "Memory trace: cannot open %s\n", filename.c_str());
  }
#endif
}

std::optional<std::uint64_t>
GetPeakMemoryUsageBytes()
{
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    return std::nullopt;
  // ru_maxrss is glibc's normal POSIX-mandated way to read this field, but some libc
  // implementations define it via an anonymous union which can trip clang-tidy
  const long max_rss = usage.ru_maxrss; // NOLINT(cppcoreguidelines-pro-type-union-access)
  // A successful call with max_rss == 0 means the platform doesn't actually fill in this
  // field rather than genuine zero peak usage
  if (max_rss <= 0)
    return std::nullopt;
  // ru_maxrss is in bytes on macOS/Darwin, but kilobytes on Linux
#ifdef __APPLE__
  return static_cast<std::uint64_t>(max_rss);
#else
  return static_cast<std::uint64_t>(max_rss) * 1024;
#endif
}

std::optional<std::uint64_t>
GetTotalPeakMemoryUsageBytes()
{
  const auto local_peak = GetPeakMemoryUsageBytes();
  const std::uint64_t local_peak_bytes = local_peak.value_or(0);
  const int local_available = local_peak.has_value() ? 1 : 0;

  std::uint64_t total_peak_bytes = 0;
  int all_available = 0;
  mpi_comm.all_reduce(local_peak_bytes, total_peak_bytes, mpi::op::sum<std::uint64_t>());
  mpi_comm.all_reduce(local_available, all_available, mpi::op::min<int>());

  if (!all_available)
    return std::nullopt;
  return total_peak_bytes;
}

} // namespace opensn

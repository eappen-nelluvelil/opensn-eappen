// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <iosfwd>

namespace opensn
{

/// Returns this process's peak (high-water-mark) resident set size, in bytes.
std::optional<std::uint64_t> GetPeakMemoryUsageBytes();

/// Returns the total physical memory usage over all MPI ranks.
std::optional<std::uint64_t> GetTotalPeakMemoryUsageBytes();

/// Rank-local diagnostics enabled by OPENSN_MEMORY_TRACE_DIR. No MPI collectives or GPU barriers.
void TraceMemory(const char* stage, std::uintptr_t object = 0, bool device = false);

#ifdef __OPENSN_WITH_GPU__
void TraceDeviceMemory(std::ostream& out);
#endif

} // namespace opensn

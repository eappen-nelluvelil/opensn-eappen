// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <functional>

namespace opensn
{

/// Run one task per worker on this caller's reusable pool and wait for completion.
/// The task must not throw. No task captures are retained after return.
/// Pools are caller-thread-local, including for nested calls. Idle workers and
/// their thread-local scratch persist until release or caller-thread exit.
void RunParallelForWorkers(std::size_t num_threads, std::function<void(std::size_t)> task);

/// Join this caller's idle workers before runtime finalization or on demand.
void ReleaseParallelForWorkers();

} // namespace opensn

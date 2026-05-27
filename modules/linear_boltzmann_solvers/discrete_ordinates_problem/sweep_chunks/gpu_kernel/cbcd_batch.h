// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"

namespace opensn::gpu_kernel
{

/// A ready angle-set batch within a worker's combined launch.
struct CBCDBatch
{
  const Arguments<SweepKind::CBC>* arguments;
  const std::uint32_t* cells;
  double* saved_psi;
  std::uint64_t block_end;
  std::uint32_t num_cells;
  std::uint32_t block_size_x;
  std::uint32_t num_stride_blocks;
};

} // namespace opensn::gpu_kernel

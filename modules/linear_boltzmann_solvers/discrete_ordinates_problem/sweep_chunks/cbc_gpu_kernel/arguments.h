// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include <cstdint>

namespace opensn
{
class DiscreteOrdinatesProblem;
class LBSGroupset;
class AngleSet;
class CBCD_FLUDS;
class Task;
} // namespace opensn

namespace opensn::cbc_gpu_kernel
{

/// @brief Index for each thread.
struct Index
{
  /// @brief Constructor
  __device__ inline Index(){};

  __device__ inline void Compute(std::uint32_t thread_idx,
                                 const std::uint32_t& angleset_size,
                                 const std::uint32_t& groupset_size)
  {
    const auto angle_group_stride = angleset_size * groupset_size;
    cell_idx = thread_idx / angle_group_stride;
    angle_idx = (thread_idx % angle_group_stride) / groupset_size;
    group_idx = (thread_idx % angle_group_stride) % groupset_size;
  }

  /// @brief Index of the cell associated to the current thread in the current level vector.
  std::uint32_t cell_idx;
  /// @brief Index of the angle associated to the current thread in the current angleset.
  std::uint32_t angle_idx;
  /// @brief Index of the group associated to the current thread in the current groupset.
  std::uint32_t group_idx;
};

struct BaseArguments
{
  // Mesh and quadrature device pointers
  const char* mesh_data;
  const char* quad_data;
  // Source moments and phi device pointers
  const double* src_moment;
  double* phi;
  // Angle set info
  const std::uint32_t* directions;
  std::uint32_t angleset_size;
  // Group set info
  std::uint32_t num_groups;
  std::uint32_t groupset_start;
  std::uint32_t groupset_size;
  // Device CBC_FLUDS pointers
  CBCD_FLUDSPointerSet flud_data;
  const std::uint64_t* flud_index;
  // Batch info
  std::uint32_t batch_size;
  bool save_angular_flux;
  double* destination_psi;
};

/// Arguments for the device CBC sweep chunk kernel
struct Arguments : public BaseArguments
{
  Arguments(DiscreteOrdinatesProblem& problem,
            const LBSGroupset& groupset,
            AngleSet& angle_set,
            CBCD_FLUDS& fluds,
            std::vector<Task*>& tasks);

  // Cell local IDs corresponding to the current set of ready cells that can be swept
  const std::uint64_t* cell_local_ids;
};

/// Arguments for the device CBC sweep chunk kernel
struct GraphArguments : public BaseArguments
{
  GraphArguments(DiscreteOrdinatesProblem& problem,
            const LBSGroupset& groupset,
            AngleSet& angle_set,
            CBCD_FLUDS& fluds,
            const std::uint64_t cell_local_id);

  std::uint64_t cell_local_id;
};

} // namespace opensn::cbc_gpu_kernel
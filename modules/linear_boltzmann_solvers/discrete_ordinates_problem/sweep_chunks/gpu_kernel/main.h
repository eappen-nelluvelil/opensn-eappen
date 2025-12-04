// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/buffer.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include <utility>

namespace opensn::gpu_kernel
{

template <std::uint32_t... I, class F>
__device__ inline void
ForDOFs1ToN(std::integer_sequence<std::uint32_t, I...>, F&& f)
{
  (f(std::integral_constant<std::uint32_t, I + 1>{}), ...);
}

template <class F>
__device__ inline void
ForDOFs1ToMax(F&& f)
{
  ForDOFs1ToN(std::make_integer_sequence<std::uint32_t, LBSProblem::max_dofs_gpu>{},
              std::forward<F>(f));
}

template <class... Args>
__device__ inline void
SweepDispatch(std::uint32_t n, Args&&... args)
{
  bool done = false;
  ForDOFs1ToMax(
    [&](auto dof_c)
    {
      constexpr std::uint32_t dof = decltype(dof_c)::value;
      if (!done && n == dof)
      {
        gpu_kernel::Sweep<dof>(std::forward<Args>(args)...);
        done = true;
      }
    });
}

} // namespace opensn::gpu_kernel

namespace opensn::cbc_gpu_kernel
{

using SweepFunc = std::add_pointer_t<void(const cbc_gpu_kernel::CBC_Arguments&,
                                          CellView&,
                                          DirectionView&,
                                          const std::uint64_t*,
                                          const unsigned int&,
                                          const unsigned int&,
                                          const std::uint32_t&,
                                          double*)>;
template <std::size_t... IntSequence>
__device__ constexpr std::array<SweepFunc, sizeof...(IntSequence)>
MakeCBCSweepSpecMap(std::index_sequence<IntSequence...>)
{
  return std::array<SweepFunc, sizeof...(IntSequence)>{
    &gpu_kernel::Sweep<IntSequence, CBCD_NodeIndex, cbc_gpu_kernel::CBC_Arguments>...};
}
__device__ std::array<SweepFunc, LBSProblem::max_dofs_gpu> cbc_sweep_spec_map =
  MakeCBCSweepSpecMap(gpu_kernel::MakeIndexSequenceFromRange<1, LBSProblem::max_dofs_gpu + 1>{});

} // namespace opensn::cbc_gpu_kernel
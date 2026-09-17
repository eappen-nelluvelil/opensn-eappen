// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <new>

namespace opensn
{

constexpr std::size_t HardwareInterferenceSize =
#ifdef __cpp_lib_hardware_interference_size
  std::hardware_destructive_interference_size;
#else
  64;
#endif

} // namespace opensn

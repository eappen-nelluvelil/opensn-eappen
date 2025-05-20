// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include <cstddef>   // For std::size_t
#include <stdexcept> // For std::bad_alloc, std::invalid_argument
#include "framework/logging/log.h" // If you want to use opensn::log

namespace opensn
{

class AngularFluxMemoryPool
{
public:
  AngularFluxMemoryPool(size_t num_slots, size_t slot_size_in_doubles);
  ~AngularFluxMemoryPool();

  AngularFluxMemoryPool(const AngularFluxMemoryPool&) = delete;
  AngularFluxMemoryPool& operator=(const AngularFluxMemoryPool&) = delete;
  AngularFluxMemoryPool(AngularFluxMemoryPool&&) = delete;
  AngularFluxMemoryPool& operator=(AngularFluxMemoryPool&&) = delete;

  double* allocate_slot();
  void deallocate_slot(double* slot_ptr);

  size_t get_slot_size_in_doubles() const { return slot_size_in_doubles_; }
  size_t get_num_total_slots() const { return num_slots_; }
  size_t get_num_free_slots() const;
  size_t get_num_allocated_slots() const { return num_allocated_slots_; }

private:
  const size_t num_slots_;
  const size_t slot_size_in_doubles_;
  const size_t slot_size_in_bytes_;

  std::vector<double> backing_buffer_;
  std::vector<double*> free_list_;
  size_t num_allocated_slots_ = 0;
};

} // namespace opensn
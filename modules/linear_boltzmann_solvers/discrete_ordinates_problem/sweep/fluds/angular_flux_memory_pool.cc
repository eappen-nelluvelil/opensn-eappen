// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/angular_flux_memory_pool.h" // Adjust path as necessary
#include "framework/logging/log.h"

#include <algorithm>

namespace opensn
{

AngularFluxMemoryPool::AngularFluxMemoryPool(size_t num_slots, size_t slot_size_in_doubles)
  : num_slots_(num_slots),
    slot_size_in_doubles_(slot_size_in_doubles),
    slot_size_in_bytes_(slot_size_in_doubles * sizeof(double))
{
  if (num_slots_ == 0) // Allow slot_size_in_doubles to be 0 if num_slots is 0 (e.g. no live cells)
  {
    // opensn::log.Log0Warning() << "AngularFluxMemoryPool initialized with 0 slots.";
    return;
  }
  if (slot_size_in_doubles_ == 0)
  {
    throw std::invalid_argument(
      "AngularFluxMemoryPool: slot_size_in_doubles cannot be 0 if num_slots > 0.");
  }

  try
  {
    backing_buffer_.resize(num_slots_ * slot_size_in_doubles_);
  }
  catch (const std::bad_alloc& e)
  {
    throw std::runtime_error(
      std::string("AngularFluxMemoryPool: Failed to allocate backing buffer of size ") +
      std::to_string(num_slots_ * slot_size_in_bytes_ / (1024.0 * 1024.0)) +
      " MB. Exception: " + e.what());
  }

  free_list_.reserve(num_slots_);
  for (size_t i = 0; i < num_slots_; ++i)
  {
    free_list_.push_back(backing_buffer_.data() + (i * slot_size_in_doubles_));
  }
  // To mimic pool behavior (often allocates from a preferred end),
  // reversing makes pop_back take from the "beginning" of the conceptual pool blocks.
  std::reverse(free_list_.begin(), free_list_.end());
}

AngularFluxMemoryPool::~AngularFluxMemoryPool()
{
  if (num_allocated_slots_ != 0)
  {
    // Do something here
  }
}

double*
AngularFluxMemoryPool::allocate_slot()
{
  if (free_list_.empty())
  {
    // This is a critical error if the liveness analysis was correct.
    throw std::runtime_error("AngularFluxMemoryPool: Pool exhausted. Peak liveness count likely "
                             "underestimated or logic error in deallocation. "
                             "Total slots: " +
                             std::to_string(num_slots_) +
                             ", Allocated: " + std::to_string(num_allocated_slots_));
  }
  double* slot = free_list_.back();
  free_list_.pop_back();
  num_allocated_slots_++;
  return slot;
}

void
AngularFluxMemoryPool::deallocate_slot(double* slot_ptr)
{
  if (slot_ptr == nullptr)
  {
    // opensn::log.Log0Warning() << "AngularFluxMemoryPool: Attempt to deallocate a nullptr.";
    return; // Or throw, depending on desired strictness
  }

  // Basic check: ensure pointer is within the backing_buffer range
  // More advanced checks could ensure it's on a slot boundary.
  if (!backing_buffer_.empty()) // Avoid issues if buffer was never allocated (0 slots)
  {
    if (slot_ptr < backing_buffer_.data() ||
        slot_ptr >= (backing_buffer_.data() + backing_buffer_.size()))
    {
      throw std::invalid_argument(
        "AngularFluxMemoryPool: Attempt to deallocate pointer outside of pool's backing buffer.");
    }
  }
  else if (num_slots_ > 0)
  { // If num_slots > 0, buffer should not be empty
    throw std::logic_error("AngularFluxMemoryPool: Deallocating from an uninitialized or "
                           "zero-sized pool with non-zero slot count expectation.");
  }

  // TODO: For robust debugging, one could check if slot_ptr is already in free_list_ (double free)
  // This is O(N) with std::find, so only for debug builds.
  // if (std::find(free_list_.begin(), free_list_.end(), slot_ptr) != free_list_.end()) {
  //     throw std::runtime_error("AngularFluxMemoryPool: Double deallocation attempt.");
  // }

  free_list_.push_back(slot_ptr);
  if (num_allocated_slots_ == 0)
  { // Should not happen if deallocating a valid slot
    throw std::logic_error(
      "AngularFluxMemoryPool: Deallocated when no slots were thought to be allocated.");
  }
  num_allocated_slots_--;
}

size_t
AngularFluxMemoryPool::get_num_free_slots() const
{
  return free_list_.size();
}

} // namespace opensn
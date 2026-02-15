// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cassert>

namespace opensn
{

CBCD_AngleSet::CBCD_AngleSet(size_t id,
                             size_t num_groups,
                             const SPDS& spds,
                             std::shared_ptr<FLUDS>& fluds,
                             const std::vector<size_t>& angle_indices,
                             std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                             const MPICommunicatorSet& comm_set)
  : CBC_AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries, comm_set),
    stream_(crb::Stream::create()),
    device_angle_indices_(angles_.size(), stream_),
    cbcd_fluds_(static_cast<CBCD_FLUDS&>(*fluds_))
{
  crb::MemoryPinningManager angle_indices_pinner_(angles_);
  crb::copy(device_angle_indices_, angle_indices_pinner_, angles_.size(), 0, 0, stream_);
  cbcd_fluds_.GetStream() = stream_;
  cbcd_fluds_.AllocateLocalAndSavedPsi();
}

CBCD_AngleSet::~CBCD_AngleSet()
{
  device_angle_indices_.async_free(stream_);
}

void
CBCD_AngleSet::SetStartingLatch()
{
  starting_latch_ = std::make_unique<std::latch>(num_dependencies_);
}

void
CBCD_AngleSet::UpdateSweepDependencies(std::set<AngleSet*>& following_angle_sets)
{
  for (auto* as : following_angle_sets)
  {
    auto* cbcd_as = static_cast<CBCD_AngleSet*>(as);
    following_angle_sets_.push_back(cbcd_as);
    ++(cbcd_as->num_dependencies_);
  }
}

void
CBCD_AngleSet::ResetSweepBuffers()
{
  current_task_list_.clear();
  async_comm_.Reset();
  fluds_->ClearLocalAndReceivePsi();
  executed_ = false;
  // Reset pool allocator state
  auto& slot_map = cbcd_fluds_.GetCellToSlotMap();
  std::fill(slot_map.begin(), slot_map.end(), std::numeric_limits<std::uint32_t>::max());
  cbcd_fluds_.ResetFreeSlots();
}

AngleSetStatus
CBCD_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  assert(cbcd_sweep_chunk_ != nullptr);
  auto* comm = static_cast<CBC_AsynchronousCommunicator*>(GetCommunicator());

  // Initialize task list
  if (current_task_list_.empty())
    current_task_list_ = static_cast<const CBC_SPDS&>(spds_).GetTaskList();

  // For non-reflecting problems, num_dependencies_ == 0, so latch(0) is
  // immediately released and wait() returns without blocking.
  starting_latch_->wait();

  cbcd_fluds_.CopyIncomingBoundaryPsiToDevice(*cbcd_sweep_chunk_, this);

  bool kernel_in_flight = false;
  std::vector<Task*> in_flight_tasks;
  std::vector<std::uint64_t> in_flight_cell_ids;

  bool all_tasks_completed = false;
  while (not all_tasks_completed)
  {
    // Poll for kernel completion
    if (kernel_in_flight and stream_.is_completed())
    {
      // Copy back outgoing boundary (reflecting) and non-local psi to host
      cbcd_fluds_.CopyOutgoingPsiBackToHost(*cbcd_sweep_chunk_, this, in_flight_cell_ids);

      // Update task dependencies and pool allocator state
      for (auto* task : in_flight_tasks)
      {
        for (uint64_t succ : task->successors)
          --current_task_list_[succ].num_dependencies;
        task->completed = true;

        // Deallocate predecessor slots whose successors are all satisfied
        for (const auto& pred : task->predecessors)
        {
          ++current_task_list_[pred].num_satisfied_dependencies;
          if (current_task_list_[pred].num_satisfied_dependencies >=
              current_task_list_[pred].successors.size())
          {
            cbcd_fluds_.DeallocateDeviceSlot(current_task_list_[pred].reference_id);
          }
        }

        // If this cell has no local successors, deallocate immediately
        if (task->successors.empty())
          cbcd_fluds_.DeallocateDeviceSlot(task->reference_id);
      }

      // Send MPI data
      comm->SendData();
      in_flight_tasks.clear();
      in_flight_cell_ids.clear();
      kernel_in_flight = false;
    }

    // Receive MPI data
    {
      auto received = comm->ReceiveData();
      for (uint64_t t : received)
        --current_task_list_[t].num_dependencies;
      comm->SendData();
    }

    // Collect ready tasks and launch kernel
    if (not kernel_in_flight)
    {
      std::vector<Task*> ready_tasks;
      std::vector<std::uint64_t> ready_cell_ids;
      for (auto& task : current_task_list_)
      {
        if (task.num_dependencies == 0 and not task.completed)
        {
          ready_tasks.push_back(&task);
          ready_cell_ids.push_back(task.reference_id);
        }
      }

      if (not ready_tasks.empty())
      {
        // Allocate pool slots for ready cells
        for (const auto& cell_id : ready_cell_ids)
          cbcd_fluds_.AllocateDeviceSlot(cell_id);

        // Copy incoming non-local data for these cells to device
        cbcd_fluds_.CopyIncomingNonlocalPsiToDevice(this, ready_cell_ids);

        // Launch kernel on this angle set's caribou stream
        cbcd_sweep_chunk_->GPUSweep(*this, ready_cell_ids);

        in_flight_tasks = std::move(ready_tasks);
        in_flight_cell_ids = std::move(ready_cell_ids);
        kernel_in_flight = true;
      }
    }

    // Check overall completion
    if (not kernel_in_flight)
    {
      all_tasks_completed = std::all_of(
        current_task_list_.begin(),
        current_task_list_.end(),
        [](const Task& t) { return t.completed; });
    }
  } // while not all_tasks_completed

  // Flush all MPI sends
  while (not comm->SendData())
    ; // spin until all sends complete

  // Update boundary readiness and notify following angle sets
  for (auto& [bid, boundary] : boundaries_)
    boundary->UpdateAnglesReadyStatus(angles_);

  for (auto* following_as : following_angle_sets_)
    following_as->starting_latch_->count_down();

  // Copy saved psi from device to host
  cbcd_fluds_.CopySavedPsiFromDevice();
  stream_.add_callback([&]() {
    cbcd_fluds_.CopySavedPsiToDestinationPsi(*cbcd_sweep_chunk_, this);
  });

  executed_ = true;
  return AngleSetStatus::FINISHED;
}

} // namespace opensn
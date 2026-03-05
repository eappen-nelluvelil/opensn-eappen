// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_aggregated_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"
#include <cassert>
#include <thread>

namespace opensn
{

CBCD_AngleSet::CBCD_AngleSet(size_t id,
                             size_t num_groups,
                             const SPDS& spds,
                             std::shared_ptr<FLUDS>& fluds,
                             const std::vector<size_t>& angle_indices,
                             std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                             const MPICommunicatorSet& comm_set)
  : AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries),
    stream_(crb::Stream::create()),
    device_angle_indices_(angles_.size(), stream_),
    cbcd_fluds_(static_cast<CBCD_FLUDS&>(*fluds_)),
    cbcd_sweep_chunk_(nullptr),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds_))
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

bool
CBCD_AngleSet::TryInitialize()
{
  if (initialized_)
    return true;

  assert(cbcd_sweep_chunk_ != nullptr);
  assert(agg_comm_ != nullptr);

  // Non-blocking latch check. 
  // For non-reflecting problems, num_dependencies_ == 0,
  // so latch(0).try_wait() returns true immediately.
  if (not starting_latch_->try_wait())
    return false;

  // Initialize task list
  if (current_task_list_.empty())
    current_task_list_ = cbc_spds_.GetTaskList();

  cbcd_fluds_.CopyIncomingBoundaryPsiToDevice(*cbcd_sweep_chunk_, this);

  // Build initial ready queue from zero-dependency tasks.
  // After this, the queue is maintained incrementally.
  ready_queue_.reserve(current_task_list_.size());
  for (size_t i = 0; i < current_task_list_.size(); ++i)
    if (current_task_list_[i].num_dependencies == 0)
      ready_queue_.push_back(i);

  total_tasks_ = current_task_list_.size();
  completed_count_ = 0;
  kernel_in_flight_ = false;

  // Pre-compute which cells have outgoing reflecting boundary faces.
  // This allows counting down following angle set latches as soon as all
  // reflecting boundary data has been written, rather than waiting for
  // the entire sweep to complete.
  reflecting_boundary_cells_.clear();
  reflecting_boundary_completed_ = 0;
  latch_counted_down_ = false;
  if (not following_angle_sets_.empty())
  {
    const auto& outgoing_boundary_nodes = cbcd_fluds_.GetOutgoingBoundaryNodeMap();
    for (const auto& [cell_id, nodes] : outgoing_boundary_nodes)
    {
      for (const auto& node : nodes)
      {
        auto it = boundaries_.find(node.boundary_id);
        if (it != boundaries_.end() and it->second->IsReflecting())
        {
          reflecting_boundary_cells_.insert(cell_id);
          break;
        }
      }
    }
  }
  total_reflecting_boundary_cells_ = reflecting_boundary_cells_.size();

  // If there are no reflecting boundary cells but we have followers,
  // count down immediately — we have no reflecting data to produce.
  if (not following_angle_sets_.empty() and total_reflecting_boundary_cells_ == 0)
  {
    for (auto& [bid, boundary] : boundaries_)
      boundary->UpdateAnglesReadyStatus(angles_);
    for (auto* following_as : following_angle_sets_)
      following_as->starting_latch_->count_down();
    latch_counted_down_ = true;
  }

  initialized_ = true;
  return true;
}

bool
CBCD_AngleSet::TryAdvanceOneStep()
{
  if (not initialized_ or executed_)
    return false;

  bool any_work_done = false;

  // Poll for kernel completion
  if (kernel_in_flight_ and stream_.is_completed())
  {
    // Copy back outgoing boundary (reflecting) and non-local psi to host
    cbcd_fluds_.CopyOutgoingPsiBackToHost(*cbcd_sweep_chunk_, this, in_flight_cell_ids_);

    // Update task dependencies and push newly-ready successors to the queue
    for (auto* task : in_flight_tasks_)
    {
      for (uint64_t succ : task->successors)
      {
        if (--current_task_list_[succ].num_dependencies == 0)
          ready_queue_.push_back(succ);
      }
      task->completed = true;
      ++completed_count_;

      // Track reflecting boundary cell completions for early latch count-down
      if (not latch_counted_down_ and reflecting_boundary_cells_.count(task->reference_id))
        ++reflecting_boundary_completed_;
    }

    // Early latch count-down: once all reflecting boundary cells are done,
    // notify following angle sets so they can begin initialization.
    if (not latch_counted_down_ and
        reflecting_boundary_completed_ >= total_reflecting_boundary_cells_ and
        total_reflecting_boundary_cells_ > 0)
    {
      for (auto& [bid, boundary] : boundaries_)
        boundary->UpdateAnglesReadyStatus(angles_);
      for (auto* following_as : following_angle_sets_)
        following_as->starting_latch_->count_down();
      latch_counted_down_ = true;
    }

    in_flight_tasks_.clear();
    in_flight_cell_ids_.clear();
    kernel_in_flight_ = false;
    any_work_done = true;
  }

  // Pull received data from aggregated comm
  {
    auto received_batches = agg_comm_->DequeueIncoming(id_);
    if (not received_batches.empty())
    {
      for (auto& batch : received_batches)
      {
        for (auto& entry : batch)
        {
          cbcd_fluds_.ScatterReceivedFaceData(
            entry.cell_global_id, entry.face_id, entry.psi_data
          );
          auto local_id = spds_.GetGrid()->MapCellGlobalID2LocalID(entry.cell_global_id);
          if (--current_task_list_[local_id].num_dependencies == 0)
            ready_queue_.push_back(local_id);
        }
      }
      any_work_done = true;
    }
  }

  // Drain ready queue and launch kernel
  if (not kernel_in_flight_ and not ready_queue_.empty())
  {
    std::vector<Task*> ready_tasks;
    std::vector<std::uint64_t> ready_cell_ids;
    ready_tasks.reserve(ready_queue_.size());
    ready_cell_ids.reserve(ready_queue_.size());

    for (uint64_t task_idx : ready_queue_)
    {
      auto& task = current_task_list_[task_idx];
      ready_tasks.push_back(&task);
      ready_cell_ids.push_back(task.reference_id);
    }
    ready_queue_.clear();

    cbcd_sweep_chunk_->GPUSweep(*this, ready_cell_ids);

    in_flight_tasks_ = std::move(ready_tasks);
    in_flight_cell_ids_ = std::move(ready_cell_ids);
    kernel_in_flight_ = true;
    any_work_done = true;
  }

  // Check completion
  if (completed_count_ >= total_tasks_)
    FinalizeSweep();

  return any_work_done;
}

void
CBCD_AngleSet::FinalizeSweep()
{
  // Signal to the aggregated communicator that this angle set has no more outgoing data
  agg_comm_->SignalAngleSetComplete(id_);

  // Count down following angle set latches if not already done by early latch logic
  if (not latch_counted_down_)
  {
    for (auto& [bid, boundary] : boundaries_)
      boundary->UpdateAnglesReadyStatus(angles_);
    for (auto* following_as : following_angle_sets_)
      following_as->starting_latch_->count_down();
  }

  // Copy saved psi from device to host
  cbcd_fluds_.CopySavedPsiFromDevice();
  cbcd_fluds_.CopySavedPsiToDestinationPsi(*cbcd_sweep_chunk_, this);

  executed_ = true;
}

AngleSetStatus
CBCD_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  // Block until initialized (backward-compatible with one-thread-per-angle-set)
  while (not TryInitialize())
    std::this_thread::yield();

  // Main loop
  while (not executed_)
  {
    if (not TryAdvanceOneStep())
      std::this_thread::yield();
  }

  return AngleSetStatus::FINISHED;
}

void
CBCD_AngleSet::ResetSweepBuffers()
{
  current_task_list_.clear();
  cbcd_fluds_.ClearLocalAndReceivePsi();
  executed_ = false;
  initialized_ = false;
  ready_queue_.clear();
  kernel_in_flight_ = false;
  in_flight_tasks_.clear();
  in_flight_cell_ids_.clear();
  completed_count_ = 0;
  total_tasks_ = 0;
  reflecting_boundary_cells_.clear();
  reflecting_boundary_completed_ = 0;
  total_reflecting_boundary_cells_ = 0;
  latch_counted_down_ = false;
}

const double*
CBCD_AngleSet::PsiBoundary(uint64_t boundary_id,
                           unsigned int angle_num,
                           uint64_t cell_local_id,
                           unsigned int face_num,
                           unsigned int fi,
                           unsigned int g,
                           bool surface_source_active)
{
  if (boundaries_[boundary_id]->IsReflecting())
    return boundaries_[boundary_id]->PsiIncoming(cell_local_id, face_num, fi, angle_num, g);

  if (not surface_source_active)
    return boundaries_[boundary_id]->ZeroFlux(g);

  return boundaries_[boundary_id]->PsiIncoming(cell_local_id, face_num, fi, angle_num, g);
}

double*
CBCD_AngleSet::PsiReflected(uint64_t boundary_id,
                            unsigned int angle_num,
                            uint64_t cell_local_id,
                            unsigned int face_num,
                            unsigned int fi)
{
  return boundaries_[boundary_id]->PsiOutgoing(cell_local_id, face_num, fi, angle_num);
}

} // namespace opensn

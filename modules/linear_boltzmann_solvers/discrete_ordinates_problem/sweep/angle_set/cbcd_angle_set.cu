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

AngleSetStatus
CBCD_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  assert(cbcd_sweep_chunk_ != nullptr);
  assert(agg_comm_ != nullptr);

  // Initialize task list
  if (current_task_list_.empty())
    current_task_list_ = cbc_spds_.GetTaskList();

  // For non-reflecting problems, num_dependencies_ == 0, so latch(0) is
  // immediately released and wait() returns without blocking.
  starting_latch_->wait();

  cbcd_fluds_.CopyIncomingBoundaryPsiToDevice(*cbcd_sweep_chunk_, this);

  // Build initial ready queue from zero-dependency tasks (O(N) once at start).
  // After this, the queue is maintained incrementally — O(1) per task completion.
  std::vector<uint64_t> ready_queue;
  ready_queue.reserve(current_task_list_.size());
  for (size_t i = 0; i < current_task_list_.size(); ++i)
    if (current_task_list_[i].num_dependencies == 0)
      ready_queue.push_back(i);

  bool kernel_in_flight = false;
  std::vector<Task*> in_flight_tasks;
  std::vector<std::uint64_t> in_flight_cell_ids;

  size_t completed_count = 0;
  const size_t total_tasks = current_task_list_.size();

  while (completed_count < total_tasks)
  {
    bool any_work_done = false;

    // Poll for kernel completion
    if (kernel_in_flight and stream_.is_completed())
    {
      // Copy back outgoing boundary (reflecting) and non-local psi to host
      cbcd_fluds_.CopyOutgoingPsiBackToHost(*cbcd_sweep_chunk_, this, in_flight_cell_ids);

      // Update task dependencies and push newly-ready successors to the queue
      for (auto* task : in_flight_tasks)
      {
        for (uint64_t succ : task->successors)
        {
          if (--current_task_list_[succ].num_dependencies == 0)
            ready_queue.push_back(succ); // O(1) push — no full scan needed
        }
        task->completed = true;
        ++completed_count;
      }

      // Outgoing data is now enqueued via aggregated comm (done inside CopyOutgoingPsiBackToHost)
      in_flight_tasks.clear();
      in_flight_cell_ids.clear();
      kernel_in_flight = false;
      any_work_done = true;
    }

    // Pull received data from aggregated comm
    {
      auto received_entries = agg_comm_->DequeueIncoming(id_);
      if (not received_entries.empty())
      {
        for (auto& entry : received_entries)
        {
          // Merge into deplocs_outgoing_messages_ for NLUpwindPsi access
          cbcd_fluds_.GetDeplocsOutgoingMessages()[{entry.cell_global_id, entry.face_id}] =
            std::move(entry.psi_data);
          // Map cell_global_id to local task and push to ready queue if now unblocked
          auto local_id = spds_.GetGrid()->MapCellGlobalID2LocalID(entry.cell_global_id);
          if (--current_task_list_[local_id].num_dependencies == 0)
            ready_queue.push_back(local_id); // O(1) push — no full scan needed
        }
        any_work_done = true;
      }
    }

    // Drain ready queue and launch kernel (no O(N) scan!)
    if (not kernel_in_flight and not ready_queue.empty())
    {
      std::vector<Task*> ready_tasks;
      std::vector<std::uint64_t> ready_cell_ids;
      ready_tasks.reserve(ready_queue.size());
      ready_cell_ids.reserve(ready_queue.size());

      for (uint64_t task_idx : ready_queue)
      {
        auto& task = current_task_list_[task_idx];
        ready_tasks.push_back(&task);
        ready_cell_ids.push_back(task.reference_id);
      }
      ready_queue.clear();

      // Copy incoming non-local data for these cells to device
      cbcd_fluds_.CopyIncomingNonlocalPsiToDevice(this, ready_cell_ids);

      // Launch kernel on this angle set's caribou stream
      cbcd_sweep_chunk_->GPUSweep(*this, ready_cell_ids);

      in_flight_tasks = std::move(ready_tasks);
      in_flight_cell_ids = std::move(ready_cell_ids);
      kernel_in_flight = true;
      any_work_done = true;
    }

    if (not any_work_done)
      std::this_thread::yield();
  } // while completed_count < total_tasks

  // Signal to the aggregated communicator that this angle set has no more outgoing data
  agg_comm_->SignalAngleSetComplete(id_);

  // Update boundary readiness and notify following angle sets
  for (auto& [bid, boundary] : boundaries_)
    boundary->UpdateAnglesReadyStatus(angles_);

  for (auto* following_as : following_angle_sets_)
    following_as->starting_latch_->count_down();

  // Copy saved psi from device to host
  cbcd_fluds_.CopySavedPsiFromDevice();
  cbcd_fluds_.CopySavedPsiToDestinationPsi(*cbcd_sweep_chunk_, this);

  executed_ = true;
  return AngleSetStatus::FINISHED;
}

void
CBCD_AngleSet::ResetSweepBuffers()
{
  current_task_list_.clear();
  cbcd_fluds_.ClearLocalAndReceivePsi();
  executed_ = false;
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

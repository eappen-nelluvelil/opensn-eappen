// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbcd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_aggregated_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"
#include <cassert>
#include <cstring>
#include <thread>

namespace opensn
{

CBCD_AngleSet::CBCD_AngleSet(size_t id,
                             size_t num_groups,
                             const SPDS& spds,
                             std::shared_ptr<FLUDS>& fluds,
                             const std::vector<size_t>& angle_indices,
                             std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries)
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
  cbcd_fluds_.InitializeReflectingBoundaryNodes(boundaries_, angles_);
}

CBCD_AngleSet::~CBCD_AngleSet()
{
  device_angle_indices_.async_free(stream_);
}

void
CBCD_AngleSet::ResetDependencyCounter()
{
  dependency_counter_.store(num_dependencies_, std::memory_order_release);
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

  if (dependency_counter_.load(std::memory_order_acquire) != 0)
    return false;

  if (reference_ids_.empty())
  {
    const auto& tasks = cbc_spds_.GetTaskList();
    const size_t N = tasks.size();
    reference_ids_.resize(N);
    initial_deps_.resize(N);
    successor_offsets_.resize(N + 1, 0);

    for (size_t i = 0; i < N; ++i)
    {
      reference_ids_[i] = tasks[i].reference_id;
      initial_deps_[i] = static_cast<int>(tasks[i].num_dependencies);
      successor_offsets_[i + 1] = static_cast<uint32_t>(tasks[i].successors.size());
    }

    for (size_t i = 0; i < N; ++i)
      successor_offsets_[i + 1] += successor_offsets_[i];

    successor_data_.resize(successor_offsets_[N]);
    for (size_t i = 0; i < N; ++i)
    {
      uint32_t off = successor_offsets_[i];
      for (size_t j = 0; j < tasks[i].successors.size(); ++j)
        successor_data_[off + j] = tasks[i].successors[j];
    }

    for (size_t i = 0; i < N; ++i)
      if (initial_deps_[i] == 0)
        initial_ready_tasks_.push_back(static_cast<uint32_t>(i));

    ready_queue_.reserve(N);
    in_flight_task_indices_.reserve(N);
    deferred_cell_ids_.reserve(N);

    if (not following_angle_sets_.empty())
    {
      is_reflecting_task_.assign(N, 0);
      for (size_t i = 0; i < N; ++i)
      {
        uint64_t cell_id = reference_ids_[i];
        if (not cbcd_fluds_.GetReflectingOutgoingBoundaryFaces(cell_id).empty())
        {
          is_reflecting_task_[i] = 1;
          ++total_reflecting_tasks_;
        }
      }
    }
  }

  remaining_deps_.resize(initial_deps_.size());
  std::memcpy(remaining_deps_.data(), initial_deps_.data(), initial_deps_.size() * sizeof(int));

  cbcd_fluds_.CopyIncomingBoundaryPsiToDevice(*cbcd_sweep_chunk_, this);

  ready_queue_.assign(initial_ready_tasks_.begin(), initial_ready_tasks_.end());

  total_tasks_ = reference_ids_.size();
  completed_count_ = 0;
  kernel_in_flight_ = false;
  reflecting_tasks_completed_ = 0;
  followers_notified_ = false;

  TryNotifyFollowers();

  initialized_ = true;
  return true;
}

bool
CBCD_AngleSet::TryAdvanceOneStep()
{
  if (not initialized_ or executed_)
    return false;

  bool any_work_done = false;
  bool has_deferred_outgoing = false;

  if (kernel_in_flight_ and stream_.is_completed())
  {
    deferred_cell_ids_.clear();
    for (uint32_t task_idx : in_flight_task_indices_)
    {
      deferred_cell_ids_.push_back(reference_ids_[task_idx]);

      const uint32_t succ_begin = successor_offsets_[task_idx];
      const uint32_t succ_end = successor_offsets_[task_idx + 1];
      for (uint32_t j = succ_begin; j < succ_end; ++j)
      {
        if (--remaining_deps_[successor_data_[j]] == 0)
          ready_queue_.push_back(successor_data_[j]);
      }
      ++completed_count_;

      if (not followers_notified_ and is_reflecting_task_[task_idx])
        ++reflecting_tasks_completed_;
    }

    in_flight_task_indices_.clear();
    kernel_in_flight_ = false;
    has_deferred_outgoing = true;
    any_work_done = true;
  }

  any_work_done |= agg_comm_->DrainIncoming(id_,
    [this](const CBCD_AggregatedCommunicator::IncomingSection& section)
    {
      const auto* ptr = section.Data();
      const size_t num_entries = cbcd_wire::LoadUnalignedAndAdvance<size_t>(ptr);

      for (size_t e = 0; e < num_entries; ++e)
      {
        const auto entry_header = cbcd_wire::LoadUnalignedAndAdvance<cbcd_wire::EntryHeader>(ptr);
        const auto* psi_data = reinterpret_cast<const double*>(ptr);
        ptr += entry_header.data_size * sizeof(double);

        auto local_id =
          cbcd_fluds_.ScatterReceivedFaceData(entry_header.cell_global_id, entry_header.face_id, psi_data);
        if (--remaining_deps_[local_id] == 0)
          ready_queue_.push_back(static_cast<uint32_t>(local_id));
      }
    });

  if (not kernel_in_flight_ and not ready_queue_.empty())
  {
    auto& host_cell_ids = cbcd_fluds_.GetLocalCellIDs();
    unsigned int ready_count = 0;
    in_flight_task_indices_.clear();

    for (uint32_t task_idx : ready_queue_)
    {
      in_flight_task_indices_.push_back(task_idx);
      host_cell_ids[ready_count++] = static_cast<uint32_t>(reference_ids_[task_idx]);
    }
    ready_queue_.clear();

    cbcd_sweep_chunk_->Sweep(*this, ready_count);
    kernel_in_flight_ = true;
    any_work_done = true;
  }

  if (has_deferred_outgoing)
  {
    cbcd_fluds_.CopyOutgoingPsiBackToHost(this, deferred_cell_ids_);
    TryNotifyFollowers();
  }

  if (completed_count_ >= total_tasks_)
    FinalizeSweep();

  return any_work_done;
}

void
CBCD_AngleSet::TryNotifyFollowers()
{
  if (followers_notified_)
    return;

  if (following_angle_sets_.empty())
  {
    followers_notified_ = true;
    return;
  }

  if (reflecting_tasks_completed_ < total_reflecting_tasks_)
    return;

  for (auto& [bid, boundary] : boundaries_)
    boundary->UpdateAnglesReadyStatus(angles_);
  for (auto* following_as : following_angle_sets_)
    following_as->dependency_counter_.fetch_sub(1, std::memory_order_release);
  followers_notified_ = true;
}

void
CBCD_AngleSet::FinalizeSweep()
{
  agg_comm_->SignalAngleSetComplete(id_);

  TryNotifyFollowers();
  cbcd_fluds_.CopySavedPsiToDestinationPsi(*cbcd_sweep_chunk_, this);

  executed_ = true;
}

AngleSetStatus
CBCD_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBCD_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  while (not TryInitialize())
    std::this_thread::yield();

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
  cbcd_fluds_.ClearLocalAndReceivePsi();
  executed_ = false;
  initialized_ = false;
  ready_queue_.clear();
  kernel_in_flight_ = false;
  in_flight_task_indices_.clear();
  deferred_cell_ids_.clear();
  completed_count_ = 0;
  total_tasks_ = 0;
  reflecting_tasks_completed_ = 0;
  followers_notified_ = false;
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
  if (not boundaries_[boundary_id]->IsReflecting() and not surface_source_active)
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

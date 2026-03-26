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

  // Non-blocking dependency check.
  // For non-reflecting problems, num_dependencies_ == 0,
  // so dependency_counter_ is 0 immediately.
  if (dependency_counter_.load(std::memory_order_acquire) != 0)
    return false;

  // Build CSR representation of the task DAG on first use.
  // Extracts only the fields used in the hot path (reference_id, successors, num_dependencies),
  // laid out as flat contiguous arrays for cache-friendly traversal.
  if (reference_ids_.empty())
  {
    const auto& tasks = cbc_spds_.GetTaskList();
    const size_t N = tasks.size();
    reference_ids_.resize(N);
    initial_deps_.resize(N);
    successor_offsets_.resize(N + 1, 0);

    // First pass: extract per-task data and count successors.
    for (size_t i = 0; i < N; ++i)
    {
      reference_ids_[i] = tasks[i].reference_id;
      initial_deps_[i] = static_cast<int>(tasks[i].num_dependencies);
      successor_offsets_[i + 1] = static_cast<uint32_t>(tasks[i].successors.size());
    }

    // Prefix sum to build CSR offsets.
    for (size_t i = 0; i < N; ++i)
      successor_offsets_[i + 1] += successor_offsets_[i];

    // Second pass: flatten successor data into contiguous array.
    successor_data_.resize(successor_offsets_[N]);
    for (size_t i = 0; i < N; ++i)
    {
      uint32_t off = successor_offsets_[i];
      for (size_t j = 0; j < tasks[i].successors.size(); ++j)
        successor_data_[off + j] = tasks[i].successors[j];
    }

    // Pre-compute initial ready tasks (zero-dependency) — constant across sweeps.
    for (size_t i = 0; i < N; ++i)
      if (initial_deps_[i] == 0)
        initial_ready_tasks_.push_back(static_cast<uint32_t>(i));

    // Pre-allocate working buffers to full capacity (never re-allocates during sweep).
    ready_queue_.reserve(N);
    in_flight_task_indices_.reserve(N);
    deferred_cell_ids_.reserve(N);
  }

  // Reset dependency counts from cached initial values (fast memcpy of ints).
  remaining_deps_.resize(initial_deps_.size());
  std::copy(initial_deps_.begin(), initial_deps_.end(), remaining_deps_.begin());

  cbcd_fluds_.CopyIncomingBoundaryPsiToDevice(*cbcd_sweep_chunk_, this);

  // Populate initial ready queue from pre-computed zero-dependency tasks.
  for (uint32_t task_idx : initial_ready_tasks_)
    ready_queue_.push_back(task_idx);

  total_tasks_ = reference_ids_.size();
  completed_count_ = 0;
  kernel_in_flight_ = false;

  // Pre-compute which cells have outgoing reflecting boundary faces (cached across sweeps).
  // Uses vector<bool> indexed by cell_local_id for O(1) lookup instead of unordered_set.
  reflecting_boundary_completed_ = 0;
  latch_counted_down_ = false;
  if (is_reflecting_boundary_cell_.empty() and not following_angle_sets_.empty())
  {
    is_reflecting_boundary_cell_.assign(reference_ids_.size(), false);
    const auto& outgoing_boundary_nodes = cbcd_fluds_.GetOutgoingBoundaryNodeMap();
    for (size_t cell_id = 0; cell_id < outgoing_boundary_nodes.size(); ++cell_id)
    {
      const auto& nodes = outgoing_boundary_nodes[cell_id];
      for (const auto& node : nodes)
      {
        auto it = boundaries_.find(node.boundary_id);
        if (it != boundaries_.end() and it->second->IsReflecting())
        {
          is_reflecting_boundary_cell_[cell_id] = true;
          total_reflecting_boundary_cells_++;
          break;
        }
      }
    }
  }

  // If there are no reflecting boundary cells but we have followers,
  // count down immediately — we have no reflecting data to produce.
  if (not following_angle_sets_.empty() and total_reflecting_boundary_cells_ == 0)
  {
    for (auto& [bid, boundary] : boundaries_)
      boundary->UpdateAnglesReadyStatus(angles_);
    for (auto* following_as : following_angle_sets_)
      following_as->dependency_counter_.fetch_sub(1, std::memory_order_release);
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
  bool has_deferred_outgoing = false;

  // A: Poll for kernel completion — update dependencies only (defer outgoing data processing
  //    so the next GPU kernel can be launched sooner, overlapping GPU compute with host work).
  if (kernel_in_flight_ and stream_.is_completed())
  {
    // Update task dependencies and build deferred cell IDs for step D in one pass.
    // Uses CSR successor layout for cache-friendly contiguous traversal.
    deferred_cell_ids_.clear();
    for (uint32_t task_idx : in_flight_task_indices_)
    {
      deferred_cell_ids_.push_back(reference_ids_[task_idx]);

      const uint32_t succ_begin = successor_offsets_[task_idx];
      const uint32_t succ_end = successor_offsets_[task_idx + 1];
      for (uint32_t j = succ_begin; j < succ_end; ++j)
      {
        uint32_t succ = successor_data_[j];
        if (--remaining_deps_[succ] == 0)
          ready_queue_.push_back(succ);
      }
      ++completed_count_;

      // Track reflecting boundary cell completions for early dependency countdown
      if (not latch_counted_down_ and
          not is_reflecting_boundary_cell_.empty() and
          is_reflecting_boundary_cell_[reference_ids_[task_idx]])
        ++reflecting_boundary_completed_;
    }

    in_flight_task_indices_.clear();
    kernel_in_flight_ = false;
    has_deferred_outgoing = true;
    any_work_done = true;
  }

  // B: Pull received data from aggregated comm (lock-free callback drain).
  //    Each ByteArray is a raw wire-format section:
  //      [num_entries : size_t]
  //      per entry: [cell_global_id : uint64_t][face_id : uint][data_size : size_t][psi doubles]
  //    Psi data is read directly from the ByteArray buffer (zero per-face heap allocations).
  any_work_done |= agg_comm_->DrainIncoming(id_,
    [this](ByteArray&& section)
    {
      const auto* raw = section.Data().data();
      size_t offset = 0;

      size_t num_entries;
      std::memcpy(&num_entries, raw + offset, sizeof(size_t));
      offset += sizeof(size_t);

      for (size_t e = 0; e < num_entries; ++e)
      {
        uint64_t cell_global_id;
        std::memcpy(&cell_global_id, raw + offset, sizeof(uint64_t));
        offset += sizeof(uint64_t);

        unsigned int face_id;
        std::memcpy(&face_id, raw + offset, sizeof(unsigned int));
        offset += sizeof(unsigned int);

        size_t data_size;
        std::memcpy(&data_size, raw + offset, sizeof(size_t));
        offset += sizeof(size_t);

        // Read psi directly from the wire-format buffer — no intermediate copy.
        const auto* psi_data = reinterpret_cast<const double*>(raw + offset);
        offset += data_size * sizeof(double);

        auto local_id = cbcd_fluds_.ScatterReceivedFaceData(
          cell_global_id, face_id, psi_data);
        if (--remaining_deps_[local_id] == 0)
          ready_queue_.push_back(static_cast<uint32_t>(local_id));
      }
    });

  // C: Launch next kernel IMMEDIATELY (GPU starts working while we process outgoing data below).
  //    Write cell IDs directly to the FLUDS MappedHostVector (eliminates intermediate staging
  //    vector and the std::copy inside GPUSweep).
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

  // D: Process deferred outgoing data (overlaps with GPU kernel from step C).
  //    Safe because different kernel batches write to different cell positions in outgoing buffers.
  if (has_deferred_outgoing)
  {
    cbcd_fluds_.CopyOutgoingPsiBackToHost(*cbcd_sweep_chunk_, this, deferred_cell_ids_);

    // Early dependency countdown: once all reflecting boundary cells are done,
    // notify following angle sets so they can begin initialization.
    if (not latch_counted_down_ and
        reflecting_boundary_completed_ >= total_reflecting_boundary_cells_ and
        total_reflecting_boundary_cells_ > 0)
    {
      for (auto& [bid, boundary] : boundaries_)
        boundary->UpdateAnglesReadyStatus(angles_);
      for (auto* following_as : following_angle_sets_)
        following_as->dependency_counter_.fetch_sub(1, std::memory_order_release);
      latch_counted_down_ = true;
    }
  }

  // E: Check completion
  if (completed_count_ >= total_tasks_)
    FinalizeSweep();

  return any_work_done;
}

void
CBCD_AngleSet::FinalizeSweep()
{
  // Signal to the aggregated communicator that this angle set has no more outgoing data
  agg_comm_->SignalAngleSetComplete(id_);

  // Count down following angle set dependency counters if not already done by early logic
  if (not latch_counted_down_)
  {
    for (auto& [bid, boundary] : boundaries_)
      boundary->UpdateAnglesReadyStatus(angles_);
    for (auto* following_as : following_angle_sets_)
      following_as->dependency_counter_.fetch_sub(1, std::memory_order_release);
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
  // Note: CSR arrays (reference_ids_, successor_offsets_, successor_data_, initial_deps_,
  // initial_ready_tasks_) and is_reflecting_boundary_cell_ are cached across sweeps
  // (topology doesn't change). Only the working state is reset.
  cbcd_fluds_.ClearLocalAndReceivePsi();
  executed_ = false;
  initialized_ = false;
  ready_queue_.clear();
  kernel_in_flight_ = false;
  in_flight_task_indices_.clear();
  deferred_cell_ids_.clear();
  completed_count_ = 0;
  total_tasks_ = 0;
  reflecting_boundary_completed_ = 0;
  // total_reflecting_boundary_cells_ is preserved (computed once from cached data)
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

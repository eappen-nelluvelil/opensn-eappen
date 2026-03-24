// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/data_types/range.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"

namespace opensn
{

CBC_AngleSet::CBC_AngleSet(size_t id,
                           unsigned int num_groups,
                           const SPDS& spds,
                           std::shared_ptr<FLUDS>& fluds,
                           const std::vector<size_t>& angle_indices,
                           std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                           const MPICommunicatorSet& comm_set)
  : AngleSet(id, num_groups, spds, fluds, angle_indices, boundaries),
    cbc_spds_(dynamic_cast<const CBC_SPDS&>(spds_)),
    async_comm_(id, *fluds, comm_set)
{
}

AsynchronousCommunicator*
CBC_AngleSet::GetCommunicator()
{
  return static_cast<AsynchronousCommunicator*>(&async_comm_);
}

AngleSetStatus
CBC_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  CALI_CXX_MARK_SCOPE("CBC_AngleSet::AngleSetAdvance");

  if (executed_)
    return AngleSetStatus::FINISHED;

  // First call: initialize from immutable topology
  if (task_topology_ == nullptr)
  {
    task_topology_ = &cbc_spds_.GetTaskTopology();
    const size_t n = task_topology_->size();
    task_states_.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
      task_states_[i].num_dependencies = (*task_topology_)[i].initial_num_dependencies;
      task_states_[i].completed = false;
    }
    num_completed_ = 0;
    ready_tasks_.clear();
    ready_tasks_.reserve(n);
    for (size_t i = 0; i < n; ++i)
      if (task_states_[i].num_dependencies == 0)
        ready_tasks_.push_back(i);
  }

  sweep_chunk.SetAngleSet(*this);

  // Receive remote data and enqueue newly ready tasks
  auto tasks_who_received_data = async_comm_.ReceiveData();
  for (const uint64_t task_number : tasks_who_received_data)
  {
    if (--task_states_[task_number].num_dependencies == 0 and
        not task_states_[task_number].completed)
      ready_tasks_.push_back(task_number);
  }

  async_comm_.SendData();

  // Check if boundaries allow for execution
  for (auto& [bid, boundary] : boundaries_)
    if (not boundary->CheckAnglesReadyStatus(angles_))
      return AngleSetStatus::NOT_FINISHED;

  // Process all ready tasks
  const auto& topology = *task_topology_;
  while (not ready_tasks_.empty())
  {
    const auto task_idx = ready_tasks_.back();
    ready_tasks_.pop_back();

    auto& state = task_states_[task_idx];
    if (state.completed)
      continue;

    const auto& topo = topology[task_idx];

    sweep_chunk.SetCell(topo.cell_ptr, *this);
    sweep_chunk.Sweep(*this);

    // Notify communicator that this cell's outgoing non-local faces are written
    async_comm_.CellSwept(topo.cell_ptr->local_id);

    for (uint64_t local_task_num : topo.successors)
    {
      if (--task_states_[local_task_num].num_dependencies == 0 and
          not task_states_[local_task_num].completed)
        ready_tasks_.push_back(local_task_num);
    }

    state.completed = true;
    ++num_completed_;
    async_comm_.SendData();
  }

  async_comm_.SendData();

  const bool all_tasks_completed = (num_completed_ == task_topology_->size());
  const bool all_messages_sent = async_comm_.SendData();

  if (all_tasks_completed and all_messages_sent)
  {
    // Update boundary readiness
    for (auto& [bid, boundary] : boundaries_)
      boundary->UpdateAnglesReadyStatus(angles_);
    executed_ = true;
    return AngleSetStatus::FINISHED;
  }

  return AngleSetStatus::NOT_FINISHED;
}

void
CBC_AngleSet::ResetSweepBuffers()
{
  if (task_topology_)
  {
    const size_t n = task_topology_->size();
    for (size_t i = 0; i < n; ++i)
    {
      task_states_[i].num_dependencies = (*task_topology_)[i].initial_num_dependencies;
      task_states_[i].completed = false;
    }
  }
  ready_tasks_.clear();
  num_completed_ = 0;
  async_comm_.Reset();
  fluds_->ClearLocalAndReceivePsi();
  executed_ = false;
}

const double*
CBC_AngleSet::PsiBoundary(uint64_t boundary_id,
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
CBC_AngleSet::PsiReflected(uint64_t boundary_id,
                           unsigned int angle_num,
                           uint64_t cell_local_id,
                           unsigned int face_num,
                           unsigned int fi)
{
  return boundaries_[boundary_id]->PsiOutgoing(cell_local_id, face_num, fi, angle_num);
}

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/sweep_scheduler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/aah.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aah_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "framework/math/quadratures/angular/product_quadrature.h"
#include "framework/math/quadratures/angular/curvilinear_product_quadrature.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <unordered_map>

namespace opensn
{

namespace
{

bool
Compare(const RuleValues& a, const RuleValues& b)
{
  if (a.depth_of_graph != b.depth_of_graph)
    return a.depth_of_graph > b.depth_of_graph;
  if (a.sign_of_omegax != b.sign_of_omegax)
    return a.sign_of_omegax > b.sign_of_omegax;
  if (a.sign_of_omegay != b.sign_of_omegay)
    return a.sign_of_omegay > b.sign_of_omegay;
  return a.sign_of_omegaz > b.sign_of_omegaz;
}

bool
CompareCylindrical(const RuleValues& a, const RuleValues& b)
{
  if (a.sign_of_omegax != b.sign_of_omegax)
    return a.sign_of_omegax > b.sign_of_omegax;
  if (a.depth_of_graph != b.depth_of_graph)
    return a.depth_of_graph > b.depth_of_graph;
  if (a.sign_of_omegay != b.sign_of_omegay)
    return a.sign_of_omegay > b.sign_of_omegay;
  if (a.sign_of_omegaz != b.sign_of_omegaz)
    return a.sign_of_omegaz > b.sign_of_omegaz;
  if (a.azimuthal_order != b.azimuthal_order)
    return a.azimuthal_order < b.azimuthal_order;
  return a.set_index < b.set_index;
}

} // namespace

SweepScheduler::SweepScheduler(SchedulingAlgorithm scheduler_type,
                               AngleAggregation& angle_agg,
                               SweepChunk& sweep_chunk)
  : scheduler_type_(scheduler_type), angle_agg_(angle_agg), sweep_chunk_(sweep_chunk)
{
  if (scheduler_type_ == SchedulingAlgorithm::DEPTH_OF_GRAPH)
    InitializeAlgoDOG();

  if (scheduler_type_ == SchedulingAlgorithm::ALL_AT_ONCE)
  {
    pool_.Resize(angle_agg_.GetNumAngleSets());
    execution_order_.reserve(angle_agg_.GetNumAngleSets());
  }

  // Initialize delayed upstream data
  for (auto& angset : angle_agg_)
    angset->InitializeDelayedUpstreamData();

  if (scheduler_type_ == SchedulingAlgorithm::DEPTH_OF_GRAPH)
  {
    const auto* curvi_quad =
      dynamic_cast<const CurvilinearProductQuadrature*>(angle_agg_.GetQuadrature().get());
    if (curvi_quad && angle_agg_.GetQuadrature()->GetDimension() == 2 &&
        angle_agg_.GetCoordinateSystem() == CoordinateSystemType::CYLINDRICAL)
    {
      const auto& following_map = angle_agg_.GetFollowingAngleSetsMap();
      for (const auto& [from, to_set] : following_map)
        for (auto* to : to_set)
          preceding_angle_sets_[to].insert(from);
    }
  }

  // Get local max num messages accross anglesets
  int local_max_num_messages = 0;
  for (auto& angset : angle_agg_)
    local_max_num_messages = std::max(angset->GetMaxBufferMessages(), local_max_num_messages);

  // Reconcile all local maximums
  int global_max_num_messages = 0;
  mpi_comm.all_reduce(local_max_num_messages, global_max_num_messages, mpi::op::max<int>());

  // Propogate items back to sweep buffers
  for (auto& angset : angle_agg_)
    angset->SetMaxBufferMessages(global_max_num_messages);
}

void
SweepScheduler::InitializeAlgoDOG()
{
  const bool is_cylindrical = angle_agg_.GetQuadrature()->GetDimension() == 2 &&
                              angle_agg_.GetCoordinateSystem() == CoordinateSystemType::CYLINDRICAL;

  std::unordered_map<unsigned int, int> angle_order;
  const auto* curvi_quad =
    dynamic_cast<const CurvilinearProductQuadrature*>(angle_agg_.GetQuadrature().get());
  const auto* product_quad =
    dynamic_cast<const ProductQuadrature*>(angle_agg_.GetQuadrature().get());
  if (is_cylindrical && curvi_quad && product_quad)
  {
    int order = 0;
    for (const auto& dir_set : product_quad->GetDirectionMap())
      for (const auto dir_id : dir_set.second)
        angle_order.emplace(dir_id, order++);
  }
  // Load all anglesets in preperation for sorting
  size_t num_anglesets = angle_agg_.GetNumAngleSets();
  for (size_t as = 0; as < num_anglesets; ++as)
  {
    auto angleset = angle_agg_[as];
    const auto& spds = dynamic_cast<const AAH_SPDS&>(angleset->GetSPDS());

    const std::vector<STDG>& leveled_graph = spds.GetGlobalSweepPlanes();

    // Find location depth
    int loc_depth = -1;
    for (size_t level = 0; level < leveled_graph.size(); ++level)
    {
      for (size_t index = 0; index < leveled_graph[level].item_id.size(); ++index)
      {
        if (leveled_graph[level].item_id[index] == opensn::mpi_comm.rank())
        {
          loc_depth = static_cast<int>(leveled_graph.size() - level);
          break;
        }
      } // for locations in plane
    } // for sweep planes

    // Set up rule values
    if (loc_depth >= 0)
    {
      RuleValues new_rule_vals(angleset);
      new_rule_vals.depth_of_graph = loc_depth;
      new_rule_vals.set_index = as;

      const auto& omega = spds.GetOmega();
      new_rule_vals.sign_of_omegax = (omega.x >= 0) ? 2 : 1;
      new_rule_vals.sign_of_omegay = (omega.y >= 0) ? 2 : 1;
      new_rule_vals.sign_of_omegaz = (omega.z >= 0) ? 2 : 1;
      if (is_cylindrical && !angle_order.empty() && angleset->GetNumAngles() == 1)
      {
        const auto angle_idx = angleset->GetAngleIndices().front();
        const auto it = angle_order.find(angle_idx);
        if (it != angle_order.end())
          new_rule_vals.azimuthal_order = it->second;
      }

      rule_values_.push_back(new_rule_vals);
    }
    else
      throw std::runtime_error("InitializeAlgoDOG: Failed to find location depth");
  } // for anglesets

  std::stable_sort(
    rule_values_.begin(), rule_values_.end(), is_cylindrical ? &CompareCylindrical : &Compare);
}

void
SweepScheduler::ScheduleAlgoDOG(SweepChunk& sweep_chunk)
{
  // Reset dependency counter
  for (auto& angle_set : angle_agg_)
    angle_set->ResetDependencyCounter();

  bool finished = false;
  while (not finished)
  {
    finished = true;
    for (auto& rule_value : rule_values_)
    {
      auto angleset = rule_value.angle_set;
      AngleSetStatus status = angleset->AngleSetAdvance(sweep_chunk, AngleSetStatus::EXECUTE);
      if (status != AngleSetStatus::FINISHED)
        finished = false;
    }
  }

  // Receive delayed data
  opensn::mpi_comm.barrier();
  bool received_delayed_data = false;
  while (not received_delayed_data)
  {
    received_delayed_data = true;

    for (auto& angle_set : angle_agg_)
    {
      if (angle_set->FlushSendBuffers() == AngleSetStatus::MESSAGES_PENDING)
        received_delayed_data = false;

      if (not angle_set->ReceiveDelayedData())
        received_delayed_data = false;
    }
  }

  // Reset all
  for (auto& angle_set : angle_agg_)
    angle_set->ResetSweepBuffers();
}

void
SweepScheduler::ScheduleAlgoFIFO(SweepChunk& sweep_chunk)
{
  CALI_CXX_MARK_SCOPE("HostSweepProfile/CBC/Sweep");

  std::vector<CBC_AngleSet*> ready_angle_sets;
  std::vector<CBC_AngleSet*> blocked_angle_sets;
  std::vector<CBC_AngleSet*> next_ready_angle_sets;
  std::vector<CBC_AngleSet*> next_blocked_angle_sets;
  const auto num_angle_sets = angle_agg_.GetNumAngleSets();
  ready_angle_sets.reserve(num_angle_sets);
  blocked_angle_sets.reserve(num_angle_sets);
  next_ready_angle_sets.reserve(num_angle_sets);
  next_blocked_angle_sets.reserve(num_angle_sets);
  {
    CALI_CXX_MARK_SCOPE("HostSweepProfile/CBC/Reset");
    for (auto& angle_set : angle_agg_)
    {
      angle_set->ResetDependencyCounter();
      auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(*angle_set);
      cbc_angle_set.InitializeSweep();
      auto& queue = cbc_angle_set.HasReadyTasks() ? ready_angle_sets : blocked_angle_sets;
      queue.push_back(&cbc_angle_set);
    }
  }

  while (not ready_angle_sets.empty() or not blocked_angle_sets.empty())
  {
    ++cbc_scheduler_passes_;

    if (not ready_angle_sets.empty())
    {
      CALI_CXX_MARK_SCOPE("HostSweepProfile/CBC/Compute");
      for (auto* angle_set : ready_angle_sets)
      {
        ++cbc_active_angle_set_visits_;

        angle_set->AdvanceReadyTasks(sweep_chunk);
        if (angle_set->IsFinished())
          continue;

        auto& queue = angle_set->HasReadyTasks() ? next_ready_angle_sets : next_blocked_angle_sets;
        queue.push_back(angle_set);
      }
    }

    if (not blocked_angle_sets.empty())
    {
      CALI_CXX_MARK_SCOPE("HostSweepProfile/CBC/Communication");
      for (auto* angle_set : blocked_angle_sets)
      {
        ++cbc_active_angle_set_visits_;

        angle_set->ProgressCommunication();
        if (angle_set->IsFinished())
          continue;

        auto& queue = angle_set->HasReadyTasks() ? next_ready_angle_sets : next_blocked_angle_sets;
        queue.push_back(angle_set);
      }
    }

    ready_angle_sets.swap(next_ready_angle_sets);
    blocked_angle_sets.swap(next_blocked_angle_sets);
    next_ready_angle_sets.clear();
    next_blocked_angle_sets.clear();
  }

  {
    CALI_CXX_MARK_SCOPE("HostSweepProfile/CBC/DelayedDrain");
    opensn::mpi_comm.barrier();

    std::vector<CBC_AngleSet*> delayed_angle_sets;
    std::vector<CBC_AngleSet*> next_delayed_angle_sets;
    delayed_angle_sets.reserve(num_angle_sets);
    next_delayed_angle_sets.reserve(num_angle_sets);
    for (auto& angle_set : angle_agg_)
    {
      auto& cbc_angle_set = dynamic_cast<CBC_AngleSet&>(*angle_set);
      if (cbc_angle_set.NeedsDelayedDrain())
        delayed_angle_sets.push_back(&cbc_angle_set);
    }

    while (not delayed_angle_sets.empty())
    {
      for (auto* angle_set : delayed_angle_sets)
      {
        const bool sends_complete = angle_set->FlushSendBuffers() == AngleSetStatus::MESSAGES_SENT;
        const bool receives_complete = angle_set->ReceiveDelayedData();
        if (not sends_complete or not receives_complete)
          next_delayed_angle_sets.push_back(angle_set);
      }
      delayed_angle_sets.swap(next_delayed_angle_sets);
      next_delayed_angle_sets.clear();
    }
  }

  {
    CALI_CXX_MARK_SCOPE("HostSweepProfile/CBC/Reset");
    for (auto& angle_set : angle_agg_)
      angle_set->ResetSweepBuffers();
  }

  std::uint64_t ready_bursts = 0;
  std::uint64_t ready_tasks = 0;
  CBC_AsynchronousCommunicator::Statistics communication;
  for (const auto& angle_set : angle_agg_)
  {
    const auto& cbc_angle_set = dynamic_cast<const CBC_AngleSet&>(*angle_set);
    const auto& angle_statistics = cbc_angle_set.GetStatistics();
    const auto& comm_statistics = cbc_angle_set.GetCommunicationStatistics();
    ready_bursts += angle_statistics.ready_bursts;
    ready_tasks += angle_statistics.ready_tasks;
    communication.empty_probes += comm_statistics.empty_probes;
    communication.messages_sent += comm_statistics.messages_sent;
    communication.messages_received += comm_statistics.messages_received;
    communication.bytes_sent += comm_statistics.bytes_sent;
    communication.bytes_received += comm_statistics.bytes_received;
    communication.records_sent += comm_statistics.records_sent;
    communication.records_received += comm_statistics.records_received;
    communication.send_progress_calls += comm_statistics.send_progress_calls;
  }

  const auto messages = communication.messages_sent + communication.messages_received;
  const auto records = communication.records_sent + communication.records_received;
  cali_set_global_uint_byname("opensn.cbc.sched_passes", cbc_scheduler_passes_);
  cali_set_global_uint_byname("opensn.cbc.aset_visits", cbc_active_angle_set_visits_);
  cali_set_global_uint_byname("opensn.cbc.empty_probes", communication.empty_probes);
  cali_set_global_uint_byname("opensn.cbc.msg_sent", communication.messages_sent);
  cali_set_global_uint_byname("opensn.cbc.msg_recv", communication.messages_received);
  cali_set_global_uint_byname("opensn.cbc.bytes_sent", communication.bytes_sent);
  cali_set_global_uint_byname("opensn.cbc.bytes_recv", communication.bytes_received);
  cali_set_global_uint_byname("opensn.cbc.records_sent", communication.records_sent);
  cali_set_global_uint_byname("opensn.cbc.records_recv", communication.records_received);
  cali_set_global_uint_byname("opensn.cbc.ready_bursts", ready_bursts);
  cali_set_global_uint_byname("opensn.cbc.ready_tasks", ready_tasks);
  cali_set_global_uint_byname("opensn.cbc.send_tests", communication.send_progress_calls);
  cali_set_global_double_byname(
    "opensn.cbc.recs_per_msg",
    messages == 0 ? 0.0 : static_cast<double>(records) / static_cast<double>(messages));
  cali_set_global_double_byname(
    "opensn.cbc.burst_avg",
    ready_bursts == 0 ? 0.0 : static_cast<double>(ready_tasks) / static_cast<double>(ready_bursts));
}

#ifndef __OPENSN_WITH_GPU__

void
SweepScheduler::ScheduleAlgoAAO(SweepChunk& sweep_chunk)
{
  throw std::runtime_error("SweepScheduler::ScheduleAlgoAAO: AAO scheduling is only "
                           "available for builds with GPU support.");
}

void
SweepScheduler::ScheduleAlgoAsyncFIFO(SweepChunk& sweep_chunk)
{
  throw std::runtime_error("SweepScheduler::ScheduleAlgoAsyncFIFO: ASYNC_FIFO scheduling is only "
                           "available for builds with GPU support.");
}

#endif // __OPENSN_WITH_GPU__

void
SweepScheduler::Sweep()
{
  if (scheduler_type_ == SchedulingAlgorithm::ASYNC_FIFO)
    ScheduleAlgoAsyncFIFO(sweep_chunk_);
  else if (scheduler_type_ == SchedulingAlgorithm::FIRST_IN_FIRST_OUT)
    ScheduleAlgoFIFO(sweep_chunk_);
  else if (scheduler_type_ == SchedulingAlgorithm::ALL_AT_ONCE)
    ScheduleAlgoAAO(sweep_chunk_);
  else if (scheduler_type_ == SchedulingAlgorithm::DEPTH_OF_GRAPH)
    ScheduleAlgoDOG(sweep_chunk_);
}

void
SweepScheduler::PrepareForSweep(bool use_boundary_source, bool zero_incoming_delayed_psi)
{
  if (zero_incoming_delayed_psi)
    angle_agg_.ZeroIncomingDelayedPsi();
  angle_agg_.ZeroOutgoingDelayedPsi();
  sweep_chunk_.ZeroDestinationPsi();
  sweep_chunk_.ZeroDestinationPhi();
  sweep_chunk_.SetBoundarySourceActiveFlag(use_boundary_source);
}

} // namespace opensn

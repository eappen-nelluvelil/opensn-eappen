// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/sweep_scheduler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_message_transport.h"
#include "framework/mpi/sweep_communicator.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/aah.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aah_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "framework/math/quadratures/angular/product_quadrature.h"
#include "framework/math/quadratures/angular/curvilinear_product_quadrature.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include <algorithm>
#include <bit>
#include <cstring>
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

  if (scheduler_type_ == SchedulingAlgorithm::RECEIVE_EVENTS)
    InitializeReceiveEvents();

  if (scheduler_type_ == SchedulingAlgorithm::ALL_AT_ONCE)
  {
    pool_.Resize(angle_agg_.GetNumAngleSets());
    execution_order_.reserve(angle_agg_.GetNumAngleSets());
  }
  else if (scheduler_type_ == SchedulingAlgorithm::ASYNC_FIFO)
  {
    constexpr std::size_t num_communicator_threads = 1;
    const auto worker_limit = opensn_num_threads > num_communicator_threads
                                ? opensn_num_threads - num_communicator_threads
                                : 1;
    const auto num_workers =
      std::max<std::size_t>(1, std::min(angle_agg_.GetNumAngleSets(), worker_limit));
    pool_.Resize(num_workers);
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
SweepScheduler::InitializeReceiveEvents()
{
  const auto num_sets = angle_agg_.GetNumAngleSets();
  event_angle_sets_.reserve(num_sets);
  for (std::size_t i = 0; i < num_sets; ++i)
  {
    auto* angle_set = dynamic_cast<CBC_AngleSet*>(angle_agg_[i].get());
    OpenSnLogicalErrorIf(
      angle_set == nullptr or angle_set->GetID() != i or
        angle_set->GetEventCommunicator() == nullptr,
      "CBC receive events require dense angle IDs and a private groupset context.");
    if (i != 0)
    {
      const auto* first = event_angle_sets_.front();
      OpenSnLogicalErrorIf(
        angle_set->GetEventCommunicator() != first->GetEventCommunicator() or
          static_cast<std::size_t>(angle_set->GetMessageTag() - first->GetMessageTag()) != i,
        "CBC receive events require one context and contiguous message tags.");
    }
    event_angle_sets_.push_back(angle_set);
  }
  event_ready_.resize(num_sets / 64 + (num_sets % 64 != 0));
  event_finished_.resize(num_sets);
  if (num_sets != 0)
  {
    event_transport_ =
      std::make_shared<CBC_MessageTransport>(event_angle_sets_.front()->GetEventCommunicatorPtr(),
                                             event_angle_sets_.front()->GetPacketLimit());
    for (auto* angle_set : event_angle_sets_)
      angle_set->SetMessageTransport(event_transport_);
  }
}

void
SweepScheduler::ScheduleReceiveEvents(SweepChunk& sweep_chunk)
{
  const auto num_sets = event_angle_sets_.size();
  std::fill(event_ready_.begin(), event_ready_.end(), 0);
  std::fill(event_finished_.begin(), event_finished_.end(), 0);
  std::size_t cursor = 0;
  std::size_t queued = 0;
  std::size_t completed = 0;
  std::size_t pending_faces = 0;
  const auto enqueue = [&](std::size_t id)
  {
    auto& word = event_ready_[id / 64];
    const auto mask = std::uint64_t{1} << (id % 64);
    if ((word & mask) != 0 or event_finished_[id])
      return;
    word |= mask;
    ++queued;
  };

  // Every set gets an initial visit, including sets with zero local cells.
  for (std::size_t i = 0; i < num_sets; ++i)
  {
    event_angle_sets_[i]->ResetDependencyCounter();
    pending_faces += event_angle_sets_[i]->GetPendingNormalFaces();
    enqueue(i);
  }

  if (num_sets != 0)
  {
    const auto& comm = event_transport_->GetCommunicator();
    const auto first_tag = event_angle_sets_.front()->GetMessageTag();
    const auto dispatch = [&](const mpi::Status& status)
    {
      auto packet = event_transport_->Receive(status);
      while (not packet.empty())
      {
        OpenSnLogicalErrorIf(packet.size() < CBC_MessageTransport::FRAME_BYTES,
                             "Truncated host CBC frame header.");
        std::uint32_t tag = 0;
        std::uint32_t count = 0;
        std::memcpy(&tag, packet.data(), sizeof(tag));
        std::memcpy(&count, packet.data() + sizeof(tag), sizeof(count));
        packet = packet.subspan(CBC_MessageTransport::FRAME_BYTES);
        OpenSnLogicalErrorIf(
          count == 0 or count > packet.size() or tag < static_cast<std::uint32_t>(first_tag) or
            static_cast<std::size_t>(tag - static_cast<std::uint32_t>(first_tag)) >= num_sets,
          "Invalid host CBC frame length or angle tag.");
        const auto id = static_cast<std::size_t>(tag - static_cast<std::uint32_t>(first_tag));
        OpenSnLogicalErrorIf(event_finished_[id],
                             "CBC normal data arrived after local completion.");
        const auto previous_faces = event_angle_sets_[id]->GetPendingNormalFaces();
        const bool ready =
          event_angle_sets_[id]->ReceivePacket(status.source(), packet.first(count));
        pending_faces -= previous_faces - event_angle_sets_[id]->GetPendingNormalFaces();
        if (ready)
          enqueue(id);
        packet = packet.subspan(count);
      }
    };

    while (completed != num_sets)
    {
      mpi::Status status;
      // One dispatcher receives all normal packets. Exact source/tag receives follow
      // each probe; no other thread or groupset can consume a matched message.
      while (pending_faces != 0 and comm.iprobe(mpi::ANY_SOURCE, mpi::ANY_TAG, status))
        dispatch(status);
      event_transport_->Progress();

      if (queued == 0)
      {
        event_transport_->Flush();
        OpenSnLogicalErrorIf(pending_faces == 0,
                             "Host CBC has blocked tasks but no pending normal receives.");
        // All partial peer packets start before waiting. With an acyclic current-
        // sweep task graph, unfinished work now depends on a normal receive. Blocking
        // MPI provides progress without eager buffers or a background progress thread.
        OpenSnMPICall(
          MPI_Probe(mpi::ANY_SOURCE, mpi::ANY_TAG, static_cast<MPI_Comm>(comm), status));
        dispatch(status);
        continue;
      }

      // Pick the next ready ID in cyclic order, skipping entire empty words.
      // Arrival order must not replace the original round-robin angle priority.
      auto word_id = cursor / 64;
      auto word = event_ready_[word_id] & (~std::uint64_t{0} << (cursor % 64));
      while (word == 0)
      {
        if (++word_id == event_ready_.size())
          word_id = 0;
        word = event_ready_[word_id];
      }
      const auto bit = static_cast<std::size_t>(std::countr_zero(word));
      const auto id = word_id * 64 + bit;
      event_ready_[word_id] &= ~(std::uint64_t{1} << bit);
      cursor = id + 1;
      if (cursor == num_sets)
        cursor = 0;
      --queued;
      auto* angle_set = event_angle_sets_[id];
      if (angle_set->AdvanceReadyTasks(sweep_chunk, AngleSetStatus::EXECUTE) ==
          AngleSetStatus::FINISHED)
      {
        event_finished_[id] = 1;
        ++completed;
        const auto& following = angle_agg_.GetFollowingAngleSetsMap();
        const auto it = following.find(angle_set);
        if (it != following.end())
          for (const auto* next : it->second)
            if (next->IsDependencyResolved())
              enqueue(next->GetID());
      }
    }
  }

  // All normal face data has been consumed globally. Delayed sends begin only
  // after this barrier; they cannot be mistaken for normal-phase wakeup events.
  if (event_transport_)
    event_transport_->Flush();
  opensn::mpi_comm.barrier();
  if (event_transport_)
    event_transport_->Finish();
  bool drained = false;
  while (not drained)
  {
    drained = true;
    for (auto& angle_set : angle_agg_)
    {
      if (angle_set->FlushSendBuffers() == AngleSetStatus::MESSAGES_PENDING)
        drained = false;
      if (not angle_set->ReceiveDelayedData())
        drained = false;
    }
  }
  for (auto& angle_set : angle_agg_)
    angle_set->ResetSweepBuffers();
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

    const int loc_depth = spds.GetLocationDepth();

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
  // Reset dependency counter
  for (auto& angle_set : angle_agg_)
    angle_set->ResetDependencyCounter();

  // Loop over AngleSetGroups
  bool finished = false;
  while (not finished)
  {
    finished = true;

    for (auto& angle_set : angle_agg_)
    {
      AngleSetStatus status = angle_set->AngleSetAdvance(sweep_chunk, AngleSetStatus::EXECUTE);
      if (status != AngleSetStatus::FINISHED)
        finished = false;
    } // for angleset
  } // while not finished

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
  if (scheduler_type_ == SchedulingAlgorithm::RECEIVE_EVENTS)
    ScheduleReceiveEvents(sweep_chunk_);
  else if (scheduler_type_ == SchedulingAlgorithm::ASYNC_FIFO)
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

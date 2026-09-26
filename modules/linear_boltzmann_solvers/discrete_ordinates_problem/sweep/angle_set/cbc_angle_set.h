// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include <cstdint>
#include <vector>

namespace opensn
{

class CBC_SPDS;

/// Host CBC angle set.
class CBC_AngleSet : public AngleSet
{
public:
  CBC_AngleSet(std::size_t id,
               const LBSGroupset& groupset,
               const SPDS& spds,
               std::shared_ptr<FLUDS>& fluds,
               const std::vector<std::size_t>& angle_indices,
               std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
               int max_mpi_message_size,
               const SweepCommunicator& sweep_communicator,
               std::shared_ptr<const SweepCommunicator> event_communicator = nullptr);

  /// Return the CBC asynchronous communicator.
  AsynchronousCommunicator* GetCommunicator() override;

  /// Initialize delayed upstream storage and receive state.
  void InitializeDelayedUpstreamData() override { async_comm_.InitializeDelayedUpstreamData(); }

  /// Return the maximum number of buffered messages.
  int GetMaxBufferMessages() const override { return max_buffer_messages_; }

  /// Set the maximum number of buffered messages.
  void SetMaxBufferMessages(int new_max) override { max_buffer_messages_ = new_max; }

  /// Advance ready CBC tasks and progress asynchronous communication.
  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  /// Advance local tasks without probing receives; the event scheduler dispatches them.
  AngleSetStatus AdvanceReadyTasks(SweepChunk& sweep_chunk, AngleSetStatus permission);

  /// Consume one normal-flux frame from the shared transport.
  bool ReceivePacket(int source_rank, std::span<const char> packet);

  /// Install a groupset transport before execution.
  void SetMessageTransport(std::shared_ptr<CBC_MessageTransport> transport)
  {
    async_comm_.SetMessageTransport(std::move(transport));
  }

  std::size_t GetPacketLimit() const { return async_comm_.GetPacketLimit(); }

  const std::shared_ptr<const SweepCommunicator>& GetEventCommunicatorPtr() const
  {
    return event_communicator_;
  }

  /// Private groupset context, owned until all angle-set requests have been drained.
  const SweepCommunicator* GetEventCommunicator() const { return event_communicator_.get(); }

  /// Message tag used by the groupset event dispatcher.
  int GetMessageTag() const { return async_comm_.GetMessageTag(); }

  /// Normal faces not yet completely received in this sweep.
  std::size_t GetPendingNormalFaces() const { return async_comm_.GetPendingNormalFaces(); }

  /// Flush pending CBC send buffers.
  AngleSetStatus FlushSendBuffers() override
  {
    const bool all_messages_sent = async_comm_.FlushSendBuffers();
    return all_messages_sent ? AngleSetStatus::MESSAGES_SENT : AngleSetStatus::MESSAGES_PENDING;
  }

  /// Reset task and communication state before another sweep.
  void ResetSweepBuffers() override;

  /// Receive delayed data until all delayed upstream locations are complete.
  bool ReceiveDelayedData() override { return async_comm_.ReceiveDelayedData(); }

protected:
  /// Initialize task dependencies once per sweep, including when a receive arrives first.
  void InitializeTasks();
  /// Apply one notification for each fully received nonlocal face.
  void UpdateReceivedDependencies();
  /// CBC sweep-plane data structure.
  const CBC_SPDS& cbc_spds_;
  /// Current CBC task list.
  const std::vector<Task>* task_list_ = nullptr;
  /// Unsatisfied dependency count by task.
  std::vector<unsigned int> remaining_dependency_counts_;
  /// Ready task stack.
  std::vector<std::uint32_t> ready_tasks_;
  /// Reusable buffer for newly unlocked received tasks.
  std::vector<std::uint32_t> received_task_buffer_;
  /// Number of completed tasks in the current sweep.
  std::size_t num_completed_tasks_ = 0;
  /// Maximum number of buffered messages.
  int max_buffer_messages_ = 0;
  /// Declared before the communicator so the MPI context outlives its requests.
  std::shared_ptr<const SweepCommunicator> event_communicator_;
  /// CBC asynchronous communicator.
  CBC_AsynchronousCommunicator async_comm_;
};

} // namespace opensn

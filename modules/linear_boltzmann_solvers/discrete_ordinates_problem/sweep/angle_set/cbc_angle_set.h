// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"

namespace opensn
{

class CBC_SPDS;

/// Angle set for the host cell-by-cell sweep algorithm.
class CBC_AngleSet : public AngleSet
{
public:
  /// Construct a host CBC angle set.
  CBC_AngleSet(size_t id,
               unsigned int num_groups,
               const SPDS& spds,
               std::shared_ptr<FLUDS>& fluds,
               const std::vector<size_t>& angle_indices,
               std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
               const MPICommunicatorSet& comm_set);

  /// Return the host CBC asynchronous communicator.
  AsynchronousCommunicator* GetCommunicator() override;

  /// Initialize delayed upstream data.
  void InitializeDelayedUpstreamData() override {}

  /// Return the maximum number of buffered messages.
  int GetMaxBufferMessages() const override { return 0; }

  /// Set the maximum number of buffered messages.
  void SetMaxBufferMessages(int) override {}

  /// Advance the CBC angle set by sweeping all currently ready cells.
  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  /// Progress pending host CBC send buffers.
  AngleSetStatus FlushSendBuffers() override
  {
    const bool all_messages_sent =
      (not async_comm_.HasPendingCommunication()) or async_comm_.SendData();
    return all_messages_sent ? AngleSetStatus::MESSAGES_SENT : AngleSetStatus::MESSAGES_PENDING;
  }

  /// Reset transient sweep buffers.
  void ResetSweepBuffers() override;

  /// Receive delayed data.
  bool ReceiveDelayedData() override { return true; }

  /// Return boundary angular-flux data.
  const double* PsiBoundary(uint64_t boundary_id,
                            unsigned int angle_num,
                            uint64_t cell_local_id,
                            unsigned int face_num,
                            unsigned int fi,
                            unsigned int g,
                            bool surface_source_active) override;

  /// Return reflected boundary angular-flux data.
  double* PsiReflected(uint64_t boundary_id,
                       unsigned int angle_num,
                       uint64_t cell_local_id,
                       unsigned int face_num,
                       unsigned int fi) override;

protected:
  /// CBC sweep-plane data structure.
  const CBC_SPDS& cbc_spds_;

  /// Mutable task list for the active sweep.
  std::vector<Task> current_task_list_;

  /// Ready task stack.
  std::vector<std::uint64_t> ready_tasks_;

  /// Scratch buffer for tasks that received nonlocal data.
  std::vector<std::uint64_t> received_task_buffer_;

  /// Number of completed local tasks in the active sweep.
  size_t num_completed_tasks = 0;

  /// Host CBC asynchronous communicator.
  CBC_AsynchronousCommunicator async_comm_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include <cstdint>
#include <vector>

namespace opensn
{

class Cell;
class CBC_SPDS;

/// CBC angle set implementation.
class CBC_AngleSet : public AngleSet
{
public:
  CBC_AngleSet(size_t id,
               unsigned int num_groups,
               const SPDS& spds,
               std::shared_ptr<FLUDS>& fluds,
               const std::vector<size_t>& angle_indices,
               std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
               int max_mpi_message_size,
               const MPICommunicatorSet& comm_set);

  AsynchronousCommunicator* GetCommunicator() override;

  void InitializeDelayedUpstreamData() override { async_comm_.InitializeDelayedUpstreamData(); }

  int GetMaxBufferMessages() const override { return 0; }

  void SetMaxBufferMessages(int /*new_max*/) override {}

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  AngleSetStatus FlushSendBuffers() override
  {
    const bool all_messages_sent = async_comm_.FlushSendBuffers();
    return all_messages_sent ? AngleSetStatus::MESSAGES_SENT : AngleSetStatus::MESSAGES_PENDING;
  }

  void ResetSweepBuffers() override;

  bool ReceiveDelayedData() override { return async_comm_.ReceiveDelayedData(); }

  const double* PsiBoundary(uint64_t boundary_id,
                            unsigned int angle_num,
                            uint64_t cell_local_id,
                            unsigned int face_num,
                            unsigned int fi,
                            unsigned int g,
                            bool surface_source_active) override;

  double* PsiReflected(uint64_t boundary_id,
                       unsigned int angle_num,
                       uint64_t cell_local_id,
                       unsigned int face_num,
                       unsigned int fi) override;

protected:
  bool IncomingBoundaryFacesReady(const Cell& cell) const;

  void InitializeTaskState();

  void ResetTaskState();

  const CBC_SPDS& cbc_spds_;
  /// Immutable CBC task graph for the active sweep.
  const std::vector<Task>* task_list_ = nullptr;
  /// Immutable dependency counts keyed by task index.
  std::vector<unsigned int> initial_dependencies_;
  /// Immutable initial ready-task stack.
  std::vector<std::uint32_t> initial_ready_tasks_;
  /// Mutable dependency counts keyed by task index.
  std::vector<unsigned int> remaining_dependencies_;
  /// Mutable completion flags keyed by task index.
  std::vector<unsigned char> completed_tasks_;
  /// Ready task stack.
  std::vector<std::uint32_t> ready_tasks_;
  /// Scratch buffer for tasks that received nonlocal data.
  std::vector<std::uint32_t> received_task_buffer_;
  /// Number of completed local tasks in the active sweep.
  size_t num_completed_tasks_ = 0;
  CBC_AsynchronousCommunicator async_comm_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/communicators/cbc_async_comm.h"

namespace opensn
{

struct Task;
class CBC_SPDS;

class CBC_AngleSet : public AngleSet
{
protected:
  // NEW: Structure to track runtime dependencies
  struct TaskRuntimeInfo
  {
    unsigned int remaining_local_dependencies = 0;
    unsigned int remaining_non_local_dependencies = 0;
    bool completed = false;
    // Future idea: Add TaskStatus enum if needed for more complex states
  };

  const CBC_SPDS& cbc_spds_;
  std::vector<Task> current_task_list_;

  // NEW: Runtime tracking parallel to current_task_list_
  std::vector<TaskRuntimeInfo> task_runtime_info_;

  CBC_ASynchronousCommunicator async_comm_;

public:
  CBC_AngleSet(size_t id,
               size_t num_groups,
               const SPDS& spds,
               std::shared_ptr<FLUDS>& fluds,
               const std::vector<size_t>& angle_indices,
               std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
               const MPICommunicatorSet& comm_set);

  // NEW: Initialize the task_runtime_info_ structure
  void InitializeTaskRuntimeInfo();

  AsynchronousCommunicator* GetCommunicator() override;

  void InitializeDelayedUpstreamData() override {}

  int GetMaxBufferMessages() const override { return 0; }

  void SetMaxBufferMessages(int new_max) override {}

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  AngleSetStatus FlushSendBuffers() override
  {
    const bool all_messages_sent = async_comm_.SendData();
    return all_messages_sent ? AngleSetStatus::MESSAGES_SENT : AngleSetStatus::MESSAGES_PENDING;
  }

  void ResetSweepBuffers() override;

  bool ReceiveDelayedData() override { return true; }

  const double* PsiBoundary(uint64_t boundary_id,
                            unsigned int angle_num,
                            uint64_t cell_local_id,
                            unsigned int face_num,
                            unsigned int fi,
                            int g,
                            bool surface_source_active) override;

  double* PsiReflected(uint64_t boundary_id,
                       unsigned int angle_num,
                       uint64_t cell_local_id,
                       unsigned int face_num,
                       unsigned int fi) override;
};

} // namespace opensn

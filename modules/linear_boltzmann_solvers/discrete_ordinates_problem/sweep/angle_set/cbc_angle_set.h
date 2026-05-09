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

/// CBC angle set.
class CBC_AngleSet : public AngleSet
{
public:
  CBC_AngleSet(size_t id,
               const LBSGroupset& groupset,
               const SPDS& spds,
               std::shared_ptr<FLUDS>& fluds,
               const std::vector<size_t>& angle_indices,
               std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
               const MPICommunicatorSet& comm_set);

  AsynchronousCommunicator* GetCommunicator() override;

  void InitializeDelayedUpstreamData() override { async_comm_.InitializeDelayedUpstreamData(); }

  int GetMaxBufferMessages() const override { return 0; }

  void SetMaxBufferMessages(int max_buffer_messages) override { (void)max_buffer_messages; }

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  AngleSetStatus FlushSendBuffers() override
  {
    const bool all_messages_sent = async_comm_.FlushSendBuffers();
    return all_messages_sent ? AngleSetStatus::MESSAGES_SENT : AngleSetStatus::MESSAGES_PENDING;
  }

  void ResetSweepBuffers() override;

  bool ReceiveDelayedData() override { return async_comm_.ReceiveDelayedData(); }

protected:
  const CBC_SPDS& cbc_spds_;
  const std::vector<Task>* task_list_ = nullptr;
  std::vector<unsigned int> remaining_dependencies_;
  std::vector<unsigned char> completed_tasks_;
  std::vector<std::uint32_t> ready_tasks_;
  std::vector<std::uint32_t> received_task_buffer_;
  size_t num_completed_tasks_ = 0;
  CBC_AsynchronousCommunicator async_comm_;
};

} // namespace opensn

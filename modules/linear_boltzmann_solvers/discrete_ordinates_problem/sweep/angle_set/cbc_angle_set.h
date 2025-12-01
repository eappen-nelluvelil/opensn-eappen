// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"

namespace opensn
{

struct Task;
class CBC_SPDS;

class CBC_AngleSet : public AngleSet
{
protected:
  const CBC_SPDS& cbc_spds_;
  std::vector<Task> current_task_list_;
  CBC_ASynchronousCommunicator async_comm_;
  CBC_FLUDS& cbc_fluds_;
  bool use_gpus_ = false;

public:
  CBC_AngleSet(size_t id,
               size_t num_groups,
               const SPDS& spds,
               std::shared_ptr<FLUDS>& fluds,
               const std::vector<size_t>& angle_indices,
               std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
               const MPICommunicatorSet& comm_set,
               bool use_gpu);
  
  ~CBC_AngleSet();

  AsynchronousCommunicator* GetCommunicator() override;

  void InitializeDelayedUpstreamData() override {}

  int GetMaxBufferMessages() const override { return 0; }

  void SetMaxBufferMessages(int new_max) override {}

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  AngleSetStatus CPUAngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission);

  AngleSetStatus GPUAngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission);

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

  const CBC_SPDS& GetSPDS() { return cbc_spds_; }

  CBC_FLUDS& GetFLUDS() { return cbc_fluds_; }

  std::vector<Task>& GetCurrentTaskList() { return current_task_list_; }

  void ReceiveData() { tasks_who_received_data_ = async_comm_.ReceiveData(); }

  void SendData() { async_comm_.SendData(); }

  bool HasSentBoundaryData() const { return has_set_boundary_data_; }

  void SetBoundaryDataFlag(bool flag) { has_set_boundary_data_ = flag; }

  bool GetExecutionStatus() const { return executed_; }

  void SetExecutionStatus(bool status) { executed_ = status; }

  void CreateCUDAStream();

  void DestroyCUDAStream();

  std::vector<uint64_t> tasks_who_received_data_;
  std::vector<Task*> ready_tasks_;

  void* stream_ptr = nullptr;
  bool has_set_boundary_data_ = false;
};

} // namespace opensn
// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include <any>

namespace caribou
{
class Stream;
}

namespace opensn
{

namespace cbc_gpu_kernel
{
struct GraphArguments;
}

struct Task;
class CBC_SPDS;

class CBC_AngleSet : public AngleSet
{
protected:
  const CBC_SPDS& cbc_spds_;
  std::vector<Task> current_task_list_;
  CBC_ASynchronousCommunicator async_comm_;
  bool use_gpus_;
  void* stream_ = nullptr;
  std::any cuda_graph_;
  std::any cuda_graph_exec_;
  std::any boundary_event_;
  std::any cell_params_;

public:
  CBC_AngleSet(size_t id,
               size_t num_groups,
               const SPDS& spds,
               std::shared_ptr<FLUDS>& fluds,
               const std::vector<size_t>& angle_indices,
               std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
               const MPICommunicatorSet& comm_set,
               bool use_gpus);

  ~CBC_AngleSet() override;

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

  void AssociateAngleSetWithFLUDS();

  /// Create caribou stream for asynchronous kernel launches and data transfers
  void CreateStream();

  /// Create CUDA graph for sweep chunk execution
  void CreateCUDAGraph();

  void InitializeBoundaryEvent();

  void DestroyBoundaryEvent();

  std::any GetBoundaryEvent() const { return boundary_event_; }

  std::any GetCUDAGraphExec() const { return cuda_graph_exec_; }

  /// Build and instantiate CUDA graph
  void BuildAndInstantiateCUDAGraph(std::vector<cbc_gpu_kernel::GraphArguments>& graph_args);

  /// Destroy caribou stream
  void DestroyStream();

  /// Destroy CUDA graph
  void DestroyCUDAGraph();

  /// Get the void pointer to the caribou stream, which can be casted to caribou::Stream
  void* GetStream() const { return stream_; }

  std::vector<Task>& GetCurrentTaskList() { return current_task_list_; }
};

} // namespace opensn

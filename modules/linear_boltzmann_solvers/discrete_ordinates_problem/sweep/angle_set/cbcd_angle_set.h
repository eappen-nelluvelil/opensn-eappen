// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "caribou/main.hpp"
#include <latch>
#include <memory>
#include <set>

namespace crb = caribou;

namespace opensn
{

class CBCDSweepChunk;

/// CBC angle set for device.
class CBCD_AngleSet : public CBC_AngleSet
{
public:
  CBCD_AngleSet(size_t id,
                size_t num_groups,
                const SPDS& spds,
                std::shared_ptr<FLUDS>& fluds,
                const std::vector<size_t>& angle_indices,
                std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                const MPICommunicatorSet& comm_set);

  ~CBCD_AngleSet();

  crb::Stream& GetStream() { return stream_; }

  std::uint32_t* GetDeviceAngleIndices() { return device_angle_indices_.get(); }

  std::vector<Task>& GetCurrentTaskList() { return current_task_list_; }

  void SetSweepChunk(CBCDSweepChunk* sweep_chunk) { cbcd_sweep_chunk_ = sweep_chunk; }

  /// Initialize the starting latch.
  void SetStartingLatch();
  
  /// Must be called after UpdateSweepDependencies and before launching threads.
  void UpdateSweepDependencies(std::set<AngleSet*>& following_angle_sets) override;

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus exec_status) override;

  /// Reset sweep buffers for next sweep.
  void ResetSweepBuffers() override;

private:
  /// Associated crb::Stream.
  crb::Stream stream_;
  /// Angle indices on GPU.
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;
  /// Reference to CBCD_FLUDS owned by this angleset.
  CBCD_FLUDS& cbcd_fluds_;
  /// Pointer to the sweep chunk.
  CBCDSweepChunk* cbcd_sweep_chunk_;
  /// Number of angle sets this one must wait for before starting.
  std::size_t num_dependencies_ = 0;
  /// A starting latch.
  /// Thread waits here until all predecessors count_down.
  /// A latch(0) is immediately released.
  std::unique_ptr<std::latch> starting_latch_;
  /// Anglesets whose latches this angleset counts down upon completion.
  std::vector<CBCD_AngleSet*> following_angle_sets_; 
};

} // namespace opensn
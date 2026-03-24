// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace mpi = mpicpp_lite;

namespace opensn
{

class MPICommunicatorSet;
class CBC_FLUDS;

class CBC_ASynchronousCommunicator : public AsynchronousCommunicator
{
public:
  explicit CBC_ASynchronousCommunicator(size_t angle_set_id,
                                        FLUDS& fluds,
                                        const MPICommunicatorSet& comm_set)
    : AsynchronousCommunicator(fluds, comm_set),
      angle_set_id_(angle_set_id),
      cbc_fluds_(dynamic_cast<CBC_FLUDS&>(fluds))
  {
    InitStates();
  }

  /// Notify that a cell has been swept — decrements per-rank send counters.
  void CellSwept(uint64_t cell_local_id);

  /// Posts sends for any rank whose buffer is fully populated, tests completion.
  bool SendData();

  /// Probes for incoming messages and receives directly into the FLUDS recv buffer.
  std::vector<uint64_t> ReceiveData();

  void Reset();

protected:
  const size_t angle_set_id_;
  CBC_FLUDS& cbc_fluds_;

  struct SendState
  {
    mpi::Request request;
    bool send_initiated = false;
    bool completed = false;
  };
  std::vector<SendState> send_states_;
  std::vector<size_t> send_remaining_; ///< Per-message face counter.

  struct RecvState
  {
    bool completed = false;
  };
  std::vector<RecvState> recv_states_;

  void InitStates();
};

} // namespace opensn

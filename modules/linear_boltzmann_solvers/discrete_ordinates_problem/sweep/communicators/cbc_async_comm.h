// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/async_comm.h"
#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <map>
#include <set>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace mpi = mpicpp_lite;

namespace opensn
{

class MPICommunicatorSet;
class ByteArray;

class CBC_AsynchronousCommunicator : public AsynchronousCommunicator
{
public:
  explicit CBC_AsynchronousCommunicator(size_t angle_set_id,
                                        FLUDS& fluds,
                                        const MPICommunicatorSet& comm_set);

  std::vector<double>& InitGetDownwindMessageData(int location_id,
                                                   uint64_t cell_global_id,
                                                   unsigned int face_id,
                                                   size_t angle_set_id,
                                                   size_t data_size) override;

  bool SendData();

  /// Sends accumulated delayed outgoing data as a single message per destination.
  /// Must be called after the sweep loop completes and before the barrier.
  bool SendDelayedData();

  std::vector<uint64_t> ReceiveData();

  /// Receives delayed data from delayed location dependencies after the main sweep.
  bool ReceiveDelayedData();

  void Reset()
  {
    outgoing_message_queue_.clear();
    send_buffer_.clear();
    delayed_send_buffer_.clear();
    delayed_sends_initiated_ = false;
    delayed_deps_received_.clear();
  }

protected:
  const size_t angle_set_id_;

  /// Set of partition IDs that are delayed location successors for this SPDS.
  std::set<int> delayed_successor_set_;

  // location_id, cell_global_id, face_id
  using MessageKey = std::tuple<int, uint64_t, unsigned int>;
  std::map<MessageKey, std::vector<double>> outgoing_message_queue_;

  struct BufferItem
  {
    int destination = 0;
    mpi::Request mpi_request;
    bool send_initiated = false;
    bool completed = false;
    ByteArray data_array;
  };
  std::vector<BufferItem> send_buffer_;

  /// Separate send buffer for delayed outgoing data (one message per destination).
  std::vector<BufferItem> delayed_send_buffer_;
  bool delayed_sends_initiated_ = false;

  /// Tracks which delayed dependency locations have already been received.
  std::set<int> delayed_deps_received_;

  // cell_global_id, face_id
  using CellFaceKey = std::pair<uint64_t, unsigned int>;
  void MergeDeplocsOutgoingMessages(std::map<CellFaceKey, std::vector<double>>& received_messages);
};

} // namespace opensn

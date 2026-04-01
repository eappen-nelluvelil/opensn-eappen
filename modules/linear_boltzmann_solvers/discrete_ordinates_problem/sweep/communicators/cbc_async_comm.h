// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <limits>
#include <unordered_map>
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
                                        const MPICommunicatorSet& comm_set)
    : AsynchronousCommunicator(fluds, comm_set), angle_set_id_(angle_set_id)
  {
  }

  std::vector<double>& InitGetDownwindMessageData(int location_id,
                                                  uint64_t cell_global_id,
                                                  unsigned int face_id,
                                                  size_t angle_set_id,
                                                  size_t data_size);

  void InitializeDelayedUpstreamData();
  bool SendData();
  bool FlushSendBuffers();
  std::vector<uint64_t> ReceiveData();
  bool ReceiveDelayedData();

  void Reset()
  {
    outgoing_message_queue_.clear();
    delayed_outgoing_message_queue_.clear();
    send_buffer_.clear();
    std::fill(delayed_recv_done_.begin(), delayed_recv_done_.end(), false);
    delayed_completion_markers_queued_ = false;
  }

protected:
  static constexpr std::uint64_t delayed_done_cell_id_ = std::numeric_limits<std::uint64_t>::max();
  static constexpr unsigned int delayed_done_face_id_ = std::numeric_limits<unsigned int>::max();

  const size_t angle_set_id_;

  using MessageKey = std::tuple<int, std::uint64_t, unsigned int>;

  struct MessageKeyHash
  {
    std::size_t operator()(const MessageKey& key) const noexcept
    {
      size_t h = std::hash<int>{}(std::get<0>(key));
      h ^= std::hash<std::uint64_t>{}(std::get<1>(key)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<unsigned int>{}(std::get<2>(key)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  struct BufferItem
  {
    int destination = 0;
    mpi::Request mpi_request;
    bool send_initiated = false;
    bool completed = false;
    ByteArray data_array;
  };

  std::unordered_map<MessageKey, std::vector<double>, MessageKeyHash> outgoing_message_queue_;
  std::unordered_map<MessageKey, std::vector<double>, MessageKeyHash>
    delayed_outgoing_message_queue_;
  std::vector<BufferItem> send_buffer_;
  std::vector<bool> delayed_recv_done_;
  bool delayed_completion_markers_queued_ = false;

private:
  void QueueMessagesForSend(
    std::unordered_map<MessageKey, std::vector<double>, MessageKeyHash>& message_queue);
  bool AllBufferedSendsCompleted();
  void QueueDelayedCompletionMarkers();
};

} // namespace opensn

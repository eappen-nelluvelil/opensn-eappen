// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/async_comm.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace opensn
{

namespace mpi = mpicpp_lite;

class MPICommunicatorSet;
class CBC_FLUDS;

/// CBC asynchronous communicator.
class CBC_AsynchronousCommunicator : public AsynchronousCommunicator
{
public:
  explicit CBC_AsynchronousCommunicator(size_t angle_set_id,
                                        FLUDS& fluds,
                                        const MPICommunicatorSet& comm_set);

  /// Queue a complete downwind face payload for sending.
  void QueueDownwindMessage(size_t peer_index,
                            size_t incoming_face_slot,
                            std::span<const double> payload);

  bool SendData();

  /// Receive available nonlocal face payloads and return unlocked cells.
  void ReceiveData(std::vector<std::uint32_t>& cells_who_received_data);

  bool HasPendingCommunication() const noexcept { return not send_buffer_.empty(); }

  void Reset();

protected:
  const size_t angle_set_id_;
  const int location_id_;
  const mpi::Communicator& receive_comm_;
  CBC_FLUDS& cbc_fluds_;
  std::size_t num_receive_sources_ = 0;

  /// Destination-batched nonblocking send buffer.
  struct BufferItem
  {
    size_t peer_index = 0;
    const mpi::Communicator* comm = nullptr;
    int rank = 0;
    bool send_initiated = false;
    std::vector<char> data;
  };

  struct SendPeer
  {
    const mpi::Communicator* comm = nullptr;
    int rank = 0;
  };

  BufferItem& GetOpenSendBuffer(size_t peer_index);

  std::vector<BufferItem> send_buffer_;
  std::vector<mpi::Request> send_requests_;
  std::vector<BufferItem> reusable_send_buffers_;
  std::vector<char> receive_buffer_;
  std::vector<SendPeer> send_peers_;
  std::vector<size_t> open_send_buffer_indices_;
  static constexpr size_t INVALID_BUFFER_INDEX = std::numeric_limits<size_t>::max();
};

} // namespace opensn

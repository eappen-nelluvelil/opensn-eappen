// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <boost/unordered/unordered_flat_map.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class MPICommunicatorSet;

class CBC_AsynchronousCommunicator : public AsynchronousCommunicator
{
public:
  explicit CBC_AsynchronousCommunicator(size_t angle_set_id,
                                        FLUDS& fluds,
                                        const MPICommunicatorSet& comm_set);

  void QueueDownwindMessage(int destination,
                            std::uint64_t cell_global_id,
                            unsigned int face_id,
                            std::span<const double> data);

  bool SendData();

  std::vector<uint64_t> ReceiveData();
  void ReceiveData(std::vector<std::uint64_t>& cells_who_received_data);

  [[nodiscard]] bool HasPendingCommunication() const noexcept { return not send_buffer_.empty(); }

  void Reset();

protected:
  const size_t angle_set_id_;
  const int location_id_;
  const mpi::Communicator& receive_comm_;
  std::vector<int> receive_source_ranks_;

  struct BufferItem
  {
    int destination = 0;
    mpi::Request mpi_request;
    bool send_initiated = false;
    bool completed = false;
    std::vector<std::byte> data;
  };

  struct SendPeer
  {
    const mpi::Communicator* comm = nullptr;
    int rank = 0;
  };

  const SendPeer& GetSendPeer(int destination);
  BufferItem& GetOpenSendBuffer(int destination);

  std::vector<BufferItem> send_buffer_;
  std::vector<BufferItem> reusable_send_buffers_;
  std::vector<std::byte> receive_buffer_;
  boost::unordered_flat_map<int, SendPeer> send_peers_;
  boost::unordered_flat_map<int, size_t> open_send_buffer_indices_;
};

} // namespace opensn

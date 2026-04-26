// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <vector>
#include <cstddef>
#include <cstdint>

namespace mpi = mpicpp_lite;

namespace opensn
{

class MPICommunicatorSet;

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

  bool SendData();

  std::vector<uint64_t> ReceiveData();
  void ReceiveData(std::vector<std::uint64_t>& cells_who_received_data);

  [[nodiscard]] bool HasPendingCommunication() const noexcept
  {
    return (not outgoing_message_queue_.empty()) or (not send_buffer_.empty());
  }

  void Reset();

protected:
  const size_t angle_set_id_;

  struct PendingMessage
  {
    int destination = 0;
    std::uint64_t cell_global_id = 0;
    unsigned int face_id = 0;
    std::vector<double> data;
  };

  std::vector<PendingMessage> outgoing_message_queue_;

  struct BufferItem
  {
    int destination = 0;
    mpi::Request mpi_request;
    bool send_initiated = false;
    bool completed = false;
    std::vector<std::byte> data;
  };
  std::vector<BufferItem> send_buffer_;
  std::vector<BufferItem> reusable_send_buffers_;
  std::vector<std::byte> receive_buffer_;
};

} // namespace opensn

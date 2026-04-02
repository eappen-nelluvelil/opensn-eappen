// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/async_comm.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class CBC_FLUDS;
class MPICommunicatorSet;

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
                                                  size_t data_size);

  bool SendData();

  std::vector<uint64_t> ReceiveData();

  void Reset()
  {
    outgoing_message_queue_.clear();
    send_buffer_.clear();
  }

protected:
  const size_t angle_set_id_;

  /// location_id, cell_global_id, face_id
  using MessageKey = std::tuple<int, std::uint64_t, unsigned int>;

  /// boost::hash_combine hash function for MessageKey.
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

  std::unordered_map<MessageKey, std::vector<double>, MessageKeyHash> outgoing_message_queue_;

  struct BufferItem
  {
    int destination = 0;
    mpi::Request mpi_request;
    bool send_initiated = false;
    bool completed = false;
    std::vector<std::byte> data;
  };
  std::vector<BufferItem> send_buffer_;
  CBC_FLUDS& cbc_fluds_;
  std::vector<std::byte> receive_buffer_;
  std::vector<size_t> destination_buffer_bytes_;
  std::vector<size_t> destination_buffer_indices_;

private:
  void QueueOutgoingMessages();
};

} // namespace opensn

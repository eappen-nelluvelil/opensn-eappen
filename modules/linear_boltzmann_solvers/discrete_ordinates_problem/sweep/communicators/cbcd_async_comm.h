// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/async_comm.h"
#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace opensn
{

namespace mpi = mpicpp_lite;

class MPICommunicatorSet;
class ByteArray;
class FLUDS;
class CBCD_FLUDS;

/// CBCD asynchronous communicator.
class CBCD_AsynchronousCommunicator : public AsynchronousCommunicator
{
public:
  explicit CBCD_AsynchronousCommunicator(size_t angle_set_id,
                                         FLUDS& fluds,
                                         const MPICommunicatorSet& comm_set);

  /// Return outgoing face payload storage for a downwind location.
  std::vector<double>& InitGetDownwindMessageData(int location_id,
                                                  std::uint64_t cell_global_id,
                                                  unsigned int face_id,
                                                  size_t data_size);

  bool SendData();

  std::vector<std::uint64_t> ReceiveData();

  void Reset()
  {
    outgoing_message_queue_.clear();
    send_buffer_.clear();
  }

protected:
  const size_t angle_set_id_;
  CBCD_FLUDS& cbcd_fluds_;

  struct MessageKey
  {
    int location_id = 0;
    std::uint64_t cell_global_id = 0;
    unsigned int face_id = 0;
    bool operator==(const MessageKey&) const = default;
  };

  struct MessageKeyHash
  {
    std::size_t operator()(const MessageKey& key) const noexcept
    {
      auto seed = Mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.location_id)));
      seed = Combine(seed, Mix(key.cell_global_id));
      seed = Combine(seed, Mix(static_cast<std::uint64_t>(key.face_id)));
      return static_cast<std::size_t>(seed);
    }

  private:
    static std::uint64_t Combine(std::uint64_t seed, std::uint64_t value) noexcept
    {
      constexpr std::uint64_t offset = 0x9e3779b97f4a7c15ULL;
      return Mix(seed ^ (value + offset + (seed << 6) + (seed >> 2)));
    }

    static std::uint64_t Mix(std::uint64_t value) noexcept
    {
      constexpr std::uint64_t multiplier = 0xe9846afb1a615dULL;
      value ^= value >> 32;
      value *= multiplier;
      value ^= value >> 32;
      value *= multiplier;
      value ^= value >> 28;
      return value;
    }
  };

  std::unordered_map<MessageKey, std::vector<double>, MessageKeyHash> outgoing_message_queue_;

  /// Destination-batched nonblocking send buffer.
  struct BufferItem
  {
    int destination = 0;
    mpi::Request mpi_request;
    bool send_initiated = false;
    bool completed = false;
    ByteArray data_array;
  };

  std::vector<BufferItem> send_buffer_;
};

} // namespace opensn

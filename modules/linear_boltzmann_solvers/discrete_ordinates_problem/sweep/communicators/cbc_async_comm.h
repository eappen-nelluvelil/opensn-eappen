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

namespace opensn
{

namespace mpi = mpicpp_lite;

class MPICommunicatorSet;
class CBC_FLUDS;

/// Nonblocking MPI communicator for host CBC nonlocal face flux payloads.
class CBC_AsynchronousCommunicator : public AsynchronousCommunicator
{
public:
  explicit CBC_AsynchronousCommunicator(size_t angle_set_id,
                                        FLUDS& fluds,
                                        const MPICommunicatorSet& comm_set);

  /// Queue a complete downwind face payload for asynchronous transmission.
  void QueueDownwindMessage(int destination,
                            size_t incoming_face_slot,
                            std::span<const double> payload);

  [[nodiscard]] bool SendData();

  /// Receive all currently available nonlocal face payloads into a caller-owned buffer.
  void ReceiveData(std::vector<std::uint32_t>& cells_who_received_data);

  [[nodiscard]] bool HasPendingCommunication() const noexcept { return not send_buffer_.empty(); }

  void Reset();

protected:
  /// MPI tag shared by the angle set.
  const size_t angle_set_id_;

  /// Current location.
  const int location_id_;

  /// Receiver-side communicator.
  const mpi::Communicator& receive_comm_;

  /// Slot-addressed receive storage.
  CBC_FLUDS& cbc_fluds_;

  /// Ranks that may send nonlocal payloads to this location.
  std::vector<int> receive_source_ranks_;

  /// Destination-batched nonblocking send buffer.
  struct BufferItem
  {
    /// Destination location.
    int destination = 0;

    /// Destination communicator.
    const mpi::Communicator* comm = nullptr;

    /// Destination rank in `comm`.
    int rank = 0;

    /// Active nonblocking send.
    mpi::Request mpi_request;

    /// Posted-send flag.
    bool send_initiated = false;

    /// Completed-send flag.
    bool completed = false;

    /// Packed face records.
    std::vector<std::byte> data;
  };

  /// Cached destination routing.
  struct SendPeer
  {
    /// Destination communicator.
    const mpi::Communicator* comm = nullptr;

    /// Destination rank in `comm`.
    int rank = 0;
  };

  /// Return cached MPI routing data for a destination location.
  [[nodiscard]] const SendPeer& GetSendPeer(int destination);

  /// Return the open send buffer for a destination location.
  [[nodiscard]] BufferItem& GetOpenSendBuffer(int destination);

  /// Queued or in-flight sends.
  std::vector<BufferItem> send_buffer_;

  /// Completed buffers retained for reuse.
  std::vector<BufferItem> reusable_send_buffers_;

  /// Packed receive scratch buffer.
  std::vector<std::byte> receive_buffer_;

  /// Destination routing cache.
  boost::unordered_flat_map<int, SendPeer> send_peers_;

  /// Open destination-batch indices.
  boost::unordered_flat_map<int, size_t> open_send_buffer_indices_;
};

} // namespace opensn

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

/// Nonblocking MPI communicator for host CBC nonlocal face flux payloads.
class CBC_AsynchronousCommunicator : public AsynchronousCommunicator
{
public:
  explicit CBC_AsynchronousCommunicator(size_t angle_set_id,
                                        FLUDS& fluds,
                                        const MPICommunicatorSet& comm_set);

  /// Queue a complete downwind face payload for asynchronous transmission.
  void QueueDownwindMessage(int destination,
                            std::uint64_t cell_global_id,
                            unsigned int face_id,
                            std::span<const double> payload);

  /// Progress all queued nonblocking sends.
  [[nodiscard]] bool SendData();

  /// Receive all currently available nonlocal face payloads.
  [[nodiscard]] std::vector<std::uint64_t> ReceiveData();

  /// Receive all currently available nonlocal face payloads into a caller-owned buffer.
  void ReceiveData(std::vector<std::uint64_t>& cells_who_received_data);

  /// Return whether there are queued or in-flight sends.
  [[nodiscard]] bool HasPendingCommunication() const noexcept { return not send_buffer_.empty(); }

  /// Clear all transient send and receive buffers.
  void Reset();

protected:
  /// Angle-set identifier used as the MPI message tag.
  const size_t angle_set_id_;

  /// Current MPI location identifier.
  const int location_id_;

  /// Communicator used to receive messages targeting the current location.
  const mpi::Communicator& receive_comm_;

  /// Source ranks that may send data to the current location.
  std::vector<int> receive_source_ranks_;

  /// Buffered nonblocking send request.
  struct BufferItem
  {
    /// Destination OpenSn location.
    int destination = 0;

    /// Nonblocking MPI send request.
    mpi::Request mpi_request;

    /// Whether the nonblocking send has been posted.
    bool send_initiated = false;

    /// Whether the nonblocking send has completed.
    bool completed = false;

    /// Packed message bytes for one destination.
    std::vector<std::byte> data;
  };

  /// Cached MPI routing data for a destination location.
  struct SendPeer
  {
    /// Communicator used to send to the destination location.
    const mpi::Communicator* comm = nullptr;

    /// Destination rank within the communicator.
    int rank = 0;
  };

  /// Return cached MPI routing data for a destination location.
  [[nodiscard]] const SendPeer& GetSendPeer(int destination);

  /// Return the open send buffer for a destination location.
  [[nodiscard]] BufferItem& GetOpenSendBuffer(int destination);

  /// Queued or in-flight send buffers.
  std::vector<BufferItem> send_buffer_;

  /// Completed send buffers retained for storage reuse.
  std::vector<BufferItem> reusable_send_buffers_;

  /// Scratch receive buffer for packed message bytes.
  std::vector<std::byte> receive_buffer_;

  /// Cached MPI routing data keyed by destination location.
  boost::unordered_flat_map<int, SendPeer> send_peers_;

  /// Indices of open send buffers keyed by destination location.
  boost::unordered_flat_map<int, size_t> open_send_buffer_indices_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/async_comm.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class CBC_FLUDS;
class MPICommunicatorSet;

/**
 * Host-side CBC delayed-data communicator.
 *
 * Packs outgoing non-local face data by destination locality, performs asynchronous
 * sends, and receives upwind data needed by the host CBC sweep.
 */
class CBC_AsynchronousCommunicator : public AsynchronousCommunicator
{
public:
  /**
   * Construct the CBC delayed-data communicator.
   *
   * \param angle_set_id Owning angle-set ID.
   * \param fluds CBC FLUDS instance served by this communicator.
   * \param comm_set MPI communicator set.
   */
  explicit CBC_AsynchronousCommunicator(size_t angle_set_id,
                                        FLUDS& fluds,
                                        const MPICommunicatorSet& comm_set);

  /**
   * Initialize one outgoing message payload and return its writable data vector.
   *
   * \param location_id Destination locality ID.
   * \param cell_global_id Destination cell global ID.
   * \param face_id Destination face ID.
   * \param angle_set_id Producing angle-set ID.
   * \param data_size Number of doubles to store in the payload.
   * \return Writable payload vector for the outgoing face data.
   */
  std::vector<double>& InitGetDownwindMessageData(int location_id,
                                                  uint64_t cell_global_id,
                                                  unsigned int face_id,
                                                  size_t angle_set_id,
                                                  size_t data_size);

  void InitializeDelayedUpstreamData();

  /// Send all currently queued outgoing messages.
  bool SendData();

  bool FlushSendBuffers();

  /// Receive all currently available upwind messages.
  std::vector<uint64_t> ReceiveData();

  bool ReceiveDelayedData();

  /// Clear all queued outgoing state.
  void Reset()
  {
    outgoing_message_queue_.clear();
    delayed_outgoing_message_queue_.clear();
    send_buffer_.clear();
    std::fill(delayed_recv_done_.begin(), delayed_recv_done_.end(), false);
    delayed_completion_markers_queued_ = false;
  }

protected:
  /// Owning angle-set ID.
  const size_t angle_set_id_;

  /// Outgoing message key: `(location_id, cell_global_id, face_id)`.
  struct QueuedMessage
  {
    int location_id = 0;
    std::uint64_t cell_global_id = 0;
    unsigned int face_id = 0;
    std::vector<double> data;
  };

  /// Outgoing face payloads queued in sweep order.
  std::vector<QueuedMessage> outgoing_message_queue_;
  std::vector<QueuedMessage> delayed_outgoing_message_queue_;

  /// In-flight send buffer record.
  struct BufferItem
  {
    /// Destination locality.
    int destination = 0;
    /// MPI request for the send.
    mpi::Request mpi_request;
    /// Flag indicating that the send was posted.
    bool send_initiated = false;
    /// Flag indicating that the send completed.
    bool completed = false;
    /// Packed outgoing message bytes.
    std::vector<std::byte> data;
  };
  /// In-flight outgoing message buffers.
  std::vector<BufferItem> send_buffer_;
  /// CBC FLUDS instance served by this communicator.
  CBC_FLUDS& cbc_fluds_;
  /// Scratch receive buffer for incoming messages.
  std::vector<std::byte> receive_buffer_;
  /// Packed byte counts per destination locality.
  std::vector<size_t> destination_buffer_bytes_;
  /// Send-buffer indices grouped by destination locality.
  std::vector<size_t> destination_buffer_indices_;
  /// Delayed-successor locality flags indexed by MPI rank.
  std::vector<std::uint8_t> delayed_successor_flags_;
  /// Completion flags for delayed dependency receives.
  std::vector<bool> delayed_recv_done_;
  /// Whether completion markers were queued for delayed-successor sends.
  bool delayed_completion_markers_queued_ = false;
  static constexpr std::uint64_t delayed_done_cell_id_ = std::numeric_limits<std::uint64_t>::max();
  static constexpr unsigned int delayed_done_face_id_ = std::numeric_limits<unsigned int>::max();

private:
  /// Pack the queued outgoing face payloads into send buffers.
  void QueueOutgoingMessages(std::vector<QueuedMessage>& message_queue);

  bool AllBufferedSendsCompleted();

  void QueueDelayedCompletionMarkers();
};

} // namespace opensn

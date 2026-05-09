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
  enum class DownwindPayloadType : std::uint8_t
  {
    NORMAL = 0,
    DELAYED = 1
  };

  explicit CBC_AsynchronousCommunicator(size_t angle_set_id,
                                        FLUDS& fluds,
                                        const MPICommunicatorSet& comm_set);

  /// Queue a complete downwind face payload for sending.
  void QueueDownwindMessage(DownwindPayloadType payload_type,
                            size_t target,
                            size_t face_slot,
                            std::span<const double> payload);

  void InitializeDelayedUpstreamData();

  bool SendData();

  bool FlushSendBuffers();

  /// Receive all currently available nonlocal face payloads into a caller-owned buffer.
  void ReceiveData(std::vector<std::uint32_t>& cells_who_received_data);

  bool ReceiveDelayedData();

  bool HasPendingCommunication() const noexcept { return not send_buffer_.empty(); }

  void Reset();

private:
  const size_t angle_set_id_;
  const mpi::Communicator& receive_comm_;
  CBC_FLUDS& cbc_fluds_;

  struct SendPeer
  {
    const mpi::Communicator* comm = nullptr;
    int rank = 0;
  };

  enum class MessageKind : std::uint8_t
  {
    NORMAL_PAYLOAD = 0,
    DELAYED_PAYLOAD = 1,
    DELAYED_COMPLETION = 2
  };

  struct BufferItem
  {
    const mpi::Communicator* comm = nullptr;
    int rank = 0;
    bool send_initiated = false;
    std::vector<char> data;
  };

  static constexpr std::size_t CBC_MESSAGE_HEADER_SIZE =
    sizeof(std::uint8_t) + sizeof(std::size_t) + sizeof(std::size_t);

  static void AppendDownwindMessage(std::vector<char>& raw,
                                    MessageKind kind,
                                    size_t face_slot,
                                    std::span<const double> payload);

  BufferItem& GetOpenSendBuffer(size_t peer_index,
                                size_t record_size,
                                const std::vector<SendPeer>& peers,
                                std::vector<BufferItem>& buffers,
                                std::vector<mpi::Request>& requests,
                                std::vector<size_t>& open_buffer_indices);

  bool SendMessages(std::vector<BufferItem>& buffers,
                    std::vector<mpi::Request>& requests,
                    std::vector<size_t>& open_buffer_indices);

  void QueueDelayedCompletionMarkers();

  void ReceiveAvailableMessages(std::vector<std::uint32_t>& cells_who_received_data);

  void MarkDelayedReceiveComplete(int source_rank);

  void StorePayload(MessageKind kind,
                    size_t face_slot,
                    const char* payload,
                    size_t num_values,
                    std::vector<std::uint32_t>& cells_who_received_data);

  std::vector<BufferItem> send_buffer_;
  std::vector<mpi::Request> send_requests_;
  std::vector<BufferItem> delayed_send_buffer_;
  std::vector<mpi::Request> delayed_send_requests_;
  std::vector<BufferItem> reusable_send_buffers_;
  std::vector<char> receive_buffer_;
  std::vector<std::uint32_t> received_task_scratch_;
  std::vector<SendPeer> send_peers_;
  std::vector<SendPeer> delayed_send_peers_;
  std::vector<size_t> open_send_buffer_indices_;
  std::vector<size_t> open_delayed_send_buffer_indices_;
  std::vector<size_t> delayed_peer_indices_by_location_;
  std::vector<unsigned char> delayed_recv_done_;
  std::vector<size_t> delayed_dependency_index_by_source_rank_;
  std::vector<unsigned char> delayed_payload_received_;
  bool delayed_completion_markers_queued_ = false;
  static constexpr size_t INVALID_BUFFER_INDEX = std::numeric_limits<size_t>::max();
};

} // namespace opensn

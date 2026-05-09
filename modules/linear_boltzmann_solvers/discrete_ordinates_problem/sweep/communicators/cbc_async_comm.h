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
    NORMAL,
    DELAYED
  };

  explicit CBC_AsynchronousCommunicator(size_t angle_set_id,
                                        FLUDS& fluds,
                                        int max_mpi_message_size,
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
  const int location_id_;
  const mpi::Communicator& receive_comm_;
  CBC_FLUDS& cbc_fluds_;

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

  enum class MessageKind : std::uint8_t
  {
    NORMAL_PAYLOAD = 0,
    DELAYED_PAYLOAD = 1,
    DELAYED_COMPLETION = 2
  };

  // Size of one packed CBC record header before optional payload data.
  static constexpr std::size_t CBC_MESSAGE_HEADER_SIZE = sizeof(std::uint8_t) +
                                                         sizeof(std::size_t) + sizeof(std::size_t) +
                                                         sizeof(std::size_t) + sizeof(std::size_t);

  static constexpr std::size_t CBC_MAX_IMMEDIATE_MESSAGE_BYTES = 3072;

  static void AppendDownwindMessage(std::vector<char>& raw,
                                    MessageKind kind,
                                    size_t face_slot,
                                    size_t total_size,
                                    size_t offset,
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

  void ProgressPayloadReadyAcks();

  void ProgressPayloadReadyAckSends();

  void ProgressPayloadReceives(std::vector<std::uint32_t>& cells_who_received_data);

  void SendPayloadReadyAck(int source_rank, size_t message_id);

  void MarkDelayedReceiveComplete(int source_rank);

  void StorePayload(MessageKind kind,
                    size_t face_slot,
                    size_t total_size,
                    size_t chunk_offset,
                    size_t chunk_size,
                    const char* payload,
                    std::vector<std::uint32_t>& cells_who_received_data);

  void StoreCompletePayload(MessageKind kind,
                            size_t face_slot,
                            size_t total_size,
                            const char* payload,
                            std::span<const double> assembled_payload,
                            std::vector<std::uint32_t>& cells_who_received_data);

  std::vector<BufferItem> send_buffer_;
  std::vector<mpi::Request> send_requests_;
  std::vector<BufferItem> delayed_send_buffer_;
  std::vector<mpi::Request> delayed_send_requests_;
  std::vector<BufferItem> reusable_send_buffers_;
  std::vector<char> receive_buffer_;
  std::vector<std::uint32_t> received_task_scratch_;
  std::vector<SendPeer> send_peers_;
  /// Delayed SPDS-successor-indexed routing cache.
  std::vector<SendPeer> delayed_send_peers_;
  std::vector<size_t> open_send_buffer_indices_;
  std::vector<size_t> open_delayed_send_buffer_indices_;
  /// MPI-location-indexed map to delayed peer indices.
  std::vector<size_t> delayed_peer_indices_by_location_;
  /// Completion flags for delayed predecessor receives.
  std::vector<unsigned char> delayed_recv_done_;
  std::vector<size_t> delayed_dependency_index_by_source_rank_;
  struct PartialIncomingPayload
  {
    std::vector<double> data;
    std::vector<unsigned char> received_chunks;
    size_t total_size = 0;
    size_t received = 0;
  };

  static void ResetPartialPayload(PartialIncomingPayload& partial);

  static bool StorePartialPayload(PartialIncomingPayload& partial,
                                  size_t total_size,
                                  size_t chunk_offset,
                                  size_t chunk_size,
                                  size_t max_payload_chunk_size,
                                  const char* payload);

  std::vector<PartialIncomingPayload> incoming_partials_;
  std::vector<PartialIncomingPayload> delayed_partials_;
  std::vector<unsigned char> delayed_payload_received_;
  bool delayed_completion_markers_queued_ = false;
  std::size_t max_mpi_message_size_ = 0;
  std::size_t max_payload_chunk_size_ = 1;
  static constexpr size_t INVALID_BUFFER_INDEX = std::numeric_limits<size_t>::max();
};

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

namespace opensn
{

namespace
{

template <typename T>
  requires std::is_trivially_copyable_v<T>
void
WriteMessageValue(char*& buffer, const T& value)
{
  std::memcpy(static_cast<void*>(buffer), static_cast<const void*>(&value), sizeof(T));
  buffer += sizeof(T);
}

template <typename T>
  requires std::is_trivially_copyable_v<T>
T
ReadMessageValue(char*& buffer)
{
  T value;
  std::memcpy(static_cast<void*>(&value), static_cast<const void*>(buffer), sizeof(T));
  buffer += sizeof(T);
  return value;
}

} // namespace

CBC_AsynchronousCommunicator::MessageHeader
CBC_AsynchronousCommunicator::MakeHeader(MessageKind kind,
                                         size_t message_id,
                                         size_t face_slot,
                                         size_t total_size,
                                         size_t offset,
                                         size_t chunk_size)
{
  MessageHeader header{};
  auto* write_ptr = header.data();
  WriteMessageValue(write_ptr, static_cast<std::uint8_t>(kind));
  WriteMessageValue(write_ptr, message_id);
  WriteMessageValue(write_ptr, face_slot);
  WriteMessageValue(write_ptr, total_size);
  WriteMessageValue(write_ptr, offset);
  WriteMessageValue(write_ptr, chunk_size);
  return header;
}

int
CBC_AsynchronousCommunicator::HeaderTag() const noexcept
{
  assert(angle_set_id_ <= static_cast<size_t>((std::numeric_limits<int>::max() - 2) / 3));
  return 3 * static_cast<int>(angle_set_id_);
}

int
CBC_AsynchronousCommunicator::PayloadReadyTag() const noexcept
{
  return HeaderTag() + 1;
}

int
CBC_AsynchronousCommunicator::PayloadTag() const noexcept
{
  return HeaderTag() + 2;
}

void
CBC_AsynchronousCommunicator::ResetPartialPayload(PartialIncomingPayload& partial)
{
  partial.data.clear();
  partial.received_chunks.clear();
  partial.total_size = 0;
  partial.received = 0;
}

bool
CBC_AsynchronousCommunicator::StorePartialPayload(PartialIncomingPayload& partial,
                                                  std::size_t total_size,
                                                  std::size_t chunk_offset,
                                                  std::size_t chunk_size,
                                                  std::size_t max_payload_chunk_size,
                                                  std::span<const double> payload)
{
  if (partial.total_size == 0)
  {
    partial.data.assign(total_size, 0.0);
    partial.received_chunks.assign((total_size - 1) / max_payload_chunk_size + 1, 0);
    partial.total_size = total_size;
    partial.received = 0;
  }
  assert(partial.total_size == total_size);
  const auto chunk_index = chunk_offset / max_payload_chunk_size;
  assert(chunk_index < partial.received_chunks.size());
  assert(partial.received_chunks[chunk_index] == 0);
  std::copy(payload.begin(), payload.end(), partial.data.begin() + chunk_offset);
  partial.received_chunks[chunk_index] = 1;
  partial.received += chunk_size;
  assert(partial.received <= total_size);
  return partial.received == total_size;
}

CBC_AsynchronousCommunicator::CBC_AsynchronousCommunicator(size_t angle_set_id,
                                                           FLUDS& fluds,
                                                           int max_mpi_message_size,
                                                           const MPICommunicatorSet& comm_set)
  : AsynchronousCommunicator(fluds, comm_set),
    angle_set_id_(angle_set_id),
    location_id_(opensn::mpi_comm.rank()),
    receive_comm_(comm_set.LocICommunicator(location_id_)),
    cbc_fluds_(dynamic_cast<CBC_FLUDS&>(fluds)),
    max_payload_chunk_size_(std::max<std::size_t>(
      static_cast<std::size_t>(std::max(max_mpi_message_size, 1)) / sizeof(double), 1))
{
  incoming_partials_.resize(cbc_fluds_.GetCommonData().GetNumIncomingNonlocalFaces());
  delayed_partials_.resize(cbc_fluds_.GetCommonData().GetNumDelayedNonlocalFaces());
  delayed_payload_received_.assign(cbc_fluds_.GetCommonData().GetNumDelayedNonlocalFaces(), 0);

  const auto& location_successors = fluds_.GetSPDS().GetLocationSuccessors();
  send_peers_.reserve(location_successors.size());
  for (const int successor : location_successors)
  {
    auto& peer = send_peers_.emplace_back();
    peer.comm = &comm_set_.LocICommunicator(successor);
    peer.rank = comm_set_.MapIonJ(successor, successor);
  }

  const auto& delayed_location_successors = fluds_.GetSPDS().GetDelayedLocationSuccessors();
  const auto& delayed_location_dependencies = fluds_.GetSPDS().GetDelayedLocationDependencies();
  delayed_peer_indices_by_location_.assign(static_cast<size_t>(opensn::mpi_comm.size()),
                                           INVALID_BUFFER_INDEX);
  delayed_send_peers_.reserve(delayed_location_successors.size());
  for (const int successor : delayed_location_successors)
  {
    const auto peer_index = delayed_send_peers_.size();
    auto& peer = delayed_send_peers_.emplace_back();
    peer.comm = &comm_set_.LocICommunicator(successor);
    peer.rank = comm_set_.MapIonJ(successor, successor);
    delayed_peer_indices_by_location_[static_cast<size_t>(successor)] = peer_index;
  }

  delayed_dependency_index_by_source_rank_.assign(static_cast<size_t>(opensn::mpi_comm.size()),
                                                  INVALID_BUFFER_INDEX);
  for (std::size_t dependency_index = 0; dependency_index < delayed_location_dependencies.size();
       ++dependency_index)
  {
    const auto source_rank =
      comm_set_.MapIonJ(delayed_location_dependencies[dependency_index], location_id_);
    assert(source_rank >= 0);
    const auto source_rank_index = static_cast<std::size_t>(source_rank);
    if (source_rank_index >= delayed_dependency_index_by_source_rank_.size())
      delayed_dependency_index_by_source_rank_.resize(source_rank_index + 1, INVALID_BUFFER_INDEX);
    delayed_dependency_index_by_source_rank_[source_rank_index] = dependency_index;
  }
}

void
CBC_AsynchronousCommunicator::AddOutgoingMessage(size_t peer_index,
                                                 const std::vector<SendPeer>& peers,
                                                 std::vector<OutgoingMessage>& buffer,
                                                 MessageKind kind,
                                                 size_t face_slot,
                                                 size_t total_size,
                                                 size_t offset,
                                                 std::span<const double> payload)
{
  assert(peer_index < peers.size());

  if (reusable_send_buffers_.empty())
    buffer.emplace_back();
  else
  {
    buffer.push_back(std::move(reusable_send_buffers_.back()));
    reusable_send_buffers_.pop_back();
  }

  auto& message = buffer.back();
  const auto& peer = peers[peer_index];
  const auto message_id = next_message_id_++;
  if (message_id >= payload_ready_by_message_id_.size())
    payload_ready_by_message_id_.resize(message_id + 1, 0);
  payload_ready_by_message_id_[message_id] = payload.empty() ? 1 : 0;

  message.comm = peer.comm;
  message.rank = peer.rank;
  message.message_id = message_id;
  if (message.header == nullptr)
    message.header = std::make_unique<MessageHeader>();
  *message.header = MakeHeader(kind, message_id, face_slot, total_size, offset, payload.size());
  message.payload.assign(payload.begin(), payload.end());
  message.header_request = mpi::Request();
  message.payload_request = mpi::Request();
  message.header_send_initiated = false;
  message.payload_send_initiated = false;
  message.header_complete = false;
  message.payload_complete = payload.empty();
}

void
CBC_AsynchronousCommunicator::QueueDownwindMessage(DownwindPayloadType payload_type,
                                                   size_t target,
                                                   size_t face_slot,
                                                   std::span<const double> payload)
{
  assert(not payload.empty());
  const bool delayed = payload_type == DownwindPayloadType::DELAYED;
  const auto kind = delayed ? MessageKind::DELAYED_PAYLOAD : MessageKind::NORMAL_PAYLOAD;
  auto peer_index = target;
  const auto* peers = &send_peers_;

  if (delayed)
  {
    assert(target < delayed_peer_indices_by_location_.size());
    peer_index = delayed_peer_indices_by_location_[target];
    assert(peer_index != INVALID_BUFFER_INDEX);
    peers = &delayed_send_peers_;
  }

  const auto total_size = payload.size();
  for (size_t offset = 0; offset < total_size; offset += max_payload_chunk_size_)
  {
    const auto chunk_size = std::min(max_payload_chunk_size_, total_size - offset);
    AddOutgoingMessage(peer_index,
                       *peers,
                       delayed ? delayed_send_buffer_ : send_buffer_,
                       kind,
                       face_slot,
                       total_size,
                       offset,
                       payload.subspan(offset, chunk_size));
  }
}

void
CBC_AsynchronousCommunicator::InitializeDelayedUpstreamData()
{
  cbc_fluds_.AllocateDelayedLocalPsi();
  cbc_fluds_.AllocateDelayedPrelocIOutgoingPsi();
  delayed_recv_done_.assign(fluds_.GetSPDS().GetDelayedLocationDependencies().size(), 0);
  std::fill(delayed_payload_received_.begin(), delayed_payload_received_.end(), 0);
  delayed_completion_markers_queued_ = false;
}

bool
CBC_AsynchronousCommunicator::SendMessages(std::vector<OutgoingMessage>& buffer)
{
  if (buffer.empty())
  {
    ProgressPayloadReadyAcks();
    ProgressPayloadReadyAckSends();
    return payload_ready_ack_sends_.empty();
  }

  for (auto& message : buffer)
  {
    if (not message.header_send_initiated)
    {
      assert(message.header != nullptr);
      message.header_request = message.comm->isend(message.rank,
                                                   HeaderTag(),
                                                   message.header->data(),
                                                   static_cast<int>(message.header->size()));
      message.header_send_initiated = true;
    }
  }

  ProgressPayloadReadyAcks();
  ProgressPayloadReadyAckSends();

  for (auto& message : buffer)
  {
    if (message.payload.empty() or message.payload_send_initiated)
      continue;

    assert(message.message_id < payload_ready_by_message_id_.size());
    if (payload_ready_by_message_id_[message.message_id] == 0)
      continue;

    assert(message.payload.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
    message.payload_request = message.comm->isend(
      message.rank, PayloadTag(), message.payload.data(), static_cast<int>(message.payload.size()));
    message.payload_send_initiated = true;
  }

  for (auto& message : buffer)
  {
    if (not message.header_complete)
      message.header_complete = mpi::test(message.header_request);
    if (message.payload_send_initiated and not message.payload_complete)
      message.payload_complete = mpi::test(message.payload_request);
  }

  std::size_t num_pending = 0;
  for (auto& message : buffer)
  {
    if (message.header_complete and message.payload_complete)
    {
      message.header_send_initiated = false;
      message.payload_send_initiated = false;
      message.header_complete = false;
      message.payload_complete = true;
      message.payload.clear();
      reusable_send_buffers_.push_back(std::move(message));
    }
    else
    {
      if (&message != &buffer[num_pending])
        buffer[num_pending] = std::move(message);
      ++num_pending;
    }
  }
  buffer.resize(num_pending);

  return buffer.empty() and payload_ready_ack_sends_.empty();
}

bool
CBC_AsynchronousCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::SendData");

  return SendMessages(send_buffer_);
}

void
CBC_AsynchronousCommunicator::QueueDelayedCompletionMarkers()
{
  for (std::size_t delayed_peer_index = 0; delayed_peer_index < delayed_send_peers_.size();
       ++delayed_peer_index)
  {
    AddOutgoingMessage(delayed_peer_index,
                       delayed_send_peers_,
                       delayed_send_buffer_,
                       MessageKind::DELAYED_COMPLETION,
                       CBC_FLUDSCommonData::INVALID_FACE_SLOT,
                       0,
                       0,
                       std::span<const double>());
  }
  delayed_completion_markers_queued_ = true;
}

bool
CBC_AsynchronousCommunicator::FlushSendBuffers()
{
  if (not SendData())
    return false;

  if (not delayed_completion_markers_queued_)
    QueueDelayedCompletionMarkers();

  return SendMessages(delayed_send_buffer_);
}

void
CBC_AsynchronousCommunicator::Reset()
{
  while (not payload_ready_ack_sends_.empty())
    ProgressPayloadReadyAckSends();

  send_buffer_.clear();
  delayed_send_buffer_.clear();
  reusable_send_buffers_.clear();
  payload_receives_.clear();
  payload_ready_ack_sends_.clear();
  payload_ready_by_message_id_.clear();
  for (auto& partial : incoming_partials_)
    ResetPartialPayload(partial);
  for (auto& partial : delayed_partials_)
    ResetPartialPayload(partial);
  std::fill(delayed_payload_received_.begin(), delayed_payload_received_.end(), 0);
  std::fill(delayed_recv_done_.begin(), delayed_recv_done_.end(), 0);
  delayed_completion_markers_queued_ = false;
  next_message_id_ = 0;
}

void
CBC_AsynchronousCommunicator::MarkDelayedReceiveComplete(int source_rank)
{
  assert(source_rank >= 0);
  const auto source_rank_index = static_cast<std::size_t>(source_rank);
  assert(source_rank_index < delayed_dependency_index_by_source_rank_.size());
  const auto dependency_index = delayed_dependency_index_by_source_rank_[source_rank_index];
  assert(dependency_index != INVALID_BUFFER_INDEX);
  if (delayed_recv_done_.size() <= dependency_index)
    delayed_recv_done_.resize(dependency_index + 1, 0);
  delayed_recv_done_[dependency_index] = 1;
}

void
CBC_AsynchronousCommunicator::StoreCompletePayload(
  MessageKind kind,
  size_t face_slot,
  std::span<const double> payload,
  std::vector<std::uint32_t>& cells_who_received_data)
{
  assert(kind == MessageKind::NORMAL_PAYLOAD or kind == MessageKind::DELAYED_PAYLOAD);
  assert(not payload.empty());

  if (kind == MessageKind::DELAYED_PAYLOAD)
  {
    assert(delayed_payload_received_[face_slot] == 0);
    auto incoming = cbc_fluds_.PrepareIncomingDelayedNonlocalPsiBySlot(face_slot, payload.size());
    std::copy(payload.begin(), payload.end(), incoming.begin());
    delayed_payload_received_[face_slot] = 1;
  }
  else
  {
    auto incoming = cbc_fluds_.PrepareIncomingNonlocalPsiBySlot(face_slot, payload.size());
    std::copy(payload.begin(), payload.end(), incoming.psi.begin());
    cells_who_received_data.push_back(incoming.cell_local_id);
  }
}

void
CBC_AsynchronousCommunicator::StorePayload(MessageKind kind,
                                           size_t face_slot,
                                           size_t total_size,
                                           size_t chunk_offset,
                                           std::span<const double> payload,
                                           std::vector<std::uint32_t>& cells_who_received_data)
{
  assert(kind == MessageKind::NORMAL_PAYLOAD or kind == MessageKind::DELAYED_PAYLOAD);
  assert(face_slot != CBC_FLUDSCommonData::INVALID_FACE_SLOT);
  assert(total_size > 0);
  assert(not payload.empty());
  assert(chunk_offset < total_size);
  assert(payload.size() <= total_size - chunk_offset);

  const bool delayed = (kind == MessageKind::DELAYED_PAYLOAD);
  auto& partials = delayed ? delayed_partials_ : incoming_partials_;
  assert(face_slot < partials.size());

  if (chunk_offset == 0 and payload.size() == total_size)
    StoreCompletePayload(kind, face_slot, payload, cells_who_received_data);
  else
  {
    auto& partial = partials[face_slot];
    if (StorePartialPayload(
          partial, total_size, chunk_offset, payload.size(), max_payload_chunk_size_, payload))
    {
      StoreCompletePayload(kind,
                           face_slot,
                           std::span<const double>(partial.data.data(), partial.data.size()),
                           cells_who_received_data);
      ResetPartialPayload(partial);
    }
  }
}

void
CBC_AsynchronousCommunicator::ProgressPayloadReadyAcks()
{
  auto progress = [this](const std::vector<SendPeer>& peers)
  {
    mpi::Status status;
    for (const auto& peer : peers)
    {
      while (peer.comm->iprobe(peer.rank, PayloadReadyTag(), status))
      {
        size_t message_id = 0;
        peer.comm->recv(peer.rank, PayloadReadyTag(), message_id);
        assert(message_id < payload_ready_by_message_id_.size());
        payload_ready_by_message_id_[message_id] = 1;
      }
    }
  };

  progress(send_peers_);
  progress(delayed_send_peers_);
}

void
CBC_AsynchronousCommunicator::ProgressPayloadReadyAckSends()
{
  for (std::size_t i = 0; i < payload_ready_ack_sends_.size();)
  {
    if (mpi::test(payload_ready_ack_sends_[i].request))
    {
      if (i != payload_ready_ack_sends_.size() - 1)
        payload_ready_ack_sends_[i] = std::move(payload_ready_ack_sends_.back());
      payload_ready_ack_sends_.pop_back();
    }
    else
      ++i;
  }
}

void
CBC_AsynchronousCommunicator::SendPayloadReadyAck(int source_rank, size_t message_id)
{
  auto& ack = payload_ready_ack_sends_.emplace_back();
  ack.message_id = std::make_unique<size_t>(message_id);
  ack.request = receive_comm_.isend(source_rank, PayloadReadyTag(), *ack.message_id);
}

void
CBC_AsynchronousCommunicator::ProgressPayloadReceives(
  std::vector<std::uint32_t>& cells_who_received_data)
{
  for (std::size_t i = 0; i < payload_receives_.size();)
  {
    auto& receive = payload_receives_[i];
    if (mpi::test(receive.request))
    {
      StorePayload(receive.kind,
                   receive.face_slot,
                   receive.total_size,
                   receive.chunk_offset,
                   std::span<const double>(receive.payload.data(), receive.payload.size()),
                   cells_who_received_data);
      if (i != payload_receives_.size() - 1)
        payload_receives_[i] = std::move(payload_receives_.back());
      payload_receives_.pop_back();
    }
    else
      ++i;
  }
}

void
CBC_AsynchronousCommunicator::ReceiveAvailableMessages(
  std::vector<std::uint32_t>& cells_who_received_data)
{
  mpi::Status status;

  ProgressPayloadReadyAckSends();
  ProgressPayloadReceives(cells_who_received_data);

  while (receive_comm_.iprobe(ANY_SOURCE, HeaderTag(), status))
  {
    const auto source_rank = status.source();
    const auto num_items = status.count<char>();
    assert(num_items == static_cast<int>(CBC_MESSAGE_HEADER_SIZE));
    receive_comm_.recv(
      source_rank, HeaderTag(), receive_header_.data(), static_cast<int>(receive_header_.size()));

    auto* read_ptr = receive_header_.data();
    const auto kind = static_cast<MessageKind>(ReadMessageValue<std::uint8_t>(read_ptr));
    const auto message_id = ReadMessageValue<std::size_t>(read_ptr);
    const auto face_slot = ReadMessageValue<std::size_t>(read_ptr);
    const auto total_size = ReadMessageValue<std::size_t>(read_ptr);
    const auto chunk_offset = ReadMessageValue<std::size_t>(read_ptr);
    const auto chunk_size = ReadMessageValue<std::size_t>(read_ptr);

    switch (kind)
    {
      case MessageKind::NORMAL_PAYLOAD:
      case MessageKind::DELAYED_PAYLOAD:
      {
        assert(face_slot != CBC_FLUDSCommonData::INVALID_FACE_SLOT);
        assert(total_size > 0);
        assert(chunk_offset < total_size);
        assert(chunk_size > 0);
        assert(chunk_size <= total_size - chunk_offset);
        assert(chunk_size <= static_cast<std::size_t>(std::numeric_limits<int>::max()));

        auto& receive = payload_receives_.emplace_back();
        receive.kind = kind;
        receive.face_slot = face_slot;
        receive.total_size = total_size;
        receive.chunk_offset = chunk_offset;
        receive.payload.resize(chunk_size);
        receive.request = receive_comm_.irecv(source_rank,
                                              PayloadTag(),
                                              receive.payload.data(),
                                              static_cast<int>(receive.payload.size()));
        SendPayloadReadyAck(source_rank, message_id);
        break;
      }
      case MessageKind::DELAYED_COMPLETION:
        assert((face_slot == CBC_FLUDSCommonData::INVALID_FACE_SLOT) and (total_size == 0) and
               (chunk_offset == 0) and (chunk_size == 0));
        MarkDelayedReceiveComplete(source_rank);
        break;
      default:
        assert(false and "Invalid message kind.");
        break;
    }

    ProgressPayloadReceives(cells_who_received_data);
    ProgressPayloadReadyAckSends();
  }

  ProgressPayloadReceives(cells_who_received_data);
  ProgressPayloadReadyAckSends();
}

void
CBC_AsynchronousCommunicator::ReceiveData(std::vector<std::uint32_t>& cells_who_received_data)
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveData");

  cells_who_received_data.clear();
  ReceiveAvailableMessages(cells_who_received_data);
}

bool
CBC_AsynchronousCommunicator::ReceiveDelayedData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveDelayedData");

  const auto& delayed_location_dependencies = fluds_.GetSPDS().GetDelayedLocationDependencies();
  if (delayed_recv_done_.size() != delayed_location_dependencies.size())
    delayed_recv_done_.assign(delayed_location_dependencies.size(), 0);

  received_task_scratch_.clear();
  ReceiveAvailableMessages(received_task_scratch_);

  return payload_receives_.empty() and std::all_of(delayed_recv_done_.begin(),
                                                   delayed_recv_done_.end(),
                                                   [](const auto done) { return done != 0; });
}

} // namespace opensn

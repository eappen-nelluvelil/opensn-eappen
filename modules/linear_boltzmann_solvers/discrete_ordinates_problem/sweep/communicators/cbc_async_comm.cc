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

void
CBC_AsynchronousCommunicator::AppendDownwindMessage(std::vector<char>& raw,
                                                    MessageKind kind,
                                                    size_t face_slot,
                                                    size_t total_size,
                                                    size_t offset,
                                                    std::span<const double> payload)
{
  const auto chunk_size = payload.size();
  const auto old_size = raw.size();
  const auto num_bytes = chunk_size * sizeof(double);
  const auto required_size = old_size + CBC_MESSAGE_HEADER_SIZE + num_bytes;
  if (raw.capacity() < required_size)
    raw.reserve(required_size);
  raw.resize(required_size);

  auto* write_ptr = raw.data() + old_size;
  WriteMessageValue(write_ptr, static_cast<std::uint8_t>(kind));
  WriteMessageValue(write_ptr, face_slot);
  WriteMessageValue(write_ptr, total_size);
  WriteMessageValue(write_ptr, offset);
  WriteMessageValue(write_ptr, chunk_size);
  if (num_bytes != 0)
    std::memcpy(write_ptr, payload.data(), num_bytes);
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
                                                  const char* payload)
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
  const auto num_bytes = chunk_size * sizeof(double);
  std::memcpy(partial.data.data() + chunk_offset, payload, num_bytes);
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
    max_mpi_message_size_(std::max(
      std::min(static_cast<std::size_t>(max_mpi_message_size), CBC_MAX_IMMEDIATE_MESSAGE_BYTES),
      CBC_MESSAGE_HEADER_SIZE + sizeof(double))),
    max_payload_chunk_size_(max_mpi_message_size_ <= CBC_MESSAGE_HEADER_SIZE + sizeof(double)
                              ? 1
                              : (max_mpi_message_size_ - CBC_MESSAGE_HEADER_SIZE) / sizeof(double))
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
  open_send_buffer_indices_.assign(send_peers_.size(), INVALID_BUFFER_INDEX);

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
  open_delayed_send_buffer_indices_.assign(delayed_send_peers_.size(), INVALID_BUFFER_INDEX);

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

CBC_AsynchronousCommunicator::BufferItem&
CBC_AsynchronousCommunicator::GetOpenSendBuffer(size_t peer_index,
                                                size_t record_size,
                                                const std::vector<SendPeer>& peers,
                                                std::vector<BufferItem>& buffers,
                                                std::vector<mpi::Request>& requests,
                                                std::vector<size_t>& open_buffer_indices)
{
  assert(peer_index < open_buffer_indices.size());
  auto& open_buffer_index = open_buffer_indices[peer_index];
  if (open_buffer_index != INVALID_BUFFER_INDEX)
  {
    auto& buffer = buffers[open_buffer_index];
    if ((not buffer.data.empty()) and buffer.data.size() + record_size > max_mpi_message_size_)
      open_buffer_index = INVALID_BUFFER_INDEX;
    else
      return buffer;
  }

  if (reusable_send_buffers_.empty())
    buffers.emplace_back();
  else
  {
    buffers.push_back(std::move(reusable_send_buffers_.back()));
    reusable_send_buffers_.pop_back();
  }
  requests.emplace_back();

  const auto buffer_index = buffers.size() - 1;
  auto& buffer = buffers.back();
  const auto& peer = peers[peer_index];
  buffer.peer_index = peer_index;
  buffer.comm = peer.comm;
  buffer.rank = peer.rank;
  buffer.send_initiated = false;
  buffer.data.clear();
  open_buffer_index = buffer_index;
  return buffer;
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
  auto* buffers = &send_buffer_;
  auto* requests = &send_requests_;
  auto* open_buffer_indices = &open_send_buffer_indices_;

  if (delayed)
  {
    assert(target < delayed_peer_indices_by_location_.size());
    peer_index = delayed_peer_indices_by_location_[target];
    assert(peer_index != INVALID_BUFFER_INDEX);
    peers = &delayed_send_peers_;
    buffers = &delayed_send_buffer_;
    requests = &delayed_send_requests_;
    open_buffer_indices = &open_delayed_send_buffer_indices_;
  }

  const auto total_size = payload.size();
  for (size_t offset = 0; offset < total_size; offset += max_payload_chunk_size_)
  {
    const auto chunk_size = std::min(max_payload_chunk_size_, total_size - offset);
    auto& raw = GetOpenSendBuffer(peer_index,
                                  CBC_MESSAGE_HEADER_SIZE + chunk_size * sizeof(double),
                                  *peers,
                                  *buffers,
                                  *requests,
                                  *open_buffer_indices)
                  .data;
    AppendDownwindMessage(
      raw, kind, face_slot, total_size, offset, payload.subspan(offset, chunk_size));
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
CBC_AsynchronousCommunicator::SendMessages(std::vector<BufferItem>& buffers,
                                           std::vector<mpi::Request>& requests,
                                           std::vector<size_t>& open_buffer_indices)
{
  assert(buffers.size() == requests.size());

  if (buffers.empty())
    return true;

  const auto tag = static_cast<int>(angle_set_id_);
  for (std::size_t i = 0; i < buffers.size(); ++i)
  {
    auto& buffer_item = buffers[i];
    if (not buffer_item.send_initiated)
    {
      requests[i] = buffer_item.comm->isend(buffer_item.rank, tag, buffer_item.data);
      buffer_item.send_initiated = true;
    }
  }

  for (std::size_t i = 0; i < buffers.size();)
  {
    auto& buffer_item = buffers[i];
    if (mpi::test(requests[i]))
    {
      buffer_item.send_initiated = false;
      buffer_item.data.clear();
      reusable_send_buffers_.push_back(std::move(buffer_item));
      if (i != buffers.size() - 1)
      {
        buffers[i] = std::move(buffers.back());
        requests[i] = requests.back();
      }
      buffers.pop_back();
      requests.pop_back();
    }
    else
      ++i;
  }

  std::fill(open_buffer_indices.begin(), open_buffer_indices.end(), INVALID_BUFFER_INDEX);
  return buffers.empty();
}

bool
CBC_AsynchronousCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::SendData");

  return SendMessages(send_buffer_, send_requests_, open_send_buffer_indices_);
}

void
CBC_AsynchronousCommunicator::QueueDelayedCompletionMarkers()
{
  for (std::size_t delayed_peer_index = 0; delayed_peer_index < delayed_send_peers_.size();
       ++delayed_peer_index)
  {
    auto& raw = GetOpenSendBuffer(delayed_peer_index,
                                  CBC_MESSAGE_HEADER_SIZE,
                                  delayed_send_peers_,
                                  delayed_send_buffer_,
                                  delayed_send_requests_,
                                  open_delayed_send_buffer_indices_)
                  .data;
    AppendDownwindMessage(raw,
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

  return SendMessages(
    delayed_send_buffer_, delayed_send_requests_, open_delayed_send_buffer_indices_);
}

void
CBC_AsynchronousCommunicator::Reset()
{
  send_buffer_.clear();
  send_requests_.clear();
  delayed_send_buffer_.clear();
  delayed_send_requests_.clear();
  reusable_send_buffers_.clear();
  receive_buffer_.clear();
  received_task_scratch_.clear();
  for (auto& partial : incoming_partials_)
    ResetPartialPayload(partial);
  for (auto& partial : delayed_partials_)
    ResetPartialPayload(partial);
  std::fill(delayed_payload_received_.begin(), delayed_payload_received_.end(), 0);
  std::fill(
    open_send_buffer_indices_.begin(), open_send_buffer_indices_.end(), INVALID_BUFFER_INDEX);
  std::fill(open_delayed_send_buffer_indices_.begin(),
            open_delayed_send_buffer_indices_.end(),
            INVALID_BUFFER_INDEX);
  std::fill(delayed_recv_done_.begin(), delayed_recv_done_.end(), 0);
  delayed_completion_markers_queued_ = false;
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
  size_t total_size,
  const char* payload,
  std::span<const double> assembled_payload,
  std::vector<std::uint32_t>& cells_who_received_data)
{
  assert(kind == MessageKind::NORMAL_PAYLOAD or kind == MessageKind::DELAYED_PAYLOAD);
  assert((not assembled_payload.empty() and assembled_payload.size() == total_size) or
         payload != nullptr);

  const auto num_bytes = total_size * sizeof(double);
  if (kind == MessageKind::DELAYED_PAYLOAD)
  {
    assert(delayed_payload_received_[face_slot] == 0);
    auto incoming = cbc_fluds_.PrepareIncomingDelayedNonlocalPsiBySlot(face_slot, total_size);
    if (not assembled_payload.empty())
      std::copy(assembled_payload.begin(), assembled_payload.end(), incoming.begin());
    else
      std::memcpy(incoming.data(), payload, num_bytes);
    delayed_payload_received_[face_slot] = 1;
  }
  else
  {
    auto incoming = cbc_fluds_.PrepareIncomingNonlocalPsiBySlot(face_slot, total_size);
    if (not assembled_payload.empty())
      std::copy(assembled_payload.begin(), assembled_payload.end(), incoming.psi.begin());
    else
      std::memcpy(incoming.psi.data(), payload, num_bytes);
    cells_who_received_data.push_back(incoming.cell_local_id);
  }
}

void
CBC_AsynchronousCommunicator::StorePayload(MessageKind kind,
                                           size_t face_slot,
                                           size_t total_size,
                                           size_t chunk_offset,
                                           size_t chunk_size,
                                           const char* payload,
                                           std::vector<std::uint32_t>& cells_who_received_data)
{
  assert(kind == MessageKind::NORMAL_PAYLOAD or kind == MessageKind::DELAYED_PAYLOAD);
  assert(face_slot != CBC_FLUDSCommonData::INVALID_FACE_SLOT);
  assert(total_size > 0);
  assert(chunk_size > 0);
  assert(chunk_offset < total_size);
  assert(chunk_size <= total_size - chunk_offset);

  const bool delayed = (kind == MessageKind::DELAYED_PAYLOAD);
  auto& partials = delayed ? delayed_partials_ : incoming_partials_;
  assert(face_slot < partials.size());

  if (chunk_offset == 0 and chunk_size == total_size)
    StoreCompletePayload(
      kind, face_slot, total_size, payload, std::span<const double>(), cells_who_received_data);
  else
  {
    auto& partial = partials[face_slot];
    if (StorePartialPayload(
          partial, total_size, chunk_offset, chunk_size, max_payload_chunk_size_, payload))
    {
      StoreCompletePayload(kind,
                           face_slot,
                           total_size,
                           nullptr,
                           std::span<const double>(partial.data.data(), partial.data.size()),
                           cells_who_received_data);
      ResetPartialPayload(partial);
    }
  }
}

void
CBC_AsynchronousCommunicator::ReceiveAvailableMessages(
  std::vector<std::uint32_t>& cells_who_received_data)
{
  const auto tag = static_cast<int>(angle_set_id_);
  mpi::Status status;
  while (receive_comm_.iprobe(ANY_SOURCE, tag, status))
  {
    const auto source_rank = status.source();
    const auto num_items = status.count<char>();
    receive_buffer_.resize(num_items);
    receive_comm_.recv(source_rank, status.tag(), receive_buffer_.data(), num_items);

    auto* read_ptr = receive_buffer_.data();
    const auto* const read_end = read_ptr + receive_buffer_.size();

    while (read_ptr < read_end)
    {
      assert(read_ptr + CBC_MESSAGE_HEADER_SIZE <= read_end);
      const auto kind = static_cast<MessageKind>(ReadMessageValue<std::uint8_t>(read_ptr));
      const auto face_slot = ReadMessageValue<std::size_t>(read_ptr);
      const auto total_size = ReadMessageValue<std::size_t>(read_ptr);
      const auto chunk_offset = ReadMessageValue<std::size_t>(read_ptr);
      const auto chunk_size = ReadMessageValue<std::size_t>(read_ptr);

      const auto num_bytes = chunk_size * sizeof(double);
      assert(read_ptr + num_bytes <= read_end);

      switch (kind)
      {
        case MessageKind::NORMAL_PAYLOAD:
        case MessageKind::DELAYED_PAYLOAD:
          StorePayload(kind,
                       face_slot,
                       total_size,
                       chunk_offset,
                       chunk_size,
                       read_ptr,
                       cells_who_received_data);
          break;
        case MessageKind::DELAYED_COMPLETION:
          assert((face_slot == CBC_FLUDSCommonData::INVALID_FACE_SLOT) and (total_size == 0) and
                 (chunk_offset == 0) and (chunk_size == 0));
          MarkDelayedReceiveComplete(source_rank);
          break;
        default:
          assert(false and "Invalid message kind.");
          break;
      }

      read_ptr += num_bytes;
    }
  }
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

  return std::all_of(delayed_recv_done_.begin(),
                     delayed_recv_done_.end(),
                     [](const auto done) { return done != 0; });
}

} // namespace opensn

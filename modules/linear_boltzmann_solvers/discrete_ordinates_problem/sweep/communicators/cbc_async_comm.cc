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
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace opensn
{

namespace
{

template <typename T>
void
WriteMessageValue(std::byte*& buffer, const T& value)
{
  static_assert(std::is_trivially_copyable_v<T>,
                "CBC message serialization requires trivially copyable values.");
  std::memcpy(buffer, &value, sizeof(T));
  buffer += sizeof(T);
}

template <typename T>
T
ReadMessageValue(const std::vector<std::byte>& buffer, std::size_t& offset)
{
  assert(offset + sizeof(T) <= buffer.size());
  T value;
  std::memcpy(&value, buffer.data() + offset, sizeof(T));
  offset += sizeof(T);
  return value;
}

constexpr std::size_t CBC_MESSAGE_HEADER_SIZE =
  sizeof(std::size_t) + sizeof(std::size_t) + sizeof(std::size_t) + sizeof(std::size_t);
constexpr std::size_t CBC_MAX_IMMEDIATE_MESSAGE_BYTES = 3072;

void
AppendDownwindMessage(std::vector<std::byte>& raw,
                      size_t incoming_face_slot,
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
  WriteMessageValue(write_ptr, incoming_face_slot);
  WriteMessageValue(write_ptr, total_size);
  WriteMessageValue(write_ptr, offset);
  WriteMessageValue(write_ptr, chunk_size);
  if (num_bytes != 0)
    std::memcpy(write_ptr, payload.data(), num_bytes);
}

std::size_t
DownwindMessageSize(std::size_t chunk_size)
{
  return CBC_MESSAGE_HEADER_SIZE + chunk_size * sizeof(double);
}

std::size_t
MaxPayloadChunkSize(std::size_t max_mpi_message_size)
{
  if (max_mpi_message_size <= CBC_MESSAGE_HEADER_SIZE + sizeof(double))
    return 1;
  return (max_mpi_message_size - CBC_MESSAGE_HEADER_SIZE) / sizeof(double);
}

std::size_t
NumPayloadChunks(std::size_t payload_size, std::size_t max_payload_chunk_size)
{
  assert(max_payload_chunk_size > 0);
  assert(payload_size > 0);
  return (payload_size - 1) / max_payload_chunk_size + 1;
}

void
ValidatePayloadChunk(std::size_t expected_size,
                     std::size_t total_size,
                     std::size_t chunk_offset,
                     std::size_t chunk_size,
                     std::size_t max_payload_chunk_size,
                     const char* context)
{
  if (total_size != expected_size)
    throw std::logic_error(context);

  if (total_size == 0 or chunk_size == 0 or chunk_offset >= total_size or
      chunk_size > total_size - chunk_offset)
    throw std::logic_error(context);

  if (chunk_offset % max_payload_chunk_size != 0)
    throw std::logic_error(context);

  const auto expected_chunk_size = std::min(max_payload_chunk_size, total_size - chunk_offset);
  if (chunk_size != expected_chunk_size)
    throw std::logic_error(context);
}

} // namespace

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
                                                  const std::byte* payload,
                                                  const char* context)
{
  if (partial.total_size == 0)
  {
    partial.data.assign(total_size, 0.0);
    partial.received_chunks.assign(NumPayloadChunks(total_size, max_payload_chunk_size), 0);
    partial.total_size = total_size;
    partial.received = 0;
  }
  else if (partial.total_size != total_size)
    throw std::logic_error(context);

  const auto chunk_index = chunk_offset / max_payload_chunk_size;
  if (chunk_index >= partial.received_chunks.size() or partial.received_chunks[chunk_index] != 0)
    throw std::logic_error(context);

  const auto num_bytes = chunk_size * sizeof(double);
  std::memcpy(partial.data.data() + chunk_offset, payload, num_bytes);
  partial.received_chunks[chunk_index] = 1;
  partial.received += chunk_size;

  if (partial.received > total_size)
    throw std::logic_error(context);

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
    max_payload_chunk_size_(MaxPayloadChunkSize(max_mpi_message_size_))
{
  const auto& location_dependencies = fluds_.GetSPDS().GetLocationDependencies();
  num_receive_sources_ = location_dependencies.size();
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
}

CBC_AsynchronousCommunicator::BufferItem&
CBC_AsynchronousCommunicator::GetOpenSendBuffer(size_t peer_index, size_t record_size)
{
  assert(peer_index < open_send_buffer_indices_.size());
  auto& open_buffer_index = open_send_buffer_indices_[peer_index];
  if (open_buffer_index != INVALID_BUFFER_INDEX)
  {
    auto& buffer = send_buffer_[open_buffer_index];
    if ((not buffer.data.empty()) and buffer.data.size() + record_size > max_mpi_message_size_)
      open_buffer_index = INVALID_BUFFER_INDEX;
    else
      return buffer;
  }

  if (reusable_send_buffers_.empty())
  {
    send_buffer_.emplace_back();
    send_requests_.emplace_back();
  }
  else
  {
    send_buffer_.push_back(std::move(reusable_send_buffers_.back()));
    reusable_send_buffers_.pop_back();
    send_requests_.emplace_back();
  }

  const auto buffer_index = send_buffer_.size() - 1;
  auto& buffer = send_buffer_.back();
  const auto& peer = send_peers_[peer_index];
  buffer.peer_index = peer_index;
  buffer.comm = peer.comm;
  buffer.rank = peer.rank;
  buffer.send_initiated = false;
  buffer.data.clear();
  open_buffer_index = buffer_index;
  return buffer;
}

CBC_AsynchronousCommunicator::BufferItem&
CBC_AsynchronousCommunicator::GetOpenDelayedSendBuffer(size_t delayed_peer_index,
                                                       size_t record_size)
{
  assert(delayed_peer_index < open_delayed_send_buffer_indices_.size());
  auto& open_buffer_index = open_delayed_send_buffer_indices_[delayed_peer_index];
  if (open_buffer_index != INVALID_BUFFER_INDEX)
  {
    auto& buffer = send_buffer_[open_buffer_index];
    if ((not buffer.data.empty()) and buffer.data.size() + record_size > max_mpi_message_size_)
      open_buffer_index = INVALID_BUFFER_INDEX;
    else
      return buffer;
  }

  if (reusable_send_buffers_.empty())
  {
    send_buffer_.emplace_back();
    send_requests_.emplace_back();
  }
  else
  {
    send_buffer_.push_back(std::move(reusable_send_buffers_.back()));
    reusable_send_buffers_.pop_back();
    send_requests_.emplace_back();
  }

  const auto buffer_index = send_buffer_.size() - 1;
  auto& buffer = send_buffer_.back();
  const auto& peer = delayed_send_peers_[delayed_peer_index];
  buffer.peer_index = delayed_peer_index;
  buffer.comm = peer.comm;
  buffer.rank = peer.rank;
  buffer.send_initiated = false;
  buffer.data.clear();
  open_buffer_index = buffer_index;
  return buffer;
}

void
CBC_AsynchronousCommunicator::QueueDownwindMessage(size_t peer_index,
                                                   size_t incoming_face_slot,
                                                   std::span<const double> payload)
{
  const auto total_size = payload.size();
  for (size_t offset = 0; offset < total_size; offset += max_payload_chunk_size_)
  {
    const auto chunk_size = std::min(max_payload_chunk_size_, total_size - offset);
    auto& raw = GetOpenSendBuffer(peer_index, DownwindMessageSize(chunk_size)).data;
    AppendDownwindMessage(
      raw, incoming_face_slot, total_size, offset, payload.subspan(offset, chunk_size));
  }
}

void
CBC_AsynchronousCommunicator::QueueDelayedDownwindMessage(int destination_location,
                                                          size_t delayed_face_slot,
                                                          std::span<const double> payload)
{
  assert(destination_location >= 0);
  const auto location = static_cast<size_t>(destination_location);
  assert(location < delayed_peer_indices_by_location_.size());
  const auto delayed_peer_index = delayed_peer_indices_by_location_[location];
  assert(delayed_peer_index != INVALID_BUFFER_INDEX);

  const auto total_size = payload.size();
  for (size_t offset = 0; offset < total_size; offset += max_payload_chunk_size_)
  {
    const auto chunk_size = std::min(max_payload_chunk_size_, total_size - offset);
    auto& raw = GetOpenDelayedSendBuffer(delayed_peer_index, DownwindMessageSize(chunk_size)).data;
    AppendDownwindMessage(
      raw, delayed_face_slot, total_size, offset, payload.subspan(offset, chunk_size));
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
CBC_AsynchronousCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::SendData");

  if (send_buffer_.empty())
    return true;

  for (std::size_t i = 0; i < send_buffer_.size();)
  {
    auto& buffer_item = send_buffer_[i];
    if (not buffer_item.send_initiated)
    {
      const auto tag = static_cast<int>(angle_set_id_);
      send_requests_[i] = buffer_item.comm->isend(buffer_item.rank, tag, buffer_item.data);
      buffer_item.send_initiated = true;
    }

    if (mpi::test(send_requests_[i]))
    {
      send_buffer_[i].send_initiated = false;
      send_buffer_[i].data.clear();
      reusable_send_buffers_.push_back(std::move(send_buffer_[i]));
      if (i != send_buffer_.size() - 1)
      {
        send_buffer_[i] = std::move(send_buffer_.back());
        send_requests_[i] = send_requests_.back();
      }
      send_buffer_.pop_back();
      send_requests_.pop_back();
    }
    else
      ++i;
  }
  std::fill(
    open_send_buffer_indices_.begin(), open_send_buffer_indices_.end(), INVALID_BUFFER_INDEX);
  std::fill(open_delayed_send_buffer_indices_.begin(),
            open_delayed_send_buffer_indices_.end(),
            INVALID_BUFFER_INDEX);

  return send_buffer_.empty();
}

void
CBC_AsynchronousCommunicator::QueueDelayedCompletionMarkers()
{
  for (std::size_t delayed_peer_index = 0; delayed_peer_index < delayed_send_peers_.size();
       ++delayed_peer_index)
  {
    auto& raw = GetOpenDelayedSendBuffer(delayed_peer_index, DownwindMessageSize(0)).data;
    AppendDownwindMessage(
      raw, CBC_FLUDSCommonData::INVALID_FACE_SLOT, 0, 0, std::span<const double>());
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

  return SendData();
}

void
CBC_AsynchronousCommunicator::Reset()
{
  send_buffer_.clear();
  send_requests_.clear();
  reusable_send_buffers_.clear();
  receive_buffer_.clear();
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
CBC_AsynchronousCommunicator::ReceiveData(std::vector<std::uint32_t>& cells_who_received_data)
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveData");

  cells_who_received_data.clear();
  if (cells_who_received_data.capacity() < num_receive_sources_)
    cells_who_received_data.reserve(num_receive_sources_);

  const auto tag = static_cast<int>(angle_set_id_);
  const auto& location_dependencies = fluds_.GetSPDS().GetLocationDependencies();
  for (const int locJ : location_dependencies)
  {
    const auto source_rank = comm_set_.MapIonJ(locJ, location_id_);
    mpi::Status status;
    while (receive_comm_.iprobe(source_rank, tag, status))
    {
      const auto num_items = status.count<std::byte>();
      receive_buffer_.resize(num_items);
      receive_comm_.recv(status.source(), status.tag(), receive_buffer_.data(), num_items);

      std::size_t offset = 0;

      while (offset < receive_buffer_.size())
      {
        const auto incoming_face_slot = ReadMessageValue<std::size_t>(receive_buffer_, offset);
        const auto total_size = ReadMessageValue<std::size_t>(receive_buffer_, offset);
        const auto chunk_offset = ReadMessageValue<std::size_t>(receive_buffer_, offset);
        const auto chunk_size = ReadMessageValue<std::size_t>(receive_buffer_, offset);
        assert(incoming_face_slot != CBC_FLUDSCommonData::INVALID_FACE_SLOT);

        if (incoming_face_slot >= incoming_partials_.size())
          throw std::logic_error("CBC received non-local psi for an unknown face slot.");

        const auto expected_size = cbc_fluds_.GetIncomingNonlocalPsiSize(incoming_face_slot);
        ValidatePayloadChunk(expected_size,
                             total_size,
                             chunk_offset,
                             chunk_size,
                             max_payload_chunk_size_,
                             "CBC received a malformed non-local psi chunk.");

        const auto num_bytes = chunk_size * sizeof(double);
        assert(offset + num_bytes <= receive_buffer_.size());

        if (chunk_offset == 0 and chunk_size == total_size)
        {
          auto incoming =
            cbc_fluds_.PrepareIncomingNonlocalPsiBySlot(incoming_face_slot, total_size);
          if (num_bytes != 0)
            std::memcpy(incoming.psi.data(), receive_buffer_.data() + offset, num_bytes);
          cells_who_received_data.push_back(incoming.cell_local_id);
        }
        else
        {
          auto& partial = incoming_partials_[incoming_face_slot];

          if (StorePartialPayload(partial,
                                  total_size,
                                  chunk_offset,
                                  chunk_size,
                                  max_payload_chunk_size_,
                                  receive_buffer_.data() + offset,
                                  "CBC received a duplicate non-local psi chunk."))
          {
            auto incoming =
              cbc_fluds_.PrepareIncomingNonlocalPsiBySlot(incoming_face_slot, total_size);
            std::copy(partial.data.begin(), partial.data.end(), incoming.psi.begin());
            ResetPartialPayload(partial);
            cells_who_received_data.push_back(incoming.cell_local_id);
          }
        }

        offset += num_bytes;
      } // while not at end of buffer
    }
  }
}

bool
CBC_AsynchronousCommunicator::ReceiveDelayedData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveDelayedData");

  const auto& delayed_location_dependencies = fluds_.GetSPDS().GetDelayedLocationDependencies();
  if (delayed_recv_done_.size() != delayed_location_dependencies.size())
    delayed_recv_done_.assign(delayed_location_dependencies.size(), 0);

  const auto tag = static_cast<int>(angle_set_id_);
  for (std::size_t dependency_index = 0; dependency_index < delayed_location_dependencies.size();
       ++dependency_index)
  {
    if (delayed_recv_done_[dependency_index] != 0)
      continue;

    const int locJ = delayed_location_dependencies[dependency_index];
    const auto source_rank = comm_set_.MapIonJ(locJ, location_id_);
    mpi::Status status;
    while (receive_comm_.iprobe(source_rank, tag, status))
    {
      const auto num_items = status.count<std::byte>();
      receive_buffer_.resize(num_items);
      receive_comm_.recv(status.source(), status.tag(), receive_buffer_.data(), num_items);

      std::size_t offset = 0;

      while (offset < receive_buffer_.size())
      {
        const auto delayed_face_slot = ReadMessageValue<std::size_t>(receive_buffer_, offset);
        const auto total_size = ReadMessageValue<std::size_t>(receive_buffer_, offset);
        const auto chunk_offset = ReadMessageValue<std::size_t>(receive_buffer_, offset);
        const auto chunk_size = ReadMessageValue<std::size_t>(receive_buffer_, offset);

        if (delayed_face_slot == CBC_FLUDSCommonData::INVALID_FACE_SLOT)
        {
          if (total_size != 0 or chunk_offset != 0 or chunk_size != 0)
            throw std::logic_error("CBC received a malformed delayed completion marker.");
          delayed_recv_done_[dependency_index] = 1;
          continue;
        }

        if (delayed_face_slot >= delayed_partials_.size())
          throw std::logic_error("CBC received delayed non-local psi for an unknown face slot.");

        const auto expected_size = cbc_fluds_.GetDelayedNonlocalPsiSize(delayed_face_slot);
        ValidatePayloadChunk(expected_size,
                             total_size,
                             chunk_offset,
                             chunk_size,
                             max_payload_chunk_size_,
                             "CBC received a malformed delayed non-local psi chunk.");

        const auto num_bytes = chunk_size * sizeof(double);
        assert(offset + num_bytes <= receive_buffer_.size());

        if (chunk_offset == 0 and chunk_size == total_size)
        {
          if (delayed_payload_received_[delayed_face_slot] != 0)
            throw std::logic_error("CBC received duplicate delayed non-local psi.");

          auto incoming =
            cbc_fluds_.PrepareIncomingDelayedNonlocalPsiBySlot(delayed_face_slot, total_size);
          if (num_bytes != 0)
            std::memcpy(incoming.data(), receive_buffer_.data() + offset, num_bytes);
          delayed_payload_received_[delayed_face_slot] = 1;
        }
        else
        {
          auto& partial = delayed_partials_[delayed_face_slot];

          if (StorePartialPayload(partial,
                                  total_size,
                                  chunk_offset,
                                  chunk_size,
                                  max_payload_chunk_size_,
                                  receive_buffer_.data() + offset,
                                  "CBC received a duplicate delayed non-local psi chunk."))
          {
            if (delayed_payload_received_[delayed_face_slot] != 0)
              throw std::logic_error("CBC received duplicate delayed non-local psi.");

            auto incoming =
              cbc_fluds_.PrepareIncomingDelayedNonlocalPsiBySlot(delayed_face_slot, total_size);
            std::copy(partial.data.begin(), partial.data.end(), incoming.begin());
            ResetPartialPayload(partial);
            delayed_payload_received_[delayed_face_slot] = 1;
          }
        }

        offset += num_bytes;
      }
    }
  }

  return std::all_of(delayed_recv_done_.begin(),
                     delayed_recv_done_.end(),
                     [](const auto done) { return done != 0; });
}

} // namespace opensn

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

void
AppendDownwindMessage(std::vector<std::byte>& raw,
                      size_t incoming_face_slot,
                      std::span<const double> payload)
{
  const auto data_size = payload.size();
  constexpr size_t header_size = sizeof(std::size_t) + sizeof(std::size_t);
  const auto old_size = raw.size();
  const auto num_bytes = data_size * sizeof(double);
  const auto required_size = old_size + header_size + num_bytes;
  if (raw.capacity() < required_size)
    raw.reserve(required_size);
  raw.resize(required_size);

  auto* write_ptr = raw.data() + old_size;
  WriteMessageValue(write_ptr, incoming_face_slot);
  WriteMessageValue(write_ptr, data_size);
  if (num_bytes != 0)
    std::memcpy(write_ptr, payload.data(), num_bytes);
}

} // namespace

CBC_AsynchronousCommunicator::CBC_AsynchronousCommunicator(size_t angle_set_id,
                                                           FLUDS& fluds,
                                                           const MPICommunicatorSet& comm_set)
  : AsynchronousCommunicator(fluds, comm_set),
    angle_set_id_(angle_set_id),
    location_id_(opensn::mpi_comm.rank()),
    receive_comm_(comm_set.LocICommunicator(location_id_)),
    cbc_fluds_(dynamic_cast<CBC_FLUDS&>(fluds))
{
  const auto& location_dependencies = fluds_.GetSPDS().GetLocationDependencies();
  num_receive_sources_ = location_dependencies.size();

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
CBC_AsynchronousCommunicator::GetOpenSendBuffer(size_t peer_index)
{
  assert(peer_index < open_send_buffer_indices_.size());
  auto& open_buffer_index = open_send_buffer_indices_[peer_index];
  if (open_buffer_index != INVALID_BUFFER_INDEX)
    return send_buffer_[open_buffer_index];

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
CBC_AsynchronousCommunicator::GetOpenDelayedSendBuffer(size_t delayed_peer_index)
{
  assert(delayed_peer_index < open_delayed_send_buffer_indices_.size());
  auto& open_buffer_index = open_delayed_send_buffer_indices_[delayed_peer_index];
  if (open_buffer_index != INVALID_BUFFER_INDEX)
    return send_buffer_[open_buffer_index];

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
  auto& raw = GetOpenSendBuffer(peer_index).data;
  AppendDownwindMessage(raw, incoming_face_slot, payload);
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

  auto& raw = GetOpenDelayedSendBuffer(delayed_peer_index).data;
  AppendDownwindMessage(raw, delayed_face_slot, payload);
}

void
CBC_AsynchronousCommunicator::InitializeDelayedUpstreamData()
{
  cbc_fluds_.AllocateDelayedLocalPsi();
  cbc_fluds_.AllocateDelayedPrelocIOutgoingPsi();
  delayed_recv_done_.assign(fluds_.GetSPDS().GetDelayedLocationDependencies().size(), 0);
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
    auto& raw = GetOpenDelayedSendBuffer(delayed_peer_index).data;
    AppendDownwindMessage(raw, CBC_FLUDSCommonData::INVALID_FACE_SLOT, std::span<const double>());
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
        const auto data_size = ReadMessageValue<std::size_t>(receive_buffer_, offset);
        assert(incoming_face_slot != CBC_FLUDSCommonData::INVALID_FACE_SLOT);

        auto incoming = cbc_fluds_.PrepareIncomingNonlocalPsiBySlot(incoming_face_slot, data_size);
        const auto num_bytes = data_size * sizeof(double);
        assert(offset + num_bytes <= receive_buffer_.size());
        std::memcpy(incoming.psi.data(), receive_buffer_.data() + offset, num_bytes);
        offset += num_bytes;

        cells_who_received_data.push_back(incoming.cell_local_id);
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
        const auto data_size = ReadMessageValue<std::size_t>(receive_buffer_, offset);

        if (delayed_face_slot == CBC_FLUDSCommonData::INVALID_FACE_SLOT)
        {
          assert(data_size == 0);
          delayed_recv_done_[dependency_index] = 1;
          continue;
        }

        auto incoming =
          cbc_fluds_.PrepareIncomingDelayedNonlocalPsiBySlot(delayed_face_slot, data_size);
        const auto num_bytes = data_size * sizeof(double);
        assert(offset + num_bytes <= receive_buffer_.size());
        std::memcpy(incoming.data(), receive_buffer_.data() + offset, num_bytes);
        offset += num_bytes;
      }
    }
  }

  return std::all_of(delayed_recv_done_.begin(),
                     delayed_recv_done_.end(),
                     [](const auto done) { return done != 0; });
}

} // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
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
AppendMessageValue(std::vector<std::byte>& buffer, const T& value)
{
  static_assert(std::is_trivially_copyable_v<T>,
                "CBC message serialization requires trivially copyable values.");
  const auto offset = buffer.size();
  buffer.resize(offset + sizeof(T));
  std::memcpy(buffer.data() + offset, &value, sizeof(T));
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

} // namespace

CBC_AsynchronousCommunicator::CBC_AsynchronousCommunicator(size_t angle_set_id,
                                                           FLUDS& fluds,
                                                           const MPICommunicatorSet& comm_set)
  : AsynchronousCommunicator(fluds, comm_set),
    angle_set_id_(angle_set_id),
    location_id_(opensn::mpi_comm.rank()),
    receive_comm_(comm_set.LocICommunicator(location_id_))
{
  const auto& location_dependencies = fluds_.GetSPDS().GetLocationDependencies();
  receive_source_ranks_.reserve(location_dependencies.size());
  for (int location : location_dependencies)
    receive_source_ranks_.push_back(comm_set_.MapIonJ(location, location_id_));

  send_peers_.reserve(fluds_.GetSPDS().GetLocationSuccessors().size());
  open_send_buffer_indices_.reserve(fluds_.GetSPDS().GetLocationSuccessors().size());
}

const CBC_AsynchronousCommunicator::SendPeer&
CBC_AsynchronousCommunicator::GetSendPeer(int destination)
{
  auto [it, inserted] = send_peers_.try_emplace(destination);
  if (inserted)
  {
    it->second.comm = &comm_set_.LocICommunicator(destination);
    it->second.rank = comm_set_.MapIonJ(destination, destination);
  }

  return it->second;
}

CBC_AsynchronousCommunicator::BufferItem&
CBC_AsynchronousCommunicator::GetOpenSendBuffer(int destination)
{
  const auto lookup_it = open_send_buffer_indices_.find(destination);
  if (lookup_it != open_send_buffer_indices_.end())
    return send_buffer_[lookup_it->second];

  if (reusable_send_buffers_.empty())
    send_buffer_.emplace_back();
  else
  {
    send_buffer_.push_back(std::move(reusable_send_buffers_.back()));
    reusable_send_buffers_.pop_back();
  }

  const auto buffer_index = send_buffer_.size() - 1;
  auto& buffer = send_buffer_.back();
  buffer.destination = destination;
  buffer.send_initiated = false;
  buffer.completed = false;
  buffer.data.clear();
  open_send_buffer_indices_.emplace(destination, buffer_index);
  return buffer;
}

void
CBC_AsynchronousCommunicator::QueueDownwindMessage(int destination,
                                                   std::uint64_t cell_global_id,
                                                   unsigned int face_id,
                                                   std::span<const double> payload)
{
  auto& raw = GetOpenSendBuffer(destination).data;
  const auto data_size = payload.size();
  raw.reserve(raw.size() + sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(std::size_t) +
              data_size * sizeof(double));

  AppendMessageValue(raw, cell_global_id);
  AppendMessageValue(raw, face_id);
  AppendMessageValue(raw, data_size);

  const auto old_size = raw.size();
  const auto num_bytes = data_size * sizeof(double);
  raw.resize(old_size + num_bytes);
  if (num_bytes != 0)
    std::memcpy(raw.data() + old_size, payload.data(), num_bytes);
}

bool
CBC_AsynchronousCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::SendData");

  // Now we attempt to flush items in the send buffer
  bool all_messages_sent = true;
  for (auto& buffer_item : send_buffer_)
  {
    if (not buffer_item.send_initiated)
    {
      const auto& peer = GetSendPeer(buffer_item.destination);
      const auto tag = static_cast<int>(angle_set_id_);
      buffer_item.mpi_request = peer.comm->isend(peer.rank, tag, buffer_item.data);
      buffer_item.send_initiated = true;
    }

    if (not buffer_item.completed)
    {
      if (mpi::test(buffer_item.mpi_request))
        buffer_item.completed = true;
      else
        all_messages_sent = false;
    }
  }

  std::size_t next_active = 0;
  for (std::size_t i = 0; i < send_buffer_.size(); ++i)
  {
    if (send_buffer_[i].completed)
    {
      send_buffer_[i].send_initiated = false;
      send_buffer_[i].data.clear();
      reusable_send_buffers_.push_back(std::move(send_buffer_[i]));
    }
    else
    {
      if (next_active != i)
        send_buffer_[next_active] = std::move(send_buffer_[i]);
      ++next_active;
    }
  }
  send_buffer_.erase(send_buffer_.begin() + static_cast<std::ptrdiff_t>(next_active),
                     send_buffer_.end());
  open_send_buffer_indices_.clear();

  return all_messages_sent;
}

void
CBC_AsynchronousCommunicator::Reset()
{
  send_buffer_.clear();
  reusable_send_buffers_.clear();
  receive_buffer_.clear();
  open_send_buffer_indices_.clear();
}

std::vector<std::uint64_t>
CBC_AsynchronousCommunicator::ReceiveData()
{
  std::vector<std::uint64_t> cells_who_received_data;
  ReceiveData(cells_who_received_data);
  return cells_who_received_data;
}

void
CBC_AsynchronousCommunicator::ReceiveData(std::vector<std::uint64_t>& cells_who_received_data)
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveData");

  cells_who_received_data.clear();
  if (cells_who_received_data.capacity() < receive_source_ranks_.size())
    cells_who_received_data.reserve(receive_source_ranks_.size());

  const auto tag = static_cast<int>(angle_set_id_);
  for (const auto source_rank : receive_source_ranks_)
  {
    mpi::Status status;
    while (receive_comm_.iprobe(source_rank, tag, status))
    {
      const auto num_items = status.count<std::byte>();
      receive_buffer_.resize(num_items);
      receive_comm_.recv(source_rank, status.tag(), receive_buffer_.data(), num_items);

      std::size_t offset = 0;

      while (offset < receive_buffer_.size())
      {
        const auto cell_global_id = ReadMessageValue<std::uint64_t>(receive_buffer_, offset);
        const auto face_id = ReadMessageValue<unsigned int>(receive_buffer_, offset);
        const auto data_size = ReadMessageValue<std::size_t>(receive_buffer_, offset);

        auto& psi_data = fluds_.PrepareIncomingNonlocalPsi(cell_global_id, face_id, data_size);
        const auto num_bytes = data_size * sizeof(double);
        assert(offset + num_bytes <= receive_buffer_.size());
        std::memcpy(psi_data.data(), receive_buffer_.data() + offset, num_bytes);
        offset += num_bytes;

        cells_who_received_data.push_back(
          fluds_.GetSPDS().GetGrid()->MapCellGlobalID2LocalID(cell_global_id));
      } // while not at end of buffer
    }
  }
}

} // namespace opensn

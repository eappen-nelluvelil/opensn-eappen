// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <utility>

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
ReadMessageValue(const std::vector<std::byte>& buffer, size_t& offset)
{
  assert(offset + sizeof(T) <= buffer.size());
  T value;
  std::memcpy(&value, buffer.data() + offset, sizeof(T));
  offset += sizeof(T);
  return value;
}

} // namespace

namespace opensn
{

std::vector<double>&
CBC_AsynchronousCommunicator::InitGetDownwindMessageData(int location_id,
                                                         uint64_t cell_global_id,
                                                         unsigned int face_id,
                                                         size_t /*angle_set_id*/,
                                                         size_t data_size)
{
  auto pending = std::find_if(outgoing_message_queue_.begin(),
                              outgoing_message_queue_.end(),
                              [location_id, cell_global_id, face_id](const PendingMessage& message)
                              {
                                return message.destination == location_id and
                                       message.cell_global_id == cell_global_id and
                                       message.face_id == face_id;
                              });

  if (pending == outgoing_message_queue_.end())
  {
    pending = outgoing_message_queue_.emplace(outgoing_message_queue_.end());
    pending->destination = location_id;
    pending->cell_global_id = cell_global_id;
    pending->face_id = face_id;
  }

  if (pending->data.empty())
    pending->data.assign(data_size, 0.0);

  return pending->data;
}

bool
CBC_AsynchronousCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::SendData");

  // First we convert any new outgoing messages from the queue into
  // buffer messages. We aggregate these messages per location-id
  // they need to be sent to
  if (not outgoing_message_queue_.empty())
  {
    const auto new_buffer_start = send_buffer_.size();

    auto append_send_buffer = [this](int destination) -> BufferItem&
    {
      if (reusable_send_buffers_.empty())
        send_buffer_.emplace_back();
      else
      {
        send_buffer_.push_back(std::move(reusable_send_buffers_.back()));
        reusable_send_buffers_.pop_back();
      }

      auto& buffer = send_buffer_.back();
      buffer.destination = destination;
      buffer.send_initiated = false;
      buffer.completed = false;
      buffer.data.clear();
      return buffer;
    };

    for (const auto& message : outgoing_message_queue_)
    {
      auto buffer_index = send_buffer_.size();
      for (size_t i = new_buffer_start; i < send_buffer_.size(); ++i)
        if (send_buffer_[i].destination == message.destination)
        {
          buffer_index = i;
          break;
        }

      auto& buffer = (buffer_index == send_buffer_.size()) ? append_send_buffer(message.destination)
                                                           : send_buffer_[buffer_index];

      const size_t data_size = message.data.size();
      auto& raw = buffer.data;
      raw.reserve(raw.size() + sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(size_t) +
                  data_size * sizeof(double));

      AppendMessageValue(raw, message.cell_global_id);
      AppendMessageValue(raw, message.face_id);
      AppendMessageValue(raw, data_size);

      const size_t old_size = raw.size();
      const size_t num_bytes = data_size * sizeof(double);
      raw.resize(old_size + num_bytes);
      std::memcpy(raw.data() + old_size, message.data.data(), num_bytes);
    }

    outgoing_message_queue_.clear();
  } // if there are outgoing messages

  // Now we attempt to flush items in the send buffer
  bool all_messages_sent = true;
  for (auto& buffer_item : send_buffer_)
  {
    if (not buffer_item.send_initiated)
    {
      const int locJ = buffer_item.destination;
      const auto& comm = comm_set_.LocICommunicator(locJ);
      auto dest = comm_set_.MapIonJ(locJ, locJ);
      auto tag = static_cast<int>(angle_set_id_);
      buffer_item.mpi_request = comm.isend(dest, tag, buffer_item.data);
      buffer_item.send_initiated = true;
    }

    if (not buffer_item.completed)
    {
      if (mpi::test(buffer_item.mpi_request))
        buffer_item.completed = true;
      else
        all_messages_sent = false;
    }
  } // for item in buffer

  size_t next_active = 0;
  for (size_t i = 0; i < send_buffer_.size(); ++i)
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

  return all_messages_sent;
}

void
CBC_AsynchronousCommunicator::Reset()
{
  outgoing_message_queue_.clear();
  send_buffer_.clear();
  reusable_send_buffers_.clear();
  receive_buffer_.clear();
}

std::vector<uint64_t>
CBC_AsynchronousCommunicator::ReceiveData()
{
  std::vector<uint64_t> cells_who_received_data;
  ReceiveData(cells_who_received_data);
  return cells_who_received_data;
}

void
CBC_AsynchronousCommunicator::ReceiveData(std::vector<std::uint64_t>& cells_who_received_data)
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveData");

  cells_who_received_data.clear();
  const auto& location_dependencies = fluds_.GetSPDS().GetLocationDependencies();
  if (cells_who_received_data.capacity() < location_dependencies.size())
    cells_who_received_data.reserve(location_dependencies.size());

  const auto& comm = comm_set_.LocICommunicator(opensn::mpi_comm.rank());
  const auto tag = static_cast<int>(angle_set_id_);
  for (int locJ : location_dependencies)
  {
    auto source_rank = comm_set_.MapIonJ(locJ, opensn::mpi_comm.rank());
    mpi::Status status;
    while (comm.iprobe(source_rank, tag, status))
    {
      int num_items = status.count<std::byte>();
      receive_buffer_.resize(num_items);
      comm.recv(source_rank, status.tag(), receive_buffer_.data(), num_items);

      size_t offset = 0;

      while (offset < receive_buffer_.size())
      {
        const auto cell_global_id = ReadMessageValue<uint64_t>(receive_buffer_, offset);
        const auto face_id = ReadMessageValue<unsigned int>(receive_buffer_, offset);
        const auto data_size = ReadMessageValue<size_t>(receive_buffer_, offset);

        auto& psi_data = fluds_.PrepareIncomingNonlocalPsi(cell_global_id, face_id, data_size);
        const size_t num_bytes = data_size * sizeof(double);
        assert(offset + num_bytes <= receive_buffer_.size());
        std::memcpy(psi_data.data(), receive_buffer_.data() + offset, num_bytes);
        offset += num_bytes;

        cells_who_received_data.push_back(
          fluds_.GetSPDS().GetGrid()->MapCellGlobalID2LocalID(cell_global_id));
      } // while not at end of buffer
    } // Process each message embedded in buffer
  }
}

} // namespace opensn

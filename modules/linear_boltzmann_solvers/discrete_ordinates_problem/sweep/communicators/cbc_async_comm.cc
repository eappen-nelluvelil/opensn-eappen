// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <cstring>
#include <utility>

namespace opensn
{

CBC_AsynchronousCommunicator::CBC_AsynchronousCommunicator(size_t angle_set_id,
                                                           FLUDS& fluds,
                                                           const MPICommunicatorSet& comm_set)
  : AsynchronousCommunicator(fluds, comm_set),
    angle_set_id_(angle_set_id),
    cbc_fluds_(dynamic_cast<CBC_FLUDS&>(fluds))
{
}

std::vector<double>&
CBC_AsynchronousCommunicator::InitGetDownwindMessageData(int location_id,
                                                         uint64_t cell_global_id,
                                                         unsigned int face_id,
                                                         size_t angle_set_id,
                                                         size_t data_size)
{
  MessageKey key{location_id, cell_global_id, face_id};
  std::vector<double>& data = outgoing_message_queue_[key];
  if (data.empty())
    data.assign(data_size, 0.0);
  return data;
}

void
CBC_AsynchronousCommunicator::QueueOutgoingMessages()
{
  if (outgoing_message_queue_.empty())
    return;

  std::unordered_map<int, size_t> buffer_indices;
  buffer_indices.reserve(outgoing_message_queue_.size());

  for (const auto& [msg_key, data] : outgoing_message_queue_)
  {
    const int locI = std::get<0>(msg_key);
    const uint64_t cell_global_id = std::get<1>(msg_key);
    const unsigned int face_id = std::get<2>(msg_key);
    const size_t data_size = data.size();

    const auto [it, inserted] = buffer_indices.try_emplace(locI, send_buffer_.size());
    if (inserted)
    {
      send_buffer_.emplace_back();
      send_buffer_.back().destination = locI;
    }

    auto& buffer_item = send_buffer_[it->second];
    auto& buffer_array = buffer_item.data_array;
    buffer_array.Write(cell_global_id);
    buffer_array.Write(face_id);
    buffer_array.Write(data_size);

    auto& raw = buffer_array.Data();
    const size_t old_size = raw.size();
    const size_t num_bytes = data_size * sizeof(double);
    raw.resize(old_size + num_bytes);
    std::memcpy(raw.data() + old_size, data.data(), num_bytes);
  }

  outgoing_message_queue_.clear();
}

bool
CBC_AsynchronousCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::SendData");

  QueueOutgoingMessages();

  bool all_messages_sent = true;
  for (auto& buffer_item : send_buffer_)
  {
    if (not buffer_item.send_initiated)
    {
      const int locJ = buffer_item.destination;
      const auto& comm = comm_set_.LocICommunicator(locJ);
      auto dest = comm_set_.MapIonJ(locJ, locJ);
      auto tag = static_cast<int>(angle_set_id_);
      buffer_item.mpi_request = comm.isend(dest, tag, buffer_item.data_array.Data());
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

  if (all_messages_sent)
    send_buffer_.clear();

  return all_messages_sent;
}

std::vector<uint64_t>
CBC_AsynchronousCommunicator::ReceiveData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveData");

  std::vector<uint64_t> cells_who_received_data;
  const auto& location_dependencies = fluds_.GetSPDS().GetLocationDependencies();
  for (int locJ : location_dependencies)
  {
    const auto& comm = comm_set_.LocICommunicator(opensn::mpi_comm.rank());
    auto source_rank = comm_set_.MapIonJ(locJ, opensn::mpi_comm.rank());
    auto tag = static_cast<int>(angle_set_id_);
    mpi::Status status;
    while (comm.iprobe(source_rank, tag, status))
    {
      int num_items = status.count<std::byte>();
      std::vector<std::byte> recv_buffer(num_items);
      comm.recv(source_rank, status.tag(), recv_buffer.data(), num_items);
      ByteArray data_array(recv_buffer);

      while (not data_array.EndOfBuffer())
      {
        const auto cell_global_id = data_array.Read<uint64_t>();
        const auto face_id = data_array.Read<unsigned int>();
        const auto data_size = data_array.Read<size_t>();

        const size_t num_bytes = data_size * sizeof(double);
        cbc_fluds_.StoreIncomingFaceData(
          cell_global_id,
          face_id,
          reinterpret_cast<const double*>(&data_array.Data()[data_array.Offset()]),
          data_size);
        data_array.Seek(data_array.Offset() + num_bytes);

        cells_who_received_data.push_back(
          fluds_.GetSPDS().GetGrid()->MapCellGlobalID2LocalID(cell_global_id));
      } // while not at end of buffer
    } // Process each message embedded in buffer
  }

  return cells_who_received_data;
}

} // namespace opensn

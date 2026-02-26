// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_direct_comm.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "caliper/cali.h"

namespace opensn
{

CBCD_DirectCommunicator::CBCD_DirectCommunicator(size_t angle_set_id,
                                                   const MPICommunicatorSet& comm_set,
                                                   const std::vector<int>& location_dependencies,
                                                   const MeshContinuum& grid)
  : angle_set_id_(angle_set_id),
    comm_set_(comm_set),
    location_dependencies_(location_dependencies),
    grid_(grid)
{
}

std::vector<double>&
CBCD_DirectCommunicator::InitGetDownwindMessageData(int location_id,
                                                     uint64_t cell_global_id,
                                                     unsigned int face_id,
                                                     size_t data_size)
{
  MessageKey key{location_id, cell_global_id, face_id};
  std::vector<double>& data = outgoing_message_queue_[key];
  if (data.empty())
    data.assign(data_size, 0.0);
  return data;
}

bool
CBCD_DirectCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBCD_DirectCommunicator::SendData");

  // Convert outgoing messages from queue into send buffer items,
  // aggregated per destination location.
  if (not outgoing_message_queue_.empty())
  {
    std::map<int, BufferItem> locI_buffer_map;

    for (const auto& [msg_key, data] : outgoing_message_queue_)
    {
      const int locI = std::get<0>(msg_key);
      const uint64_t cell_global_id = std::get<1>(msg_key);
      const unsigned int face_id = std::get<2>(msg_key);
      const size_t data_size = data.size();

      BufferItem& buffer_item = locI_buffer_map[locI];
      buffer_item.destination = locI;
      auto& buffer_array = buffer_item.data_array;
      buffer_array.Write(cell_global_id);
      buffer_array.Write(face_id);
      buffer_array.Write(data_size);
      for (const double value : data)
        buffer_array.Write(value);
    }

    for (auto& [locI, buffer] : locI_buffer_map)
      send_buffer_.push_back(std::move(buffer));

    outgoing_message_queue_.clear();
  }

  // Flush items in the send buffer
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
  }

  return all_messages_sent;
}

std::vector<ReceivedMessage>
CBCD_DirectCommunicator::ReceiveData()
{
  CALI_CXX_MARK_SCOPE("CBCD_DirectCommunicator::ReceiveData");

  std::vector<ReceivedMessage> received;

  for (int locJ : location_dependencies_)
  {
    const auto& comm = comm_set_.LocICommunicator(opensn::mpi_comm.rank());
    auto source_rank = comm_set_.MapIonJ(locJ, opensn::mpi_comm.rank());
    auto tag = static_cast<int>(angle_set_id_);

    mpi::Status status;
    if (comm.iprobe(source_rank, tag, status))
    {
      int num_items = status.count<std::byte>();
      std::vector<std::byte> recv_buffer(num_items);
      comm.recv(source_rank, status.tag(), recv_buffer.data(), num_items);

      ByteArray data_array(recv_buffer);

      while (not data_array.EndOfBuffer())
      {
        ReceivedMessage msg;
        msg.cell_global_id = data_array.Read<uint64_t>();
        msg.face_id = data_array.Read<unsigned int>();
        const auto data_size = data_array.Read<size_t>();

        msg.psi_data.reserve(data_size);
        for (size_t k = 0; k < data_size; ++k)
          msg.psi_data.push_back(data_array.Read<double>());

        msg.cell_local_id = grid_.MapCellGlobalID2LocalID(msg.cell_global_id);
        received.push_back(std::move(msg));
      }
    }
  }

  return received;
}

void
CBCD_DirectCommunicator::Reset()
{
  outgoing_message_queue_.clear();
  send_buffer_.clear();
}

} // namespace opensn

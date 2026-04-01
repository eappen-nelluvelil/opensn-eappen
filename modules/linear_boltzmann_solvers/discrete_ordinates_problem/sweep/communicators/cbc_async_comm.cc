// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <cstring>
#include <map>

namespace opensn
{

std::vector<double>&
CBC_AsynchronousCommunicator::InitGetDownwindMessageData(int location_id,
                                                         uint64_t cell_global_id,
                                                         unsigned int face_id,
                                                         size_t angle_set_id,
                                                         size_t data_size)
{
  MessageKey key{location_id, cell_global_id, face_id};
  const auto& delayed_successors = fluds_.GetSPDS().GetDelayedLocationSuccessors();
  auto& queue = (std::find(delayed_successors.begin(), delayed_successors.end(), location_id) !=
                 delayed_successors.end())
                  ? delayed_outgoing_message_queue_
                  : outgoing_message_queue_;
  std::vector<double>& data = queue[key];
  if (data.empty())
    data.assign(data_size, 0.0);
  return data;
}

void
CBC_AsynchronousCommunicator::QueueMessagesForSend(
  std::unordered_map<MessageKey, std::vector<double>, MessageKeyHash>& message_queue)
{
  if (message_queue.empty())
    return;

  std::map<int, BufferItem> locI_buffer_map;

  for (const auto& [msg_key, data] : message_queue)
  {
    const int locI = std::get<0>(msg_key);
    const uint64_t cell_global_id = std::get<1>(msg_key);
    const unsigned int face_id = std::get<2>(msg_key);
    const size_t data_size = data.size();

    auto& buffer_item = locI_buffer_map[locI];
    buffer_item.destination = locI;
    buffer_item.data_array.Write(cell_global_id);
    buffer_item.data_array.Write(face_id);
    buffer_item.data_array.Write(data_size);

    auto& raw = buffer_item.data_array.Data();
    const size_t old_size = raw.size();
    const size_t num_bytes = data_size * sizeof(double);
    raw.resize(old_size + num_bytes);
    std::memcpy(raw.data() + old_size, data.data(), num_bytes);
  }

  for (auto& [locI, buffer] : locI_buffer_map)
    send_buffer_.push_back(std::move(buffer));

  message_queue.clear();
}

void
CBC_AsynchronousCommunicator::InitializeDelayedUpstreamData()
{
  auto& cbc_fluds = dynamic_cast<CBC_FLUDS&>(fluds_);
  cbc_fluds.AllocateDelayedLocalPsi();
  cbc_fluds.AllocateDelayedPrelocIOutgoingPsi();
  delayed_recv_done_.assign(fluds_.GetSPDS().GetDelayedLocationDependencies().size(), false);
  delayed_completion_markers_queued_ = false;
}

bool
CBC_AsynchronousCommunicator::AllBufferedSendsCompleted()
{
  bool all_messages_sent = true;
  for (auto& buffer_item : send_buffer_)
  {
    if (not buffer_item.send_initiated)
    {
      const int locJ = buffer_item.destination;
      const auto& comm = comm_set_.LocICommunicator(locJ);
      const auto dest = comm_set_.MapIonJ(locJ, locJ);
      const auto tag = static_cast<int>(angle_set_id_);
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

bool
CBC_AsynchronousCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::SendData");

  QueueMessagesForSend(outgoing_message_queue_);

  return AllBufferedSendsCompleted();
}

void
CBC_AsynchronousCommunicator::QueueDelayedCompletionMarkers()
{
  const auto& delayed_successors = fluds_.GetSPDS().GetDelayedLocationSuccessors();
  for (const int locI : delayed_successors)
  {
    BufferItem item;
    item.destination = locI;
    item.data_array.Write(delayed_done_cell_id_);
    item.data_array.Write(delayed_done_face_id_);
    item.data_array.Write(static_cast<size_t>(0));
    send_buffer_.push_back(std::move(item));
  }
  delayed_completion_markers_queued_ = true;
}

bool
CBC_AsynchronousCommunicator::FlushSendBuffers()
{
  const bool normal_sent = SendData();
  if (not normal_sent)
    return false;

  QueueMessagesForSend(delayed_outgoing_message_queue_);

  if (not delayed_completion_markers_queued_)
    QueueDelayedCompletionMarkers();

  return AllBufferedSendsCompleted();
}

std::vector<uint64_t>
CBC_AsynchronousCommunicator::ReceiveData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveData");

  std::vector<uint64_t> cells_who_received_data;
  const auto& location_dependencies = fluds_.GetSPDS().GetLocationDependencies();
  auto& cbc_fluds = dynamic_cast<CBC_FLUDS&>(fluds_);

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

        std::vector<double> psi_data(data_size);
        const size_t num_bytes = data_size * sizeof(double);
        std::memcpy(psi_data.data(), &data_array.Data()[data_array.Offset()], num_bytes);
        data_array.Seek(data_array.Offset() + num_bytes);

        cbc_fluds.StoreIncomingFaceData(cell_global_id, face_id, std::move(psi_data));
        cells_who_received_data.push_back(
          fluds_.GetSPDS().GetGrid()->MapCellGlobalID2LocalID(cell_global_id));
      }
    }
  }

  return cells_who_received_data;
}

bool
CBC_AsynchronousCommunicator::ReceiveDelayedData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveDelayedData");

  auto& cbc_fluds = dynamic_cast<CBC_FLUDS&>(fluds_);
  const auto& delayed_dependencies = fluds_.GetSPDS().GetDelayedLocationDependencies();

  for (size_t loc_idx = 0; loc_idx < delayed_dependencies.size(); ++loc_idx)
  {
    const int locJ = delayed_dependencies[loc_idx];
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

        if (cell_global_id == delayed_done_cell_id_ and face_id == delayed_done_face_id_)
        {
          delayed_recv_done_[loc_idx] = true;
          continue;
        }

        const size_t num_bytes = data_size * sizeof(double);
        cbc_fluds.StoreDelayedIncomingFaceData(
          cell_global_id,
          face_id,
          reinterpret_cast<const double*>(&data_array.Data()[data_array.Offset()]),
          data_size);
        data_array.Seek(data_array.Offset() + num_bytes);
      }
    }
  }

  return std::all_of(
    delayed_recv_done_.begin(), delayed_recv_done_.end(), [](bool done) { return done; });
}

} // namespace opensn

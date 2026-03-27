// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <memory>

namespace opensn
{

CBC_AsynchronousCommunicator::CBC_AsynchronousCommunicator(size_t angle_set_id,
                                                           FLUDS& fluds,
                                                           const MPICommunicatorSet& comm_set)
  : AsynchronousCommunicator(fluds, comm_set), angle_set_id_(angle_set_id)
{
  const auto& delayed_successors = fluds.GetSPDS().GetDelayedLocationSuccessors();
  delayed_successor_set_.insert(delayed_successors.begin(), delayed_successors.end());
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

bool
CBC_AsynchronousCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::SendData");

  // Convert new outgoing messages from the queue into buffer messages,
  // aggregated per destination. Messages to delayed successor locations
  // are left in the queue for batch sending via SendDelayedData().
  if (not outgoing_message_queue_.empty())
  {
    std::map<int, BufferItem> locI_buffer_map;

    auto it = outgoing_message_queue_.begin();
    while (it != outgoing_message_queue_.end())
    {
      const int locI = std::get<0>(it->first);

      // Skip delayed successor destinations — accumulate for batch send later
      if (delayed_successor_set_.count(locI) > 0)
      {
        ++it;
        continue;
      }

      const uint64_t cell_global_id = std::get<1>(it->first);
      const unsigned int face_id = std::get<2>(it->first);
      const size_t data_size = it->second.size();

      BufferItem& buffer_item = locI_buffer_map[locI];
      buffer_item.destination = locI;
      auto& buffer_array = buffer_item.data_array;
      buffer_array.Write(cell_global_id);
      buffer_array.Write(face_id);
      buffer_array.Write(data_size);
      for (const double value : it->second)
        buffer_array.Write(value);

      it = outgoing_message_queue_.erase(it);
    }

    for (auto& [locI, buffer] : locI_buffer_map)
      send_buffer_.push_back(std::move(buffer));
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

  return all_messages_sent;
}

bool
CBC_AsynchronousCommunicator::SendDelayedData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::SendDelayedData");

  // Aggregate accumulated delayed outgoing messages into one message per destination
  if (not delayed_sends_initiated_)
  {
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

      outgoing_message_queue_.clear();

      for (auto& [locI, buffer] : locI_buffer_map)
        delayed_send_buffer_.push_back(std::move(buffer));
    }

    delayed_sends_initiated_ = true;
  }

  // Flush delayed send buffer
  bool all_sent = true;
  for (auto& buffer_item : delayed_send_buffer_)
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
        all_sent = false;
    }
  }

  return all_sent;
}

std::vector<uint64_t>
CBC_AsynchronousCommunicator::ReceiveData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveData");

  using CellFaceKey = std::pair<uint64_t, unsigned int>; // cell_gid + face_id
  std::map<CellFaceKey, std::vector<double>> received_messages;
  std::vector<uint64_t> cells_who_received_data;
  const auto& location_dependencies = fluds_.GetSPDS().GetLocationDependencies();
  for (int locJ : location_dependencies)
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
        const auto cell_global_id = data_array.Read<uint64_t>();
        const auto face_id = data_array.Read<unsigned int>();
        const auto data_size = data_array.Read<size_t>();

        std::vector<double> psi_data;
        psi_data.reserve(data_size);
        for (size_t k = 0; k < data_size; ++k)
          psi_data.push_back(data_array.Read<double>());

        received_messages[{cell_global_id, face_id}] = std::move(psi_data);
        cells_who_received_data.push_back(
          fluds_.GetSPDS().GetGrid()->MapCellGlobalID2LocalID(cell_global_id));
      } // while not at end of buffer
    } // Process each message embedded in buffer
  }

  auto* cbc_fluds = dynamic_cast<CBC_FLUDS*>(&fluds_);
  if (cbc_fluds != nullptr)
    cbc_fluds->GetDeplocsOutgoingMessages().merge(received_messages);
  else
    MergeDeplocsOutgoingMessages(received_messages);

  return cells_who_received_data;
}

bool
CBC_AsynchronousCommunicator::ReceiveDelayedData()
{
  CALI_CXX_MARK_SCOPE("CBC_AsynchronousCommunicator::ReceiveDelayedData");

  const auto& delayed_deps = fluds_.GetSPDS().GetDelayedLocationDependencies();
  if (delayed_deps.empty())
    return true;

  auto* cbc_fluds = dynamic_cast<CBC_FLUDS*>(&fluds_);
  if (cbc_fluds == nullptr)
    return true;

  bool all_received = true;
  for (int locJ : delayed_deps)
  {
    // Skip sources we have already received from (one batched message per source)
    if (delayed_deps_received_.count(locJ) > 0)
      continue;

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
        const auto cell_global_id = data_array.Read<uint64_t>();
        const auto face_id = data_array.Read<unsigned int>();
        const auto data_size = data_array.Read<size_t>();

        std::vector<double> psi_data;
        psi_data.reserve(data_size);
        for (size_t k = 0; k < data_size; ++k)
          psi_data.push_back(data_array.Read<double>());

        // Store in the delayed non-local psi (new) flat array
        cbc_fluds->StoreDelayedNonlocalData(
          cell_global_id, face_id, psi_data.data(), data_size);
      }
      delayed_deps_received_.insert(locJ);
    }
    else
    {
      all_received = false;
    }
  }

  return all_received;
}

#ifndef __OPENSN_WITH_GPU__
void
CBC_AsynchronousCommunicator::MergeDeplocsOutgoingMessages(
  std::map<CBC_FLUDS::CellFaceKey, std::vector<double>>& received_messages)
{
}
#endif

} // namespace opensn

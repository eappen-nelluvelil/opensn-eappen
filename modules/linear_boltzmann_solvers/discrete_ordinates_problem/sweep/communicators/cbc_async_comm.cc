// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"

namespace opensn
{

void
CBC_ASynchronousCommunicator::InitStates()
{
  const auto& send_msgs = cbc_fluds_.GetNLSendMessages();
  send_states_.resize(send_msgs.size());
  send_remaining_ = cbc_fluds_.GetNLSendTotalFacesPerMsg();

  const auto& recv_msgs = cbc_fluds_.GetNLRecvMessages();
  recv_states_.resize(recv_msgs.size());
}

void
CBC_ASynchronousCommunicator::CellSwept(uint64_t cell_local_id)
{
  for (const auto& contrib : cbc_fluds_.GetCellSendContributions(cell_local_id))
    send_remaining_[contrib.msg_index] -= contrib.face_count;
}

bool
CBC_ASynchronousCommunicator::SendData()
{
  CALI_CXX_MARK_SCOPE("CBC_ASynchronousCommunicator::SendData");

  const auto& send_msgs = cbc_fluds_.GetNLSendMessages();
  bool all_done = true;

  for (size_t i = 0; i < send_msgs.size(); ++i)
  {
    auto& state = send_states_[i];

    // Post the send once all contributing faces have been written
    if (not state.send_initiated and send_remaining_[i] == 0)
    {
      const auto& msg = send_msgs[i];
      const auto& comm = comm_set_.LocICommunicator(msg.rank);
      auto dest = comm_set_.MapIonJ(msg.rank, msg.rank);
      auto tag = static_cast<int>(angle_set_id_);
      double* buf = cbc_fluds_.GetNLSendBufferPtr() + msg.buf_start;
      auto count = static_cast<int>(msg.buf_size);
      state.request = comm.isend(dest, tag, buf, count);
      state.send_initiated = true;
    }

    if (state.send_initiated and not state.completed)
    {
      if (mpi::test(state.request))
        state.completed = true;
      else
        all_done = false;
    }
    else if (not state.send_initiated)
    {
      all_done = false;
    }
  }

  return all_done;
}

std::vector<uint64_t>
CBC_ASynchronousCommunicator::ReceiveData()
{
  CALI_CXX_MARK_SCOPE("CBC_ASynchronousCommunicator::ReceiveData");

  std::vector<uint64_t> cells_who_received_data;
  const auto& recv_msgs = cbc_fluds_.GetNLRecvMessages();
  const auto& recv_cell_ids = cbc_fluds_.GetRecvMsgCellLocalIDs();

  for (size_t i = 0; i < recv_msgs.size(); ++i)
  {
    if (recv_states_[i].completed)
      continue;

    const auto& msg = recv_msgs[i];
    const auto& comm = comm_set_.LocICommunicator(opensn::mpi_comm.rank());
    auto source_rank = comm_set_.MapIonJ(msg.rank, opensn::mpi_comm.rank());
    auto tag = static_cast<int>(angle_set_id_);
    mpi::Status status;

    if (comm.iprobe(source_rank, tag, status))
    {
      double* buf = cbc_fluds_.GetNLRecvBufferPtr() + msg.buf_start;
      auto count = static_cast<int>(msg.buf_size);
      comm.recv(source_rank, status.tag(), buf, count);
      recv_states_[i].completed = true;

      // Report which local cells received data (one entry per face)
      for (const auto cell_local_id : recv_cell_ids[i])
        cells_who_received_data.push_back(cell_local_id);
    }
  }

  return cells_who_received_data;
}

void
CBC_ASynchronousCommunicator::Reset()
{
  std::fill(cbc_fluds_.GetNLSendBufferPtr(),
            cbc_fluds_.GetNLSendBufferPtr() +
              (cbc_fluds_.GetNLSendMessages().empty()
                 ? 0
                 : (cbc_fluds_.GetNLSendMessages().back().buf_start +
                    cbc_fluds_.GetNLSendMessages().back().buf_size)),
            0.0);

  for (auto& s : send_states_)
  {
    s.send_initiated = false;
    s.completed = false;
  }
  send_remaining_ = cbc_fluds_.GetNLSendTotalFacesPerMsg();

  for (auto& r : recv_states_)
    r.completed = false;
}

} // namespace opensn

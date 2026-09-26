// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_message_transport.h"
#include "framework/mpi/sweep_communicator.h"
#include "framework/runtime.h"
#include "gtest/gtest.h"
#include <cstring>
#include <limits>
#include <memory>

using namespace opensn;

TEST(CBCMessageTransportTest, ValidatesPacketLimit)
{
  auto context = std::make_shared<SweepCommunicator>(mpi_comm, 1);
  EXPECT_THROW(CBC_MessageTransport(nullptr, 64), std::invalid_argument);
  EXPECT_THROW(CBC_MessageTransport(context, CBC_MessageTransport::FRAME_BYTES),
               std::invalid_argument);
  EXPECT_THROW(
    CBC_MessageTransport(context, static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1),
    std::invalid_argument);
  CBC_MessageTransport transport(context, 64);
  EXPECT_THROW(transport.GetMessageBuffer(0, -1, 1), std::logic_error);
  EXPECT_THROW(transport.GetMessageBuffer(0, 0, 0), std::logic_error);
  EXPECT_THROW(transport.GetMessageBuffer(0, 0, 57), std::logic_error);
  transport.Finish();
}

TEST(CBCMessageTransportTest, CoalescesSplitsAndReusesAcrossSweeps)
{
  auto context = std::make_shared<SweepCommunicator>(mpi_comm, 1);
  const auto& comm = context->GetCommunicator();
  const int next = (mpi_comm.rank() + 1) % mpi_comm.size();
  const int previous = (mpi_comm.rank() + mpi_comm.size() - 1) % mpi_comm.size();
  CBC_MessageTransport transport(context, 64);

  for (int epoch = 0; epoch < 3; ++epoch)
  {
    // Two 32-byte frames fit exactly; the third closes the first packet.
    // This also exercises nonblocking self sends in the one-rank unit run.
    for (int frame = 0; frame < 3; ++frame)
    {
      auto& packet = transport.GetMessageBuffer(next, frame + 7, 24);
      packet.insert(packet.end(), 24, static_cast<char>(epoch * 3 + frame));
    }
    transport.Flush(); // Partial packet must be visible before blocking.
    int frame = 0;
    for (int packet_index = 0; packet_index < 2; ++packet_index)
    {
      mpi::Status status;
      ASSERT_EQ(MPI_Probe(previous, 0, static_cast<MPI_Comm>(comm), status), MPI_SUCCESS);
      const auto packet = transport.Receive(status);
      const std::size_t expected_frames = packet_index == 0 ? 2 : 1;
      EXPECT_EQ(packet.size(), 32 * expected_frames);
      // Do not read past a malformed packet even if a test assertion fails.
      for (std::size_t offset = 0; offset + 32 <= packet.size(); offset += 32, ++frame)
      {
        std::uint32_t tag = 0;
        std::uint32_t count = 0;
        std::memcpy(&tag, packet.data() + offset, sizeof(tag));
        std::memcpy(&count, packet.data() + offset + sizeof(tag), sizeof(count));
        EXPECT_EQ(tag, static_cast<std::uint32_t>(frame + 7));
        EXPECT_EQ(count, 24);
        for (std::size_t i = 8; i < 32; ++i)
          EXPECT_EQ(packet[offset + i], static_cast<char>(epoch * 3 + frame));
      }
      transport.Progress();
    }
    EXPECT_EQ(frame, 3);
    transport.Finish();
    comm.barrier();
  }
}

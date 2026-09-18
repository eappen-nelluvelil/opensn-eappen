// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_peer_transport.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include "framework/runtime.h"
#include <gtest/gtest.h>
#include <cstring>
#include <limits>
#include <set>
#include <thread>

using opensn::CBCDPeerTransport;
using opensn::CBCDReceivePacket;

TEST(CBCDPeerTransport, PacketCapacity)
{
  constexpr auto target = CBCDPeerTransport::TARGET_PACKET_BYTES;
  constexpr auto header = sizeof(std::size_t) + opensn::CBCDSectionHeader::SERIALIZED_SIZE;
  EXPECT_EQ(CBCDPeerTransport::PacketBytes(0, 0), 0);
  EXPECT_EQ(CBCDPeerTransport::PacketBytes(64, 32), 64);
  EXPECT_EQ(CBCDPeerTransport::PacketBytes(4 * target, 64), target);
  EXPECT_EQ(CBCDPeerTransport::PacketBytes(4 * target, 2 * target), 2 * target + header);
  EXPECT_THROW(CBCDPeerTransport::PacketBytes(std::numeric_limits<std::size_t>::max(),
                                              std::numeric_limits<int>::max()),
               std::length_error);
}

TEST(CBCDPeerTransport, EmptyTopology)
{
  CBCDPeerTransport transport({}, {}, 0);
  for (int sweep = 0; sweep < 3; ++sweep)
  {
    transport.Start();
    EXPECT_FALSE(transport.Progress(
      [](auto&, int)
      {
        ADD_FAILURE();
        return false;
      }));
    EXPECT_FALSE(transport.HasSends());
  }
}

TEST(CBCDPeerTransport, RepeatedSweepsWithSlowReader)
{
  const auto& comm = opensn::mpi_comm;
  const int previous = (comm.rank() + comm.size() - 1) % comm.size();
  const int next = (comm.rank() + 1) % comm.size();
  constexpr std::size_t bytes = 256UL * 1024;
  CBCDPeerTransport transport({{&comm, previous, bytes}}, {{&comm, next, bytes}}, 17);
  for (const std::size_t messages : {1, 2, 31, 64})
  {
    transport.Start();
    std::size_t sent = 0;
    std::set<std::size_t> received;
    CBCDReceivePacket* held = nullptr;
    std::size_t held_id = 0;
    const auto Send = [&]()
    {
      const auto slot = transport.GetSendSlot(0);
      if (slot.data.empty() or sent == messages)
        return;
      std::fill(slot.data.begin(), slot.data.end(), std::byte{93});
      std::memcpy(slot.data.data(), &sent, sizeof(sent));
      transport.Send(slot, bytes);
      ++sent;
    };
    Send();
    if (messages > 1)
    {
      Send();
      EXPECT_TRUE(transport.GetSendSlot(0).data.empty());
    }
    while (sent < messages or received.size() < messages or transport.HasSends())
    {
      Send();
      transport.Progress(
        [&](CBCDReceivePacket& packet, int count)
        {
          EXPECT_EQ(count, bytes);
          std::size_t id = 0;
          std::memcpy(&id, packet.data.data(), sizeof(id));
          EXPECT_LT(id, messages);
          EXPECT_TRUE(received.insert(id).second);
          EXPECT_TRUE(std::all_of(packet.data.begin() + sizeof(id),
                                  packet.data.end(),
                                  [](auto value) { return value == std::byte{93}; }));
          if (held == nullptr)
          {
            held = &packet;
            held_id = id;
          }
          else
          {
            EXPECT_NE(held, &packet);
            packet.readers.fetch_sub(1, std::memory_order_release);
          }
          return received.size() < messages;
        });
    }
    ASSERT_NE(held, nullptr);
    std::size_t id = 0;
    std::memcpy(&id, held->data.data(), sizeof(id));
    EXPECT_EQ(id, held_id);
    held->readers.fetch_sub(1, std::memory_order_release);
    EXPECT_FALSE(transport.HasSends());
    comm.barrier();
  }
}

TEST(CBCDPeerTransport, OneWayAndZeroFaceSources)
{
  const auto& comm = opensn::mpi_comm;
  if (comm.size() < 2)
    GTEST_SKIP() << "Requires at least two MPI ranks.";
  const bool sender = comm.rank() == 0;
  const bool receiver = comm.rank() == 1;
  std::vector<CBCDPeerTransport::Peer> sources;
  std::vector<CBCDPeerTransport::Peer> destinations;
  if (sender)
    destinations.push_back({&comm, 1, 32});
  if (receiver)
  {
    sources.push_back({&comm, 0, 32});
    sources.push_back({&comm, 1, 0});
  }
  CBCDPeerTransport transport(sources, destinations, 18);
  for (int sweep = 0; sweep < 5; ++sweep)
  {
    transport.Start();
    bool received = not receiver;
    if (sender)
    {
      const auto slot = transport.GetSendSlot(0);
      slot.data.front() = std::byte{79};
      transport.Send(slot, 1);
    }
    while (not received or transport.HasSends())
      transport.Progress(
        [&](CBCDReceivePacket& packet, int count)
        {
          EXPECT_TRUE(receiver);
          EXPECT_EQ(count, 1);
          EXPECT_EQ(packet.source_index, 0);
          EXPECT_EQ(packet.data.front(), std::byte{79});
          packet.readers.fetch_sub(1, std::memory_order_release);
          received = true;
          return false;
        });
    comm.barrier();
  }
}

TEST(CBCDPeerTransport, RepostsAfterAllBuffersAreHeld)
{
  const auto& comm = opensn::mpi_comm;
  const int previous = (comm.rank() + comm.size() - 1) % comm.size();
  const int next = (comm.rank() + 1) % comm.size();
  CBCDPeerTransport transport({{&comm, previous, 32}}, {{&comm, next, 32}}, 20);
  for (int sweep = 0; sweep < 3; ++sweep)
  {
    transport.Start();
    std::vector<CBCDReceivePacket*> held;
    for (std::size_t i = 0; i < CBCDPeerTransport::WINDOW; ++i)
    {
      const auto slot = transport.GetSendSlot(0);
      ASSERT_FALSE(slot.data.empty());
      slot.data.front() = std::byte{57};
      transport.Send(slot, 1);
    }
    while (held.size() < CBCDPeerTransport::WINDOW or transport.HasSends())
      transport.Progress(
        [&](CBCDReceivePacket& packet, int count)
        {
          EXPECT_EQ(count, 1);
          EXPECT_EQ(packet.data.front(), std::byte{57});
          held.push_back(&packet);
          return true;
        });
    for (int i = 0; i < 100; ++i)
      EXPECT_FALSE(transport.Progress(
        [](auto&, int)
        {
          ADD_FAILURE();
          return true;
        }));
    held.front()->readers.fetch_sub(1, std::memory_order_release);
    const auto slot = transport.GetSendSlot(0);
    ASSERT_FALSE(slot.data.empty());
    slot.data.front() = std::byte{83};
    transport.Send(slot, 1);
    bool received = false;
    while (not received or transport.HasSends())
      transport.Progress(
        [&](CBCDReceivePacket& packet, int count)
        {
          EXPECT_EQ(count, 1);
          EXPECT_EQ(&packet, held.front());
          EXPECT_EQ(packet.data.front(), std::byte{83});
          packet.readers.fetch_sub(1, std::memory_order_release);
          received = true;
          return false;
        });
    EXPECT_EQ(held.back()->data.front(), std::byte{57});
    held.back()->readers.fetch_sub(1, std::memory_order_release);
    comm.barrier();
  }
}

TEST(CBCDPeerTransport, ConcurrentSectionReaders)
{
  const auto& comm = opensn::mpi_comm;
  const int previous = (comm.rank() + comm.size() - 1) % comm.size();
  const int next = (comm.rank() + 1) % comm.size();
  constexpr std::size_t messages = 128;
  constexpr std::size_t bytes = 1024;
  CBCDPeerTransport transport({{&comm, previous, bytes}}, {{&comm, next, bytes}}, 19);
  struct Section
  {
    CBCDReceivePacket* packet;
    std::size_t id;
  };
  std::array<opensn::LockFreeSPSCSlotQueue<Section>, 2> queues;
  std::array<std::thread, 2> workers;
  std::atomic<std::size_t> errors{0};
  for (std::size_t worker = 0; worker < workers.size(); ++worker)
  {
    queues[worker].Preallocate(messages);
    workers[worker] = std::thread(
      [&, worker]()
      {
        std::size_t consumed = 0;
        while (consumed < messages)
        {
          queues[worker].ProcessReady(
            [&](const Section& section)
            {
              std::this_thread::yield();
              std::size_t id = 0;
              std::memcpy(&id, section.packet->data.data(), sizeof(id));
              if (id != section.id or section.packet->data.back() != std::byte{43})
                errors.fetch_add(1, std::memory_order_relaxed);
              section.packet->readers.fetch_sub(1, std::memory_order_release);
              ++consumed;
            });
        }
      });
  }
  transport.Start();
  std::size_t sent = 0;
  std::size_t received = 0;
  while (sent < messages or received < messages or transport.HasSends())
  {
    const auto slot = transport.GetSendSlot(0);
    if (sent < messages and not slot.data.empty())
    {
      std::memcpy(slot.data.data(), &sent, sizeof(sent));
      slot.data.back() = std::byte{43};
      transport.Send(slot, bytes);
      ++sent;
    }
    transport.Progress(
      [&](CBCDReceivePacket& packet, int count)
      {
        EXPECT_EQ(count, bytes);
        std::size_t id = 0;
        std::memcpy(&id, packet.data.data(), sizeof(id));
        packet.readers.store(workers.size() + 1, std::memory_order_relaxed);
        for (auto& queue : queues)
        {
          queue.ReserveSlot() = {&packet, id};
          queue.PublishSlot();
        }
        packet.readers.fetch_sub(1, std::memory_order_release);
        return ++received < messages;
      });
  }
  for (auto& worker : workers)
    worker.join();
  EXPECT_EQ(errors.load(), 0);
  comm.barrier();
}

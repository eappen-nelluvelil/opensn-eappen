// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_receive_packet.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include <gtest/gtest.h>
#include <array>
#include <thread>

TEST(CBCDReceivePacket, SectionByteCounts)
{
  opensn::CBCDReceivePacketPool pool(1);
  auto* packet = pool.Acquire(0, 4096);
  std::size_t offset = 1;
  const auto Append = [&](const auto& value)
  {
    std::memcpy(packet->data.data() + offset, &value, sizeof(value));
    offset += sizeof(value);
  };
  // Include an empty section and unequal, unaligned face payloads.
  for (std::size_t section = 0; section < 4; ++section)
  {
    const auto header_offset = offset;
    offset += opensn::CBCDSectionHeader::SERIALIZED_SIZE;
    for (std::uint32_t face = 0; face < section; ++face)
    {
      Append(face);
      Append(std::size_t{face + 1});
      for (std::size_t i = 0; i <= face; ++i)
        Append(static_cast<double>(10 * section + face + i));
    }
    opensn::CBCDSectionHeader{
      section, section, offset - header_offset - opensn::CBCDSectionHeader::SERIALIZED_SIZE}
      .Store(packet->data.data() + header_offset);
  }
  const auto* ptr = packet->data.data() + 1;
  for (std::size_t section = 0; section < 4; ++section)
  {
    const auto header = opensn::CBCDSectionHeader::Load(ptr);
    EXPECT_EQ(header.angle_set_id, section);
    EXPECT_EQ(header.num_faces, section);
    opensn::IncomingFaceBatch batch{
      packet, static_cast<std::size_t>(ptr - packet->data.data()), header.num_faces, 0};
    std::size_t faces = 0;
    std::size_t bytes = 0;
    batch.ProcessFaces(
      [&](const std::uint32_t face, const std::byte* payload)
      {
        EXPECT_EQ(face, faces++);
        for (std::size_t i = 0; i <= face; ++i)
        {
          double value;
          std::memcpy(&value, payload + i * sizeof(double), sizeof(double));
          EXPECT_EQ(value, static_cast<double>(10 * section + face + i));
        }
        bytes += sizeof(std::uint32_t) + sizeof(std::size_t) + (face + 1) * sizeof(double);
      });
    EXPECT_EQ(faces, section);
    EXPECT_EQ(bytes, header.num_bytes);
    ptr += header.num_bytes;
  }
  EXPECT_EQ(ptr, packet->data.data() + offset);
  pool.Release(packet);
}

TEST(CBCDReceivePacket, RetainsProducerAndAllSections)
{
  opensn::CBCDReceivePacketPool pool(1);
  auto* packet = pool.Acquire(0, 4096);
  packet->data[0] = std::byte{71};
  packet->readers.store(3, std::memory_order_relaxed);
  pool.Release(packet);
  pool.Release(packet);
  auto* other = pool.Acquire(0, 32);
  EXPECT_NE(packet, other);
  EXPECT_EQ(packet->data[0], std::byte{71});
  pool.Release(other);
  pool.Release(packet);
  auto* reused = pool.Acquire(0, 4096);
  EXPECT_EQ(reused, packet);
  EXPECT_EQ(reused->data[0], std::byte{71});
  pool.Release(reused);
}

TEST(CBCDReceivePacket, SmallPacketDoesNotRetainLargeBuffer)
{
  opensn::CBCDReceivePacketPool pool(2);
  auto* large = pool.Acquire(0, 4096);
  pool.Release(large);
  auto* small = pool.Acquire(0, 32);
  EXPECT_NE(large, small);
  EXPECT_LE(small->data.capacity(), 64);
  auto* reused = pool.Acquire(0, 4096);
  EXPECT_EQ(reused, large);
  auto* other_source = pool.Acquire(1, 4096);
  EXPECT_NE(other_source, reused);
  pool.Release(small);
  pool.Release(reused);
  pool.Release(other_source);
}

TEST(CBCDReceivePacket, SlowReaderDoesNotBlockOtherPackets)
{
  opensn::CBCDReceivePacketPool pool(1);
  auto* held = pool.Acquire(0, 64);
  held->data[0] = std::byte{97};
  held->readers.store(2, std::memory_order_relaxed);
  pool.Release(held);
  for (std::size_t i = 0; i < 10000; ++i)
  {
    auto* next = pool.Acquire(0, 64);
    EXPECT_NE(next, held);
    next->data[0] = std::byte{12};
    pool.Release(next);
  }
  EXPECT_EQ(held->data[0], std::byte{97});
  pool.Release(held);
}

TEST(CBCDReceivePacket, ReusesSmallPacketsAlongsideLargePackets)
{
  opensn::CBCDReceivePacketPool pool(1);
  auto* large = pool.Acquire(0, 4096);
  auto* small = pool.Acquire(0, 32);
  pool.Release(large);
  pool.Release(small);
  for (std::size_t i = 0; i < 100; ++i)
  {
    auto* reused = pool.Acquire(0, 25);
    EXPECT_EQ(reused, small);
    EXPECT_LE(reused->data.capacity(), 32);
    pool.Release(reused);
  }
  auto* reused = pool.Acquire(0, 4000);
  EXPECT_EQ(reused, large);
  pool.Release(reused);
}

TEST(CBCDReceivePacket, UnalignedFacePayloads)
{
  opensn::CBCDReceivePacketPool pool(1);
  auto* packet = pool.Acquire(0, 256);
  std::size_t offset = 1;
  const auto Append = [&](const auto& value)
  {
    std::memcpy(packet->data.data() + offset, &value, sizeof(value));
    offset += sizeof(value);
  };
  for (const std::uint32_t face : {2, 7})
  {
    Append(face);
    Append(std::size_t{3});
    for (std::size_t i = 0; i < 3; ++i)
      Append(static_cast<double>(face + i));
  }
  opensn::IncomingFaceBatch batch{packet, 1, 2, 0};
  std::size_t count = 0;
  batch.ProcessFaces(
    [&](const std::uint32_t face, const std::byte* payload)
    {
      EXPECT_EQ(face, count == 0 ? 2 : 7);
      std::array<double, 3> values;
      std::memcpy(values.data(), payload, sizeof(values));
      for (std::size_t i = 0; i < values.size(); ++i)
        EXPECT_EQ(values[i], static_cast<double>(face + i));
      ++count;
    });
  EXPECT_EQ(count, 2);
  batch.num_faces = 0;
  batch.ProcessFaces([](auto, auto) { ADD_FAILURE(); });
  pool.Release(packet);
}

TEST(CBCDReceivePacket, ConcurrentPublicationAndRecycling)
{
  constexpr std::size_t num_workers = 4;
  constexpr std::size_t num_packets = 10000;
  opensn::CBCDReceivePacketPool pool(3);
  std::array<opensn::LockFreeSPSCSlotQueue<opensn::CBCDReceivePacket*>, num_workers> queues;
  std::array<std::thread, num_workers> workers;
  std::array<bool, num_workers> valid;
  for (std::size_t sweep = 0; sweep < 2; ++sweep)
  {
    valid.fill(true);
    for (std::size_t worker = 0; worker < num_workers; ++worker)
    {
      queues[worker].Preallocate(3);
      workers[worker] = std::thread(
        [&, worker]
        {
          std::size_t received = 0;
          while (received < num_packets)
          {
            queues[worker].ProcessReady(
              [&](opensn::CBCDReceivePacket* packet)
              {
                std::size_t value;
                std::memcpy(&value, packet->data.data() + 1, sizeof(value));
                valid[worker] &= value == received + sweep * num_packets;
                if (worker == 0 and received % 7 == 0)
                  std::this_thread::yield();
                std::memcpy(&value, packet->data.data() + 1, sizeof(value));
                valid[worker] &= value == received + sweep * num_packets;
                pool.Release(packet);
                ++received;
              });
            std::this_thread::yield();
          }
        });
    }
    for (std::size_t i = 0; i < num_packets; ++i)
    {
      auto* packet = pool.Acquire(i % 3, 32 + i % 512);
      const auto value = i + sweep * num_packets;
      std::memcpy(packet->data.data() + 1, &value, sizeof(value));
      packet->readers.store(num_workers + 1, std::memory_order_relaxed);
      for (auto& queue : queues)
      {
        queue.ReserveSlot() = packet;
        queue.PublishSlot();
      }
      pool.Release(packet);
    }
    for (auto& worker : workers)
      worker.join();
    for (const auto correct : valid)
      EXPECT_TRUE(correct);
    pool.Reclaim();
  }
}

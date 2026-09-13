// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <utility>
#include <vector>

namespace opensn
{

/// Immutable received bytes shared by angle-set sections until every reader finishes.
struct CBCDReceivePacket
{
  std::vector<std::byte> data;
  std::size_t source_index = 0;
  std::atomic<std::size_t> readers{1};
  CBCDReceivePacket* next = nullptr;
};

/** Receive storage owned by the communicator, with completed packets returned by workers.
 * Acquire and Reclaim are communicator-only. Release may be called by any reader.
 * All readers must finish before destruction.
 */
class CBCDReceivePacketPool
{
public:
  explicit CBCDReceivePacketPool(std::size_t num_sources) : reusable_(num_sources) {}
  ~CBCDReceivePacketPool() { Reclaim(); }

  /// Acquire one producer reference. Set the full reader count before publishing sections.
  CBCDReceivePacket* Acquire(const std::size_t source_index, const std::size_t num_bytes)
  {
    Reclaim();
    const auto size_class = SizeClass(num_bytes);
    auto packet = std::move(reusable_[source_index][size_class]);
    if (not packet)
      packet = std::make_unique<CBCDReceivePacket>();
    if (packet->data.capacity() < num_bytes)
      packet->data = std::vector<std::byte>(num_bytes);
    else
      packet->data.resize(num_bytes);
    packet->source_index = source_index;
    packet->readers.store(1, std::memory_order_relaxed);
    return packet.release();
  }

  /// Return storage only after the last section and the publishing communicator finish.
  void Release(CBCDReceivePacket* packet)
  {
    if (packet->readers.fetch_sub(1, std::memory_order_acq_rel) != 1)
      return;
    auto* head = completed_.load(std::memory_order_relaxed);
    do
    {
      packet->next = head;
    } while (not completed_.compare_exchange_weak(
      head, packet, std::memory_order_release, std::memory_order_relaxed));
  }

  /// Detach completed packets without scanning packets still in use by workers.
  void Reclaim()
  {
    auto* packet = completed_.exchange(nullptr, std::memory_order_acquire);
    while (packet)
    {
      auto* next = packet->next;
      std::unique_ptr<CBCDReceivePacket> returned(packet);
      auto& cached = reusable_[packet->source_index][SizeClass(packet->data.size())];
      if (not cached or cached->data.capacity() < packet->data.capacity())
        cached = std::move(returned);
      packet = next;
    }
  }

private:
  static std::size_t SizeClass(const std::size_t num_bytes)
  {
    return num_bytes == 0 ? 0 : std::bit_width(num_bytes - 1);
  }

  /// One idle packet per source and power-of-two size class. Small packets cannot pin large
  /// buffers.
  using PacketCache =
    std::array<std::unique_ptr<CBCDReceivePacket>, std::numeric_limits<std::size_t>::digits + 1>;
  std::vector<PacketCache> reusable_;
  /// Intrusive multi-producer return list. Producers only push, the communicator detaches it.
  std::atomic<CBCDReceivePacket*> completed_{nullptr};
};

/// One angle-set section. Its packet reference is released after worker-side placement.
struct IncomingFaceBatch
{
  CBCDReceivePacket* packet = nullptr;
  std::size_t offset = 0;
  std::size_t num_faces = 0;
  std::uint32_t source_partition_index = 0;

  template <typename Callback>
  void ProcessFaces(Callback&& callback) const
  {
    const auto* ptr = packet->data.data() + offset;
    for (std::size_t i = 0; i < num_faces; ++i)
    {
      std::uint32_t face_index;
      std::size_t num_values;
      std::memcpy(&face_index, ptr, sizeof(face_index));
      ptr += sizeof(face_index);
      std::memcpy(&num_values, ptr, sizeof(num_values));
      ptr += sizeof(num_values);
      // Serialized doubles need not be aligned. The worker copies bytes into its FLUDS.
      callback(face_index, ptr);
      ptr += num_values * sizeof(double);
    }
  }
};

} // namespace opensn

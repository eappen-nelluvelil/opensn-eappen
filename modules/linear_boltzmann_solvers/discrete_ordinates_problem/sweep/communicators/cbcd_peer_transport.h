// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_receive_packet.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace opensn
{

/** Fixed storage for peer-to-peer packet progress. Only the communicator thread calls these
 * methods. Workers release packet references after placement. Drain sends and disable all
 * sources before destruction or the next Start. All packet readers must have finished.
 */
class CBCDPeerTransport
{
public:
  static constexpr std::size_t WINDOW = 2;
  static constexpr std::size_t TARGET_PACKET_BYTES = 1024UL * 1024;

  struct Peer
  {
    const mpicpp_lite::Communicator* communicator;
    int rank;
    std::size_t packet_bytes;
  };

  struct SendSlot
  {
    std::span<std::byte> data;
    std::size_t index = 0;
  };

  CBCDPeerTransport(std::vector<Peer> sources, std::vector<Peer> destinations, int tag)
    : sources_(std::move(sources)),
      destinations_(std::move(destinations)),
      tag_(tag),
      enabled_sources_(sources_.size(), false),
      requests_(RequestCount(sources_.size(), destinations_.size()), MPI_REQUEST_NULL),
      send_buffers_(WINDOW * destinations_.size()),
      send_cursors_(destinations_.size(), 0)
  {
    for (const auto& peers : {std::cref(sources_), std::cref(destinations_)})
      for (const auto& peer : peers.get())
        if (peer.packet_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
          throw std::length_error("CBCD peer buffer exceeds the MPI int byte-count limit.");
    for (std::size_t source = 0; source < sources_.size(); ++source)
      for (std::size_t lane = 0; lane < WINDOW; ++lane)
      {
        auto packet = std::make_unique<CBCDReceivePacket>();
        packet->data.resize(sources_[source].packet_bytes);
        packet->source_index = source;
        packet->readers.store(0, std::memory_order_relaxed);
        receive_packets_.push_back(std::move(packet));
      }
    for (std::size_t destination = 0; destination < destinations_.size(); ++destination)
      for (std::size_t lane = 0; lane < WINDOW; ++lane)
        send_buffers_[destination * WINDOW + lane].resize(destinations_[destination].packet_bytes);
    completed_.resize(requests_.size());
    statuses_.resize(requests_.size());
  }

  /// Every packet contains at least one record. Accommodate a face larger than the target.
  static std::size_t PacketBytes(std::size_t total_bytes, std::size_t largest_record_bytes)
  {
    constexpr auto header_bytes = sizeof(std::size_t) + CBCDSectionHeader::SERIALIZED_SIZE;
    constexpr auto count_limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (largest_record_bytes > count_limit - header_bytes)
      throw std::length_error("One CBCD face record exceeds the MPI int byte-count limit.");
    return std::min(total_bytes,
                    std::max(TARGET_PACKET_BYTES, header_bytes + largest_record_bytes));
  }

  void Start()
  {
    assert(active_sends_ == 0 and active_receives_ == 0);
    for (std::size_t index = 0; index < requests_.size(); ++index)
      assert(IsIdle(index));
    for (const auto& packet : receive_packets_)
      assert(packet->readers.load(std::memory_order_relaxed) == 0);
    for (std::size_t source = 0; source < sources_.size(); ++source)
      enabled_sources_[source] = sources_[source].packet_bytes != 0;
    std::fill(send_cursors_.begin(), send_cursors_.end(), 0);
    PostReceives();
  }

  /// Return an idle send buffer, or an empty span if both sends are still MPI-owned.
  SendSlot GetSendSlot(std::size_t destination)
  {
    for (std::size_t offset = 0; offset < WINDOW; ++offset)
    {
      const auto lane = (send_cursors_[destination] + offset) % WINDOW;
      const auto slot = destination * WINDOW + lane;
      const auto request = receive_packets_.size() + slot;
      if (IsIdle(request))
        return {send_buffers_[slot], request};
    }
    return {};
  }

  /// Send only the populated prefix. Its storage is unavailable until MPI completion.
  void Send(const SendSlot& slot, std::size_t num_bytes)
  {
    assert(num_bytes <= slot.data.size());
    assert(num_bytes <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
    assert(IsIdle(slot.index));
    const auto index = slot.index - receive_packets_.size();
    const auto destination = index / WINDOW;
    const auto& peer = destinations_[destination];
    MPI_CHECK(MPI_Isend(slot.data.data(),
                        static_cast<int>(num_bytes),
                        MPI_BYTE,
                        peer.rank,
                        tag_,
                        *peer.communicator,
                        &requests_[slot.index]));
    send_cursors_[destination] = (index + 1) % WINDOW;
    ++active_sends_;
  }

  /** Dispatch completed receives and reclaim sends in one MPI_Testsome call.
   * The callback publishes packet readers and releases the producer reference.
   * It returns false after the last face from that source has been received.
   */
  template <typename Callback>
  bool Progress(Callback callback)
  {
    bool progressed = false;
    int count = 0;
    if (active_sends_ != 0 or active_receives_ != 0)
      MPI_CHECK(MPI_Testsome(static_cast<int>(requests_.size()),
                             requests_.data(),
                             &count,
                             completed_.data(),
                             statuses_.data()));
    if (count != MPI_UNDEFINED)
      for (int entry = 0; entry < count; ++entry)
      {
        progressed = true;
        const auto index = static_cast<std::size_t>(completed_[entry]);
        if (index < receive_packets_.size())
        {
          --active_receives_;
          auto& packet = *receive_packets_[index];
          int bytes = 0;
          MPI_CHECK(MPI_Get_count(&statuses_[entry], MPI_BYTE, &bytes));
          if (not callback(packet, bytes))
            enabled_sources_[packet.source_index] = false;
        }
        else
          --active_sends_;
      }
    for (std::size_t source = 0; source < sources_.size(); ++source)
      if (not enabled_sources_[source])
        CancelSource(source);
    PostReceives();
    return progressed;
  }

  bool HasSends() const { return active_sends_ != 0; }

private:
  static std::size_t RequestCount(std::size_t sources, std::size_t destinations)
  {
    constexpr auto peer_limit = static_cast<std::size_t>(std::numeric_limits<int>::max()) / WINDOW;
    if (sources > peer_limit or destinations > peer_limit - sources)
      throw std::length_error("CBCD peer count exceeds the MPI request-count limit.");
    return WINDOW * (sources + destinations);
  }

  bool IsIdle(std::size_t index) const { return requests_[index] == MPI_REQUEST_NULL; }

  void PostReceives()
  {
    for (std::size_t index = 0; index < receive_packets_.size(); ++index)
    {
      auto& packet = *receive_packets_[index];
      if (enabled_sources_[packet.source_index] and IsIdle(index) and
          packet.readers.load(std::memory_order_acquire) == 0)
      {
        const auto& peer = sources_[packet.source_index];
        packet.readers.store(1, std::memory_order_relaxed);
        MPI_CHECK(MPI_Irecv(packet.data.data(),
                            static_cast<int>(packet.data.size()),
                            MPI_BYTE,
                            peer.rank,
                            tag_,
                            *peer.communicator,
                            &requests_[index]));
        ++active_receives_;
      }
    }
  }

  void CancelSource(std::size_t source)
  {
    for (std::size_t index = source * WINDOW; index < (source + 1) * WINDOW; ++index)
      if (not IsIdle(index))
      {
        // Exact face counts prove that no further packet belongs to this sweep.
        MPI_CHECK(MPI_Cancel(&requests_[index]));
        MPI_CHECK(MPI_Wait(&requests_[index], MPI_STATUS_IGNORE));
        --active_receives_;
        receive_packets_[index]->readers.store(0, std::memory_order_relaxed);
      }
  }

  std::vector<Peer> sources_;
  std::vector<Peer> destinations_;
  int tag_;
  std::vector<bool> enabled_sources_;
  std::vector<MPI_Request> requests_;
  std::vector<int> completed_;
  std::vector<MPI_Status> statuses_;
  std::vector<std::unique_ptr<CBCDReceivePacket>> receive_packets_;
  std::vector<std::vector<std::byte>> send_buffers_;
  std::vector<std::size_t> send_cursors_;
  std::size_t active_sends_ = 0;
  std::size_t active_receives_ = 0;
};

} // namespace opensn

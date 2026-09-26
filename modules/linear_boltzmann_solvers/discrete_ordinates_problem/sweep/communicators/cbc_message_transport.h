// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpicpp-lite/mpicpp-lite.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace opensn
{

class SweepCommunicator;

/**
 * Single-threaded normal-flux transport shared by one host CBC groupset.
 *
 * Each frame contains a uint32 angle tag, a uint32 byte count, and one existing
 * CBC packet. Frames from different ready angle sets share a destination packet.
 * Closed packets start immediately; all partial packets MUST be flushed before
 * waiting for incoming data or entering the end-of-sweep barrier. Delayed flux
 * continues to use the angle communicators after this normal phase is drained.
 * This is an internal, homogeneous-job wire format, not persisted restart data.
 * Finish must precede destruction; no MPI request may outlive its packet storage.
 */
class CBC_MessageTransport
{
public:
  static constexpr std::size_t FRAME_BYTES = 2 * sizeof(std::uint32_t);

  CBC_MessageTransport(std::shared_ptr<const SweepCommunicator> communicator,
                       std::size_t packet_limit);

  CBC_MessageTransport(const CBC_MessageTransport&) = delete;
  CBC_MessageTransport& operator=(const CBC_MessageTransport&) = delete;

  /// Normal-phase context, collectively duplicated and distinct from delayed traffic.
  const mpicpp_lite::Communicator& GetCommunicator() const;

  /**
   * Append an envelope header and return its peer buffer for direct serialization.
   * The caller MUST immediately append exactly num_bytes bytes before another call.
   * No intermediate angle-local packet or extra payload copy is needed.
   */
  std::vector<char>& GetMessageBuffer(int rank, int angle_tag, std::size_t num_bytes);
  /// Initiate every open partial packet; the MPI requests retain their storage.
  void Flush();
  /// Recycle completed outgoing packets without waiting for a remote consumer.
  void Progress();
  /// Wait for outstanding sends after all normal data has been consumed globally.
  void Finish();
  /// Receive the exact packet probed by the sole groupset dispatcher.
  std::span<const char> Receive(const mpicpp_lite::Status& status);
  std::size_t GetPacketLimit() const { return packet_limit_; }

private:
  void Send(int rank, std::vector<char>& data);
  void Acquire(std::vector<char>& data);

  std::shared_ptr<const SweepCommunicator> communicator_;
  const std::size_t packet_limit_;
  std::map<int, std::vector<char>> open_;
  std::vector<std::vector<char>> pending_;
  std::vector<mpicpp_lite::Request> requests_;
  std::vector<std::vector<char>> reusable_;
  std::vector<int> completed_;
  std::vector<char> incoming_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_message_transport.h"
#include "framework/mpi/sweep_communicator.h"
#include "framework/utils/error.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <utility>

namespace opensn
{

CBC_MessageTransport::CBC_MessageTransport(std::shared_ptr<const SweepCommunicator> communicator,
                                           std::size_t packet_limit)
  : communicator_(std::move(communicator)), packet_limit_(packet_limit)
{
  OpenSnInvalidArgumentIf(communicator_ == nullptr or packet_limit_ <= FRAME_BYTES or
                            packet_limit_ >
                              static_cast<std::size_t>(std::numeric_limits<int>::max()),
                          "Invalid host CBC transport context or MPI packet limit.");
}

void
CBC_MessageTransport::Acquire(std::vector<char>& data)
{
  if (data.capacity() == 0 and not reusable_.empty())
  {
    data = std::move(reusable_.back());
    reusable_.pop_back();
  }
}

void
CBC_MessageTransport::Send(int rank, std::vector<char>& data)
{
  if (data.empty())
    return;
  OpenSnLogicalErrorIf(pending_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                       "Too many outstanding host CBC transport packets for MPI.");
  pending_.push_back(std::move(data));
  requests_.emplace_back();
  requests_.back() = communicator_->GetCommunicator().isend(rank, 0, pending_.back());
  data.clear();
}

std::vector<char>&
CBC_MessageTransport::GetMessageBuffer(int rank, int angle_tag, std::size_t num_bytes)
{
  OpenSnLogicalErrorIf(angle_tag < 0 or num_bytes == 0 or num_bytes > packet_limit_ - FRAME_BYTES,
                       "Host CBC frame exceeds the transport packet limit.");
  auto& data = open_[rank];
  const auto length = FRAME_BYTES + num_bytes;
  if (length > packet_limit_ - data.size())
    Send(rank, data);
  Acquire(data);
  const auto old_size = data.size();
  data.resize(old_size + FRAME_BYTES);
  auto* out = data.data() + old_size;
  const auto tag = static_cast<std::uint32_t>(angle_tag);
  const auto count = static_cast<std::uint32_t>(num_bytes);
  std::memcpy(out, &tag, sizeof(tag));
  std::memcpy(out + sizeof(tag), &count, sizeof(count));
  return data;
}

void
CBC_MessageTransport::Flush()
{
  for (auto& [rank, data] : open_)
    Send(rank, data);
}

void
CBC_MessageTransport::Progress()
{
  if (requests_.empty())
    return;
  completed_.clear();
  mpicpp_lite::test_some(requests_, completed_);
  std::ranges::sort(completed_, std::greater<>{});
  for (const auto index : completed_)
  {
    const auto i = static_cast<std::size_t>(index);
    pending_[i].clear();
    reusable_.push_back(std::move(pending_[i]));
    if (i != pending_.size() - 1)
    {
      pending_[i] = std::move(pending_.back());
      requests_[i] = requests_.back();
    }
    pending_.pop_back();
    requests_.pop_back();
  }
}

void
CBC_MessageTransport::Finish()
{
  Flush();
  mpicpp_lite::wait_all(requests_);
  for (auto& data : pending_)
  {
    data.clear();
    reusable_.push_back(std::move(data));
  }
  pending_.clear();
  requests_.clear();
}

std::span<const char>
CBC_MessageTransport::Receive(const mpicpp_lite::Status& status)
{
  const auto count = status.count<char>();
  OpenSnLogicalErrorIf(status.tag() != 0 or count < static_cast<int>(FRAME_BYTES) or
                         static_cast<std::size_t>(count) > packet_limit_,
                       "Invalid host CBC normal transport packet.");
  incoming_.resize(static_cast<std::size_t>(count));
  communicator_->GetCommunicator().recv(status.source(), status.tag(), incoming_.data(), count);
  return incoming_;
}

} // namespace opensn

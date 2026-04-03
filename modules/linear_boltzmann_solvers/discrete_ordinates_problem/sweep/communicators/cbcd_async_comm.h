// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_treiber_stack.h"
#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class MPICommunicatorSet;

/// Dedicated MPI progress engine for the threaded CBCD sweep.
class CBCD_AsynchronousCommunicator
{
public:
  struct Wire
  {
    struct EntryHeader
    {
      std::uint64_t cell_global_id = 0;
      unsigned int face_id = 0;
      size_t data_size = 0;
    };

    static constexpr size_t AGGREGATE_HEADER_BYTES = sizeof(size_t);
    static constexpr size_t SECTION_HEADER_BYTES = 2 * sizeof(size_t);
    static constexpr size_t ENTRY_HEADER_BYTES =
      sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(size_t);

    static size_t LoadSize(const std::byte*& ptr)
    {
      size_t value;
      std::memcpy(&value, ptr, sizeof(size_t));
      ptr += sizeof(size_t);
      return value;
    }

    static void StoreSize(std::byte* ptr, size_t value) { std::memcpy(ptr, &value, sizeof(size_t)); }

    static EntryHeader LoadEntryHeader(const std::byte*& ptr)
    {
      EntryHeader header;
      std::memcpy(&header.cell_global_id, ptr, sizeof(std::uint64_t));
      ptr += sizeof(std::uint64_t);
      std::memcpy(&header.face_id, ptr, sizeof(unsigned int));
      ptr += sizeof(unsigned int);
      std::memcpy(&header.data_size, ptr, sizeof(size_t));
      ptr += sizeof(size_t);
      return header;
    }

    static void StoreSectionHeader(std::byte* ptr, size_t angle_set_id, size_t num_entries)
    {
      StoreSize(ptr, angle_set_id);
      StoreSize(ptr + sizeof(size_t), num_entries);
    }
  };

  struct IncomingSection
  {
    struct IncomingBuffer
    {
      ByteArray data;
      LockFreeTreiberStack<ByteArray>* recycler = nullptr;

      ~IncomingBuffer()
      {
        if (recycler == nullptr)
          return;
        data.Data().clear();
        recycler->Push(std::move(data));
      }
    };

    std::shared_ptr<IncomingBuffer> buffer;
    size_t offset = 0;
    size_t size = 0;

    const std::byte* Data() const { return buffer->data.Data().data() + offset; }
  };

  CBCD_AsynchronousCommunicator(const std::vector<AngleSet*>& angle_sets,
                                const MPICommunicatorSet& comm_set,
                                size_t max_message_bytes);

  int GetQueueIndex(int dest_location) const;

  void EnqueuePrepackedByIndex(int queue_index, size_t producer_id, ByteArray&& data);

  template <class F>
  bool DrainIncoming(size_t angle_set_id, F&& callback)
  {
    assert(angle_set_id < num_angle_sets_);
    return incoming_mailboxes_[angle_set_id].DrainAndProcess(
      [&callback](IncomingSection&& section) { callback(section); });
  }

  void SignalAngleSetComplete(size_t angle_set_id);

  void Start();

  void Stop();

  ~CBCD_AsynchronousCommunicator();

private:
  struct NeighborQueue
  {
    int dest_location = 0;
    int dest_rank = 0;
    std::vector<std::unique_ptr<LockFreeTreiberStack<ByteArray>>> shards;
  };

  struct InFlightSend
  {
    mpi::Request request;
    ByteArray data;
  };

  void CommThreadLoop();
  bool FlushOutgoing();
  bool ProbeAndReceive();
  bool PollInFlightSends();
  bool AllWorkComplete() const;
  bool HasQueuedSections(const NeighborQueue& queue) const;
  bool DrainSections(NeighborQueue& queue, size_t& num_sections, size_t& total_bytes);
  std::shared_ptr<IncomingSection::IncomingBuffer> AcquireReceiveBuffer(size_t num_bytes);

  ByteArray AcquireSendBuffer()
  {
    if (not send_buffer_pool_.empty())
    {
      ByteArray buf = std::move(send_buffer_pool_.back());
      send_buffer_pool_.pop_back();
      return buf;
    }
    return ByteArray();
  }

  void ReleaseSendBuffer(ByteArray&& buf)
  {
    buf.Data().clear();
    send_buffer_pool_.push_back(std::move(buf));
  }

private:
  const MPICommunicatorSet& comm_set_;
  size_t num_angle_sets_ = 0;
  size_t max_message_bytes_ = 0;
  int mpi_tag_ = 0;
  std::vector<int> source_ranks_;
  std::vector<NeighborQueue> outgoing_queues_;
  std::unordered_map<int, int> dest_to_queue_index_;
  LockFreeTreiberStack<ByteArray> recv_buffer_recycler_;
  std::vector<LockFreeTreiberStack<IncomingSection>> incoming_mailboxes_;
  std::vector<ByteArray> recv_buffer_reuse_cache_;
  std::vector<InFlightSend> in_flight_sends_;
  std::vector<ByteArray> send_buffer_pool_;
  std::vector<ByteArray> drained_sections_;
  size_t num_outgoing_shards_ = 0;
  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_done_;
  std::thread comm_thread_;
};

} // namespace opensn

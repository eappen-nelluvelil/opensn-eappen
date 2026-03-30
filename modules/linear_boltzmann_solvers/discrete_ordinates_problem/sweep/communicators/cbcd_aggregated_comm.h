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

/**
 * Dedicated MPI progress engine for the threaded CBCD sweep.
 *
 * The communicator owns one progress thread for the sweep. Worker threads
 * publish pre-packed outgoing sections into sharded mailboxes, and the
 * progress thread aggregates those sections into one MPI message per
 * destination. Incoming aggregate messages are split into angle-set sections
 * and forwarded through per-angle-set mailboxes.
 */
class CBCD_AggregatedCommunicator
{
public:
  /// CBCD aggregate-message wire helpers.
  struct Wire
  {
    /// Parsed face-payload header.
    struct EntryHeader
    {
      /// Receiving cell global identifier.
      std::uint64_t cell_global_id = 0;
      /// Receiving face identifier.
      unsigned int face_id = 0;
      /// Payload size in doubles.
      size_t data_size = 0;
    };

    /// Aggregate-header size in bytes.
    static constexpr size_t AGGREGATE_HEADER_BYTES = sizeof(size_t);
    /// Section-header size in bytes.
    static constexpr size_t SECTION_HEADER_BYTES = 2 * sizeof(size_t);
    /// Entry-header size in bytes.
    static constexpr size_t ENTRY_HEADER_BYTES =
      sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(size_t);

    /// Load one `size_t` field and advance the cursor.
    static size_t LoadSize(const std::byte*& ptr)
    {
      size_t value;
      std::memcpy(&value, ptr, sizeof(size_t));
      ptr += sizeof(size_t);
      return value;
    }

    /// Store one `size_t` field.
    static void StoreSize(std::byte* ptr, size_t value)
    {
      std::memcpy(ptr, &value, sizeof(size_t));
    }

    /// Load one face-payload header and advance the cursor.
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

    /// Store one section header.
    static void StoreSectionHeader(std::byte* ptr, size_t angle_set_id, size_t num_entries)
    {
      StoreSize(ptr, angle_set_id);
      StoreSize(ptr + sizeof(size_t), num_entries);
    }
  };

  /// Incoming section view into one aggregate receive buffer.
  struct IncomingSection
  {
    /// Shared aggregate receive buffer.
    struct IncomingBuffer
    {
      /// Aggregate message bytes.
      ByteArray data;
      /// Recycler for the aggregate message buffer.
      LockFreeTreiberStack<ByteArray>* recycler = nullptr;

      ~IncomingBuffer()
      {
        if (recycler == nullptr)
          return;
        data.Data().clear();
        recycler->Push(std::move(data));
      }
    };

    /// Shared aggregate-message storage.
    std::shared_ptr<IncomingBuffer> buffer;
    /// Section start offset within `buffer`.
    size_t offset = 0;
    /// Section size in bytes.
    size_t size = 0;

    /// Return the first byte of the section payload.
    const std::byte* Data() const { return buffer->data.Data().data() + offset; }
  };

  /// Construct the communicator for one CBCD groupset.
  ///
  /// \param angle_sets Angle sets served by this communicator.
  /// \param comm_set Communicator mapping between mesh locations.
  /// \param max_message_bytes Worst-case receive message size in bytes.
  CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                              const MPICommunicatorSet& comm_set,
                              size_t max_message_bytes);

  /// Resolve the outgoing queue index for one destination location.
  ///
  /// \param dest_location Destination mesh location.
  /// \return Queue index, or `-1` if the destination is not present.
  int GetQueueIndex(int dest_location) const;

  /// Publish one pre-packed outgoing section.
  ///
  /// \param queue_index Destination queue index returned by GetQueueIndex().
  /// \param producer_id Stable producer identifier used to select a shard.
  /// \param data Packed section buffer.
  void EnqueuePrepackedByIndex(int queue_index, size_t producer_id, ByteArray&& data);

  /// Drain all received sections for one angle set.
  ///
  /// \tparam F Callback type.
  /// \param angle_set_id Angle-set identifier.
  /// \param callback Callback invoked once per received section.
  /// \return `true` if any section was processed.
  template <class F>
  bool DrainIncoming(size_t angle_set_id, F&& callback)
  {
    assert(angle_set_id < num_angle_sets_);
    return incoming_mailboxes_[angle_set_id].DrainAndProcess(
      [&callback](IncomingSection&& section) { callback(section); });
  }

  /// Mark one angle set as complete on the outgoing side.
  ///
  /// \param angle_set_id Angle-set identifier.
  void SignalAngleSetComplete(size_t angle_set_id);

  /// Start the communication thread for one sweep.
  void Start();

  /// Stop the communication thread after queued work drains.
  void Stop();

  ~CBCD_AggregatedCommunicator();

private:
  /// Outgoing mailbox set for one destination location.
  struct NeighborQueue
  {
    /// Destination mesh location.
    int dest_location;
    /// Communicator-local destination rank.
    int dest_rank;
    /// Sharded queues of pre-packed outgoing sections.
    std::vector<std::unique_ptr<LockFreeTreiberStack<ByteArray>>> shards;
  };

  /// In-flight send and its backing storage.
  struct InFlightSend
  {
    /// MPI request handle.
    mpi::Request request;
    /// Serialized send buffer kept alive until completion.
    ByteArray data;
  };

  /// Run the communication-thread progress loop.
  void CommThreadLoop();
  /// Aggregate and launch pending outgoing sends.
  bool FlushOutgoing();
  /// Probe for incoming MPI messages and route them to angle-set mailboxes.
  bool ProbeAndReceive();
  /// Recycle completed sends.
  bool PollInFlightSends();
  /// Check whether all angle sets are complete and no outgoing work remains.
  bool AllWorkComplete() const;

  /// Communicator mapping between mesh locations.
  const MPICommunicatorSet& comm_set_;
  /// Number of angle sets served by this communicator.
  size_t num_angle_sets_;
  /// Worst-case aggregate receive message size in bytes.
  size_t max_message_bytes_;
  /// MPI tag reserved for aggregated CBCD messages.
  int mpi_tag_;
  /// Communicator-local source ranks.
  std::vector<int> source_ranks_;
  /// Outgoing queues indexed by destination slot.
  std::vector<NeighborQueue> outgoing_queues_;
  /// Destination-location to outgoing-queue index map.
  std::unordered_map<int, int> dest_to_queue_index_;
  /// Recycler for aggregate receive buffers.
  LockFreeTreiberStack<ByteArray> recv_buffer_recycler_;
  /// Incoming mailboxes indexed by angle-set identifier.
  std::vector<LockFreeTreiberStack<IncomingSection>> incoming_mailboxes_;
  /// Recycled aggregate receive buffers cached on the communication thread.
  std::vector<ByteArray> recv_buffer_reuse_cache_;
  /// Outstanding nonblocking sends.
  std::vector<InFlightSend> in_flight_sends_;
  /// Recycled outgoing aggregate buffers.
  std::vector<ByteArray> send_buffer_pool_;
  /// Scratch storage for drained outgoing sections on the communication thread.
  std::vector<ByteArray> drained_sections_;
  /// Number of outgoing shards per destination queue.
  size_t num_outgoing_shards_;

  /// Acquire one outgoing aggregate buffer.
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

  /// Return one outgoing aggregate buffer to the recycle pool.
  void ReleaseSendBuffer(ByteArray&& buf)
  {
    buf.Data().clear();
    send_buffer_pool_.push_back(std::move(buf));
  }

  /// Stop flag observed by the communication thread.
  std::atomic<bool> stop_requested_{false};
  /// Per-angle-set completion flags.
  std::vector<std::atomic<bool>> angle_set_done_;
  /// Dedicated communication thread.
  std::thread comm_thread_;
};

} // namespace opensn

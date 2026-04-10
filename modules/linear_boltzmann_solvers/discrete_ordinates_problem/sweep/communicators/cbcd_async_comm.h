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
#include <deque>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class MPICommunicatorSet;

/**
 * Aggregated MPI progress engine for the threaded CBCD sweep.
 *
 * A single communicator instance services all angle sets for a given groupset,
 * running a dedicated communication thread that interleaves MPI_Isend,
 * MPI_Iprobe/MPI_Recv, and in-flight send retirement in a tight poll loop.
 * This design is motivated by Cray MPICH threading limitations on Tuolumne,
 * where per-angle-set MPI threads cause contention.
 *
 * ## Outgoing path
 *
 * Sweep threads enqueue pre-packed ByteArray sections into per-destination
 * lock-free Treiber stacks (sharded by angle-set ID for contention
 * reduction). The communication thread drains all shards for a destination,
 * aggregates sections into a single send buffer (up to max_message_bytes),
 * and issues an MPI_Isend.
 *
 * ## Incoming path
 *
 * The communication thread probes for incoming messages, receives them into
 * recycled buffers, splits them into per-angle-set sections, and pushes
 * each section into the corresponding angle set's lock-free incoming
 * mailbox. Sweep threads drain their mailbox via DrainIncoming.
 *
 * ## Wire format
 *
 * Each aggregate message contains: an aggregate header (num_sections),
 * followed by per-angle-set sections, each with a section header
 * (angle_set_id, num_entries) and a sequence of per-face entries
 * (cell_global_id, face_id, data_size, payload doubles).
 */
class CBCD_AsynchronousCommunicator
{
public:
  /// Wire-format encoding/decoding utilities for aggregate messages.
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

  /// One per-angle-set section extracted from a received aggregate message.
  struct IncomingSection
  {
    /// Reference-counted receive buffer with automatic recycling.
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
    const std::byte* data = nullptr;
    size_t num_entries = 0;
  };

  /**
   * Construct the aggregated communicator for all angle sets.
   *
   * \param angle_sets pointers to all angle sets serviced by this communicator
   * \param comm_set MPI communicator set for tag allocation
   * \param max_message_bytes maximum bytes per aggregate send message
   */
  CBCD_AsynchronousCommunicator(const std::vector<AngleSet*>& angle_sets,
                                const MPICommunicatorSet& comm_set,
                                size_t max_message_bytes);

  /// Resolve a destination MPI location to an outgoing queue index.
  int GetQueueIndex(int dest_location) const;

  /**
   * Enqueue a pre-packed wire-format section for asynchronous transmission.
   *
   * The section is pushed onto the lock-free Treiber stack for the
   * specified queue (sharded by \p producer_id for contention reduction).
   *
   * \param queue_index outgoing queue index (from GetQueueIndex)
   * \param producer_id angle set ID that produced the data (determines shard)
   * \param data pre-packed ByteArray section (consumed via move)
   */
  void EnqueuePrepackedByIndex(int queue_index, size_t producer_id, ByteArray&& data);

  /**
   * Drain all pending incoming sections for an angle set.
   *
   * \tparam F callback type accepting IncomingSection&&
   * \param angle_set_id angle set to drain
   * \param callback invoked once per incoming section
   * \return true if at least one section was drained
   */
  template <class F>
  bool DrainIncoming(size_t angle_set_id, F&& callback)
  {
    assert(angle_set_id < num_angle_sets_);
    return incoming_mailboxes_[angle_set_id].DrainAndProcess(
      [&callback](IncomingSection&& section) { callback(section); });
  }

  /// Signal that an angle set has completed its sweep.
  void SignalAngleSetComplete(size_t angle_set_id);

  /// Start the communication thread.
  void Start();

  /// Request the communication thread to stop and join it.
  void Stop();

  ~CBCD_AsynchronousCommunicator();

private:
  /// Per-destination outgoing queue with sharded lock-free stacks.
  struct NeighborQueue
  {
    /// Communicator used to reach this destination locality.
    const mpi::Communicator* communicator = nullptr;
    /// Destination MPI rank.
    int dest_rank = 0;
    /// Offset into the flattened per-destination/per-angle-set outgoing stack array.
    size_t shard_offset = 0;
  };

  /// In-flight MPI_Isend with associated data buffer.
  struct InFlightSend
  {
    /// MPI request handle.
    mpi::Request request;
    /// Send buffer (kept alive until the send completes).
    ByteArray data;
  };

  /// Main loop of the communication thread.
  void CommThreadLoop();
  /// Drain all outgoing queues and issue MPI_Isend for aggregated messages.
  bool FlushOutgoing();
  /// Probe for incoming MPI messages and dispatch sections to mailboxes.
  bool ProbeAndReceive();
  /// Retire completed in-flight sends and recycle their buffers.
  bool PollInFlightSends();
  /// Check whether all angle sets are complete and all sends are retired.
  bool AllWorkComplete() const;
  /// Acquire a receive buffer (recycled or freshly allocated).
  std::shared_ptr<IncomingSection::IncomingBuffer> AcquireReceiveBuffer(size_t num_bytes);

  /// Acquire a send buffer from the pool, or allocate a fresh one.
  ByteArray AcquireSendBuffer()
  {
    if (not send_buffer_pool_.empty())
    {
      ByteArray buf = std::move(send_buffer_pool_.back());
      send_buffer_pool_.pop_back();
      return buf;
    }
    ByteArray buf;
    if (max_message_bytes_ > 0)
      buf.Data().reserve(max_message_bytes_);
    return buf;
  }

  /// Return a completed send buffer to the pool for reuse.
  void ReleaseSendBuffer(ByteArray&& buf)
  {
    buf.Data().clear();
    send_buffer_pool_.push_back(std::move(buf));
  }

private:
  /// MPI communicator set for send/recv operations.
  const MPICommunicatorSet& comm_set_;
  /// Number of angle sets serviced by this communicator.
  size_t num_angle_sets_ = 0;
  /// Maximum bytes per aggregate send message.
  size_t max_message_bytes_ = 0;
  /// MPI tag for all sends/recvs.
  int mpi_tag_ = 0;
  /// Cached local MPI rank for receive operations.
  int my_rank_ = 0;
  /// Communicator used to receive from all source localities.
  const mpi::Communicator* recv_communicator_ = nullptr;
  /// Per-destination outgoing queues.
  std::vector<NeighborQueue> outgoing_queues_;
  /// Number of per-destination stacks, currently one per angle set.
  size_t num_outgoing_stacks_per_queue_ = 1;
  /// Flattened per-destination/per-angle-set outgoing stacks.
  std::deque<LockFreeTreiberStack<ByteArray>> outgoing_stacks_;
  /// Map from destination location to outgoing queue index.
  std::unordered_map<int, int> dest_to_queue_index_;
  /// Lock-free recycler for receive buffers.
  LockFreeTreiberStack<ByteArray> recv_buffer_recycler_;
  /// Per-angle-set lock-free incoming mailboxes.
  std::vector<LockFreeTreiberStack<IncomingSection>> incoming_mailboxes_;
  /// Comm-thread-local cache of receive buffers awaiting recycling.
  std::vector<ByteArray> recv_buffer_reuse_cache_;
  /// In-flight sends awaiting completion.
  std::vector<InFlightSend> in_flight_sends_;
  /// Pool of reusable send buffers.
  std::vector<ByteArray> send_buffer_pool_;
  /// Flag set by Stop() to request the communication thread to exit.
  std::atomic<bool> stop_requested_{false};
  /// Per-angle-set completion flags.
  std::vector<std::atomic<bool>> angle_set_done_;
  /// Communication thread handle.
  std::thread comm_thread_;
};

} // namespace opensn

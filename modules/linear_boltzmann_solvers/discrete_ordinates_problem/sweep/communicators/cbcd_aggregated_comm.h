// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <thread>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class MPICommunicatorSet;

/// A single face's angular flux data received from a remote MPI rank.
/// Deserialized from the aggregated wire format and routed to the target angle set.
struct IncomingFaceData
{
  uint64_t cell_global_id;
  unsigned int face_id;
  std::vector<double> psi_data;
};

/// A single face's angular flux data queued for sending to a remote MPI rank.
/// Produced by worker threads and consumed by the communication thread for serialization.
struct OutgoingFaceData
{
  size_t angle_set_id;
  std::uint64_t cell_global_id;
  unsigned int face_id;
  std::vector<double> psi_data;
};

/// Per-angle-set ring buffer capacity, computed by CBCDSweepChunk at construction time.
struct AngleSetCapacity
{
  size_t outgoing_faces = 0;
  size_t incoming_faces = 0;
};

/// Lock-free MPSC ring buffer for zero-allocation inter-thread communication.
/// Producers reserve slots via atomic fetch_add; the single consumer drains in FIFO order.
template <typename T>
class LockFreeRingBuffer
{
public:
  struct Slot
  {
    T payload;
    std::atomic<bool> ready{false};
  };

private:
  std::vector<Slot> buffer_;
  alignas(std::hardware_destructive_interference_size) std::atomic<size_t> head_{0};
  alignas(std::hardware_destructive_interference_size) size_t tail_{0};

public:
  void Preallocate(size_t capacity)
  {
    buffer_ = std::vector<Slot>(capacity);
  }

  /// Reserve a slot by atomic fetch_add. Spins if slot is still in use by consumer.
  Slot& ReserveSlot()
  {
    size_t idx = head_.fetch_add(1, std::memory_order_relaxed) % buffer_.size();
    while (buffer_[idx].ready.load(std::memory_order_acquire))
      std::this_thread::yield();
    return buffer_[idx];
  }

  /// Mark a slot as ready for the consumer.
  void PublishSlot(Slot& slot)
  {
    slot.ready.store(true, std::memory_order_release);
  }

  /// Output-parameter variant (avoids vector allocation in hot loops).
  void GetReadySlots(std::vector<Slot*>& out)
  {
    out.clear();
    if (buffer_.empty())
      return;
    const size_t cap = buffer_.size();
    size_t current_tail = tail_;
    while (buffer_[current_tail % cap].ready.load(std::memory_order_acquire))
    {
      out.push_back(&buffer_[current_tail % cap]);
      current_tail++;
    }
  }

  /// Free slots for producer reuse after consumer has serialized the data.
  void FreeSlots(size_t count)
  {
    const size_t cap = buffer_.size();
    for (size_t i = 0; i < count; ++i)
    {
      buffer_[tail_ % cap].ready.store(false, std::memory_order_release);
      tail_++;
    }
  }

  /// Process all ready slots in-place without allocation (consumer only).
  /// Returns the number of slots processed.
  template <typename Callback>
  size_t ProcessReady(Callback&& cb)
  {
    if (buffer_.empty())
      return 0;
    const size_t cap = buffer_.size();
    size_t count = 0;
    while (true)
    {
      auto& slot = buffer_[tail_ % cap];
      if (not slot.ready.load(std::memory_order_acquire))
        break;
      cb(slot.payload);
      slot.ready.store(false, std::memory_order_release);
      tail_++;
      count++;
    }
    return count;
  }

  bool Empty() const
  {
    if (buffer_.empty())
      return true;
    return not buffer_[tail_ % buffer_.size()].ready.load(std::memory_order_acquire);
  }
};

/**
 * Aggregated MPI communicator for threaded CBCD sweep.
 *
 * A dedicated communication thread aggregates MPI sends/receives across all angle sets,
 * reducing the number of MPI calls and eliminating MPI contention between worker threads.
 * All inter-thread data handoff uses lock-free MPSC ring buffers (zero heap allocation at
 * runtime after warm-up).
 *
 * Wire format for aggregated MPI messages (one message per destination per flush):
 *   [num_active_angle_sets : size_t]
 *   For each active angle set:
 *     [angle_set_id : size_t]
 *     [num_entries  : size_t]
 *     For each entry (one per outgoing face):
 *       [cell_global_id : uint64_t]
 *       [face_id        : unsigned int]
 *       [data_size      : size_t]         // number of doubles
 *       [psi_data       : double[data_size]]
 */
class CBCD_AggregatedCommunicator
{
public:
  CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                              const MPICommunicatorSet& comm_set,
                              size_t max_message_bytes,
                              const std::vector<AngleSetCapacity>& capacities);

  ~CBCD_AggregatedCommunicator();

  /// Enqueue outgoing face data for a remote MPI rank. The fill callback writes psi_data
  /// directly into the ring buffer slot, avoiding an intermediate staging copy.
  /// Called by worker threads; consumed by the communication thread.
  template <typename FillCallback>
  void EnqueueOutgoing(int dest_rank,
                       size_t angle_set_id,
                       uint64_t cell_global_id,
                       unsigned int face_id,
                       size_t data_size,
                       FillCallback&& fill)
  {
    auto it = dest_to_queue_index_.find(dest_rank);
    assert(it != dest_to_queue_index_.end());
    auto& queue = outgoing_queues_[it->second].queue;
    auto& slot = queue->ReserveSlot();
    slot.payload.angle_set_id = angle_set_id;
    slot.payload.cell_global_id = cell_global_id;
    slot.payload.face_id = face_id;
    slot.payload.psi_data.resize(data_size);
    fill(slot.payload.psi_data.data());
    queue->PublishSlot(slot);
  }

  /// Process all ready incoming face data batches for a given angle set.
  /// The callback receives each batch by reference; slots are freed inline (zero-allocation).
  /// Called by worker threads from CBCD_AngleSet::TryAdvanceOneStep.
  template <typename Callback>
  bool ProcessIncoming(size_t angle_set_id, Callback&& cb)
  {
    assert(angle_set_id < num_angle_sets_);
    return incoming_mailboxes_[angle_set_id].ProcessReady(std::forward<Callback>(cb)) > 0;
  }

  /// Signal that this angle set has no more outgoing data.
  void SignalAngleSetComplete(size_t angle_set_id);

  /// Launch the dedicated communication thread.
  void Start();

  /// Flush remaining sends, wait for completion, and join the communication thread.
  void Stop();

private:
  /// Per-destination outgoing ring buffer. One per MPI rank we send data to.
  struct DestinationQueue
  {
    int dest_rank;
    std::unique_ptr<LockFreeRingBuffer<OutgoingFaceData>> queue;
  };

  /// An in-flight MPI_Isend and its serialized message data.
  struct InFlightSend
  {
    mpi::Request request;
    ByteArray data;
  };

  // -- Communication thread methods ------------------------------------------

  void CommThreadLoop();
  bool SerializeAndSend();
  bool ProbeAndReceive();
  bool PollInFlightSends();
  bool AllAngleSetsComplete() const;

  // -- Configuration (immutable after construction) --------------------------

  const MPICommunicatorSet& comm_set_;
  size_t num_angle_sets_;

  /// MPI tag used for all aggregated messages (set to num_angle_sets to avoid
  /// collisions with per-angle-set tags used by other communicator types).
  int mpi_tag_;

  /// Maximum bytes per MPI message; larger batches are split. Zero means no limit.
  size_t max_message_bytes_;

  /// MPI ranks from which this rank receives face data (union over all angle sets).
  std::vector<int> source_ranks_;

  // -- Outgoing path (worker threads → comm thread) --------------------------

  /// One ring buffer per destination MPI rank for outgoing face data.
  std::vector<DestinationQueue> outgoing_queues_;

  /// Maps destination MPI rank → index into outgoing_queues_.
  std::unordered_map<int, int> dest_to_queue_index_;

  // -- Incoming path (comm thread → worker threads) --------------------------

  /// One ring buffer per angle set for incoming face data batches.
  std::vector<LockFreeRingBuffer<std::vector<IncomingFaceData>>> incoming_mailboxes_;

  // -- Communication thread state --------------------------------------------

  /// Pre-allocated receive buffer for MPI messages.
  ByteArray recv_buffer_;

  /// In-flight MPI_Isends awaiting completion.
  std::vector<InFlightSend> in_flight_sends_;

  /// Pre-allocated slot pointer cache for draining outgoing ring buffers.
  std::vector<typename LockFreeRingBuffer<OutgoingFaceData>::Slot*> slot_cache_;

  /// Scratch buffer for grouping outgoing entries by angle set during serialization.
  std::vector<std::vector<const OutgoingFaceData*>> send_batch_by_angle_set_;

  // -- Synchronization -------------------------------------------------------

  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_done_;
  std::thread comm_thread_;
};

} // namespace opensn

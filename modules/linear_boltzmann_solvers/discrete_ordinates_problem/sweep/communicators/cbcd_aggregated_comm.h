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

/// An entry received from a remote location, destined for a specific angle set.
struct IncomingEntry
{
  uint64_t cell_global_id;
  unsigned int face_id;
  std::vector<double> psi_data;
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

  /// Returns pointers to contiguous ready slots (consumer only).
  std::vector<Slot*> GetReadySlots()
  {
    std::vector<Slot*> ready_slots;
    if (buffer_.empty())
      return ready_slots;
    size_t current_tail = tail_;
    while (buffer_[current_tail % buffer_.size()].ready.load(std::memory_order_acquire))
    {
      ready_slots.push_back(&buffer_[current_tail % buffer_.size()]);
      current_tail++;
    }
    return ready_slots;
  }

  /// Output-parameter variant (avoids vector allocation in hot loops).
  void GetReadySlots(std::vector<Slot*>& out)
  {
    out.clear();
    if (buffer_.empty())
      return;
    size_t current_tail = tail_;
    while (buffer_[current_tail % buffer_.size()].ready.load(std::memory_order_acquire))
    {
      out.push_back(&buffer_[current_tail % buffer_.size()]);
      current_tail++;
    }
  }

  /// Free slots for producer reuse after consumer has serialized the data.
  void FreeSlots(size_t count)
  {
    for (size_t i = 0; i < count; ++i)
    {
      buffer_[tail_ % buffer_.size()].ready.store(false, std::memory_order_release);
      tail_++;
    }
  }

  /// Drain all ready slots (consumer only).
  std::vector<T> Drain()
  {
    std::vector<T> result;
    if (buffer_.empty())
      return result;
    while (buffer_[tail_ % buffer_.size()].ready.load(std::memory_order_acquire))
    {
      size_t idx = tail_ % buffer_.size();
      result.push_back(std::move(buffer_[idx].payload));
      buffer_[idx].ready.store(false, std::memory_order_release);
      tail_++;
    }
    return result;
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
 */
class CBCD_AggregatedCommunicator
{
public:
  /// Struct for outgoing data entries
  struct OutgoingEntry
  {
    size_t angle_set_id;
    std::uint64_t cell_global_id;
    unsigned int face_id;
    std::vector<double> psi_data;
  };

  /// Per-angle-set ring buffer capacity info, computed externally.
  struct AngleSetCapacity
  {
    size_t outgoing_faces = 0;
    size_t incoming_faces = 0;
  };

  CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                              const MPICommunicatorSet& comm_set,
                              size_t max_single_message_size_in_bytes,
                              const std::vector<AngleSetCapacity>& capacities);

  ~CBCD_AggregatedCommunicator();

  /// Enqueue a single outgoing entry directly from a raw buffer (zero-allocation path).
  void EnqueueOutgoing(int dest_location,
                       size_t angle_set_id,
                       uint64_t cell_global_id,
                       unsigned int face_id,
                       const double* psi_data,
                       size_t data_size);

  /// Enqueue an outgoing entry with a callback that fills psi_data directly in the ring
  /// buffer slot, eliminating the intermediate staging buffer copy.
  template <typename FillCallback>
  void EnqueueOutgoingDirect(int dest_location,
                             size_t angle_set_id,
                             uint64_t cell_global_id,
                             unsigned int face_id,
                             size_t data_size,
                             FillCallback&& fill)
  {
    auto it = dest_to_queue_index_.find(dest_location);
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

  /// Pull all received batches for this angle set (lock-free).
  std::vector<std::vector<IncomingEntry>> DequeueIncoming(size_t angle_set_id);

  /// Process all ready incoming batches in-place (zero-allocation path).
  /// The callback receives a const reference to each batch. Slots are freed after
  /// all batches are processed, preserving psi_data vector capacity for reuse.
  template <typename Callback>
  bool ProcessIncoming(size_t angle_set_id, Callback&& cb)
  {
    assert(angle_set_id < num_angle_sets_);
    auto& mailbox = incoming_mailboxes_[angle_set_id];
    auto ready_slots = mailbox.GetReadySlots();
    if (ready_slots.empty())
      return false;
    for (auto* slot : ready_slots)
      cb(slot->payload);
    mailbox.FreeSlots(ready_slots.size());
    return true;
  }

  /// Signal that this angle set has no more outgoing data.
  void SignalAngleSetComplete(size_t angle_set_id);

  /// Launch the dedicated communication thread.
  void Start();

  /// Flush remaining sends, wait for completion, and join the communication thread.
  void Stop();

private:
  /// Struct for per-destination queues of outgoing entries.
  struct NeighborQueue
  {
    int dest_location;
    std::unique_ptr<LockFreeRingBuffer<OutgoingEntry>> queue;
  };

  /// Struct for tracking in-flight MPI sends initiated by the comm thread.
  struct PendingSend
  {
    mpi::Request request;
    ByteArray data;
  };

  /// Main loop of the communication thread.
  void CommThreadLoop();

  bool FlushOutgoing(std::vector<std::vector<const OutgoingEntry*>>& by_angle_set);
  bool ProbeAndReceive();
  bool PollPendingSends();
  bool AllWorkComplete() const;

  const MPICommunicatorSet& comm_set_;
  size_t num_angle_sets_;

  /// Flat array of ring-buffer queues for outgoing data entries.
  std::vector<NeighborQueue> outgoing_queues_;

  /// Lookup map from destination location to outgoing queue index.
  std::unordered_map<int, int> dest_to_queue_index_;

  /// One ring-buffer mailbox per angle set for incoming data batches.
  std::vector<LockFreeRingBuffer<std::vector<IncomingEntry>>> incoming_mailboxes_;

  /// Contiguous array of location dependencies for cache-friendly polling.
  std::vector<int> location_dependencies_;

  /// Pre-allocated ByteArray for incoming MPI messages.
  ByteArray persistent_recv_buffer_;

  /// MPI tag (set to num_angle_sets to avoid collisions with per-angle-set tags).
  int aggregated_tag_;

  /// Maximum single MPI message size for message splitting.
  size_t max_single_message_size_in_bytes_;

  /// In-flight MPI_Isends.
  std::vector<PendingSend> pending_sends_;

  /// Flags to track completion status of angle sets and overall stop request.
  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_done_;

  /// Dedicated communication thread.
  std::thread comm_thread_;

  /// Pre-allocated cache for GetReadySlots in comm thread hot loop.
  std::vector<typename LockFreeRingBuffer<OutgoingEntry>::Slot*> ready_slots_cache_;
};

} // namespace opensn

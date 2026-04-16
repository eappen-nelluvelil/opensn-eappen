// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class MPICommunicatorSet;

/**
 * Aggregated asynchronous communicator for CBCD sweeps.
 *
 * Owns one communication thread for a groupset and handles all MPI communication
 * traffic for the angle sets in that groupset. Outgoing sections are produced by
 * sweep threads, pushed into sharded lock-free Treiber stacks, activated by a
 * destination bitset, aggregated by destination locality, and sent asynchronously.
 * Incoming aggregate messages are received once, split into per-angle-set sections,
 * and pushed into lock-free mailboxes that the sweep threads drain on demand.
 */
class CBCD_AsynchronousCommunicator
{
public:
  /// Wire-format helpers for CBCD aggregate messages.
  struct Wire
  {
    /// Header preceding one packed non-local face payload.
    struct EntryHeader
    {
      /// Global ID of the destination cell receiving the face data.
      std::uint64_t cell_global_id;
      /// Face ID on the destination cell.
      unsigned int face_id;
      /// Payload length in doubles.
      std::size_t data_size;
    };

    /// Size of the aggregate-message header in bytes.
    static constexpr std::size_t AGGREGATE_HEADER_BYTES = sizeof(size_t);
    /// Size of one section header in bytes.
    static constexpr std::size_t SECTION_HEADER_BYTES = 2 * sizeof(size_t);
    /// Size of one entry header in bytes.
    static constexpr std::size_t ENTRY_HEADER_BYTES =
      sizeof(std::uint64_t) + sizeof(unsigned int) + sizeof(std::size_t);

    /**
     * Load a packed `size_t` and advance the byte pointer.
     *
     * \param ptr Pointer into the packed byte stream.
     * \return Decoded value.
     */
    static std::size_t LoadSize(const std::byte*& ptr)
    {
      std::size_t value{};
      std::memcpy(&value, ptr, sizeof(std::size_t));
      ptr += sizeof(std::size_t);
      return value;
    }

    /**
     * Store a packed `size_t`.
     *
     * \param ptr Destination byte pointer.
     * \param value Value to encode.
     */
    static void StoreSize(std::byte* ptr, std::size_t value)
    {
      std::memcpy(ptr, &value, sizeof(std::size_t));
    }

    /**
     * Load one packed face-entry header and advance the byte pointer.
     *
     * \param ptr Pointer into the packed byte stream.
     * \return Decoded entry header.
     */
    static EntryHeader LoadEntryHeader(const std::byte*& ptr)
    {
      EntryHeader header{};
      std::memcpy(&header.cell_global_id, ptr, sizeof(std::uint64_t));
      ptr += sizeof(std::uint64_t);
      std::memcpy(&header.face_id, ptr, sizeof(unsigned int));
      ptr += sizeof(unsigned int);
      std::memcpy(&header.data_size, ptr, sizeof(std::size_t));
      ptr += sizeof(std::size_t);
      return header;
    }

    /**
     * Store one packed section header.
     *
     * \param ptr Destination byte pointer.
     * \param angle_set_id Angle-set ID owning the section.
     * \param num_entries Number of packed face entries in the section.
     */
    static void
    StoreSectionHeader(std::byte* ptr, std::size_t angle_set_id, std::size_t num_entries)
    {
      StoreSize(ptr, angle_set_id);
      StoreSize(ptr + sizeof(std::size_t), num_entries);
    }
  };

  /// Received section view extracted from one aggregate message.
  struct IncomingSection
  {
    /// Shared receive buffer backing one aggregate message.
    struct IncomingBuffer
    {
      /// Raw message bytes.
      ByteArray data;
    };

    /// Shared owning buffer for the aggregate message.
    std::shared_ptr<IncomingBuffer> buffer;
    /// First byte of the section payload within `buffer`.
    const std::byte* data = nullptr;
    /// Number of packed face entries in this section.
    std::size_t num_entries = 0;

    /**
     * Construct one received section view.
     *
     * \param incoming_buffer Shared backing buffer for the full aggregate message.
     * \param section_data First byte of the section payload.
     * \param section_num_entries Number of packed face entries in the section.
     */
    IncomingSection() = default;
    IncomingSection(std::shared_ptr<IncomingBuffer> incoming_buffer,
                    const std::byte* section_data,
                    std::size_t section_num_entries)
      : buffer(std::move(incoming_buffer)), data(section_data), num_entries(section_num_entries)
    {
    }

    IncomingSection(const IncomingSection&) = default;
    IncomingSection& operator=(const IncomingSection&) = default;
    IncomingSection(IncomingSection&&) noexcept = default;
    IncomingSection& operator=(IncomingSection&&) noexcept = default;
    ~IncomingSection() = default;
  };

  /**
   * Construct the aggregated communicator for all anglesets in a groupset.
   *
   * \param angle_sets Pointers to all anglesets serviced by this communicator.
   * \param comm_set MPI communicator set for tag allocation.
   * \param outgoing_dest_localities Ordered outgoing destination-locality table.
   * \param max_message_bytes Maximum bytes for aggregate send message.
   */
  CBCD_AsynchronousCommunicator(const std::vector<AngleSet*>& angle_sets,
                                const MPICommunicatorSet& comm_set,
                                const std::vector<int>& outgoing_dest_localities,
                                std::size_t max_message_bytes);

  /**
   * Resolve queue indices for an ordered list of destination localities.
   *
   * \param dest_localities Destination locality IDs.
   * \param queue_indices Output queue indices aligned with `dest_localities`.
   */
  void ResolveQueueIndices(std::span<const int> dest_localities,
                           std::span<int> queue_indices) const;

  /**
   * Enqueue a pre-packed wire-format section for sending to a destination location.
   *
   * The section is pushed onto the producer shard of the destination queue and the
   * destination's active bit is set so the communication thread can aggregate and
   * flush it.
   *
   * \param queue_index Outgoing queue index resolved during initialization.
   * \param producer_id Angleset ID that produced the data (determines shard).
   * \param data Pre-packed ByteArray section (consumed via move).
   */
  void EnqueuePrepackedByIndex(int queue_index, std::size_t producer_id, ByteArray&& data);

  /**
   * Drain all pending incoming sections for an angleset.
   *
   * \tparam F Callback type accepting IncomingSection&&.
   * \param angle_set_id Angleset to drain.
   * \param callback Invoked once per incoming section.
   * \return True if at least one section was drained.
   */
  template <typename F>
  bool DrainIncoming(std::size_t angle_set_id, F&& callback);

  /// Mark one angle set as fully completed.
  void SignalAngleSetComplete(size_t angle_set_id);

  /// Start the communication thread.
  void Start();

  /// Request the communication thread to stop and join it.
  void Stop();

  ~CBCD_AsynchronousCommunicator();

private:
  struct Impl;

  /// Run the communication-thread event loop.
  void CommThreadLoop();
  /// Dispatch one drained incoming section through a type-erased callback thunk.
  using IncomingSectionCallback = void (*)(const void*, IncomingSection&&);
  /// Drain all pending incoming sections for an angleset through a callback thunk.
  bool DrainIncomingImpl(std::size_t angle_set_id,
                         const void* callback_context,
                         IncomingSectionCallback callback);
  /// Drain all active outgoing destinations and issue aggregated sends.
  bool FlushOutgoing();
  /// Drain one active outgoing queue until no more immediately pending work remains.
  bool FlushActiveOutgoingQueue(std::size_t queue_index);
  /// Probe for incoming MPI messages and dispatch sections to mailboxes.
  bool ProbeAndReceive();
  /// Retire completed in-flight sends and recycle their buffers.
  bool PollInFlightSends();
  /// Check whether all angle sets are done and no queued outgoing work remains.
  bool AllWorkComplete() const;
  /// Acquire a receive buffer from the reuse cache or recycler.
  std::shared_ptr<IncomingSection::IncomingBuffer> AcquireReceiveBuffer(std::size_t num_bytes);
  /// Acquire a send buffer from the pool, or allocate a fresh one.
  ByteArray AcquireSendBuffer();
  /// Return a completed send buffer to the pool for reuse.
  void ReleaseSendBuffer(ByteArray&& buf);

  std::unique_ptr<Impl> impl_;
};

template <typename F>
bool
CBCD_AsynchronousCommunicator::DrainIncoming(std::size_t angle_set_id, F&& callback)
{
  using Callback = std::remove_reference_t<F>;
  const auto dispatch = [](const void* callback_context, IncomingSection&& section)
  {
    const auto* typed_callback = static_cast<const Callback*>(callback_context);
    (*typed_callback)(std::move(section));
  };
  return DrainIncomingImpl(angle_set_id, std::addressof(callback), dispatch);
}

} // namespace opensn

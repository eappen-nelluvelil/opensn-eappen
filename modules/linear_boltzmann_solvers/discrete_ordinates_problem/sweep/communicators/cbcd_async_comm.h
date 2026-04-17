// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/lock_free_queues.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class AngleSet;
class MPICommunicatorSet;

struct IncomingFaceData
{
  std::uint64_t cell_global_id = 0;
  unsigned int face_id = 0;
  std::uint32_t source_slot = 0;
  std::vector<double> psi_data;
};

struct OutgoingFaceData
{
  std::size_t angle_set_id = 0;
  std::uint64_t cell_global_id = 0;
  unsigned int face_id = 0;
  std::vector<double> psi_data;
};

struct AngleSetCapacity
{
  std::size_t outgoing_faces = 0;
  std::size_t incoming_faces = 0;
  std::size_t max_outgoing_face_values = 0;
  std::size_t max_incoming_face_values = 0;
  std::size_t max_incoming_batch_entries = 0;
};

class CBCD_AsynchronousCommunicator
{
public:
  CBCD_AsynchronousCommunicator(const std::vector<AngleSet*>& angle_sets,
                                const MPICommunicatorSet& comm_set,
                                const std::vector<std::vector<int>>& incoming_source_partitions,
                                std::size_t max_message_bytes,
                                const std::vector<AngleSetCapacity>& capacities);

  ~CBCD_AsynchronousCommunicator();

  template <typename FillCallback>
  void EnqueueOutgoing(int dest_rank,
                       std::size_t angle_set_id,
                       std::uint64_t cell_global_id,
                       unsigned int face_id,
                       std::size_t data_size,
                       FillCallback&& fill)
  {
    const auto it = dest_to_queue_index_.find(dest_rank);
    assert(it != dest_to_queue_index_.end());
    auto& queue = *outgoing_queues_[it->second]->queue;
    auto& slot = queue.ReserveSlot();
    slot.payload.angle_set_id = angle_set_id;
    slot.payload.cell_global_id = cell_global_id;
    slot.payload.face_id = face_id;
    slot.payload.psi_data.resize(data_size);
    fill(slot.payload.psi_data.data());
    queue.PublishSlot(slot);
  }

  template <typename Callback>
  bool ProcessIncoming(std::size_t angle_set_id, Callback&& callback)
  {
    assert(angle_set_id < num_angle_sets_);
    return incoming_mailboxes_[angle_set_id]->ProcessReady(std::forward<Callback>(callback)) > 0;
  }

  void SignalAngleSetComplete(std::size_t angle_set_id);
  void Start();
  void Stop();

private:
  struct DestinationQueue
  {
    int dest_rank = 0;
    std::unique_ptr<LockFreeRingBuffer<OutgoingFaceData>> queue;
  };

  struct InFlightSend
  {
    mpi::Request request;
    ByteArray data;
  };

  void CommThreadLoop();
  bool SerializeAndSend();
  bool ProbeAndReceive();
  bool PollInFlightSends();
  bool AllAngleSetsComplete() const;

  const MPICommunicatorSet& comm_set_;
  std::size_t num_angle_sets_;
  int mpi_tag_;
  std::size_t max_message_bytes_;
  int my_rank_ = 0;
  std::vector<int> source_partitions_;
  std::vector<int> source_ranks_;
  std::vector<std::unordered_map<int, std::uint32_t>> source_partition_to_slot_by_angle_set_;
  std::vector<std::unique_ptr<DestinationQueue>> outgoing_queues_;
  std::unordered_map<int, int> dest_to_queue_index_;
  std::vector<std::unique_ptr<LockFreeRingBuffer<std::vector<IncomingFaceData>>>> incoming_mailboxes_;
  std::vector<std::vector<const OutgoingFaceData*>> send_batch_by_angle_set_;
  ByteArray recv_buffer_;
  std::vector<InFlightSend> in_flight_sends_;
  std::atomic<bool> stop_requested_{false};
  std::vector<std::atomic<bool>> angle_set_done_;
  std::thread comm_thread_;
  std::vector<LockFreeRingBuffer<OutgoingFaceData>::Slot*> slot_cache_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/byte_array.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace mpi = mpicpp_lite;

namespace opensn
{

class MPICommunicatorSet;
class MeshContinuum;

/// A received MPI message parsed into its constituent fields.
struct ReceivedMessage
{
  uint64_t cell_global_id;
  unsigned int face_id;
  std::vector<double> psi_data;
  uint64_t cell_local_id;
};

/// Per-angle-set direct MPI communicator for threaded CBCD sweep.
///
/// Each worker thread owns one instance — no mutex needed (single-thread access).
/// No dedicated communication thread; MPI calls are made inline from the worker.
/// Follows the same wire format as CBC_AsynchronousCommunicator.
class CBCD_DirectCommunicator
{
public:
  CBCD_DirectCommunicator(size_t angle_set_id,
                           const MPICommunicatorSet& comm_set,
                           const std::vector<int>& location_dependencies,
                           const MeshContinuum& grid);

  /// Allocate or retrieve the buffer for face data to be sent downstream.
  std::vector<double>& InitGetDownwindMessageData(int location_id,
                                                   uint64_t cell_global_id,
                                                   unsigned int face_id,
                                                   size_t data_size);

  /// Aggregate per-destination, serialize to ByteArray, MPI_Isend, and test completion.
  /// Returns true when all pending sends have completed.
  bool SendData();

  /// MPI_Iprobe each dependency, recv, deserialize.
  /// Returns vector of parsed messages including cell_local_id.
  std::vector<ReceivedMessage> ReceiveData();

  /// Clear all internal buffers for the next sweep.
  void Reset();

private:
  size_t angle_set_id_;
  const MPICommunicatorSet& comm_set_;
  const std::vector<int>& location_dependencies_;
  const MeshContinuum& grid_;

  // location_id, cell_global_id, face_id
  using MessageKey = std::tuple<int, uint64_t, unsigned int>;
  std::map<MessageKey, std::vector<double>> outgoing_message_queue_;

  struct BufferItem
  {
    int destination = 0;
    mpi::Request mpi_request;
    bool send_initiated = false;
    bool completed = false;
    ByteArray data_array;
  };
  std::vector<BufferItem> send_buffer_;
};

} // namespace opensn

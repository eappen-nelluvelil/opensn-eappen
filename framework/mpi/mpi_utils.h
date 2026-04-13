// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/runtime.h"
#include "framework/utils/error.h"
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace opensn
{

namespace mpi_utils
{

inline std::string
MpiErrorMessage(const int ierr)
{
  char error_string[MPI_MAX_ERROR_STRING];
  int error_len = 0;
  MPI_Error_string(ierr, error_string, &error_len);
  return {error_string, static_cast<std::size_t>(error_len)};
}

inline int
TestSome(std::vector<mpi::Request>& requests, std::vector<int>& indices)
{
  if (requests.empty())
  {
    indices.clear();
    return MPI_UNDEFINED;
  }

  // mpi::Request is a thin MPI_Request wrapper with matching storage.
  // MPI_Testsome requires a mutable contiguous MPI_Request array.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* reqs = reinterpret_cast<MPI_Request*>(requests.data());
  indices.resize(requests.size());
  int outcount = MPI_UNDEFINED;
  const int ierr = MPI_Testsome(
    static_cast<int>(requests.size()), reqs, &outcount, indices.data(), MPI_STATUSES_IGNORE);
  OpenSnLogicalErrorIf(ierr != MPI_SUCCESS, "MPI_Testsome failed: " + MpiErrorMessage(ierr));
  if (outcount == MPI_UNDEFINED)
    indices.clear();
  else
    indices.resize(static_cast<std::size_t>(outcount));
  return outcount;
}

inline int
TestSome(std::vector<mpi::Request>& requests,
         std::vector<int>& indices,
         std::vector<mpi::Status>& statuses)
{
  if (requests.empty())
  {
    indices.clear();
    statuses.clear();
    return MPI_UNDEFINED;
  }

  // mpi::Request is a thin MPI_Request wrapper with matching storage.
  // MPI_Testsome requires a mutable contiguous MPI_Request array.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* reqs = reinterpret_cast<MPI_Request*>(requests.data());
  indices.resize(requests.size());
  std::vector<MPI_Status> mpi_statuses(requests.size());
  int outcount = MPI_UNDEFINED;
  const int ierr = MPI_Testsome(
    static_cast<int>(requests.size()), reqs, &outcount, indices.data(), mpi_statuses.data());
  OpenSnLogicalErrorIf(ierr != MPI_SUCCESS, "MPI_Testsome failed: " + MpiErrorMessage(ierr));
  if (outcount == MPI_UNDEFINED)
  {
    indices.clear();
    statuses.clear();
  }
  else
  {
    indices.resize(static_cast<std::size_t>(outcount));
    statuses.clear();
    statuses.reserve(static_cast<std::size_t>(outcount));
    for (int i = 0; i < outcount; ++i)
      statuses.emplace_back(mpi_statuses[static_cast<std::size_t>(i)]);
  }
  return outcount;
}

} // namespace mpi_utils

/**
 * Given each location's local size (of items), builds a vector (dimension comm-size plus 1) of
 * where each location's global indices start and end. Example: location i starts at extents[i] and
 * ends at extents[i+1]
 */
std::vector<uint64_t> BuildLocationExtents(uint64_t local_size, const mpi::Communicator& comm);

/**
 * Given a map with keys indicating the destination process-ids and the values for each key a list
 * of values of type T (T must have an MPI_Datatype). Returns a map with the keys indicating the
 * source process-ids and the values for each key a list of values of type T (sent by the respective
 * process).
 *
 * The keys must be "castable" to `int`.
 *
 * Also expects the MPI_Datatype of T.
 */
template <typename K, class T>
std::map<K, std::vector<T>>
MapAllToAll(const std::map<K, std::vector<T>>& pid_data_pairs,
            const mpi::Communicator& comm = opensn::mpi_comm)
{
  static_assert(std::is_integral_v<K>, "Integral datatype required.");

  // Make sendcounts and senddispls
  std::vector<int> sendcounts(opensn::mpi_comm.size(), 0);
  std::vector<int> senddispls(opensn::mpi_comm.size(), 0);
  {
    size_t accumulated_displ = 0;
    for (const auto& [pid, data] : pid_data_pairs)
    {
      sendcounts[pid] = static_cast<int>(data.size());
      senddispls[pid] = static_cast<int>(accumulated_displ);
      accumulated_displ += data.size();
    }
  }

  // Communicate sendcounts to get recvcounts
  std::vector<int> recvcounts(opensn::mpi_comm.size(), 0);

  comm.all_to_all(sendcounts, recvcounts);

  // Populate recvdispls, sender_pids_set, and total_recv_count
  // All three these quantities are constructed from recvcounts.
  std::vector<int> recvdispls(opensn::mpi_comm.size(), 0);
  /// set of neighbor-partitions sending data
  std::set<K> sender_pids_set;
  {
    int displacement = 0;
    for (int pid = 0; pid < opensn::mpi_comm.size(); ++pid)
    {
      recvdispls[pid] = displacement;
      displacement += recvcounts[pid];

      if (recvcounts[pid] > 0)
        sender_pids_set.insert(static_cast<K>(pid));
    } // for pid
  }

  // Make sendbuf
  // The data for each partition is now loaded into a single buffer
  std::vector<T> sendbuf;
  for (const auto& pid_data_pair : pid_data_pairs)
    sendbuf.insert(sendbuf.end(), pid_data_pair.second.begin(), pid_data_pair.second.end());

  // Make recvbuf
  std::vector<T> recvbuf;
  // Communicate serial data
  comm.all_to_all(sendbuf, sendcounts, senddispls, recvbuf, recvcounts, recvdispls);

  std::map<K, std::vector<T>> output_data;
  {
    for (K pid : sender_pids_set)
    {
      const int data_count = recvcounts.at(pid);
      const int data_displ = recvdispls.at(pid);

      auto& data = output_data[pid];
      data.resize(data_count);

      for (int i = 0; i < data_count; ++i)
        data.at(i) = recvbuf.at(data_displ + i);
    }
  }

  return output_data;
}

} // namespace opensn

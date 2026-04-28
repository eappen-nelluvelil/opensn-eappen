// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mpi/mpi_utils.h"
#include <limits>
#include <stdexcept>

namespace opensn
{

std::span<const int>
TestSomeCompleted(std::vector<mpi::Request>& requests, std::vector<int>& completed_indices)
{
  static_assert(sizeof(mpi::Request) == sizeof(MPI_Request),
                "mpicpp-lite Request must remain layout-compatible with MPI_Request.");

  if (requests.empty())
  {
    return {};
  }

  if (requests.size() == 1)
  {
    int completed = 0;
    auto* request = reinterpret_cast<MPI_Request*>(&requests.front());
    MPI_CHECK(MPI_Test(request, &completed, MPI_STATUS_IGNORE));
    if (not completed)
      return {};

    if (completed_indices.empty())
      completed_indices.resize(1);
    completed_indices.front() = 0;
    return {completed_indices.data(), 1};
  }

  if (requests.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::logic_error("MPI_Testsome request count exceeds the MPI int range.");

  auto* mpi_requests = reinterpret_cast<MPI_Request*>(requests.data());
  int num_completed = MPI_UNDEFINED;
  if (completed_indices.size() < requests.size())
    completed_indices.resize(requests.size());
  MPI_CHECK(MPI_Testsome(static_cast<int>(requests.size()),
                         mpi_requests,
                         &num_completed,
                         completed_indices.data(),
                         MPI_STATUSES_IGNORE));

  if (num_completed == MPI_UNDEFINED or num_completed == 0)
  {
    return {};
  }

  return {completed_indices.data(), static_cast<std::size_t>(num_completed)};
}

std::vector<uint64_t>
BuildLocationExtents(uint64_t local_size, const mpi::Communicator& comm)
{
  const int process_count = comm.size();
  // Get the local vector sizes per process
  std::vector<uint64_t> local_sizes;
  comm.all_gather(local_size, local_sizes);

  // With the vector sizes per processor, now the offsets for each
  // processor can be defined using a cumulative sum per processor.
  // This allows for the determination of whether a global index is
  // locally owned or not.
  std::vector<uint64_t> extents(process_count + 1, 0);
  for (int locJ = 1; locJ < process_count; ++locJ)
    extents[locJ] = extents[locJ - 1] + local_sizes[locJ - 1];
  extents[process_count] = extents[process_count - 1] + local_sizes.back();

  return extents;
}

} // namespace opensn

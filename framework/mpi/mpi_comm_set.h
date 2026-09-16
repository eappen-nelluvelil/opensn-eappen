// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/mesh.h"
#include "framework/runtime.h"
#include "framework/utils/memory.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include <utility>

namespace mpi = mpicpp_lite;

namespace opensn
{

/**
 * Simple implementation a communicator set.
 * Definitions:
 * P = total amount of processors.
 * locI = process I in [0,P]
 */
class MPICommunicatorSet
{
public:
  /// Take ownership of created MPI handles, emptying the supplied containers and group.
  MPICommunicatorSet(std::vector<mpi::Communicator>& communicators,
                     std::vector<mpi::Group>& location_groups,
                     mpi::Group& world_group)
    : world_group_(std::exchange(world_group, mpi::Group(MPI_GROUP_NULL)))
  {
    communicators_.swap(communicators);
    location_groups_.swap(location_groups);
    TraceMemory(
      "mpi.communicators.create.complete",
      reinterpret_cast<std::uintptr_t>(this),
      false,
      {{"communicator_slots", communicators_.size()}, {"groups", location_groups_.size()}});
  }

  MPICommunicatorSet(const MPICommunicatorSet&) = delete;
  MPICommunicatorSet& operator=(const MPICommunicatorSet&) = delete;

  /// Release handles after all communication finishes, before MPI finalization.
  ~MPICommunicatorSet()
  {
    if (not mpi::Environment::is_initialized() or mpi::Environment::is_finalized())
      return;

    TraceMemory("mpi.communicators.release.begin", reinterpret_cast<std::uintptr_t>(this));
    for (auto& communicator : communicators_)
      if (communicator)
        communicator.free();
    for (auto& group : location_groups_)
      if (static_cast<MPI_Group>(group) != MPI_GROUP_NULL)
        group.free();
    if (static_cast<MPI_Group>(world_group_) != MPI_GROUP_NULL)
      world_group_.free();
    TraceMemory("mpi.communicators.release.complete", reinterpret_cast<std::uintptr_t>(this));
  }

  const mpi::Communicator& LocICommunicator(int locI) const { return communicators_[locI]; }

  int MapIonJ(int locI, int locJ) const
  {
    return world_group_.translate_rank(locI, location_groups_[locJ]);
  }

private:
  /**
   * A list of communicators, size P, contains a communicator for only communicating with the
   * neighbors of locI.
   */
  std::vector<mpi::Communicator> communicators_;
  /**
   * A list of groupings, size P, allows mapping of the rank of locJ relative to the local
   * communicator.
   */
  std::vector<mpi::Group> location_groups_;
  /// Used to translate ranks.
  mpi::Group world_group_;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mpi/mpi_comm_set.h"
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace
{

int
CountReleasedCommunicator(MPI_Comm, int, void* attribute, void*)
{
  ++*static_cast<int*>(attribute);
  return MPI_SUCCESS;
}

} // namespace

TEST(MPICommunicatorSet, ReleasesCommunicatorsAfterLastOwner)
{
  static_assert(not std::is_copy_constructible_v<opensn::MPICommunicatorSet>);
  static_assert(not std::is_copy_assignable_v<opensn::MPICommunicatorSet>);
  int key = MPI_KEYVAL_INVALID;
  ASSERT_EQ(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, CountReleasedCommunicator, &key, nullptr),
            MPI_SUCCESS);
  int released = 0;
  for (int trial = 0; trial < 8; ++trial)
  {
    auto world_group = opensn::mpi_comm.group();
    std::vector<mpicpp_lite::Group> groups;
    std::vector<mpicpp_lite::Communicator> communicators;
    for (int rank = 0; rank < opensn::mpi_comm.size(); ++rank)
    {
      groups.push_back(world_group.include({rank}));
      communicators.push_back(opensn::mpi_comm.create(groups.back()));
      if (communicators.back())
        EXPECT_EQ(MPI_Comm_set_attr(communicators.back(), key, &released), MPI_SUCCESS);
    }
    auto owner = std::make_shared<opensn::MPICommunicatorSet>(communicators, groups, world_group);
    EXPECT_TRUE(communicators.empty());
    EXPECT_TRUE(groups.empty());
    EXPECT_EQ(static_cast<MPI_Group>(world_group), MPI_GROUP_NULL);
    auto other_owner = owner;
    const auto rank = opensn::mpi_comm.rank();
    EXPECT_EQ(owner->MapIonJ(rank, rank), 0);
    EXPECT_EQ(owner->LocICommunicator(rank).size(), 1);
    owner.reset();
    EXPECT_EQ(released, trial);
    other_owner.reset();
    EXPECT_EQ(released, trial + 1);
  }
  EXPECT_EQ(MPI_Comm_free_keyval(&key), MPI_SUCCESS);
  EXPECT_EQ(MPI_Barrier(opensn::mpi_comm), MPI_SUCCESS);
}

TEST(MPICommunicatorSet, CommunicatesBeforeRelease)
{
  std::vector<int> ranks(opensn::mpi_comm.size());
  std::iota(ranks.begin(), ranks.end(), 0);
  for (int trial = 0; trial < 8; ++trial)
  {
    auto world_group = opensn::mpi_comm.group();
    std::vector<mpicpp_lite::Group> groups{world_group.include(ranks)};
    std::vector<mpicpp_lite::Communicator> communicators{opensn::mpi_comm.create(groups.front())};
    opensn::MPICommunicatorSet owner(communicators, groups, world_group);
    const auto& comm = owner.LocICommunicator(0);
    const int rank = comm.rank();
    int result = -1;
    const int value = rank + trial;
    auto send = comm.isend((rank + 1) % comm.size(), 0, value);
    comm.recv((rank + comm.size() - 1) % comm.size(), 0, result);
    mpicpp_lite::wait(send);
    EXPECT_EQ(result, (rank + comm.size() - 1) % comm.size() + trial);
  }
}

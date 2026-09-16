// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include <mpi.h>
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace
{

void
Check(int result)
{
  if (result != MPI_SUCCESS)
    MPI_Abort(MPI_COMM_WORLD, result);
}

int
PositiveInt(std::string_view text)
{
  int value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} and result.ptr == text.data() + text.size() and value > 0 ? value
                                                                                            : 0;
}

void
Record(const std::filesystem::path& directory, int rank, int trial)
{
  const auto stem =
    directory / ("rank-" + std::to_string(rank) + "-trial-" + std::to_string(trial));
  std::ifstream status("/proc/self/status");
  std::ofstream snapshot(stem.string() + ".status");
  snapshot << status.rdbuf();
  snapshot.close();
  if (not status or not snapshot)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#if defined(__GLIBC__)
  auto* file = std::fopen((stem.string() + ".malloc.xml").c_str(), "w");
  if (file == nullptr)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  const auto result = malloc_info(0, file);
  const auto closed = std::fclose(file);
  if (result != 0 or closed != 0)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
}

} // namespace

int
main(int argc, char** argv)
{
  const bool matched = argc > 1 and std::string_view(argv[1]) == "matched";
  const bool probe = argc > 1 and std::string_view(argv[1]) == "probe";
  if (argc != 6 or (not matched and not probe) or PositiveInt(argv[3]) == 0 or
      PositiveInt(argv[4]) == 0 or PositiveInt(argv[5]) == 0)
  {
    std::cerr << "Usage: mpi-probe-memory probe|matched NEW_OUTPUT_DIR MESSAGES TRIALS BYTES\n";
    return EXIT_FAILURE;
  }
  const int messages = PositiveInt(argv[3]);
  const int trials = PositiveInt(argv[4]);
  const int bytes = PositiveInt(argv[5]);
  const std::filesystem::path directory(argv[2]);
  int provided = 0;
  Check(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
  int rank = 0;
  int ranks = 0;
  Check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  Check(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  if (ranks < 2)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  if (rank == 0)
  {
    std::error_code error;
    if (not std::filesystem::create_directories(directory, error) or error)
    {
      std::cerr << "Use a new writable output directory: " << directory << '\n';
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    int length = 0;
    Check(MPI_Get_library_version(version, &length));
    if (length > 0 and version[length - 1] == '\0')
      --length;
    std::ofstream metadata(directory / "metadata.txt");
    metadata.write(version, length);
    metadata << "\nmode=" << argv[1] << "\nranks=" << ranks << "\nmessages=" << messages
             << "\ntrials=" << trials << "\nbytes=" << bytes << "\nthread_level=" << provided
             << '\n';
    metadata.close();
    if (not metadata)
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  Check(MPI_Barrier(MPI_COMM_WORLD));
  const int source = rank == 0 ? ranks - 1 : rank - 1;
  const int destination = rank == ranks - 1 ? 0 : rank + 1;
  std::vector<unsigned char> send(static_cast<std::size_t>(bytes));
  std::vector<unsigned char> receive(static_cast<std::size_t>(bytes));
  Record(directory, rank, 0);
  for (int trial = 0; trial < trials; ++trial)
  {
    for (int i = 0; i < messages; ++i)
    {
      const auto value = static_cast<unsigned char>((rank % 251 + trial % 251 + i % 251) % 251);
      const auto expected =
        static_cast<unsigned char>((source % 251 + trial % 251 + i % 251) % 251);
      const int count = i % bytes;
      const int tag = i % 7;
      std::fill(send.begin(), send.begin() + count, value);
      MPI_Request request = MPI_REQUEST_NULL;
      Check(MPI_Isend(
        send.data(), count, MPI_UNSIGNED_CHAR, destination, tag, MPI_COMM_WORLD, &request));
      MPI_Status status;
      int available = 0;
      MPI_Message message = MPI_MESSAGE_NULL;
      while (not available)
      {
        if (matched)
          Check(MPI_Improbe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &available, &message, &status));
        else
          Check(MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &available, &status));
      }
      int received = 0;
      Check(MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &received));
      if (received != count or status.MPI_SOURCE != source or status.MPI_TAG != tag)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      if (matched)
      {
        Check(MPI_Mrecv(receive.data(), received, MPI_UNSIGNED_CHAR, &message, &status));
        if (message != MPI_MESSAGE_NULL)
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      else
        Check(MPI_Recv(
          receive.data(), received, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &status));
      Check(MPI_Wait(&request, MPI_STATUS_IGNORE));
      if (std::any_of(receive.begin(),
                      receive.begin() + received,
                      [expected](auto value) { return value != expected; }))
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    Check(MPI_Barrier(MPI_COMM_WORLD));
    Record(directory, rank, trial + 1);
    if (rank == 0)
      std::cout << "Completed trial " << trial + 1 << '/' << trials << std::endl;
  }
  Check(MPI_Finalize());
  return EXIT_SUCCESS;
}

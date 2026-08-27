// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace opensn
{

/** Optional, rank-local CBCD instrumentation enabled by `OPENSN_CBCD_PROFILE_DIR`. */
class CBCDProfiler
{
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  enum class CommunicationPhase : std::uint8_t
  {
    FLUSH_OUTGOING,
    PROBE_AND_RECEIVE,
    POLL_SENDS
  };

  /// Return a profiler when the output environment variable is set.
  static std::unique_ptr<CBCDProfiler> Create(std::size_t num_angle_sets);

  CBCDProfiler(std::filesystem::path output_directory, std::size_t num_angle_sets);
  ~CBCDProfiler();

  CBCDProfiler(const CBCDProfiler&) = delete;
  CBCDProfiler& operator=(const CBCDProfiler&) = delete;

  /// Reset rank-local counters immediately before a sweep.
  void BeginSweep(std::size_t num_workers);
  /// Store completed counters after workers, communication, and the final barrier finish.
  void FinishSweep();

  void RecordKernelLaunch(std::size_t angle_set_id, std::uint64_t num_cells);
  void RecordWorkerStart(std::size_t worker_id, TimePoint time);
  void RecordWorkerIdleStart(std::size_t worker_id, TimePoint time);
  void RecordWorkerIdleEnd(std::size_t worker_id, TimePoint time);
  void RecordWorkerYield(std::size_t worker_id);
  void RecordWorkerStop(std::size_t worker_id, TimePoint time);

  void RecordCommunicationIteration(bool work_done);
  void RecordCommunicationPhase(CommunicationPhase phase, std::uint64_t elapsed_ns);
  void RecordSend(std::uint64_t message_bytes, std::uint64_t face_records);
  void RecordReceive(std::uint64_t message_bytes, std::uint64_t face_records);
  void RecordCommunicatorDrain(std::uint64_t elapsed_ns);
  void RecordEndBarrier(std::uint64_t elapsed_ns);

  static std::uint64_t ElapsedNanoseconds(TimePoint start, TimePoint end);

private:
  static constexpr std::size_t NUM_HISTOGRAM_BINS = 64;

  struct SampleSummary
  {
    std::uint64_t count = 0;
    std::uint64_t sum = 0;
    std::uint64_t minimum = 0;
    std::uint64_t maximum = 0;

    void Add(std::uint64_t value);
  };

  struct Histogram
  {
    std::array<std::uint64_t, NUM_HISTOGRAM_BINS> counts{};
    void Add(std::uint64_t value);
  };

  struct AngleSetStatistics
  {
    SampleSummary cells_per_launch;
    Histogram cell_batch_histogram;
  };

  struct WorkerStatistics
  {
    TimePoint start_time{};
    TimePoint idle_start{};
    std::uint64_t wall_ns = 0;
    std::uint64_t idle_ns = 0;
    std::uint64_t yields = 0;
    bool idle = false;
  };

  struct CommunicationStatistics
  {
    std::uint64_t iterations = 0;
    std::uint64_t idle_iterations = 0;
    std::uint64_t flush_outgoing_ns = 0;
    std::uint64_t probe_and_receive_ns = 0;
    std::uint64_t poll_sends_ns = 0;
    SampleSummary send_bytes;
    SampleSummary receive_bytes;
    std::uint64_t sent_face_records = 0;
    std::uint64_t received_face_records = 0;
    Histogram send_byte_histogram;
    Histogram receive_byte_histogram;
  };

  struct SweepStatistics
  {
    std::vector<AngleSetStatistics> angle_sets;
    std::vector<WorkerStatistics> workers;
    CommunicationStatistics communication;
    std::uint64_t communicator_drain_ns = 0;
    std::uint64_t end_barrier_ns = 0;
  };

  static std::size_t HistogramBin(std::uint64_t value);
  void WriteResults() const;

  std::filesystem::path output_directory_;
  int rank_ = 0;
  std::size_t num_angle_sets_ = 0;
  std::unique_ptr<SweepStatistics> active_sweep_;
  std::vector<SweepStatistics> completed_sweeps_;
};

} // namespace opensn

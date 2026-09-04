// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/profiling/cbcd_profiler.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <utility>

namespace opensn
{

std::unique_ptr<CBCDProfiler>
CBCDProfiler::Create(const std::size_t num_angle_sets)
{
  const char* output = std::getenv("OPENSN_CBCD_PROFILE_DIR"); // NOLINT(concurrency-mt-unsafe)
  if (output == nullptr or output[0] == '\0')
    return nullptr;
  return std::make_unique<CBCDProfiler>(output, num_angle_sets);
}

CBCDProfiler::CBCDProfiler(std::filesystem::path output_directory, const std::size_t num_angle_sets)
  : output_directory_(std::move(output_directory)),
    rank_(opensn::mpi_comm.rank()),
    num_angle_sets_(num_angle_sets)
{
}

CBCDProfiler::~CBCDProfiler()
{
  try
  {
    WriteResults();
  }
  catch (const std::exception& error)
  {
    log.LogAllWarning() << "Failed to write CBCD profile for rank " << rank_ << ": "
                        << error.what();
  }
}

void
CBCDProfiler::SampleSummary::Add(const std::uint64_t value)
{
  if (count == 0)
    minimum = value;
  else
    minimum = std::min(minimum, value);
  maximum = std::max(maximum, value);
  ++count;
  sum += value;
}

std::size_t
CBCDProfiler::HistogramBin(std::uint64_t value)
{
  if (value == 0)
    return 0;
  std::size_t bin = 1;
  while (value > 1 and bin + 1 < NUM_HISTOGRAM_BINS)
  {
    value >>= 1;
    ++bin;
  }
  return bin;
}

void
CBCDProfiler::Histogram::Add(const std::uint64_t value)
{
  ++counts[HistogramBin(value)];
}

std::uint64_t
CBCDProfiler::ElapsedNanoseconds(const TimePoint start, const TimePoint end)
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void
CBCDProfiler::BeginSweep(const std::size_t num_workers)
{
  active_sweep_ = std::make_unique<SweepStatistics>();
  active_sweep_->angle_sets.resize(num_angle_sets_);
  active_sweep_->workers.resize(num_workers);
}

void
CBCDProfiler::FinishSweep()
{
  if (active_sweep_)
  {
    completed_sweeps_.push_back(std::move(*active_sweep_));
    active_sweep_.reset();
  }
}

void
CBCDProfiler::RecordKernelLaunch(const std::size_t angle_set_id, const std::uint64_t num_cells)
{
  auto& statistics = active_sweep_->angle_sets[angle_set_id];
  statistics.cells_per_launch.Add(num_cells);
  statistics.cell_batch_histogram.Add(num_cells);
}

void
CBCDProfiler::RecordDeviceDispatch(const std::size_t worker_id,
                                   const std::uint64_t num_batches,
                                   const std::uint64_t num_cells)
{
  auto& worker = active_sweep_->workers[worker_id];
  ++worker.device_dispatches;
  worker.dispatched_batches += num_batches;
  worker.dispatched_cells += num_cells;
}

void
CBCDProfiler::RecordWorkerStart(const std::size_t worker_id, const TimePoint time)
{
  active_sweep_->workers[worker_id].start_time = time;
}

void
CBCDProfiler::RecordWorkerIdleStart(const std::size_t worker_id, const TimePoint time)
{
  auto& worker = active_sweep_->workers[worker_id];
  if (not worker.idle)
  {
    worker.idle = true;
    worker.idle_start = time;
  }
}

void
CBCDProfiler::RecordWorkerIdleEnd(const std::size_t worker_id, const TimePoint time)
{
  auto& worker = active_sweep_->workers[worker_id];
  if (worker.idle)
  {
    worker.idle_ns += ElapsedNanoseconds(worker.idle_start, time);
    worker.idle = false;
  }
}

void
CBCDProfiler::RecordWorkerYield(const std::size_t worker_id)
{
  ++active_sweep_->workers[worker_id].yields;
}

void
CBCDProfiler::RecordWorkerStop(const std::size_t worker_id, const TimePoint time)
{
  RecordWorkerIdleEnd(worker_id, time);
  auto& worker = active_sweep_->workers[worker_id];
  worker.wall_ns = ElapsedNanoseconds(worker.start_time, time);
}

void
CBCDProfiler::RecordCommunicationIteration(const bool work_done)
{
  ++active_sweep_->communication.iterations;
  if (not work_done)
    ++active_sweep_->communication.idle_iterations;
}

void
CBCDProfiler::RecordCommunicationPhase(const CommunicationPhase phase,
                                       const std::uint64_t elapsed_ns)
{
  auto& communication = active_sweep_->communication;
  switch (phase)
  {
    case CommunicationPhase::FLUSH_OUTGOING:
      communication.flush_outgoing_ns += elapsed_ns;
      break;
    case CommunicationPhase::PROBE_AND_RECEIVE:
      communication.probe_and_receive_ns += elapsed_ns;
      break;
    case CommunicationPhase::POLL_SENDS:
      communication.poll_sends_ns += elapsed_ns;
      break;
  }
}

void
CBCDProfiler::RecordSend(const std::uint64_t message_bytes, const std::uint64_t face_records)
{
  auto& communication = active_sweep_->communication;
  communication.send_bytes.Add(message_bytes);
  communication.sent_face_records += face_records;
  communication.send_byte_histogram.Add(message_bytes);
}

void
CBCDProfiler::RecordReceive(const std::uint64_t message_bytes, const std::uint64_t face_records)
{
  auto& communication = active_sweep_->communication;
  communication.receive_bytes.Add(message_bytes);
  communication.received_face_records += face_records;
  communication.receive_byte_histogram.Add(message_bytes);
}

void
CBCDProfiler::RecordCommunicatorDrain(const std::uint64_t elapsed_ns)
{
  active_sweep_->communicator_drain_ns = elapsed_ns;
}

void
CBCDProfiler::RecordEndBarrier(const std::uint64_t elapsed_ns)
{
  active_sweep_->end_barrier_ns = elapsed_ns;
}

void
CBCDProfiler::WriteResults() const
{
  if (completed_sweeps_.empty())
    return;

  const auto rank_directory = output_directory_ / ("rank-" + std::to_string(rank_));
  std::filesystem::create_directories(rank_directory);

  std::ofstream sweeps;
  sweeps.exceptions(std::ios::badbit | std::ios::failbit);
  sweeps.open(rank_directory / "sweeps.csv");
  sweeps << "sweep,rank,workers,angle_sets,kernel_launches,kernel_cells,"
            "kernel_batch_min,kernel_batch_mean,kernel_batch_max,device_dispatches,"
            "device_dispatch_batches,device_dispatch_cells,device_batches_per_dispatch,"
            "worker_wall_ns,worker_idle_ns,"
            "worker_idle_fraction,worker_yields,comm_iterations,comm_idle_iterations,"
            "comm_idle_fraction,flush_outgoing_ns,probe_and_receive_ns,poll_sends_ns,"
            "send_messages,send_bytes,send_faces,send_bytes_min,send_bytes_mean,send_bytes_max,"
            "receive_messages,receive_bytes,receive_faces,receive_bytes_min,receive_bytes_mean,"
            "receive_bytes_max,communicator_drain_ns,end_barrier_ns\n";
  sweeps << std::setprecision(17);

  std::ofstream angle_sets;
  angle_sets.exceptions(std::ios::badbit | std::ios::failbit);
  angle_sets.open(rank_directory / "angle_sets.csv");
  angle_sets << "sweep,rank,angle_set,kernel_launches,kernel_cells,kernel_batch_min,"
                "kernel_batch_mean,kernel_batch_max\n";
  angle_sets << std::setprecision(17);

  std::ofstream histograms;
  histograms.exceptions(std::ios::badbit | std::ios::failbit);
  histograms.open(rank_directory / "histograms.csv");
  histograms << "sweep,rank,scope,index,metric,bin,lower_bound,upper_bound,count\n";

  for (std::size_t sweep_id = 0; sweep_id < completed_sweeps_.size(); ++sweep_id)
  {
    const auto& sweep = completed_sweeps_[sweep_id];
    SampleSummary kernel_batches;
    std::uint64_t worker_wall_ns = 0;
    std::uint64_t worker_idle_ns = 0;
    std::uint64_t worker_yields = 0;
    std::uint64_t device_dispatches = 0;
    std::uint64_t dispatched_batches = 0;
    std::uint64_t dispatched_cells = 0;
    for (std::size_t angle_set_id = 0; angle_set_id < sweep.angle_sets.size(); ++angle_set_id)
    {
      const auto& batches = sweep.angle_sets[angle_set_id].cells_per_launch;
      if (batches.count > 0)
      {
        kernel_batches.count += batches.count;
        kernel_batches.sum += batches.sum;
        kernel_batches.minimum = kernel_batches.count == batches.count
                                   ? batches.minimum
                                   : std::min(kernel_batches.minimum, batches.minimum);
        kernel_batches.maximum = std::max(kernel_batches.maximum, batches.maximum);
      }
      angle_sets << sweep_id << ',' << rank_ << ',' << angle_set_id << ',' << batches.count << ','
                 << batches.sum << ',' << batches.minimum << ','
                 << (batches.count == 0 ? 0.0 : static_cast<double>(batches.sum) / batches.count)
                 << ',' << batches.maximum << '\n';
    }
    for (const auto& worker : sweep.workers)
    {
      worker_wall_ns += worker.wall_ns;
      worker_idle_ns += worker.idle_ns;
      worker_yields += worker.yields;
      device_dispatches += worker.device_dispatches;
      dispatched_batches += worker.dispatched_batches;
      dispatched_cells += worker.dispatched_cells;
    }

    const auto& comm = sweep.communication;
    sweeps << sweep_id << ',' << rank_ << ',' << sweep.workers.size() << ','
           << sweep.angle_sets.size() << ',' << kernel_batches.count << ',' << kernel_batches.sum
           << ',' << kernel_batches.minimum << ','
           << (kernel_batches.count == 0
                 ? 0.0
                 : static_cast<double>(kernel_batches.sum) / kernel_batches.count)
           << ',' << kernel_batches.maximum << ',' << device_dispatches << ',' << dispatched_batches
           << ',' << dispatched_cells << ','
           << (device_dispatches == 0 ? 0.0
                                      : static_cast<double>(dispatched_batches) / device_dispatches)
           << ',' << worker_wall_ns << ',' << worker_idle_ns << ','
           << (worker_wall_ns == 0 ? 0.0 : static_cast<double>(worker_idle_ns) / worker_wall_ns)
           << ',' << worker_yields << ',' << comm.iterations << ',' << comm.idle_iterations << ','
           << (comm.iterations == 0 ? 0.0
                                    : static_cast<double>(comm.idle_iterations) / comm.iterations)
           << ',' << comm.flush_outgoing_ns << ',' << comm.probe_and_receive_ns << ','
           << comm.poll_sends_ns << ',' << comm.send_bytes.count << ',' << comm.send_bytes.sum
           << ',' << comm.sent_face_records << ',' << comm.send_bytes.minimum << ','
           << (comm.send_bytes.count == 0
                 ? 0.0
                 : static_cast<double>(comm.send_bytes.sum) / comm.send_bytes.count)
           << ',' << comm.send_bytes.maximum << ',' << comm.receive_bytes.count << ','
           << comm.receive_bytes.sum << ',' << comm.received_face_records << ','
           << comm.receive_bytes.minimum << ','
           << (comm.receive_bytes.count == 0
                 ? 0.0
                 : static_cast<double>(comm.receive_bytes.sum) / comm.receive_bytes.count)
           << ',' << comm.receive_bytes.maximum << ',' << sweep.communicator_drain_ns << ','
           << sweep.end_barrier_ns << '\n';

    const auto write_histogram =
      [&](
        const char* scope, const std::size_t index, const char* metric, const Histogram& histogram)
    {
      for (std::size_t bin = 0; bin < histogram.counts.size(); ++bin)
      {
        if (histogram.counts[bin] == 0)
          continue;
        const std::uint64_t lower = bin == 0 ? 0 : (std::uint64_t{1} << (bin - 1));
        const std::uint64_t upper =
          bin == 0 ? 0
                   : (bin + 1 == NUM_HISTOGRAM_BINS ? std::numeric_limits<std::uint64_t>::max()
                                                    : (std::uint64_t{1} << bin) - 1);
        histograms << sweep_id << ',' << rank_ << ',' << scope << ',' << index << ',' << metric
                   << ',' << bin << ',' << lower << ',' << upper << ',' << histogram.counts[bin]
                   << '\n';
      }
    };
    for (std::size_t angle_set_id = 0; angle_set_id < sweep.angle_sets.size(); ++angle_set_id)
      write_histogram("angle_set",
                      angle_set_id,
                      "kernel_batch_cells",
                      sweep.angle_sets[angle_set_id].cell_batch_histogram);
    write_histogram("rank", 0, "mpi_send_bytes", comm.send_byte_histogram);
    write_histogram("rank", 0, "mpi_receive_bytes", comm.receive_byte_histogram);
  }
}

} // namespace opensn

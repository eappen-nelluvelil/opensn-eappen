// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_aggregated_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mpi/mpi_comm_set.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace opensn
{

// ---------------------------------------------------------------------------
// PerDestQueue lock-free methods (Treiber stack)
// ---------------------------------------------------------------------------

void
CBCD_AggregatedCommunicator::PerDestQueue::Push(OutgoingEntry&& entry)
{
  auto* node = new OutgoingNode{std::move(entry), nullptr};
  auto* expected = head.load(std::memory_order_relaxed);
  do
  {
    node->next = expected;
  } while (not head.compare_exchange_weak(
    expected, node, std::memory_order_release, std::memory_order_relaxed));
}

std::vector<CBCD_AggregatedCommunicator::OutgoingEntry>
CBCD_AggregatedCommunicator::PerDestQueue::Drain()
{
  auto* chain = head.exchange(nullptr, std::memory_order_acquire);
  std::vector<OutgoingEntry> result;
  while (chain)
  {
    result.push_back(std::move(chain->entry));
    auto* next = chain->next;
    delete chain;
    chain = next;
  }
  return result;
}

bool
CBCD_AggregatedCommunicator::PerDestQueue::Empty() const
{
  return head.load(std::memory_order_acquire) == nullptr;
}

// ---------------------------------------------------------------------------
// PerAngleSetMailbox lock-free methods (Treiber stack)
// ---------------------------------------------------------------------------

void
CBCD_AggregatedCommunicator::PerAngleSetMailbox::Push(std::vector<IncomingEntry>&& batch)
{
  auto* node = new IncomingNode{std::move(batch), nullptr};
  auto* expected = head.load(std::memory_order_relaxed);
  do
  {
    node->next = expected;
  } while (not head.compare_exchange_weak(
    expected, node, std::memory_order_release, std::memory_order_relaxed));
}

std::vector<IncomingEntry>
CBCD_AggregatedCommunicator::PerAngleSetMailbox::Drain()
{
  auto* chain = head.exchange(nullptr, std::memory_order_acquire);
  std::vector<IncomingEntry> result;
  while (chain)
  {
    for (auto& e : chain->entries)
      result.push_back(std::move(e));
    auto* next = chain->next;
    delete chain;
    chain = next;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

CBCD_AggregatedCommunicator::CBCD_AggregatedCommunicator(const std::vector<AngleSet*>& angle_sets,
                                                         const MPICommunicatorSet& comm_set)
  : comm_set_(comm_set),
    num_angle_sets_(angle_sets.size()),
    incoming_mailboxes_(num_angle_sets_),
    aggregated_tag_(static_cast<int>(num_angle_sets_)),
    angle_set_done_(num_angle_sets_)
{
  for (const auto* as : angle_sets)
  {
    const auto& spds = as->GetSPDS();
    for (int dep : spds.GetLocationDependencies())
      all_location_dependencies_.insert(dep);
    for (int succ : spds.GetLocationSuccessors())
      all_location_successors_.insert(succ);
  }

  for (int succ : all_location_successors_)
    outgoing_queues_[succ]; // default-construct

  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);
}

CBCD_AggregatedCommunicator::~CBCD_AggregatedCommunicator()
{
  if (comm_thread_.joinable())
    Stop();

  // Drain any remaining lock-free nodes
  for (auto& [dest, queue] : outgoing_queues_)
    queue.Drain();
  for (auto& mailbox : incoming_mailboxes_)
    mailbox.Drain();
}

// ---------------------------------------------------------------------------
// Worker thread interface
// ---------------------------------------------------------------------------

void
CBCD_AggregatedCommunicator::EnqueueOutgoing(int dest_location,
                                             size_t angle_set_id,
                                             uint64_t cell_global_id,
                                             unsigned int face_id,
                                             std::vector<double>&& psi_data)
{
  auto it = outgoing_queues_.find(dest_location);
  assert(it != outgoing_queues_.end());
  it->second.Push({angle_set_id, cell_global_id, face_id, std::move(psi_data)});
}

std::vector<IncomingEntry>
CBCD_AggregatedCommunicator::DequeueIncoming(size_t angle_set_id)
{
  assert(angle_set_id < num_angle_sets_);
  return incoming_mailboxes_[angle_set_id].Drain();
}

void
CBCD_AggregatedCommunicator::SignalAngleSetComplete(size_t angle_set_id)
{
  assert(angle_set_id < num_angle_sets_);
  angle_set_done_[angle_set_id].store(true, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void
CBCD_AggregatedCommunicator::Start()
{
  stop_requested_.store(false, std::memory_order_relaxed);
  for (size_t i = 0; i < num_angle_sets_; ++i)
    angle_set_done_[i].store(false, std::memory_order_relaxed);
  pending_sends_.clear();
  comm_thread_ = std::thread(&CBCD_AggregatedCommunicator::CommThreadLoop, this);
}

void
CBCD_AggregatedCommunicator::Stop()
{
  stop_requested_.store(true, std::memory_order_release);
  if (comm_thread_.joinable())
    comm_thread_.join();
}

// ---------------------------------------------------------------------------
// Communication thread
// ---------------------------------------------------------------------------

void
CBCD_AggregatedCommunicator::CommThreadLoop()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::CommThreadLoop");

  while (true)
  {
    FlushOutgoing();
    ProbeAndReceive();
    PollPendingSends();

    if (stop_requested_.load(std::memory_order_acquire) and AllWorkComplete())
    {
      // Final flush to ensure no data is left in queues
      FlushOutgoing();

      // Wait for all pending sends to complete
      while (not pending_sends_.empty())
      {
        PollPendingSends();
        if (not pending_sends_.empty())
          std::this_thread::yield();
      }
      break;
    }

    std::this_thread::yield();
  }
}

void
CBCD_AggregatedCommunicator::FlushOutgoing()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::FlushOutgoing");

  for (auto& [dest, queue] : outgoing_queues_)
  {
    auto entries = queue.Drain();
    if (entries.empty())
      continue;

    // Group by angle set id
    std::map<size_t, std::vector<const OutgoingEntry*>> by_angle_set;
    for (const auto& entry : entries)
      by_angle_set[entry.angle_set_id].push_back(&entry);

    // Pre-compute buffer size
    size_t total_bytes = sizeof(size_t); // num_angle_sets_in_batch
    for (const auto& [as_id, ptrs] : by_angle_set)
    {
      total_bytes += sizeof(size_t) + sizeof(size_t); // as_id + num_entries
      for (const auto* e : ptrs)
        total_bytes += sizeof(uint64_t) + sizeof(unsigned int) + sizeof(size_t) +
                       e->psi_data.size() * sizeof(double);
    }

    // Serialize
    ByteArray buffer;
    buffer.Data().reserve(total_bytes);

    buffer.Write(by_angle_set.size());
    for (const auto& [as_id, ptrs] : by_angle_set)
    {
      buffer.Write(as_id);
      buffer.Write(ptrs.size());
      for (const auto* e : ptrs)
      {
        buffer.Write(e->cell_global_id);
        buffer.Write(e->face_id);
        const size_t data_size = e->psi_data.size();
        buffer.Write(data_size);
        auto& raw = buffer.Data();
        const size_t old_sz = raw.size();
        raw.resize(old_sz + data_size * sizeof(double));
        std::memcpy(&raw[old_sz], e->psi_data.data(), data_size * sizeof(double));
      }
    }

    const auto& comm = comm_set_.LocICommunicator(dest);
    auto dest_rank = comm_set_.MapIonJ(dest, dest);

    PendingSend ps;
    ps.data = std::move(buffer);
    ps.request = comm.isend(dest_rank, aggregated_tag_, ps.data.Data());
    pending_sends_.push_back(std::move(ps));
  }
}

void
CBCD_AggregatedCommunicator::ProbeAndReceive()
{
  CALI_CXX_MARK_SCOPE("CBCD_AggregatedCommunicator::ProbeAndReceive");

  const int my_rank = opensn::mpi_comm.rank();

  for (int locJ : all_location_dependencies_)
  {
    const auto& comm = comm_set_.LocICommunicator(my_rank);
    auto source_rank = comm_set_.MapIonJ(locJ, my_rank);

    mpi::Status status;
    while (comm.iprobe(source_rank, aggregated_tag_, status))
    {
      int num_bytes = status.count<std::byte>();
      std::vector<std::byte> recv_buffer(num_bytes);
      comm.recv(source_rank, status.tag(), recv_buffer.data(), num_bytes);

      ByteArray data_array(std::move(recv_buffer));
      auto num_as_in_batch = data_array.Read<size_t>();

      for (size_t as_batch = 0; as_batch < num_as_in_batch; ++as_batch)
      {
        auto as_id = data_array.Read<size_t>();
        auto num_entries = data_array.Read<size_t>();
        assert(as_id < num_angle_sets_);

        std::vector<IncomingEntry> batch;
        batch.reserve(num_entries);

        for (size_t e = 0; e < num_entries; ++e)
        {
          IncomingEntry entry;
          entry.cell_global_id = data_array.Read<uint64_t>();
          entry.face_id = data_array.Read<unsigned int>();
          auto data_size = data_array.Read<size_t>();
          entry.psi_data.resize(data_size);
          std::memcpy(entry.psi_data.data(),
                      &data_array.Data()[data_array.Offset()],
                      data_size * sizeof(double));
          data_array.Seek(data_array.Offset() + data_size * sizeof(double));
          batch.push_back(std::move(entry));
        }

        incoming_mailboxes_[as_id].Push(std::move(batch));
      }
    }
  }
}

void
CBCD_AggregatedCommunicator::PollPendingSends()
{
  pending_sends_.erase(std::remove_if(pending_sends_.begin(),
                                      pending_sends_.end(),
                                      [](PendingSend& ps) { return mpi::test(ps.request); }),
                       pending_sends_.end());
}

bool
CBCD_AggregatedCommunicator::AllWorkComplete() const
{
  for (size_t i = 0; i < num_angle_sets_; ++i)
    if (not angle_set_done_[i].load(std::memory_order_acquire))
      return false;

  for (const auto& [dest, queue] : outgoing_queues_)
    if (not queue.Empty())
      return false;

  return true;
}

} // namespace opensn

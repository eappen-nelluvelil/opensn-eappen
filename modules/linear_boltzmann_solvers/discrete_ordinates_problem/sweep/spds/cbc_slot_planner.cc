// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc_slot_planner.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace opensn::detail
{

namespace
{

constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

void
ValidatePlannerInputs(const std::vector<std::uint32_t>& successor_rank_offsets,
                      const std::vector<std::uint32_t>& successor_ranks,
                      const std::vector<std::uint32_t>& face_producer_ranks,
                      const std::vector<std::uint32_t>& face_consumer_ranks,
                      const std::vector<std::uint32_t>& producer_cell_face_offsets)
{
  if (successor_rank_offsets.empty())
    throw std::invalid_argument("CBC slot planner: successor offsets must contain one entry.");

  const auto num_tasks = successor_rank_offsets.size() - 1;
  if (num_tasks > std::numeric_limits<std::uint32_t>::max() - 2)
    throw std::length_error("CBC slot planner: task count exceeds the flow-network index range.");
  if (successor_rank_offsets.front() != 0 or
      successor_rank_offsets.back() != successor_ranks.size() or
      not std::is_sorted(successor_rank_offsets.begin(), successor_rank_offsets.end()))
    throw std::invalid_argument("CBC slot planner: malformed successor CSR offsets.");

  for (std::size_t task = 0; task < num_tasks; ++task)
    for (auto i = successor_rank_offsets[task]; i < successor_rank_offsets[task + 1]; ++i)
      if (successor_ranks[i] <= task or successor_ranks[i] >= num_tasks)
        throw std::invalid_argument(
          "CBC slot planner: successor ranks are not a strict topological adjacency.");

  if (face_producer_ranks.size() != face_consumer_ranks.size())
    throw std::invalid_argument("CBC slot planner: face endpoint tables have unequal lengths.");
  if (face_producer_ranks.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::length_error("CBC slot planner: face count exceeds the 32-bit index range.");
  if (producer_cell_face_offsets.size() != num_tasks + 1 or
      producer_cell_face_offsets.front() != 0 or
      producer_cell_face_offsets.back() != face_producer_ranks.size() or
      not std::is_sorted(producer_cell_face_offsets.begin(), producer_cell_face_offsets.end()))
    throw std::invalid_argument("CBC slot planner: malformed producer-face CSR offsets.");

  for (std::size_t producer = 0; producer < num_tasks; ++producer)
    for (auto face = producer_cell_face_offsets[producer];
         face < producer_cell_face_offsets[producer + 1];
         ++face)
      if (face_producer_ranks[face] != producer)
        throw std::invalid_argument("CBC slot planner: face producer does not match its CSR row.");

  for (std::size_t face = 0; face < face_producer_ranks.size(); ++face)
    if (face_producer_ranks[face] >= num_tasks or face_consumer_ranks[face] >= num_tasks or
        face_producer_ranks[face] >= face_consumer_ranks[face])
      throw std::invalid_argument(
        "CBC slot planner: directed face is not a strict task-DAG dependency.");
}

struct FlowArc
{
  std::uint32_t reverse = 0;
  std::uint32_t to = 0;
  std::uint32_t capacity = 0;
  std::uint32_t initial_capacity = 0;
};

struct ThreadLocalWorkspace
{
  std::vector<std::uint32_t> degrees;
  std::vector<std::uint32_t> offsets;
  std::vector<std::uint32_t> write_offsets;
  std::vector<FlowArc> arcs;
  std::vector<int> levels;
  std::vector<std::uint32_t> next_arcs;
  std::vector<std::uint32_t> queue;
  std::vector<std::uint32_t> path_arcs;
  std::vector<std::uint32_t> source_arcs_by_face;
  std::vector<std::uint32_t> sink_arcs_by_face;
  std::vector<std::uint32_t> flow_path_cursors;
  std::vector<std::uint32_t> face_mate_u;
  std::vector<std::uint32_t> face_mate_v;
};

class SparseReachabilityMatcher
{
public:
  SparseReachabilityMatcher(const std::vector<std::uint32_t>& successor_rank_offsets,
                            const std::vector<std::uint32_t>& successor_ranks,
                            const std::vector<std::uint32_t>& face_producer_ranks,
                            const std::vector<std::uint32_t>& face_consumer_ranks,
                            const std::vector<std::uint32_t>& producer_cell_face_offsets,
                            ThreadLocalWorkspace& workspace)
    : successor_rank_offsets_(successor_rank_offsets),
      successor_ranks_(successor_ranks),
      face_producer_ranks_(face_producer_ranks),
      face_consumer_ranks_(face_consumer_ranks),
      producer_cell_face_offsets_(producer_cell_face_offsets),
      workspace_(workspace),
      num_tasks_(static_cast<std::uint32_t>(successor_rank_offsets.size() - 1)),
      num_faces_(static_cast<std::uint32_t>(face_producer_ranks.size())),
      source_(num_tasks_),
      sink_(num_tasks_ + 1),
      num_nodes_(static_cast<std::size_t>(num_tasks_) + 2)
  {
    BuildNetwork();
  }

  std::size_t Solve(std::vector<std::uint32_t>& face_slot_ids)
  {
    const auto matching_size = MaxFlow();
    ExtractMatching(matching_size);
    return ExtractSlotAssignment(matching_size, face_slot_ids);
  }

private:
  void CountEdge(const std::uint32_t from, const std::uint32_t to)
  {
    if (workspace_.degrees[from] == std::numeric_limits<std::uint32_t>::max() or
        workspace_.degrees[to] == std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("CBC slot planner: flow-network degree overflow.");
    ++workspace_.degrees[from];
    ++workspace_.degrees[to];
  }

  std::uint32_t
  AddEdge(const std::uint32_t from, const std::uint32_t to, const std::uint32_t capacity)
  {
    const auto forward = workspace_.write_offsets[from]++;
    const auto reverse = workspace_.write_offsets[to]++;
    workspace_.arcs[forward] = {reverse, to, capacity, capacity};
    workspace_.arcs[reverse] = {forward, from, 0, 0};
    return forward;
  }

  void BuildNetwork()
  {
    workspace_.degrees.assign(num_nodes_, 0);
    for (std::uint32_t face = 0; face < num_faces_; ++face)
    {
      CountEdge(source_, face_consumer_ranks_[face]);
      CountEdge(face_producer_ranks_[face], sink_);
    }
    for (std::uint32_t task = 0; task < num_tasks_; ++task)
      for (auto i = successor_rank_offsets_[task]; i < successor_rank_offsets_[task + 1]; ++i)
        CountEdge(task, successor_ranks_[i]);

    workspace_.offsets.resize(num_nodes_ + 1);
    workspace_.offsets[0] = 0;
    for (std::size_t node = 0; node < num_nodes_; ++node)
    {
      if (workspace_.degrees[node] >
          std::numeric_limits<std::uint32_t>::max() - workspace_.offsets[node])
        throw std::length_error("CBC slot planner: flow-network arc-count overflow.");
      workspace_.offsets[node + 1] = workspace_.offsets[node] + workspace_.degrees[node];
    }

    workspace_.arcs.resize(workspace_.offsets.back());
    workspace_.write_offsets.assign(workspace_.offsets.begin(), workspace_.offsets.end() - 1);
    workspace_.source_arcs_by_face.resize(num_faces_);
    workspace_.sink_arcs_by_face.resize(num_faces_);

    for (std::uint32_t face = 0; face < num_faces_; ++face)
      workspace_.source_arcs_by_face[face] = AddEdge(source_, face_consumer_ranks_[face], 1);
    for (std::uint32_t face = 0; face < num_faces_; ++face)
      workspace_.sink_arcs_by_face[face] = AddEdge(face_producer_ranks_[face], sink_, 1);

    const auto internal_capacity = std::max(std::uint32_t{1}, num_faces_);
    for (std::uint32_t task = 0; task < num_tasks_; ++task)
      for (auto i = successor_rank_offsets_[task]; i < successor_rank_offsets_[task + 1]; ++i)
        AddEdge(task, successor_ranks_[i], internal_capacity);

    for (std::size_t node = 0; node < num_nodes_; ++node)
      if (workspace_.write_offsets[node] != workspace_.offsets[node + 1])
        throw std::logic_error("CBC slot planner: incomplete flow-network construction.");
  }

  bool BuildLevelGraph()
  {
    workspace_.levels.assign(num_nodes_, -1);
    if (workspace_.queue.size() < num_nodes_)
      workspace_.queue.resize(num_nodes_);

    std::size_t head = 0;
    std::size_t tail = 0;
    workspace_.levels[source_] = 0;
    workspace_.queue[tail++] = source_;
    while (head < tail)
    {
      const auto node = workspace_.queue[head++];
      if (node == sink_)
        continue;
      for (auto arc_index = workspace_.offsets[node]; arc_index < workspace_.offsets[node + 1];
           ++arc_index)
      {
        const auto& arc = workspace_.arcs[arc_index];
        if (arc.capacity == 0 or workspace_.levels[arc.to] != -1)
          continue;
        workspace_.levels[arc.to] = workspace_.levels[node] + 1;
        workspace_.queue[tail++] = arc.to;
      }
    }
    return workspace_.levels[sink_] != -1;
  }

  bool AugmentLevelGraph()
  {
    workspace_.path_arcs.clear();
    auto node = source_;
    while (true)
    {
      if (node == sink_)
      {
        for (const auto arc_index : workspace_.path_arcs)
        {
          auto& arc = workspace_.arcs[arc_index];
          --arc.capacity;
          ++workspace_.arcs[arc.reverse].capacity;
        }
        return true;
      }

      auto& next_arc = workspace_.next_arcs[node];
      const auto arc_end = workspace_.offsets[node + 1];
      while (next_arc < arc_end)
      {
        const auto& arc = workspace_.arcs[next_arc];
        if (arc.capacity != 0 and workspace_.levels[arc.to] == workspace_.levels[node] + 1)
          break;
        ++next_arc;
      }

      if (next_arc < arc_end)
      {
        workspace_.path_arcs.push_back(next_arc);
        node = workspace_.arcs[next_arc].to;
        continue;
      }

      workspace_.levels[node] = -1;
      if (workspace_.path_arcs.empty())
        return false;

      const auto failed_arc = workspace_.path_arcs.back();
      workspace_.path_arcs.pop_back();
      node = workspace_.arcs[workspace_.arcs[failed_arc].reverse].to;
      ++workspace_.next_arcs[node];
    }
  }

  std::size_t MaxFlow()
  {
    std::size_t flow = 0;
    while (BuildLevelGraph())
    {
      workspace_.next_arcs.assign(workspace_.offsets.begin(), workspace_.offsets.end() - 1);
      while (AugmentLevelGraph())
        ++flow;
    }
    return flow;
  }

  std::uint32_t FlowOnArc(const std::uint32_t arc_index) const noexcept
  {
    return workspace_.arcs[workspace_.arcs[arc_index].reverse].capacity;
  }

  void ConsumeFlowUnit(const std::uint32_t arc_index)
  {
    auto& arc = workspace_.arcs[arc_index];
    auto& reverse = workspace_.arcs[arc.reverse];
    if (arc.initial_capacity == 0 or reverse.capacity == 0)
      throw std::logic_error("CBC slot planner: invalid integral-flow decomposition.");
    ++arc.capacity;
    --reverse.capacity;
  }

  void ExtractMatching(const std::size_t matching_size)
  {
    workspace_.face_mate_u.assign(num_faces_, INVALID_INDEX);
    workspace_.face_mate_v.assign(num_faces_, INVALID_INDEX);
    workspace_.flow_path_cursors.assign(workspace_.offsets.begin(),
                                        workspace_.offsets.begin() + num_tasks_);

    std::size_t extracted = 0;
    for (std::uint32_t u_face = 0; u_face < num_faces_; ++u_face)
    {
      const auto source_arc = workspace_.source_arcs_by_face[u_face];
      if (FlowOnArc(source_arc) == 0)
        continue;

      ConsumeFlowUnit(source_arc);
      auto task = face_consumer_ranks_[u_face];
      while (true)
      {
        bool path_finished = false;
        for (auto v_face = producer_cell_face_offsets_[task];
             v_face < producer_cell_face_offsets_[task + 1];
             ++v_face)
        {
          const auto sink_arc = workspace_.sink_arcs_by_face[v_face];
          if (FlowOnArc(sink_arc) == 0)
            continue;

          ConsumeFlowUnit(sink_arc);
          if (workspace_.face_mate_v[v_face] != INVALID_INDEX)
            throw std::logic_error("CBC slot planner: flow decomposition repeated a right face.");
          workspace_.face_mate_u[u_face] = v_face;
          workspace_.face_mate_v[v_face] = u_face;
          ++extracted;
          path_finished = true;
          break;
        }
        if (path_finished)
          break;

        auto& cursor = workspace_.flow_path_cursors[task];
        const auto arc_end = workspace_.offsets[task + 1];
        while (cursor < arc_end)
        {
          const auto& arc = workspace_.arcs[cursor];
          if (arc.initial_capacity != 0 and arc.to < num_tasks_ and FlowOnArc(cursor) != 0)
            break;
          ++cursor;
        }
        if (cursor == arc_end)
          throw std::logic_error("CBC slot planner: integral flow terminated before a face.");

        const auto task_arc = cursor;
        task = workspace_.arcs[task_arc].to;
        ConsumeFlowUnit(task_arc);
      }
    }

    if (extracted != matching_size)
      throw std::logic_error("CBC slot planner: incomplete maximum-matching extraction.");
    for (const auto& arc : workspace_.arcs)
      if (arc.initial_capacity != 0 and
          (arc.capacity != arc.initial_capacity or workspace_.arcs[arc.reverse].capacity != 0))
        throw std::logic_error("CBC slot planner: residual flow remained after decomposition.");
  }

  std::size_t ExtractSlotAssignment(const std::size_t matching_size,
                                    std::vector<std::uint32_t>& face_slot_ids) const
  {
    face_slot_ids.assign(num_faces_, INVALID_INDEX);
    std::uint32_t next_slot = 0;
    for (std::uint32_t face = 0; face < num_faces_; ++face)
    {
      if (workspace_.face_mate_v[face] != INVALID_INDEX)
        continue;

      auto current = face;
      while (current != INVALID_INDEX)
      {
        if (face_slot_ids[current] != INVALID_INDEX)
          throw std::logic_error("CBC slot planner: matching extraction formed a cycle.");
        face_slot_ids[current] = next_slot;
        current = workspace_.face_mate_u[current];
      }
      ++next_slot;
    }

    const auto slot_count = static_cast<std::size_t>(num_faces_) - matching_size;
    if (next_slot != slot_count or
        std::ranges::any_of(face_slot_ids,
                            [slot_count](const std::uint32_t slot) { return slot >= slot_count; }))
      throw std::logic_error("CBC slot planner: minimum chain-cover verification failed.");
    return slot_count;
  }

  const std::vector<std::uint32_t>& successor_rank_offsets_;
  const std::vector<std::uint32_t>& successor_ranks_;
  const std::vector<std::uint32_t>& face_producer_ranks_;
  const std::vector<std::uint32_t>& face_consumer_ranks_;
  const std::vector<std::uint32_t>& producer_cell_face_offsets_;
  ThreadLocalWorkspace& workspace_;
  std::uint32_t num_tasks_ = 0;
  std::uint32_t num_faces_ = 0;
  std::uint32_t source_ = 0;
  std::uint32_t sink_ = 0;
  std::size_t num_nodes_ = 0;
};

} // namespace

std::size_t
ComputeLocalFaceSlotPlan(const std::vector<std::uint32_t>& successor_rank_offsets,
                         const std::vector<std::uint32_t>& successor_ranks,
                         const std::vector<std::uint32_t>& face_producer_ranks,
                         const std::vector<std::uint32_t>& face_consumer_ranks,
                         const std::vector<std::uint32_t>& producer_cell_face_offsets,
                         std::vector<std::uint32_t>& face_slot_ids)
{
  ValidatePlannerInputs(successor_rank_offsets,
                        successor_ranks,
                        face_producer_ranks,
                        face_consumer_ranks,
                        producer_cell_face_offsets);

  if (face_producer_ranks.empty())
  {
    face_slot_ids.clear();
    return 0;
  }

  static thread_local ThreadLocalWorkspace workspace;
  SparseReachabilityMatcher matcher(successor_rank_offsets,
                                    successor_ranks,
                                    face_producer_ranks,
                                    face_consumer_ranks,
                                    producer_cell_face_offsets,
                                    workspace);
  return matcher.Solve(face_slot_ids);
}

} // namespace opensn::detail

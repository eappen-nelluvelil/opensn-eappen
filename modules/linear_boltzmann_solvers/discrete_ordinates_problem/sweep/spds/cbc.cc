// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/logging/log.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <limits>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <bit>
#include <boost/graph/topological_sort.hpp>

namespace opensn
{

namespace
{

constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

class BitMatrix
{
public:
  BitMatrix() = default;

  void ResizeAndClear(std::size_t n)
  {
    n_ = n;
    // Pad to multiple of 8 words (512 bits) to guarantee perfect
    // boundary alignment for compiler auto-vectorization.
    words_per_row_ = (((n + 63) / 64) + 7) & ~std::size_t{7};
    const std::size_t required_words = n * words_per_row_;

    // Elide double-initialization overhead during growth: resize() value-
    // initializes new elements, so only the pre-existing prefix needs zeroing.
    if (data_.size() < required_words)
      data_.resize(required_words, 0ULL);
    else
      std::fill_n(data_.begin(), required_words, 0ULL);
  }

  std::size_t WordsPerRow() const { return words_per_row_; }
  std::uint64_t* Row(std::size_t i) { return data_.data() + i * words_per_row_; }
  const std::uint64_t* Row(std::size_t i) const { return data_.data() + i * words_per_row_; }

  void SetBit(std::size_t i, std::size_t j) { Row(i)[j / 64] |= (1ULL << (j % 64)); }
  void ClearBit(std::size_t i, std::size_t j) { Row(i)[j / 64] &= ~(1ULL << (j % 64)); }
  bool TestBit(std::size_t i, std::size_t j) const
  {
    return (Row(i)[j / 64] & (1ULL << (j % 64))) != 0ULL;
  }

  // __restrict guarantees no pointer aliasing, enabling the compiler to
  // aggressively unroll and vectorize the loops natively.
  void
  CopyRow(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
  {
    const std::size_t start_word = start_pos / 64;
    std::uint64_t* __restrict d = Row(dst) + start_word;
    const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_copy = words_per_row_ - start_word;

    std::memcpy(d, s, words_to_copy * sizeof(std::uint64_t));
  }

  void
  OrRows(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
  {
    const std::size_t start_word = start_pos / 64;
    std::uint64_t* __restrict d = Row(dst) + start_word;
    const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_process = words_per_row_ - start_word;

    for (std::size_t w = 0; w < words_to_process; ++w)
      d[w] |= s[w];
  }

  void AndRows(std::size_t dst,
               const BitMatrix& src_mat,
               std::size_t src_row,
               std::size_t start_pos = 0)
  {
    const std::size_t start_word = start_pos / 64;
    std::uint64_t* __restrict d = Row(dst) + start_word;
    const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_process = words_per_row_ - start_word;

    for (std::size_t w = 0; w < words_to_process; ++w)
      d[w] &= s[w];
  }

  std::size_t FindFirstSet(std::size_t row, std::size_t start_pos = 0) const
  {
    const std::uint64_t* __restrict r = Row(row);
    std::size_t w = start_pos / 64;

    if (w >= words_per_row_)
      return n_;

    std::uint64_t masked = r[w] & (~0ULL << (start_pos % 64));

    // Padding bits are strictly guaranteed to be 0, so a nonzero word always
    // contains a valid bit index. std::countr_zero maps to hardware TZCNT.
    if (masked)
      return w * 64 + static_cast<std::size_t>(std::countr_zero(masked));

    for (++w; w < words_per_row_; ++w)
    {
      if (r[w])
        return w * 64 + static_cast<std::size_t>(std::countr_zero(r[w]));
    }
    return n_;
  }

  std::size_t FindNextSet(std::size_t row, std::size_t pos) const
  {
    return FindFirstSet(row, pos + 1);
  }

private:
  std::size_t n_ = 0;
  std::size_t words_per_row_ = 0;
  std::vector<std::uint64_t> data_;
};

struct ThreadLocalWorkspace
{
  BitMatrix reachability;
  BitMatrix reuse_targets;
  std::vector<std::uint32_t> mate_u;
  std::vector<std::uint32_t> mate_v;
  std::vector<int> dist;
  std::vector<std::uint32_t> queue;
  std::vector<std::uint32_t> topo_rank;
  std::vector<std::uint32_t> last_rank_for_slot;

  void Prepare(std::size_t n)
  {
    reachability.ResizeAndClear(n);
    reuse_targets.ResizeAndClear(n);
    mate_u.assign(n, INVALID_INDEX);
    mate_v.assign(n, INVALID_INDEX);
    dist.assign(n, -1);

    // Conditional resize: these buffers are overwritten before use, so clearing
    // is unnecessary — only ensure they are large enough.
    if (queue.size() < n)
      queue.resize(n);
    if (topo_rank.size() < n)
      topo_rank.resize(n);
    if (last_rank_for_slot.size() < n)
      last_rank_for_slot.resize(n);
  }
};

class DenseHopcroftKarp
{
public:
  DenseHopcroftKarp(std::uint32_t num_tasks,
                    const std::vector<Task>& task_list,
                    const std::vector<std::uint32_t>& topo_order,
                    std::vector<std::uint32_t>& task_slot_ids,
                    ThreadLocalWorkspace& ws)
    : num_tasks_(num_tasks),
      task_list_(task_list),
      topo_order_(topo_order),
      task_slot_ids_(task_slot_ids),
      ws_(ws)
  {
    ws_.Prepare(num_tasks_);

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
      ws_.topo_rank[topo_order_[i]] = i;
  }

  std::size_t Solve()
  {
    BuildTransitiveClosure();
    BuildReuseTargets();

    // Dense Hopcroft-Karp bipartite matching
    std::size_t matching_size = GreedyInit();
    while (BFS())
    {
      for (std::uint32_t i = 0; i < num_tasks_; ++i)
      {
        if (ws_.mate_u[i] == INVALID_INDEX && DFS(i))
          ++matching_size;
      }
    }

    ExtractSlotAssignment();

    const std::size_t optimal_slot_count = static_cast<std::size_t>(num_tasks_) - matching_size;

    if (not VerifySlotAssignment(optimal_slot_count))
    {
      opensn::log.LogAllWarning()
        << "CBC_SPDS::ComputeMaxNumLocalPsiSlots: slot-assignment verifier rejected the planner "
        << "output; falling back to the identity assignment (one slot per local cell).";
      std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), std::uint32_t{0});
      return static_cast<std::size_t>(num_tasks_);
    }

    return optimal_slot_count;
  }

private:
  // Build the transitive closure (reachability matrix) bottom-up in reverse
  // topological order. Each task's reachability row is the union of its
  // successors' rows plus itself.
  void BuildTransitiveClosure()
  {
    for (std::uint32_t i = num_tasks_; i-- > 0;)
    {
      const std::uint32_t u = topo_order_[i];
      const auto& successors = task_list_[u].successors;

      if (successors.empty())
      {
        ws_.reachability.SetBit(i, i);
      }
      else
      {
        // Start from rank i to blanket all successors, guaranteeing
        // upper-triangular safety regardless of local successor ordering.
        ws_.reachability.CopyRow(i, ws_.reachability, ws_.topo_rank[successors[0]], i);
        ws_.reachability.SetBit(i, i);

        for (std::size_t j = 1; j < successors.size(); ++j)
          ws_.reachability.OrRows(i, ws_.reachability, ws_.topo_rank[successors[j]], i);
      }
    }
  }

  // Build the reuse-targets bitmatrix: for each task u, the reuse targets are
  // the intersection of the reachability sets of all of u's successors, minus
  // the successors themselves.
  void BuildReuseTargets()
  {
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      const std::uint32_t u = topo_order_[i];
      const auto& successors = task_list_[u].successors;
      if (successors.empty())
        continue;

      // Reuse targets can only appear after the latest successor in topological
      // order, so start all row operations at max_succ_rank.
      std::uint32_t max_succ_rank = ws_.topo_rank[successors[0]];
      for (std::size_t j = 1; j < successors.size(); ++j)
        max_succ_rank = std::max(max_succ_rank, ws_.topo_rank[successors[j]]);

      ws_.reuse_targets.CopyRow(
        i, ws_.reachability, ws_.topo_rank[successors[0]], max_succ_rank);

      for (std::size_t j = 1; j < successors.size(); ++j)
        ws_.reuse_targets.AndRows(
          i, ws_.reachability, ws_.topo_rank[successors[j]], max_succ_rank);

      for (const auto succ : successors)
        ws_.reuse_targets.ClearBit(i, ws_.topo_rank[succ]);
    }
  }

  // Walk the matching chains to produce a static slot-id per task.
  // Each chain head (unmatched on the V side) starts a new slot.
  void ExtractSlotAssignment()
  {
    task_slot_ids_.assign(num_tasks_, INVALID_INDEX);
    std::uint32_t next_slot_id = 0;

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_v[i] == INVALID_INDEX)
      {
        std::uint32_t current = i;
        while (current != INVALID_INDEX)
        {
          task_slot_ids_[topo_order_[current]] = next_slot_id;
          current = ws_.mate_u[current];
        }
        ++next_slot_id;
      }
    }
  }

  bool VerifySlotAssignment(const std::size_t slot_count) const
  {
    // INVALID_INDEX > slot_count, so one bounds check covers both unassigned
    // and out-of-range slot ids.
    for (std::uint32_t task = 0; task < num_tasks_; ++task)
    {
      if (task_slot_ids_[task] >= slot_count)
        return false;
    }

    std::fill_n(ws_.last_rank_for_slot.begin(), slot_count, INVALID_INDEX);
    for (std::uint32_t rank = 0; rank < num_tasks_; ++rank)
    {
      const auto sid = task_slot_ids_[topo_order_[rank]];
      const auto prev_rank = ws_.last_rank_for_slot[sid];
      if (prev_rank != INVALID_INDEX and not ReuseRelationHolds(prev_rank, rank))
        return false;
      ws_.last_rank_for_slot[sid] = rank;
    }
    return true;
  }

  bool ReuseRelationHolds(const std::uint32_t u_rank, const std::uint32_t v_rank) const
  {
    const auto u = topo_order_[u_rank];
    const auto& u_successors = task_list_[u].successors;

    if (u_successors.empty())
      return false;

    for (const auto succ : u_successors)
    {
      const auto succ_rank = ws_.topo_rank[succ];
      if (succ_rank == v_rank or not ws_.reachability.TestBit(succ_rank, v_rank))
        return false;
    }
    return true;
  }

  std::size_t GreedyInit()
  {
    std::size_t count = 0;
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_u[i] != INVALID_INDEX)
        continue;

      for (std::size_t v = ws_.reuse_targets.FindFirstSet(i, i + 1); v < num_tasks_;
           v = ws_.reuse_targets.FindNextSet(i, v))
      {
        if (ws_.mate_v[v] == INVALID_INDEX)
        {
          ws_.mate_u[i] = static_cast<std::uint32_t>(v);
          ws_.mate_v[v] = i;
          ++count;
          break;
        }
      }
    }
    return count;
  }

  bool BFS()
  {
    std::fill_n(ws_.dist.begin(), num_tasks_, -1);
    std::size_t head = 0, tail = 0;

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_u[i] == INVALID_INDEX)
      {
        ws_.dist[i] = 0;
        ws_.queue[tail++] = i;
      }
    }

    dist_null_ = std::numeric_limits<int>::max();

    while (head < tail)
    {
      const std::uint32_t u = ws_.queue[head++];

      if (ws_.dist[u] < dist_null_)
      {
        for (std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1); v < num_tasks_;
             v = ws_.reuse_targets.FindNextSet(u, v))
        {
          const std::uint32_t mate_of_v = ws_.mate_v[v];
          if (mate_of_v == INVALID_INDEX)
          {
            if (dist_null_ == std::numeric_limits<int>::max())
              dist_null_ = ws_.dist[u] + 1;
          }
          else if (ws_.dist[mate_of_v] == -1)
          {
            ws_.dist[mate_of_v] = ws_.dist[u] + 1;
            ws_.queue[tail++] = mate_of_v;
          }
        }
      }
    }
    return dist_null_ != std::numeric_limits<int>::max();
  }

  bool DFS(std::uint32_t u)
  {
    for (std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1); v < num_tasks_;
         v = ws_.reuse_targets.FindNextSet(u, v))
    {
      const std::uint32_t mate_of_v = ws_.mate_v[v];
      if (mate_of_v == INVALID_INDEX)
      {
        if (dist_null_ == ws_.dist[u] + 1)
        {
          ws_.mate_v[v] = u;
          ws_.mate_u[u] = static_cast<std::uint32_t>(v);
          ws_.dist[u] = -1;
          return true;
        }
      }
      else if (ws_.dist[mate_of_v] == ws_.dist[u] + 1)
      {
        if (DFS(mate_of_v))
        {
          ws_.mate_v[v] = u;
          ws_.mate_u[u] = static_cast<std::uint32_t>(v);
          ws_.dist[u] = -1;
          return true;
        }
      }
    }
    ws_.dist[u] = -1;
    return false;
  }

  std::uint32_t num_tasks_;
  const std::vector<Task>& task_list_;
  const std::vector<std::uint32_t>& topo_order_;
  std::vector<std::uint32_t>& task_slot_ids_;

  ThreadLocalWorkspace& ws_;
  int dist_null_ = 0;
};

} // namespace

void
CBC_SPDS::BuildTaskGraph()
{
  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

  const auto num_loc_cells = grid_->local_cells.size();
  task_list_.assign(num_loc_cells, Task{});

  for (const auto& cell : grid_->local_cells)
  {
    unsigned int num_dependencies = 0;
    std::vector<std::uint32_t> predecessors;
    std::vector<std::uint32_t> successors;

    predecessors.reserve(cell.faces.size());
    successors.reserve(cell.faces.size());

    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = cell_face_orientations_[cell.local_id][f];

      if (orientation == INCOMING and face.has_neighbor)
      {
        ++num_dependencies;
        if (face.IsNeighborLocal(grid_.get()))
          predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
      }
      else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
        successors.push_back(grid_->cells[face.neighbor_id].local_id);
    }

    task_list_[cell.local_id] = Task{
      0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
  }
}

CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const std::shared_ptr<MeshContinuum>& grid,
                   const bool allow_cycles)
  : SPDS(omega, grid)
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

  const auto num_loc_cells = grid->local_cells.size();

  std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
  std::set<int> location_successors;
  std::set<int> location_dependencies;

  PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

  location_successors_.assign(location_successors.begin(), location_successors.end());
  location_dependencies_.assign(location_dependencies.begin(), location_dependencies.end());

  Graph local_dg(num_loc_cells);
  for (std::size_t c = 0; c < num_loc_cells; ++c)
    for (const auto& successor : cell_successors[c])
      boost::add_edge(c, successor.first, successor.second, local_dg);

  if (allow_cycles)
  {
    const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
    for (const auto& [u, v] : edges_to_remove)
      local_sweep_fas_.emplace_back(u, v);
  }

  spls_.clear();
  boost::topological_sort(local_dg, std::back_inserter(spls_));
  std::reverse(spls_.begin(), spls_.end());
  if (spls_.empty())
  {
    throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
                           "Cycles need to be allowed by the calling application.");
  }

  topo_order_.assign(spls_.begin(), spls_.end());

  std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
  CommunicateLocationDependencies(location_dependencies_, global_dependencies);
  BuildTaskGraph();

  // Safe identity assignment: one slot per cell. ComputeMaxNumLocalPsiSlots()
  // refines this to the optimal count if called subsequently.
  max_num_local_psi_slots_ = num_loc_cells;
  task_slot_ids_.resize(num_loc_cells);
  std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), std::uint32_t{0});
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const noexcept
{
  return task_list_;
}

void
CBC_SPDS::ComputeMaxNumLocalPsiSlots()
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMaxNumLocalPsiSlots");

  const auto num_tasks = static_cast<std::uint32_t>(task_list_.size());
  if (num_tasks == 0)
  {
    max_num_local_psi_slots_ = 0;
    return;
  }

  thread_local ThreadLocalWorkspace workspace;

  DenseHopcroftKarp allocator(num_tasks, task_list_, topo_order_, task_slot_ids_, workspace);
  max_num_local_psi_slots_ = allocator.Solve();
}

} // namespace opensn

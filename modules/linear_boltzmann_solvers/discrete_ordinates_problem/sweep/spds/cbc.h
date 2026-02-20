// // SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// // SPDX-License-Identifier: MIT

// #pragma once

// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"

// namespace opensn
// {

// class CBC_SPDS : public SPDS
// {
// public:
//   /**
//    * Constructs a cell-by-cell sweep-plane data strcture (SPDS) with the given direction and grid.
//    *
//    * \param omega The angular direction vector.
//    * \param grid Reference to the grid.
//    * \param allow_cycles Whether cycles are allowed in the local sweep dependency graph.
//    */
//   CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

//   /// Returns the cell-by-cell task list.
//   const std::vector<Task>& GetTaskList() const;

// protected:
//   /// Cell-by-cell task list.
//   std::vector<Task> task_list_;
// };

// } // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <boost/graph/topological_sort.hpp>
#include <boost/dynamic_bitset.hpp>
#include <queue>
#include <limits>

namespace opensn
{

class ImplicitHopcroftKarp
{
public:
  ImplicitHopcroftKarp(size_t num_tasks,
                       const std::vector<Task>& task_list,
                       const std::vector<boost::dynamic_bitset<>>& reachability_matrix)
    : num_tasks_(num_tasks),
      task_list_(task_list),
      reachability_matrix_(reachability_matrix),
      mate_u_(num_tasks_, -1),
      mate_v_(num_tasks_, -1),
      dist_(num_tasks_),
      scratch_bitset_(num_tasks_)
  {
  }

  size_t Solve()
  {
    size_t matching_size = 0;
    // Hopcroft-Karp uses BFS to find all shortest augmenting paths, then DFS to find and augment along those paths.
    while (BFS())
    {
      for (size_t u = 0; u < num_tasks_; ++u)
      {
        if (mate_u_[u] == -1)
        {
          if (DFS(u))
            ++matching_size;
        }
      }
    }
    return matching_size;
  }

  bool VerifyMatching()
  {
    return (not BFS());
  }

private:
  // Recomputes the adjacency list (valid reuse targets) for 'u' on the fly.
  // Avoid storing an O(N^2) adjacency matrix.
  const boost::dynamic_bitset<>& GetNeighbors(size_t u)
  {
    const auto& task = task_list_[u];

    // If no successors, no reuse edges.
    // Return empty set (all zeros).
    if (task.successors.empty())
    {
      scratch_bitset_.reset();
      return scratch_bitset_;
    }

    // Start with first successor
    scratch_bitset_ = reachability_matrix_[task.successors[0]];

    // Intersect with remaining successors
    for (size_t i = 1; i < task.successors.size(); ++i)
    {
      scratch_bitset_ &= reachability_matrix_[task.successors[i]];
    }

    // Strictness: remove immediate neighbors (successors) from reuse candidates
    for (const auto& succ : task.successors)
    {
      scratch_bitset_.reset(succ);
    }

    return scratch_bitset_;
  }

  // Standard BFS for Hopcroft-Karp, using GetNeighbors to find valid reuse edges.
  bool BFS()
  {
    std::fill(dist_.begin(), dist_.end(), -1);
    std::queue<size_t> Q;

    for (size_t u = 0; u < num_tasks_; ++u)
    {
      if (mate_u_[u] == -1)
      {
        dist_[u] = 0;
        Q.push(u);
      }
    }

    dist_null_ = std::numeric_limits<int>::max();

    while (not Q.empty())
    {
      size_t u = Q.front();
      Q.pop();

      if (dist_[u] < dist_null_)
      {
        // Compute neighbors implicitly
        const auto& neighbors = GetNeighbors(u);

        // Iterate over set bits
        size_t v = neighbors.find_first();
        while (v != boost::dynamic_bitset<>::npos)
        {
          int mate_of_v = mate_v_[v];
          if (mate_of_v == -1)
          {
            if (dist_null_ == std::numeric_limits<int>::max())
              dist_null_ = dist_[u] + 1;
          }
          else
          {
            if (dist_[mate_of_v] == -1)
            {
              dist_[mate_of_v] = dist_[u] + 1;
              Q.push(mate_of_v);
            }
          }
          v = neighbors.find_next(v);
        }
      }
    }
    return dist_null_ != std::numeric_limits<int>::max();
  }

  // DFS to find augmenting paths, using GetNeighbors to find valid reuse edges.
  bool DFS(size_t u)
  {
    // Compute neighbors implicitly
    boost::dynamic_bitset<> neighbors = GetNeighbors(u);
    size_t v = neighbors.find_first();
    while (v != boost::dynamic_bitset<>::npos)
    {
      int mate_of_v = mate_v_[v];
      if (mate_of_v == -1)
      {
        if (dist_null_ == dist_[u] + 1)
        {
          mate_v_[v] = static_cast<int>(u);
          mate_u_[u] = static_cast<int>(v);
          // Mark this vertex as part of an augmenting path to avoid redundant searches in future DFS calls.
          dist_[u] = -1; // Mark as visited
          return true;
        }
      }
      else
      {
        if (dist_[mate_of_v] == dist_[u] + 1)
        {
          if (DFS(mate_of_v))
          {
            mate_v_[v] = static_cast<int>(u);
            mate_u_[u] = static_cast<int>(v);
            // Mark this vertex as part of an augmenting path to avoid redundant searches in future DFS calls.
            dist_[u] = -1; // Mark as visited
            return true;
          }
        }
      }
      v = neighbors.find_next(v);
    }
    // Mark this vertex as fully explored (no augmenting path through it) to avoid redundant searches in future DFS calls.
    dist_[u] = -1; // Mark as visited
    return false;
  }

  size_t num_tasks_;
  const std::vector<Task>& task_list_;
  const std::vector<boost::dynamic_bitset<>>& reachability_matrix_;

  std::vector<int> mate_u_;
  std::vector<int> mate_v_;
  std::vector<int> dist_;
  int dist_null_ = 0;

  boost::dynamic_bitset<> scratch_bitset_;
};

/**
 * Hopcroft-Karp maximum bipartite matching on the implicit reuse graph.
 *
 * The reuse graph encodes slot-reuse feasibility: edge (u, v) exists iff
 * task v can reuse the pool-allocator slot freed by task u. The reuse
 * condition (from the CBC slot lifecycle) requires that ALL local successors
 * of u have completed before v begins, which means v must be a descendant
 * of every immediate local successor of u.
 *
 * Instead of precomputing and storing the O(n^2) transitive-closure
 * reachability matrix, this class computes each task's reuse-neighbor set
 * ON THE FLY via forward BFS from that task's successors.
 *
 * Memory: O(n) — two temporary bitsets of n words each, plus O(n) arrays
 *         for the matching state. No reachability matrix.
 *
 * Time per GetNeighbors call: O(n + m) for BFS over the local DAG.
 * Total time: O(HK_rounds * n * (n + m)), where HK_rounds = O(sqrt(n)).
 * In practice much faster due to early termination and the fact that most
 * tasks are in simple chains and match greedily.
 */
// class ImplicitHopcroftKarp
// {
// public:
//   ImplicitHopcroftKarp(size_t num_tasks,
//                        const std::vector<Task>& task_list)
//     : num_tasks_(num_tasks),
//       task_list_(task_list),
//       mate_u_(num_tasks_, -1),
//       mate_v_(num_tasks_, -1),
//       dist_(num_tasks_),
//       // Allocate word-based bitsets: ceil(num_tasks / 64) words each
//       words_per_bitset_((num_tasks_ + 63) / 64),
//       neighbors_buf_(words_per_bitset_, 0),
//       bfs_buf_(words_per_bitset_, 0)
//   {
//   }

//   size_t Solve()
//   {
//     size_t matching_size = 0;
//     while (BFS())
//     {
//       for (size_t u = 0; u < num_tasks_; ++u)
//       {
//         if (mate_u_[u] == -1)
//         {
//           if (DFS(u))
//             ++matching_size;
//         }
//       }
//     }
//     return matching_size;
//   }

//   bool VerifyMatching()
//   {
//     return (not BFS());
//   }

// private:
//   // ---------------------------------------------------------------
//   // On-the-fly neighbor computation via forward BFS from children.
//   //
//   // For task u with children {s1, ..., sk}:
//   //   Neighbors(u) = Desc(s1) ∩ Desc(s2) ∩ ... ∩ Desc(sk) \ {s1,...,sk}
//   //
//   // Where Desc(s) = set of all tasks reachable from s via directed edges.
//   //
//   // Uses the provided buffer to store the result. Returns a reference to
//   // the buffer. The buffer is invalidated by the next call.
//   // ---------------------------------------------------------------
//   void ComputeNeighbors(size_t u, std::vector<uint64_t>& result)
//   {
//     const auto& task = task_list_[u];

//     // Clear result
//     std::fill(result.begin(), result.end(), 0ULL);

//     if (task.successors.empty())
//       return;

//     // BFS forward from first successor, mark all reachable
//     ForwardBFS(task.successors[0], result);

//     // Intersect with BFS from remaining successors
//     for (size_t i = 1; i < task.successors.size(); ++i)
//     {
//       // BFS into a temporary buffer
//       std::fill(bfs_buf_.begin(), bfs_buf_.end(), 0ULL);
//       ForwardBFS(task.successors[i], bfs_buf_);

//       // Intersect: result &= bfs_buf_
//       for (size_t w = 0; w < words_per_bitset_; ++w)
//         result[w] &= bfs_buf_[w];
//     }

//     // Remove immediate successors from the result (strictness condition)
//     for (const auto& succ : task.successors)
//       result[succ / 64] &= ~(1ULL << (succ % 64));
//   }

//   // Forward BFS from 'start', setting bits for all reachable nodes in 'bits'.
//   void ForwardBFS(uint32_t start, std::vector<uint64_t>& bits)
//   {
//     // Use a simple queue-based BFS
//     std::queue<uint32_t> queue;
//     bits[start / 64] |= (1ULL << (start % 64));
//     queue.push(start);

//     while (!queue.empty())
//     {
//       uint32_t v = queue.front();
//       queue.pop();
//       for (const auto& s : task_list_[v].successors)
//       {
//         if (!(bits[s / 64] & (1ULL << (s % 64))))
//         {
//           bits[s / 64] |= (1ULL << (s % 64));
//           queue.push(s);
//         }
//       }
//     }
//   }

//   // Helper: test if bit v is set in bitset
//   static bool TestBit(const std::vector<uint64_t>& bits, size_t v)
//   {
//     return bits[v / 64] & (1ULL << (v % 64));
//   }

//   // Helper: find first set bit >= start_pos. Returns num_tasks_ if none found.
//   size_t FindFirst(const std::vector<uint64_t>& bits, size_t start_pos = 0) const
//   {
//     size_t w = start_pos / 64;
//     if (w >= words_per_bitset_)
//       return num_tasks_;

//     // Mask off bits below start_pos in the first word
//     uint64_t masked = bits[w] & (~0ULL << (start_pos % 64));
//     if (masked)
//       return w * 64 + __builtin_ctzll(masked);

//     for (++w; w < words_per_bitset_; ++w)
//     {
//       if (bits[w])
//         return w * 64 + __builtin_ctzll(bits[w]);
//     }
//     return num_tasks_;
//   }

//   // Helper: find next set bit after pos.
//   size_t FindNext(const std::vector<uint64_t>& bits, size_t pos) const
//   {
//     return FindFirst(bits, pos + 1);
//   }

//   // BFS phase of Hopcroft-Karp
//   bool BFS()
//   {
//     std::fill(dist_.begin(), dist_.end(), -1);
//     std::queue<size_t> Q;

//     for (size_t u = 0; u < num_tasks_; ++u)
//     {
//       if (mate_u_[u] == -1)
//       {
//         dist_[u] = 0;
//         Q.push(u);
//       }
//     }

//     dist_null_ = std::numeric_limits<int>::max();

//     while (!Q.empty())
//     {
//       size_t u = Q.front();
//       Q.pop();

//       if (dist_[u] < dist_null_)
//       {
//         // Compute neighbors on the fly
//         ComputeNeighbors(u, neighbors_buf_);

//         // Iterate over set bits in neighbors_buf_
//         size_t v = FindFirst(neighbors_buf_);
//         while (v < num_tasks_)
//         {
//           int mate_of_v = mate_v_[v];
//           if (mate_of_v == -1)
//           {
//             if (dist_null_ == std::numeric_limits<int>::max())
//               dist_null_ = dist_[u] + 1;
//           }
//           else
//           {
//             if (dist_[mate_of_v] == -1)
//             {
//               dist_[mate_of_v] = dist_[u] + 1;
//               Q.push(mate_of_v);
//             }
//           }
//           v = FindNext(neighbors_buf_, v);
//         }
//       }
//     }
//     return dist_null_ != std::numeric_limits<int>::max();
//   }

//   // DFS phase of Hopcroft-Karp
//   bool DFS(size_t u)
//   {
//     // Compute neighbors on the fly (into a LOCAL buffer to handle recursion)
//     // Note: we allocate a local vector here. For the DFS recursion depth
//     // (bounded by the DAG depth), this is acceptable.
//     std::vector<uint64_t> local_neighbors(words_per_bitset_, 0);
//     ComputeNeighbors(u, local_neighbors);

//     size_t v = FindFirst(local_neighbors);
//     while (v < num_tasks_)
//     {
//       int mate_of_v = mate_v_[v];
//       if (mate_of_v == -1)
//       {
//         if (dist_null_ == dist_[u] + 1)
//         {
//           mate_v_[v] = static_cast<int>(u);
//           mate_u_[u] = static_cast<int>(v);
//           dist_[u] = -1;
//           return true;
//         }
//       }
//       else
//       {
//         if (dist_[mate_of_v] == dist_[u] + 1)
//         {
//           if (DFS(mate_of_v))
//           {
//             mate_v_[v] = static_cast<int>(u);
//             mate_u_[u] = static_cast<int>(v);
//             dist_[u] = -1;
//             return true;
//           }
//         }
//       }
//       v = FindNext(local_neighbors, v);
//     }
//     dist_[u] = -1;
//     return false;
//   }

//   size_t num_tasks_;
//   const std::vector<Task>& task_list_;

//   std::vector<int> mate_u_;
//   std::vector<int> mate_v_;
//   std::vector<int> dist_;
//   int dist_null_ = 0;

//   // Bitset dimensions
//   size_t words_per_bitset_;

//   // Reusable buffers for bitset computation (invalidated across calls)
//   std::vector<uint64_t> neighbors_buf_;
//   std::vector<uint64_t> bfs_buf_;
// };

class CBC_SPDS : public SPDS
{
public:
  /**
   * Constructs a cell-by-cell sweep-plane data strcture (SPDS) with the given direction and grid.
   *
   * \param omega The angular direction vector.
   * \param grid Reference to the grid.
   * \param allow_cycles Whether cycles are allowed in the local sweep dependency graph.
   */
  CBC_SPDS(const Vector3& omega, const std::shared_ptr<MeshContinuum>& grid, bool allow_cycles);

  /// Returns the cell-by-cell task list.
  const std::vector<Task>& GetTaskList() const;

  void SimulateLocalSweep();

protected:
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;
  /// Maximum number of pool allocator slots needed for CBC FLUDS.
  size_t max_num_pool_allocator_slots_;
};

} // namespace opensn

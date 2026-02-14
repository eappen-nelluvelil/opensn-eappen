// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <queue>

namespace opensn
{

/**
 * Helper class for the Hopcroft-Karp maximum bipartite matching algorithm.
 */
template <typename Graph, typename MateMap, typename VertexIndexMap>
class HKAugmentingPathFinder
{
public:
  using vertex_iterator = typename boost::graph_traits<Graph>::vertex_iterator;
  using out_edge_iterator = typename boost::graph_traits<Graph>::out_edge_iterator;

  HKAugmentingPathFinder(const Graph& graph, VertexIndexMap vertex_index_map, MateMap mate_map)
    : graph_(graph), vertex_index_map_(vertex_index_map), mate_map_(mate_map), distance_(boost::num_vertices(graph)),
      partition_(boost::num_vertices(graph), -1)
  {
    // Initialize partition (2-coloring) to identify the left and right sets of the bipartite graph
    vertex_iterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(graph_); vi != vi_end; ++vi)
    {
      if (partition_[boost::get(vertex_index_map_, *vi)] == -1)
      {
        // Assign color 0
        partition_[boost::get(vertex_index_map_, *vi)] = 0;
        std::queue<Vertex> q;
        q.push(*vi);
        while (not q.empty())
        {
          auto u = q.front();
          q.pop();
          auto c = partition_[boost::get(vertex_index_map_, u)];
          out_edge_iterator ei, ei_end;
          for (boost::tie(ei, ei_end) = boost::out_edges(u, graph_); ei != ei_end; ++ei)
          {
            auto v = boost::target(*ei, graph_);
            // If neighbor is uncolored, assign it the opposite color
            if (partition_[boost::get(vertex_index_map_, v)] == -1)
            {
              partition_[boost::get(vertex_index_map_, v)] = 1 - c;
              q.push(v);
            }
          }
        }
      }
    }
  }

  // Execute one phase of BFS, followed by DFS
  bool AugmentMatching()
  {
    std::queue<Vertex> Q;
    constexpr auto inf = std::numeric_limits<int>::max();
    dist_null_ = inf;

    vertex_iterator vi, vi_end;
    // BFS to build layers from free left-nodes
    for (boost::tie(vi, vi_end) = boost::vertices(graph_); vi != vi_end; ++vi)
    {
      // Only consider left partition for starting paths
      if (partition_[boost::get(vertex_index_map_, *vi)] == 0)
      {
        // If vertex is unmatched, it's a valid root for augmenting path
        if (boost::get(mate_map_, *vi) == boost::graph_traits<Graph>::null_vertex())
        {
          distance_[boost::get(vertex_index_map_, *vi)] = 0;
          Q.push(*vi);
        }
        else
        {
          distance_[boost::get(vertex_index_map_, *vi)] = inf;
        }
      }
    }

    // Standard BFS loop
    while (not Q.empty())
    {
      Vertex u = Q.front();
      Q.pop();

      // Don't search deeper than the shortest augmenting path found so far
      if (distance_[boost::get(vertex_index_map_, u)] < dist_null_)
      {
        out_edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = boost::out_edges(u, graph_); ei != ei_end; ++ei)
        {
          Vertex v = boost::target(*ei, graph_);
          Vertex v_mate = boost::get(mate_map_, v);

          // If v is free, we found an augmenting path
          if (v_mate == boost::graph_traits<Graph>::null_vertex())
          {
            dist_null_ = distance_[boost::get(vertex_index_map_, u)] + 1;
          }
          else
          {
            // If v is matched, continue path to its mate (left node)
            if (distance_[boost::get(vertex_index_map_, v_mate)] == inf)
            {
              distance_[boost::get(vertex_index_map_, v_mate)] = distance_[boost::get(vertex_index_map_, u)] + 1;
              Q.push(v_mate);
            }
          }
        }
      }
    }

    // If no augmenting path was found, we are done
    if (dist_null_ == inf)
      return false;

    // DFS to find vertex-disjoint augmenting paths
    bool augmented = false;
    for (boost::tie(vi, vi_end) = boost::vertices(graph_); vi != vi_end; ++vi)
    {
      if (partition_[boost::get(vertex_index_map_, *vi)] == 0 and
          boost::get(mate_map_, *vi) == boost::graph_traits<Graph>::null_vertex())
      {
        if (DFS(*vi, inf))
          augmented = true;
      }
    }

    return augmented;
  }

private:
  const Graph& graph_;
  VertexIndexMap vertex_index_map_;
  MateMap mate_map_;
  std::vector<int> distance_;
  std::vector<int> partition_;
  int dist_null_ = 0;

  // Recursive DFS to find augmenting paths to a free node following the layering from BFS
  bool DFS(Vertex u, const int inf)
  {
    out_edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::out_edges(u, graph_); ei != ei_end; ++ei)
    {
      auto v = boost::target(*ei, graph_);
      auto v_mate = boost::get(mate_map_, v);

      if (v_mate == boost::graph_traits<Graph>::null_vertex())
      {
        if (dist_null_ == distance_[boost::get(vertex_index_map_, u)] + 1)
        {
          // Augment: flip the matching along this edge
          boost::put(mate_map_, v, u);
          boost::put(mate_map_, u, v);
          return true;
        }
      }
      else
      {
        // Continue search through the matched edge if it follows the layer
        if (distance_[boost::get(vertex_index_map_, v_mate)] == distance_[boost::get(vertex_index_map_, u)] + 1)
        {
          if (DFS(v_mate, inf))
          {
            // Augment: flip the matching along this edge
            boost::put(mate_map_, v, u);
            boost::put(mate_map_, u, v);
            return true;
          }
        }
      }
    }

    // Mark vertex as dead end of path to prevent redundant searches
    distance_[boost::get(vertex_index_map_, u)] = inf;
    return false;
  }
};

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

  /// Returns the minimum number of slots needed for CBC_FLUDS pool allocator.
  std::size_t SimulateLocalSweep() const;

protected:
  /// Cell-by-cell task list.
  std::vector<Task> task_list_;
};

} // namespace opensn

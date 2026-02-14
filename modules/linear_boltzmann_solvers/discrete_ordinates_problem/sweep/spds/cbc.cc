// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/utils/timer.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <boost/dynamic_bitset.hpp>
#include <boost/graph/max_cardinality_matching.hpp>
#include <boost/graph/topological_sort.hpp>

namespace opensn
{



CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const std::shared_ptr<MeshContinuum>& grid,
                   bool allow_cycles)
  : SPDS(omega, grid)
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

  size_t num_loc_cells = grid->local_cells.size();

  // Populate Cell Relationships
  std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
  std::set<int> location_successors;
  std::set<int> location_dependencies;

  PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

  location_successors_.reserve(location_successors.size());
  location_dependencies_.reserve(location_dependencies.size());

  for (auto v : location_successors)
    location_successors_.push_back(v);

  for (auto v : location_dependencies)
    location_dependencies_.push_back(v);

  // Build local cell graph
  Graph local_DG(num_loc_cells);

  // Create graph edges
  for (size_t c = 0; c < num_loc_cells; ++c) // NOLINT
    for (const auto& successor : cell_successors[c])
      boost::add_edge(c, successor.first, successor.second, local_DG);

  if (allow_cycles) // NOLINT
  {
    auto edges_to_remove = RemoveCyclicDependencies(local_DG);
    for (auto& edge_to_remove : edges_to_remove)
      local_sweep_fas_.emplace_back(edge_to_remove.first, edge_to_remove.second);
  }

  // Generate topological sorting
  spls_.clear();
  boost::topological_sort(local_DG, std::back_inserter(spls_)); // NOLINT
  std::reverse(spls_.begin(), spls_.end());
  if (spls_.empty())
  {
    throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
                           "Cycles need to be allowed by the calling application.");
  }

  // Create task list
  std::vector<std::vector<int>> global_dependencies;
  global_dependencies.resize(opensn::mpi_comm.size());
  CommunicateLocationDependencies(location_dependencies_, global_dependencies);

  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

  // For each local cell create a task
  for (const auto& cell : grid_->local_cells)
  {
    const size_t num_faces = cell.faces.size();
    unsigned int num_dependencies = 0;
    unsigned int num_satisfied_successors = 0;
    std::vector<std::uint32_t> predecessors;
    std::vector<std::uint32_t> successors;

    for (size_t f = 0; f < num_faces; ++f)
    {
      const auto& face = cell.faces[f];
      const auto& face_orientation = cell_face_orientations_[cell.local_id][f];
      if (face_orientation == INCOMING)
      {
        if (face.has_neighbor)
        {
          ++num_dependencies;
          if (grid->IsCellLocal(face.neighbor_id))
            predecessors.push_back(grid->cells[face.neighbor_id].local_id);
        }
      }
      else if (face_orientation == OUTGOING)
      {
        if (face.has_neighbor and grid->IsCellLocal(face.neighbor_id))
          successors.push_back(grid->cells[face.neighbor_id].local_id);
      }
    }

    task_list_.push_back({num_dependencies, num_satisfied_successors, predecessors, successors, cell.local_id, &cell, false});
  }

  SimulateLocalSweep();

  opensn::log.Log() << "CBC_SPDS: Maximum number of pool allocator slots = " << edmonds_blossom_min_num_slots_ << "\n";
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const
{
  return task_list_;
}

void
CBC_SPDS::SimulateLocalSweep()
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::SimulateLocalSweep");

  const auto num_tasks = task_list_.size();
  if (num_tasks == 0)
    return;

  // Construct transitive closure of the local cell graph to determine the maximum number of simultaneously ready tasks
  std::vector<boost::dynamic_bitset<>> reachability(num_tasks, boost::dynamic_bitset<>(num_tasks));

  // Iterate backwards through topologically sorted tasks to populate reachability
  for (auto it = spls_.rbegin(); it != spls_.rend(); ++it)
  {
    const auto u = *it;
    const auto& task = task_list_[u];

    // Reflexive: node u reaches itself
    reachability[u].set(u);

    // Union with successors' reachability
    for (const auto& succ : task.successors)
      reachability[u] |= reachability[succ];
  }

  // Build reuse graph: edge from u to v if task v can reuse memory from task u (i.e. no path from u to v in the reachability graph)
  using BipartiteGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
  BipartiteGraph reuse_graph(2 * num_tasks);

  boost::dynamic_bitset<> valid_targets(num_tasks);

  for (int u = 0; u < num_tasks; ++u)
  {
    const auto& task_u = task_list_[u];

    // If task u has no local successors, it is a sink
    if (task_u.successors.empty())
      continue;

    // Start with the first successor's reachability
    valid_targets = reachability[task_u.successors[0]];

    // Intersect with all other successors
    for (size_t i = 1; i < task_u.successors.size(); ++i)
      valid_targets &= reachability[task_u.successors[i]];

    // Strictness check: remove immediate successors themselves
    // Buffer is live during handover to immediate successors, so it cannot be reused until after the immediate successors execute
    for (const auto& succ : task_u.successors)
      valid_targets.reset(succ);

    // Add edges to reuse graph for all valid targets
    auto v = valid_targets.find_first();
    while (v != boost::dynamic_bitset<>::npos)
    {
      // Add edge u -> v in reuse graph (with u and v in separate partitions)
      boost::add_edge(u, v + num_tasks, reuse_graph);
      v = valid_targets.find_next(v);
    }
  }

  // Run Hopcroft-Karp to find maximum matching in reuse graph, which corresponds to maximum reuse and thus minimum number of simultaneously live tasks
  // std::vector<boost::graph_traits<BipartiteGraph>::vertex_descriptor> mate_map(2 * num_tasks);
  // std::fill(mate_map.begin(), mate_map.end(), boost::graph_traits<BipartiteGraph>::null_vertex());

  // HKAugmentingPathFinder<BipartiteGraph,
  //                       boost::property_map<BipartiteGraph, boost::vertex_index_t>::type>
  //   augmenting_path_finder(reuse_graph, boost::get(boost::vertex_index, reuse_graph), mate_map);

  // // Augment until no additional augmenting paths can be found
  // while (augmenting_path_finder.AugmentMatching()) {}

  // // Count number of matched edges, which corresponds to number of reuses
  // size_t hopcroft_karp_matching_size = 0;
  // for (size_t i = 0; i < num_tasks; ++i)
  // {
  //   // Check if a vertex in the left partition (task u) is matched to a vertex in the right partition (task v)
  //   if (mate_map[i] != boost::graph_traits<BipartiteGraph>::null_vertex() and mate_map[i] >= num_tasks)
  //     ++hopcroft_karp_matching_size;
  // }

  // // Minimum number of buffers needed is total tasks minus reuses
  // hopcroft_karp_min_num_slots_ = num_tasks - hopcroft_karp_matching_size;

  // Calculate minimum number of buffers needed via Edmonds Blossom algorithm with verification
  // that a maximum matching was found, as a cross-check against the Hopcroft-Karp result
  std::vector<boost::graph_traits<BipartiteGraph>::vertex_descriptor> edmond_blossoms_mate_map(2 * num_tasks);
  bool is_maximum_matching = boost::checked_edmonds_maximum_cardinality_matching(
    reuse_graph, &edmond_blossoms_mate_map[0]);
  
  if (is_maximum_matching)
  {
    size_t edmonds_blossom_matching_size = 0;
    for (size_t i = 0; i < num_tasks; ++i)
    {
      if (edmond_blossoms_mate_map[i] != boost::graph_traits<BipartiteGraph>::null_vertex() and
          edmond_blossoms_mate_map[i] >= num_tasks)
        ++edmonds_blossom_matching_size;
    }

    edmonds_blossom_min_num_slots_ = num_tasks - edmonds_blossom_matching_size;
  }
  else
  {
    opensn::log.Log0Warning() << "Edmonds blossom algorithm did not find a maximum matching. Results may be inaccurate.\n";
  }

  // Print logging statement if two slot calculates DON'T match
  // if (hopcroft_karp_min_num_slots_ != edmonds_blossom_min_num_slots_)
  // {
  //   opensn::log.Log0Warning() 
  //      << "Hopcroft-Karp and Edmonds blossom algorithms calculated different minimum slot counts for CBC_FLUDS pool allocator. "
  //      << "Hopcroft-Karp min slots: " << hopcroft_karp_min_num_slots_ 
  //      << ", Edmonds blossom min slots: " << edmonds_blossom_min_num_slots_ << ".\n";
  // }

}

} // namespace opensn

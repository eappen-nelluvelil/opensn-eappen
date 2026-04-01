// // SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// // SPDX-License-Identifier: MIT

// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
// #include "framework/mesh/mesh_continuum/mesh_continuum.h"
// #include "framework/logging/log.h"
// #include "framework/utils/timer.h"
// #include "framework/runtime.h"
// #include "caliper/cali.h"
// #include <boost/graph/topological_sort.hpp>
// #include <algorithm>
// #include <stdexcept>

// namespace opensn
// {

// CBC_SPDS::CBC_SPDS(int id,
//                    const Vector3& omega,
//                    const std::shared_ptr<MeshContinuum>& grid,
//                    bool allow_cycles)
//   : SPDS(omega, grid),
//     id_(id),
//     allow_cycles_(allow_cycles)
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

//   size_t num_loc_cells = grid->local_cells.size();

//   // Populate Cell Relationships
//   std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
//   std::set<int> location_successors;
//   std::set<int> location_dependencies;

//   PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

//   location_successors_.assign(location_successors.begin(), location_successors.end());
//   location_dependencies_.assign(location_dependencies.begin(), location_dependencies.end());

//   // Build local cell graph
//   Graph local_DG(num_loc_cells);

//   // Create graph edges
//   for (size_t c = 0; c < num_loc_cells; ++c) // NOLINT
//     for (const auto& successor : cell_successors[c])
//       boost::add_edge(c, successor.first, successor.second, local_DG);

//   if (allow_cycles) // NOLINT
//   {
//     auto edges_to_remove = RemoveCyclicDependencies(local_DG);
//     for (const auto& [u, v] : edges_to_remove)
//     {
//       local_sweep_fas_.emplace_back(static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(v));
//       delayed_local_dependency_set_.insert(PackEdge(static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(v)));
//     }
//   }

//   // Generate topological sorting
//   spls_.clear();
//   boost::topological_sort(local_DG, std::back_inserter(spls_)); // NOLINT
//   std::reverse(spls_.begin(), spls_.end());
//   if (spls_.empty())
//     throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
//                            "Cycles need to be allowed by the calling application.");

//   global_dependencies_.resize(opensn::mpi_comm.size());
//   CommunicateLocationDependencies(location_dependencies_, global_dependencies_);

//   BuildTaskList();
// }

// std::vector<double>
// CBC_SPDS::ComputeLocalLocationEdgeWeights() const
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeLocalLocationEdgeWeights");

//   const int comm_size = opensn::mpi_comm.size();
//   std::vector<double> row(comm_size, 0.0);

//   constexpr double tolerance = 1e-16;

//   for (const auto& cell : grid_->local_cells)
//   {
//     const auto& face_orientations = cell_face_orientations_[cell.local_id];
//     std::size_t f = 0;
//     for (const auto& face : cell.faces)
//     {
//       if (face.has_neighbor and not face.IsNeighborLocal(grid_.get()) and
//           face_orientations[f] == FaceOrientation::OUTGOING)
//       {
//         const double mu = omega_.Dot(face.normal);
//         if (mu > tolerance)
//         {
//           const auto& adj_cell = grid_->cells[face.neighbor_id];
//           row[adj_cell.partition_id] += mu * mu * face.area;
//         }
//       }
//       ++f;
//     }
//   }

//   return row;
// }

// void
// CBC_SPDS::BuildGlobalSweepFAS()
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::BuildGlobalSweepFAS");

//   const int comm_size = opensn::mpi_comm.size();
//   Graph global_tdg(comm_size);

//   for (int loc = 0; loc < comm_size; ++loc)
//   {
//     for (const auto dep : global_dependencies_[loc])
//     {
//       double weight = 1.0;
//       if (not global_edge_weights_.empty())
//       {
//         const int idx = dep * comm_size + loc;
//         if (idx < static_cast<int>(global_edge_weights_.size()) and global_edge_weights_[idx] > 0.0)
//           weight = global_edge_weights_[idx];
//       }
//       boost::add_edge(loc, dep, weight, global_tdg);
//     }
//   }

//   global_sweep_fas_.clear();
//   if (allow_cycles_)
//   {
//     const auto edges_to_remove = RemoveCyclicDependencies(global_tdg);
//     for (const auto& [u, v] : edges_to_remove)
//     {
//       global_sweep_fas_.push_back(static_cast<int>(u));
//       global_sweep_fas_.push_back(static_cast<int>(v));
//     }
//   }
// }

// void
// CBC_SPDS::BuildGlobalSweepTDG()
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::BuildGlobalSweepTDG");

//   delayed_location_dependencies_.clear();
//   delayed_location_successors_.clear();

//   Graph global_tdg(opensn::mpi_comm.size());
//   for (int loc = 0; loc < opensn::mpi_comm.size(); ++loc)
//     for (const auto dep : global_dependencies_[loc])
//       boost::add_edge(loc, dep, 1.0, global_tdg);

//   std::vector<std::pair<int, int>> edges_to_remove(global_sweep_fas_.size() / 2);
//   int i = 0;
//   for (auto& edge : edges_to_remove)
//   {
//     edge.first = global_sweep_fas_[i++];
//     edge.second = global_sweep_fas_[i++];
//   }

//   for (const auto& [pred_loc, succ_loc] : edges_to_remove)
//   {
//     boost::remove_edge(pred_loc, succ_loc, global_tdg);

//     if (succ_loc == opensn::mpi_comm.rank())
//     {
//       const auto it = 
//         std::find(location_dependencies_.begin(), location_dependencies_.end(), pred_loc);
//       if (it != location_dependencies_.end())
//         location_dependencies_.erase(it);
//       delayed_location_dependencies_.push_back(pred_loc);
//     }

//     if (pred_loc == opensn::mpi_comm.rank())
//       delayed_location_successors_.push_back(succ_loc);
//   }

//   std::vector<int> global_linear_sweep_order;
//   boost::topological_sort(global_tdg, std::back_inserter(global_linear_sweep_order));
//   std::reverse(global_linear_sweep_order.begin(), global_linear_sweep_order.end());
//   if (global_linear_sweep_order.empty())
//     throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the global sweep graph.\n"
//                            "Cycles need to be allowed by the calling application.");

//   BuildTaskList();
// }

// bool
// CBC_SPDS::IsDelayedLocalDependency(std::uint32_t upwind_local_id,
//                                    std::uint32_t downwind_local_id) const noexcept
// {
//   return delayed_local_dependency_set_.contains(PackEdge(upwind_local_id, downwind_local_id));
// }

// void
// CBC_SPDS::BuildTaskList()
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::BuildTaskList");

//   task_list_.clear();
//   task_list_.reserve(grid_->local_cells.size());

//   constexpr auto INCOMING = FaceOrientation::INCOMING;
//   constexpr auto OUTGOING = FaceOrientation::OUTGOING;

//   for (const auto& cell : grid_->local_cells)
//   {
//     unsigned int num_dependencies = 0;
//     std::vector<std::uint32_t> successors;

//     for (size_t f = 0; f < cell.faces.size(); ++f)
//     {
//       const auto orientation = cell_face_orientations_[cell.local_id][f];
//       const auto& face = cell.faces[f];

//       if (orientation == INCOMING and face.has_neighbor)
//       {
//         if (face.IsNeighborLocal(grid_.get()))
//         {
//           const auto& upwind_cell = grid_->cells[face.neighbor_id];
//           if (not IsDelayedLocalDependency(upwind_cell.local_id, cell.local_id))
//             ++num_dependencies;
//         }
//         else
//         {
//           const int pred_loc = face.GetNeighborPartitionID(grid_.get());
//           const bool is_delayed = 
//             std::find(delayed_location_dependencies_.begin(),
//                       delayed_location_dependencies_.end(),
//                       pred_loc) != delayed_location_dependencies_.end();
//           if (not is_delayed)
//             ++num_dependencies;
//         }
//       }
//       else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
//       {
//         const auto& succ_cell = grid_->cells[face.neighbor_id];
//         if (not IsDelayedLocalDependency(cell.local_id, succ_cell.local_id))
//           successors.push_back(succ_cell.local_id);
//       }
//     }

//     task_list_.push_back(Task{num_dependencies, successors, cell.local_id, &cell, false});
//   }
// }

// } // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
  // SPDX-License-Identifier: MIT

  #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
  #include "framework/mesh/mesh_continuum/mesh_continuum.h"
  #include "framework/runtime.h"
  #include "caliper/cali.h"
  #include <boost/graph/topological_sort.hpp>
  #include <algorithm>
  #include <stdexcept>

  namespace opensn
  {

  CBC_SPDS::CBC_SPDS(int id,
                     const Vector3& omega,
                     const std::shared_ptr<MeshContinuum>& grid,
                     bool allow_cycles)
    : SPDS(omega, grid),
      id_(id),
      allow_cycles_(allow_cycles)
  {
    CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

    const size_t num_loc_cells = grid->local_cells.size();

    std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
    std::set<int> location_successors;
    std::set<int> location_dependencies;

    PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

    location_successors_.assign(location_successors.begin(), location_successors.end());
    location_dependencies_.assign(location_dependencies.begin(), location_dependencies.end());

    Graph local_dg(num_loc_cells);
    for (size_t c = 0; c < num_loc_cells; ++c)
      for (const auto& successor : cell_successors[c])
        boost::add_edge(c, successor.first, successor.second, local_dg);

    if (allow_cycles_)
    {
      const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
      for (const auto& [u, v] : edges_to_remove)
      {
        local_sweep_fas_.emplace_back(static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(v));
        delayed_local_dependency_set_.insert(PackEdge(static_cast<std::uint32_t>(u),
                                                      static_cast<std::uint32_t>(v)));
      }
    }

    spls_.clear();
    boost::topological_sort(local_dg, std::back_inserter(spls_));
    std::reverse(spls_.begin(), spls_.end());
    if (spls_.empty())
      throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
                             "Cycles need to be allowed by the calling application.");

    global_dependencies_.resize(opensn::mpi_comm.size());
    CommunicateLocationDependencies(location_dependencies_, global_dependencies_);

    BuildTaskList();
  }

  std::vector<double>
  CBC_SPDS::ComputeLocalLocationEdgeWeights() const
  {
    CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeLocalLocationEdgeWeights");

    const int comm_size = opensn::mpi_comm.size();
    std::vector<double> row(comm_size, 0.0);

    constexpr double tolerance = 1.0e-16;

    for (const auto& cell : grid_->local_cells)
    {
      const auto& face_orientations = cell_face_orientations_[cell.local_id];
      std::size_t f = 0;
      for (const auto& face : cell.faces)
      {
        if (face.has_neighbor and not face.IsNeighborLocal(grid_.get()) and
            face_orientations[f] == FaceOrientation::OUTGOING)
        {
          const double mu = omega_.Dot(face.normal);
          if (mu > tolerance)
          {
            const auto& adj_cell = grid_->cells[face.neighbor_id];
            row[adj_cell.partition_id] += mu * mu * face.area;
          }
        }
        ++f;
      }
    }

    return row;
  }

  void
  CBC_SPDS::BuildGlobalSweepFAS()
  {
    CALI_CXX_MARK_SCOPE("CBC_SPDS::BuildGlobalSweepFAS");

    const int comm_size = opensn::mpi_comm.size();
    Graph global_tdg(comm_size);

    for (int loc = 0; loc < comm_size; ++loc)
    {
      for (const auto dep : global_dependencies_[loc])
      {
        double weight = 1.0;
        if (not global_edge_weights_.empty())
        {
          const int idx = dep * comm_size + loc;
          if (idx < static_cast<int>(global_edge_weights_.size()) and global_edge_weights_[idx] > 0.0)
            weight = global_edge_weights_[idx];
        }
        boost::add_edge(dep, loc, weight, global_tdg);
      }
    }

    global_sweep_fas_.clear();
    if (allow_cycles_)
    {
      const auto edges_to_remove = RemoveCyclicDependencies(global_tdg);
      for (const auto& [u, v] : edges_to_remove)
      {
        global_sweep_fas_.push_back(static_cast<int>(u));
        global_sweep_fas_.push_back(static_cast<int>(v));
      }
    }
  }

  void
  CBC_SPDS::BuildGlobalSweepTDG()
  {
    CALI_CXX_MARK_SCOPE("CBC_SPDS::BuildGlobalSweepTDG");

    delayed_location_dependencies_.clear();
    delayed_location_successors_.clear();

    Graph global_tdg(opensn::mpi_comm.size());
    for (int loc = 0; loc < opensn::mpi_comm.size(); ++loc)
      for (const auto dep : global_dependencies_[loc])
        boost::add_edge(dep, loc, 1.0, global_tdg);

    std::vector<std::pair<int, int>> edges_to_remove(global_sweep_fas_.size() / 2);
    int i = 0;
    for (auto& edge : edges_to_remove)
    {
      edge.first = global_sweep_fas_[i++];
      edge.second = global_sweep_fas_[i++];
    }

    for (const auto& [pred_loc, succ_loc] : edges_to_remove)
    {
      boost::remove_edge(pred_loc, succ_loc, global_tdg);

      if (succ_loc == opensn::mpi_comm.rank())
      {
        const auto it =
          std::find(location_dependencies_.begin(), location_dependencies_.end(), pred_loc);
        if (it != location_dependencies_.end())
          location_dependencies_.erase(it);
        delayed_location_dependencies_.push_back(pred_loc);
      }

      if (pred_loc == opensn::mpi_comm.rank())
        delayed_location_successors_.push_back(succ_loc);
    }

    std::vector<int> global_linear_sweep_order;
    boost::topological_sort(global_tdg, std::back_inserter(global_linear_sweep_order));
    std::reverse(global_linear_sweep_order.begin(), global_linear_sweep_order.end());
    if (global_linear_sweep_order.empty())
      throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the global sweep graph.\n"
                             "Cycles need to be allowed by the calling application.");

    BuildTaskList();
  }

  bool
  CBC_SPDS::IsDelayedLocalDependency(std::uint32_t upwind_local_id,
                                     std::uint32_t downwind_local_id) const noexcept
  {
    return delayed_local_dependency_set_.contains(PackEdge(upwind_local_id, downwind_local_id));
  }

  void
  CBC_SPDS::BuildTaskList()
  {
    CALI_CXX_MARK_SCOPE("CBC_SPDS::BuildTaskList");

    task_list_.assign(grid_->local_cells.size(), Task{0, {}, 0, nullptr, false});

    constexpr auto INCOMING = FaceOrientation::INCOMING;
    constexpr auto OUTGOING = FaceOrientation::OUTGOING;

    for (const auto& cell : grid_->local_cells)
    {
      unsigned int num_dependencies = 0;
      std::vector<std::uint32_t> successors;

      for (size_t f = 0; f < cell.faces.size(); ++f)
      {
        const auto orientation = cell_face_orientations_[cell.local_id][f];
        const auto& face = cell.faces[f];

        if (orientation == INCOMING and face.has_neighbor)
        {
          if (face.IsNeighborLocal(grid_.get()))
          {
            const auto& upwind_cell = grid_->cells[face.neighbor_id];
            if (not IsDelayedLocalDependency(upwind_cell.local_id, cell.local_id))
              ++num_dependencies;
          }
          else
          {
            const int pred_loc = face.GetNeighborPartitionID(grid_.get());
            const bool is_delayed =
              std::find(delayed_location_dependencies_.begin(),
                        delayed_location_dependencies_.end(),
                        pred_loc) != delayed_location_dependencies_.end();
            if (not is_delayed)
              ++num_dependencies;
          }
        }
        else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
        {
          const auto& succ_cell = grid_->cells[face.neighbor_id];
          if (not IsDelayedLocalDependency(cell.local_id, succ_cell.local_id))
            successors.push_back(succ_cell.local_id);
        }
      }

      task_list_[cell.local_id] =
        Task{num_dependencies, std::move(successors), cell.local_id, &cell, false};
    }
  }

  } // namespace opensn

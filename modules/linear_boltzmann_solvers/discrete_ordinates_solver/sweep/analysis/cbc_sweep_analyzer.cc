// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/analysis/cbc_sweep_analyzer.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h" // For Cell, Face, connectivity
#include "framework/math/spatial_discretization/spatial_discretization.h" // For SpatialDiscretization
#include "framework/logging/log.h" // For potential logging
#include "framework/runtime.h" // For Exit, mpi_comm
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/sweep.h"

#include <cstdint>
#include <deque>
#include <stdexcept> // For runtime error
#include <set>
#include <sys/types.h>

namespace opensn
{
    CBC_SweepAnalyzer::CBC_SweepAnalyzer(const CBC_SPDS& spds,
                                         const SpatialDiscretization& discretization,
                                         size_t num_angles,
                                         size_t num_groups) :
        spds_(spds),
        grid_(spds.GetGrid()),
        discretization_(discretization),
        task_list_ref_(spds.GetTaskList()),
        face_orientations_(spds.GetCellFaceOrientations()),
        num_angles_(num_angles),
        num_groups_(num_groups)
    {
        total_num_local_cells_ = grid_->local_cells.size();
        if (total_num_local_cells_ == 0) return; // Skip if no local cells
            if (task_list_ref_.size() != total_num_local_cells_)
                throw std::logic_error("CBC_SweepAnalyzer: Mismatch between "
                                        "task_list size and local cell count.");
            if (face_orientations_.size() != total_num_local_cells_)
                    throw std::logic_error("CBC_SweepAnalyzer: Mismatch between "
                                            "face_orientations size and local cell count.");
    }

    /** Initializes dependency counters and task states before simulation */
    void CBC_SweepAnalyzer::InitializeAnalyticsData()
    {
        task_analytics_info_.assign(total_num_local_cells_, TaskAnalyticsInfo());
        face_analytics_info_.clear();

        // --- 1. Initialize Task statuses and initial cell dependencies ---
        std::vector<uint64_t> task_idx_to_cell_local_id_map(total_num_local_cells_);
        for (size_t task_idx = 0; task_idx < total_num_local_cells_; ++task_idx)
        {
            const auto& task = task_list_ref_[task_idx];    // Task now has num_local_predecessors

            /*
            // Store initial total count from SPDS
            task_analytics_info_[task_idx].initial_total_dependencies = task.num_dependencies;

            // Initialize remaining count with the same total count
            task_analytics_info_[task_idx].remaining_cell_dependencies = 
                                                          task.num_dependencies;
            task_analytics_info_[task_idx].status = TaskStatus::PENDING;
            task_analytics_info_[task_idx].only_external_dependencies = false;  // Initialize flag\
            */

            // Initialize remaining count based ONLY on local predecessors for simulation
            task_analytics_info_[task_idx].remaining_cell_dependencies = task.num_local_predecessors;
            task_idx_to_cell_local_id_map[task_idx] = task.reference_id;
        }

        // --- 1b. Determine which tasks depend *only* on external sources ---
        /*
        for (size_t task_idx = 0; task_idx < total_num_local_cells_; ++task_idx)
        {
            // It total deps > 0, check if *any* are local
            if (task_analytics_info_[task_idx].initial_total_dependencies > 0)
            {
                bool has_local_predecessor = false;
                uint64_t cell_local_id_down = task_idx_to_cell_local_id_map[task_idx];
                const auto& cell_down = grid_->local_cells[cell_local_id_down];

                if (cell_local_id_down < face_orientations_.size()) // Safety check
                {
                    for (int f_in = 0; f_in < cell_down.faces.size(); ++f_in)
                    {
                        if (f_in < face_orientations_[cell_local_id_down].size() && // Safety check
                            face_orientations_[cell_local_id_down][f_in] == FaceOrientation::INCOMING)
                        {
                            const auto& face = cell_down.faces[f_in];
                            // Check if it has a LOCAL neighbor providing this input
                            if (face.has_neighbor && grid_->IsCellLocal(face.neighbor_id))
                            {
                                // Need to check if this upstream face is actually outgoing for the SPDS
                                uint64_t cell_local_id_up = grid_->MapCellGlobalID2LocalID(face.neighbor_id);
                                int f_out = face.GetNeighborAdjacentFaceIndex(grid_.get());
                                if (cell_local_id_up < face_orientations_.size() && f_out >= 0 &&
                                    f_out < face_orientations_[cell_local_id_up].size() &&
                                    face_orientations_[cell_local_id_up][f_out] == FaceOrientation::OUTGOING)
                                {
                                    has_local_predecessor = true;
                                    break;  // Found a local predecessor; no need to check for further faces
                                }
                            }
                        }
                    }   // for f_in
                }   // Safety check

                // If total deps > 0, but found no local ones, set flag
                if (!has_local_predecessor)
                {
                    task_analytics_info_[task_idx].only_external_dependencies = true;
                }
            }   // if initial_total_dependencies > 0
        }   // for task_idx
        */

        // --- 2. Determine face-level dependencies
        // Build FaceKey -> list of downstream Task indices map
        std::map<FaceKey, std::vector<size_t>> face_to_downstream_tasks;

        for (size_t task_idx_down = 0; task_idx_down < total_num_local_cells_; ++task_idx_down)
        {
            uint64_t cell_local_id_down = task_idx_to_cell_local_id_map[task_idx_down];
            const auto& cell_down = grid_->local_cells[cell_local_id_down];

            if (cell_local_id_down >= face_orientations_.size()) continue; // Safety check

            for (int f_in = 0; f_in < cell_down.faces.size(); ++f_in)
            {
                // CHeck if face f_in is incoming for this SPDS direction
                if (f_in >= face_orientations_[cell_local_id_down].size()) continue; // Safey check
                if (face_orientations_[cell_local_id_down][f_in] == FaceOrientation::INCOMING)
                {
                    const auto& face = cell_down.faces[f_in];
                    if (face.has_neighbor)
                    {
                        uint64_t neighbor_global_id = face.neighbor_id;

                        // Check if neighbor is local
                        if (grid_->IsCellLocal(neighbor_global_id))
                        {
                            uint64_t cell_local_id_up = grid_->MapCellGlobalID2LocalID(neighbor_global_id);
                            int f_out = face.GetNeighborAdjacentFaceIndex(grid_.get());

                            if (cell_local_id_up >= face_orientations_.size() ||
                                f_out < 0 || f_out >= face_orientations_[cell_local_id_up].size()) continue; // Safety check
                            
                            // Check if f_out on cell_up is outgoing for this SPDS direction
                            if (face_orientations_[cell_local_id_up][f_out] == FaceOrientation::OUTGOING)
                            {
                                FaceKey upstream_face_key = {cell_local_id_up, f_out};
                                face_to_downstream_tasks[upstream_face_key].push_back(task_idx_down);
                            }
                        }
                        // Else: incoming dependency from non-local cell
                    }
                    // Else: boundary face
                }
            } // for f_in
        } // for task_idx_down

        // --- 3. Populate face_analytics_info_ using the map
        for (const auto& [face_key, downstream_tasks] : face_to_downstream_tasks)
        {
            auto& face_info = face_analytics_info_[face_key];
            face_info.total_local_dependencies = downstream_tasks.size();
            face_info.pending_local_dependencies = downstream_tasks.size();
            face_info.feeds_non_local_successor = false; // Assume false initially

            // Get face nodes count using the discretization object
            const auto& cell_of_face = grid_->local_cells[face_key.first];
            const auto& cell_mapping = discretization_.GetCellMapping(cell_of_face);
            face_info.num_nodes = cell_mapping.GetNumFaceNodes(face_key.second);
        }

        // --- 4. Identify faces feeding non-local successors and update FaceInfo
        const auto& loc_succs = spds_.GetLocationSuccessors();
        std::set<int> non_local_successor_pids(loc_succs.begin(), loc_succs.end());

        for (size_t cell_local_id_up = 0; cell_local_id_up < total_num_local_cells_; ++cell_local_id_up)
        {
            if (cell_local_id_up >= face_orientations_.size()) continue; // Safety check
            const auto& cell_up = grid_->local_cells[cell_local_id_up];
            for (int f_out = 0; f_out < cell_up.faces.size(); ++f_out) 
            {
                if (f_out >= face_orientations_[cell_local_id_up].size()) continue; // Safety check
                if (face_orientations_[cell_local_id_up][f_out] == FaceOrientation::OUTGOING)
                {
                    const auto& face = cell_up.faces[f_out];
                    if (face.has_neighbor && !grid_->IsCellLocal(face.neighbor_id))
                    {
                        int neighbor_pid = face.GetNeighborPartitionID(grid_.get());
                        if (non_local_successor_pids.count(neighbor_pid))
                        {
                            FaceKey face_key = {cell_local_id_up, f_out};
                            auto& face_info = face_analytics_info_[face_key];   // Creates if not exists
                            face_info.feeds_non_local_successor = true;
                            // If it wasn't already in the map, e.g., feeds ONLY non-local,
                            // initialize num_nodes here as well
                            if (face_info.total_local_dependencies == 0)
                            {
                                const auto& cell_of_face = grid_->local_cells[face_key.first];
                                const auto& cell_mapping = discretization_.GetCellMapping(cell_of_face);
                                face_info.num_nodes = cell_mapping.GetNumFaceNodes(face_key.second);
                            }
                        }
                    }
                }
            }
        }
    }

    /** Runs the analysis simulation */
    CBC_SweepAnalyzer::Results CBC_SweepAnalyzer::Analyze()
    {
        if (total_num_local_cells_ == 0) return {}; // Handle empty domains

        InitializeAnalyticsData();

        size_t num_executed_tasks = 0;
        std::queue<size_t> ready_queue; // Stores task_indices

        /*
        // LOGGING: Initial ready queue population
        log.Log0Verbose1() << "Analyzer: Initializing ready queue.";

        // Initialize ready queue with tasks having 0 initial dependencies
        for (size_t task_idx = 0; task_idx < total_num_local_cells_; ++task_idx)
        {
            // Use initial TOTAL dependency count from SPDS task
            if (task_analytics_info_[task_idx].initial_total_dependencies == 0)
            {
                // Analyzer treats these remaining deps as 0 for starting queue
                task_analytics_info_[task_idx].remaining_cell_dependencies = 0;
                ready_queue.push(task_idx);
                task_analytics_info_[task_idx].status = TaskStatus::READY;

                // LOGGING
                log.Log0Verbose2() << "Analyzer: Initial add task_idx=" << task_idx
                                   << " (CellLID=" << task_list_ref_[task_idx].reference_id
                                   << ") to ready queue (zero initial deps).";
            }
        }
        */

        /*
        // LOGGING: Initial ready queue population pass 2 (external deps only)
        if (ready_queue.empty() && total_num_local_cells_ > 0)
        {
            log.Log0Verbose1() << "Analyzer: initial ready queue empty; checking external-only deps (pass 2).";
            for (size_t task_idx = 0; task_idx < total_num_local_cells_; ++task_idx)
            {
                // If task only depends on external sources, kickstart it
                if (task_analytics_info_[task_idx].only_external_dependencies)
                {
                    // Analyzer treats these remaining deps as 0 for starting queue
                    task_analytics_info_[task_idx].remaining_cell_dependencies = 0;
                    ready_queue.push(task_idx);
                    task_analytics_info_[task_idx].status = TaskStatus::READY;
                    log.Log0Verbose2() << "Analyzer: initial add task_idx=" << task_idx
                                       << " (CellLID=" << task_list_ref_[task_idx].reference_id
                                       << ") to ready queue (external deps kickstart)";
                }
            }
        }
        */

        // Initialize ready queue with tasks having 0 initial *local* dependencies
        log.Log0Verbose1() << "Analyzer: Initializing ready queue (using local deps).";

        for (size_t task_idx = 0; task_idx < total_num_local_cells_; ++task_idx)
        {
            // Check the count initialized from task.num_local_predecessors
            if (task_analytics_info_[task_idx].remaining_cell_dependencies == 0)
            {
                ready_queue.push(task_idx);
                task_analytics_info_[task_idx].status = TaskStatus::READY;
                log.Log0Verbose2() << "Analyzer: Initial add task_idx=" << task_idx
                                   << " (CellLID=" << task_list_ref_[task_idx].reference_id
                                   << ") to ready queue (zero initial local deps).";
            }
        }

        // LOGGING
        log.Log0Verbose1() << "Analyzer: Initial ready queue size=" << ready_queue.size();

        std::set<FaceKey> live_face_buffers;
        Results results;

        // LOGGING
        size_t iteration = 0;   // Add iteration counter for logging

        // Simulation loop
        while (num_executed_tasks < total_num_local_cells_)
        {
            if (ready_queue.empty())
            {
                // LOGGING: Error state dump
                log.LogAllError() << "Analyzer: Ready queue empty! Iteration=" << iteration
                                  << ". Executed " << num_executed_tasks
                                  << "/" << total_num_local_cells_ << " tasks.";

                for (size_t tidx = 0; tidx < total_num_local_cells_; ++tidx) {
                    const auto& task_info = task_analytics_info_[tidx];
                    if (task_info.status != TaskStatus::EXECUTED) {
                         log.LogAllError() << "  PENDING/READY Task Idx=" << tidx
                                           << " (CellLID=" << task_list_ref_[tidx].reference_id
                                           << ") Status=" << static_cast<int>(task_info.status) // Assuming PENDING=0, READY=1
                                           << " RemainingCellDeps=" << task_info.remaining_cell_dependencies;
                    }
                }
                // Log live buffer state at error
                log.LogAllError() << "Analyzer: Live face buffers at error (" << live_face_buffers.size() << " total):";
                for (const auto& key : live_face_buffers) {
                    log.LogAllError() << "  Face (CellLID=" << key.first << ", FaceIdx=" << key.second << ")";
                }

                // This indicates a deadlock or an error in the dependency graph logic
                throw std::runtime_error("CBC_SweepAnalyzer::Analyze: Ready queue became empty "
                                         "before all tasks were processed. Possible graph error.");
            }

            size_t current_task_idx = ready_queue.front();
            ready_queue.pop();

            // Safety check: ensure we do not process already executed task
            if (task_analytics_info_[current_task_idx].status == TaskStatus::EXECUTED)
                continue;

                // LOGGING: Task dequeued
            log.Log0Verbose1() << "Analyzer: Iter " << iteration << " Dequeued Task Idx=" << current_task_idx
                               << " (CellLID=" << task_list_ref_[current_task_idx].reference_id << ")";

            task_analytics_info_[current_task_idx].status = TaskStatus::EXECUTED;
            num_executed_tasks++;

            const auto& current_task_def = task_list_ref_[current_task_idx];
            uint64_t cell_local_id_current = current_task_def.reference_id;
            const auto& cell_current = grid_->local_cells[cell_local_id_current];

            // --- 1. Decrement dependencies for faces providing INPUT to current_task
            if (cell_local_id_current < face_orientations_.size())  // Safety check
            {
                for (int f_in = 0; f_in < cell_current.faces.size(); ++f_in)
                {
                    if (f_in < face_orientations_[cell_local_id_current].size() &&
                        face_orientations_[cell_local_id_current][f_in] == FaceOrientation::INCOMING)
                    {
                        const auto& face = cell_current.faces[f_in];
                        if (face.has_neighbor && grid_->IsCellLocal(face.neighbor_id))
                        {
                            uint64_t cell_local_id_up = grid_->MapCellGlobalID2LocalID(face.neighbor_id);
                            int f_out = face.GetNeighborAdjacentFaceIndex(grid_.get());

                            // Check upstream face is valid and outgoing
                            if (cell_local_id_up < face_orientations_.size() && f_out >= 0 &&
                                f_out < face_orientations_[cell_local_id_up].size() &&
                                face_orientations_[cell_local_id_up][f_out] == FaceOrientation::OUTGOING)
                            {
                                FaceKey upstream_face_key = {cell_local_id_up, f_out};
                                if (face_analytics_info_.count(upstream_face_key))
                                {
                                    auto& face_info = face_analytics_info_.at(upstream_face_key);

                                    // LOGGING: Before decrementing face dep
                                    log.Log0Verbose2() << "Analyzer: Task " << current_task_idx << " consuming Upstream Face ("
                                                       << cell_local_id_up << "," << f_out
                                                       << "). PendingLocalDeps BEFORE=" << face_info.pending_local_dependencies;

                                    if (face_info.pending_local_dependencies > 0)  // Avoid underflow
                                    {
                                        face_info.pending_local_dependencies--;
                                    }

                                    // LOGGING: After decrementing face dep
                                    log.Log0Verbose2() << "Analyzer: Upstream Face (" << cell_local_id_up << "," << f_out
                                                       << "). PendingLocalDeps AFTER=" << face_info.pending_local_dependencies
                                                       << ", FeedsNL=" << face_info.feeds_non_local_successor;


                                    // Release buffer IF local deps met AND does not feed non-local
                                    // (simple approach for non-local lifetime)
                                    if (face_info.pending_local_dependencies == 0 &&
                                        !face_info.feeds_non_local_successor)
                                    {
                                        live_face_buffers.erase(upstream_face_key);

                                        // LOGGING: Buffer release
                                        log.Log0Verbose2() << "Analyzer: Released buffer for Upstream Face ("
                                                           << cell_local_id_up << "," << f_out << ")";
                                    }
                                }
                            }
                        }

                    }
                }
            }   // Safety check end

            // --- 2. Add outgoing faces of current_task to live set if needed
            if (cell_local_id_current < face_orientations_.size())  // Safety check
            {
                for (int f_out = 0; f_out < cell_current.faces.size(); ++f_out)
                {
                    if (f_out < face_orientations_[cell_local_id_current].size() &&
                        face_orientations_[cell_local_id_current][f_out] == FaceOrientation::OUTGOING)
                    {
                        FaceKey current_face_key = {cell_local_id_current, f_out};

                        // Check if this face actually has any downstream dependencies (local or non-local)
                        if (face_analytics_info_.count(current_face_key))
                        {
                            const auto& face_info = face_analytics_info_.at(current_face_key);

                            // Need storage if EITHER local deps remain OR it feeds non-local 
                            if (face_info.pending_local_dependencies > 0 || face_info.feeds_non_local_successor)
                            {
                                bool inserted = live_face_buffers.insert(current_face_key).second;
                                
                                // LOGGING: Buffer activation
                                if (inserted) 
                                {
                                    log.Log0Verbose2() << "Analyzer: Activated buffer for Outgoing Face ("
                                                       << cell_local_id_current << "," << f_out << ")";
                                }
                                
                                // live_face_buffers.insert(current_face_key);
                            }
                        }
                        // Else, this outgoing face has no downstream dependencies recorded, e.g., boundary face
                    }
                }
            }   // Safety check end

            // --- 3. Update peak counts
            results.max_live_faces_count = std::max(results.max_live_faces_count,
                                                    live_face_buffers.size());

            // Update max data size estimate
            size_t current_data_size_nodes = 0; // Accumulate nodes first
            for (const auto& live_face_key : live_face_buffers)
            {
                if (face_analytics_info_.count(live_face_key))
                {
                    const auto& face_info = face_analytics_info_.at(live_face_key);
                    current_data_size_nodes += face_info.num_nodes; // Accumulate nodes first 
                }
            }

            size_t current_data_size_bytes = current_data_size_nodes * 
                                                num_angles_ * num_groups_ * sizeof(double);  // Scale at the end
            results.max_live_data_size_estimate = std::max(results.max_live_data_size_estimate,
                                                           current_data_size_bytes);
                          
            // --- 4. Update and enqueue successors
            for (uint64_t successor_task_idx : current_task_def.successors)
            {
                // successor_task_idx is the index in the main task_list_ref_
                if (successor_task_idx < total_num_local_cells_ && // Bounds check
                    task_analytics_info_[successor_task_idx].status == TaskStatus::PENDING)
                {
                    // LOGGING: Before decrementing cell dep
                    log.Log0Verbose2() << "Analyzer: Task " << current_task_idx << " updating Successor Idx=" << successor_task_idx
                                                            << " (CellLID=" << task_list_ref_[successor_task_idx].reference_id
                                                            << "). CellDeps BEFORE=" << task_analytics_info_[successor_task_idx].remaining_cell_dependencies;

                    task_analytics_info_[successor_task_idx].remaining_cell_dependencies--;

                    // LOGGING: After decrementing cell dep
                    log.Log0Verbose2() << "Analyzer: Successor Idx=" << successor_task_idx
                                       << ". CellDeps AFTER=" << task_analytics_info_[successor_task_idx].remaining_cell_dependencies;

                    if (task_analytics_info_[successor_task_idx].remaining_cell_dependencies == 0)
                    {
                        ready_queue.push(successor_task_idx);
                        task_analytics_info_[successor_task_idx].status = TaskStatus::READY;

                        // LOGGING: Task enqueued
                        log.Log0Verbose1() << "Analyzer: Enqueued Successor Idx=" << successor_task_idx
                                           << " (CellLID=" << task_list_ref_[successor_task_idx].reference_id << ")";
                    }
                }
            }   // end successor loop

            // LOGGING: Live buffer count at end of iteration
            log.Log0Verbose2() << "Analyzer: End of Iter " << iteration << ". Live buffers=" << live_face_buffers.size();
        }   // while not all tasks executed

        if (num_executed_tasks != total_num_local_cells_)
        {
            log.LogAllWarning() << "CBC_SweepAnalyzer::Analyze: Simulation completed but not all "
                                << total_num_local_cells_ << " tasks were executed("
                                << num_executed_tasks << "done)."
                                << " Check for graph inconsistencies or deadlocks";
        }

        log.Log() << "CBC Sweep Analysis Results for SPDS (Omega="
                  << spds_.GetOmega().PrintStr() << ")";
        log.Log() << " Max concurrent live faces: " << results.max_live_faces_count;
        log.Log() << " Estimated max live data size (bytes): " << results.max_live_data_size_estimate;

        return results;
    }
}   // namespace opensn
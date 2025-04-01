// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/cbc.h" // For CBC_SPDS
#include "modules/linear_boltzmann_solvers/lbs_solver/lbs_structs.h" // For FaceOrientation
#include "framework/math/spatial_discretization/spatial_discretization.h" // For SpatialDiscretization

#include <vector>
#include <map>
#include <set>
#include <queue>
#include <memory> // For shared_ptr


namespace opensn
{
    class MehshContiuum;

    /**
    * Analyzes the task dependency graph of a CBC_SPDS to determine the peak concurrent memory
    * requirement for storing outoing angular fluxes on cell faces during a sweep.
    * This analysis is non-invase and simulates the control flow without performing
    * actual transport solves.
    */
    class CBC_SweepAnalyzer
    {
        public:
            /** Structure to hold the analysis results */
            struct Results
            {
                /** Peak number of faces requiring simultaneous storage */
                size_t max_live_faces_count = 0;
                /** Estimated peak data size (bytes) based on face nodes, angles, groups */
                size_t max_live_data_size_estimate = 0;
            };

        private:
            // Internal structures for tracking analysis state
            enum class TaskStatus {PENDING, READY, EXECUTED };

            struct TaskAnalyticsInfo
            {
                TaskStatus status = TaskStatus::PENDING;
                /** Number of direct cell/boundary dependencies remaining */
                unsigned int remaining_cell_dependencies = 0;
            };

            // Key: pair<unint64_t cell_local_id, int face_index>
            using FaceKey = std::pair<uint64_t, int>;

            struct FaceAnalyticsInfo
            {
                /** Total number of local downstream tasks depending on this face */
                size_t total_local_dependencies = 0;
                /** Remaining number of local downstream tasks depending on this face */
                size_t pending_local_dependencies = 0;
                /** Flag indicating if this face feeds a non-local successor */
                bool feeds_non_local_successor = false;
                /** Number of nodes on this face (for data size calculation) */
                size_t num_nodes = 0;
            };

            const CBC_SPDS& spds_;  // Reference to the SPDS being analyzed
            const std::shared_ptr<MeshContinuum> grid_; // Reference to the grid
            const SpatialDiscretization& discretization_; // Reference to the discretization
            const std::vector<Task>& task_list_ref_; // Reference to the SPDS task list
            const std::vector<std::vector<FaceOrientation>>& face_orientations_; // Reference to face orientations

            // Parameters for data size calculation (optional)
            const size_t num_angles_;
            const size_t num_groups_;

            // Internal state for the analysis simulation 
            std::vector<TaskAnalyticsInfo> task_analytics_info_; // Task analytics information
            std::map<FaceKey, FaceAnalyticsInfo> face_analytics_info_; // Face analytics information
            size_t total_num_local_cells_ = 0; // Total number of local cells

            /** Initialize dependency counters and task states before simulation */
            void InitializeAnalyticsData();

        public:
            /**
            * Constructor
            * \param spds The CBC_SPDS object to analyze
            * \param discretization The spatial discretization object providing cell mappings
            * \param num_angles Number of angles associated with sweeps using this SPDS
                                (used for data size estimation)
            * \param num_groups Number of groups associated with sweeps using this SPDS
                                (used for data size estimation)
            */

            explicit CBC_SweepAnalyzer(const CBC_SPDS& spds,
                                       const SpatialDiscretization& discretization,
                                       size_t num_angles,
                                       size_t num_groups);
            
            /**
            * Runs the analysis simulation
            * \return Results structure containing peak face count and data size
            */
            Results Analyze();
    };  // class CBC_SweepAnalyzer
}   // namespace opensn
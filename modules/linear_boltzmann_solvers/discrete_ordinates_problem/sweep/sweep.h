// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <set>
#include <memory>
#include <cstdint>

namespace opensn
{

class Cell;
class SPDS;
class MeshContinuum;

enum class FaceOrientation : short
{
  PARALLEL = -1,
  INCOMING = 0,
  OUTGOING = 1
};

enum class AngleSetStatus
{
  NOT_FINISHED = 0,
  FINISHED = 1,
  RECEIVING = 2,
  READY_TO_EXECUTE = 3,
  EXECUTE = 4,
  NO_EXEC_IF_READY = 5,
  MESSAGES_SENT = 6,
  MESSAGES_PENDING = 7
};

/// Per-cell task descriptor for the CBC sweep task graph.
struct Task
{
  /// Number of successors whose dependencies have been satisfied by this task.
  unsigned int num_satisfied_successors = 0;
  /// Total number of predecessor dependencies (local + nonlocal incoming).
  unsigned int num_dependencies;
  /// Local cell IDs of predecessor tasks.
  std::vector<std::uint32_t> predecessors;
  /// Local cell IDs of successor tasks.
  std::vector<std::uint32_t> successors;
  /// Cell local ID that this task corresponds to.
  uint64_t reference_id;
  /// Pointer to the cell in the mesh continuum.
  const Cell* cell_ptr;
  /// Whether this task has been completed in the current sweep.
  bool completed = false;
};

/// Stage Task Dependency Graphs
struct STDG
{
  std::vector<int> item_id;
};

/// Communicates location by location dependencies.
void CommunicateLocationDependencies(const std::vector<int>& location_dependencies,
                                     std::vector<std::vector<int>>& global_dependencies);

/// Print a sweep ordering to file.
void PrintSweepOrdering(SPDS* sweep_order, std::shared_ptr<MeshContinuum> vol_continuum);

} // namespace opensn

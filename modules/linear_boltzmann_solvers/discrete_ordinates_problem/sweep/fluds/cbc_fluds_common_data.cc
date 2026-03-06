// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mesh/cell/cell.h"

namespace opensn
{

CBC_FLUDSCommonData::CBC_FLUDSCommonData(
  const SPDS& spds, const std::vector<CellFaceNodalMapping>& grid_nodal_mappings)
  : FLUDSCommonData(spds, grid_nodal_mappings)
{
  // Pre-compute non-local face counts for hash map capacity reservation.
  // This follows the pattern used in AAH_FLUDSCommonData and AAHD_FLUDSCommonData.
  const auto& grid = *spds.GetGrid();
  const auto& face_orientations = spds.GetCellFaceOrientations();

  for (const auto& cell : grid.local_cells)
  {
    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = face_orientations[cell.local_id][f];

      // Skip local faces and boundary faces
      if (not face.has_neighbor or face.IsNeighborLocal(&grid))
        continue;

      if (orientation == FaceOrientation::OUTGOING)
        ++num_nonlocal_outgoing_faces_;
      else if (orientation == FaceOrientation::INCOMING)
        ++num_nonlocal_incoming_faces_;
    }
  }
}

} // namespace opensn

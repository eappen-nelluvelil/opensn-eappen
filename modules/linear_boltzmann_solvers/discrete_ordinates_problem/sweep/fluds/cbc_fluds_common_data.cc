// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/mesh/cell/cell.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include <cassert>

namespace opensn
{

CBC_FLUDSCommonData::CBC_FLUDSCommonData(
  const SPDS& spds, const std::vector<CellFaceNodalMapping>& grid_nodal_mappings)
  : FLUDSCommonData(spds, grid_nodal_mappings),
    num_incoming_nonlocal_faces_(0),
    num_outgoing_nonlocal_faces_(0)
{
  // Pre-compute non-local face counts for hash map capacity reservation
  const auto& grid = *spds.GetGrid();
  const auto& face_orientations = spds.GetCellFaceOrientations();
  incoming_nonlocal_face_slot_offsets_.resize(grid.local_cells.size(), 0);

  std::size_t num_local_faces = 0;
  std::size_t num_incoming_nonlocal_faces = 0;
  for (const auto& cell : grid.local_cells)
  {
    assert(cell.local_id < incoming_nonlocal_face_slot_offsets_.size());
    incoming_nonlocal_face_slot_offsets_[cell.local_id] = num_local_faces;
    num_local_faces += cell.faces.size();

    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      if ((not face.has_neighbor) or (face.IsNeighborLocal(&grid)))
        continue;

      const auto orientation = face_orientations[cell.local_id][f];
      if (orientation == FaceOrientation::INCOMING)
        ++num_incoming_nonlocal_faces;
      else if (orientation == FaceOrientation::OUTGOING)
        ++num_outgoing_nonlocal_faces_;
    }
  }

  incoming_nonlocal_face_slots_by_local_face_.assign(num_local_faces, INVALID_FACE_SLOT);
  incoming_nonlocal_face_slots_.reserve(num_incoming_nonlocal_faces);
  incoming_nonlocal_face_local_cells_.reserve(num_incoming_nonlocal_faces);

  for (const auto& cell : grid.local_cells)
  {
    const auto local_face_slot_offset = incoming_nonlocal_face_slot_offsets_[cell.local_id];
    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = face_orientations[cell.local_id][f];

      if ((not face.has_neighbor) or (face.IsNeighborLocal(&grid)))
        continue;

      if (orientation == FaceOrientation::INCOMING)
      {
        const auto slot = num_incoming_nonlocal_faces_;
        incoming_nonlocal_face_slots_.emplace(
          CellFaceKey{cell.global_id, static_cast<unsigned int>(f)}, slot);
        incoming_nonlocal_face_slots_by_local_face_[local_face_slot_offset + f] = slot;
        incoming_nonlocal_face_local_cells_.push_back(cell.local_id);
        ++num_incoming_nonlocal_faces_;
      }
    }
  }
}

size_t
CBC_FLUDSCommonData::GetIncomingNonlocalFaceSlot(std::uint64_t cell_global_id,
                                                 unsigned int face_id) const
{
  const auto it = incoming_nonlocal_face_slots_.find(CellFaceKey{cell_global_id, face_id});
  return it == incoming_nonlocal_face_slots_.end() ? INVALID_FACE_SLOT : it->second;
}

size_t
CBC_FLUDSCommonData::GetIncomingNonlocalFaceSlotByLocalFace(std::uint32_t cell_local_id,
                                                            unsigned int face_id) const
{
  assert(cell_local_id < incoming_nonlocal_face_slot_offsets_.size());
  const auto slot_offset = incoming_nonlocal_face_slot_offsets_[cell_local_id] + face_id;
  assert(slot_offset < incoming_nonlocal_face_slots_by_local_face_.size());
  return incoming_nonlocal_face_slots_by_local_face_[slot_offset];
}

std::uint32_t
CBC_FLUDSCommonData::GetIncomingNonlocalFaceLocalCell(size_t incoming_face_slot) const
{
  assert(incoming_face_slot < incoming_nonlocal_face_local_cells_.size());
  return incoming_nonlocal_face_local_cells_[incoming_face_slot];
}

} // namespace opensn

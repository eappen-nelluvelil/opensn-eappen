// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include "framework/mesh/cell/cell.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace opensn
{

class CBC_FLUDSCommonData : public FLUDSCommonData
{
public:
  CBC_FLUDSCommonData(const SPDS& spds,
                      const std::vector<CellFaceNodalMapping>& grid_nodal_mappings);

  size_t GetNumIncomingNonlocalFaces() const { return num_incoming_nonlocal_faces_; }

  size_t GetNumOutgoingNonlocalFaces() const { return num_outgoing_nonlocal_faces_; }

  size_t GetIncomingNonlocalFaceSlot(std::uint64_t cell_global_id, unsigned int face_id) const;
  size_t GetIncomingNonlocalFaceSlotByLocalFace(std::uint32_t cell_local_id,
                                                unsigned int face_id) const;

  static constexpr size_t invalid_face_slot = std::numeric_limits<size_t>::max();

private:
  size_t num_incoming_nonlocal_faces_;
  size_t num_outgoing_nonlocal_faces_;
  std::unordered_map<CellFaceKey, size_t> incoming_nonlocal_face_slots_;
  std::vector<size_t> incoming_nonlocal_face_slot_offsets_;
  std::vector<size_t> incoming_nonlocal_face_slots_by_local_face_;
};

} // namespace opensn

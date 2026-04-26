// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include "framework/mesh/cell/cell.h"
#include <boost/unordered/unordered_flat_map.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace opensn
{

class CBC_FLUDSCommonData : public FLUDSCommonData
{
public:
  /// Construct common host CBC FLUDS metadata.
  CBC_FLUDSCommonData(const SPDS& spds,
                      const std::vector<CellFaceNodalMapping>& grid_nodal_mappings);

  /// Return the number of incoming nonlocal faces.
  [[nodiscard]] size_t GetNumIncomingNonlocalFaces() const { return num_incoming_nonlocal_faces_; }

  /// Return the number of outgoing nonlocal faces.
  [[nodiscard]] size_t GetNumOutgoingNonlocalFaces() const { return num_outgoing_nonlocal_faces_; }

  /// Return the incoming nonlocal face slot for a remote cell-face pair.
  [[nodiscard]] size_t GetIncomingNonlocalFaceSlot(std::uint64_t cell_global_id,
                                                   unsigned int face_id) const;

  /// Return the incoming nonlocal face slot for a local cell-face pair.
  [[nodiscard]] size_t GetIncomingNonlocalFaceSlotByLocalFace(std::uint32_t cell_local_id,
                                                              unsigned int face_id) const;

  /// Invalid incoming nonlocal face slot sentinel.
  static constexpr size_t INVALID_FACE_SLOT = std::numeric_limits<size_t>::max();

private:
  /// Number of incoming nonlocal faces.
  size_t num_incoming_nonlocal_faces_;

  /// Number of outgoing nonlocal faces.
  size_t num_outgoing_nonlocal_faces_;

  /// Incoming nonlocal face slots keyed by remote cell-face pair.
  boost::unordered_flat_map<CellFaceKey, size_t, std::hash<CellFaceKey>>
    incoming_nonlocal_face_slots_;

  /// Incoming nonlocal face slot offsets indexed by local cell.
  std::vector<size_t> incoming_nonlocal_face_slot_offsets_;

  /// Incoming nonlocal face slot lookup indexed by local face.
  std::vector<size_t> incoming_nonlocal_face_slots_by_local_face_;
};

} // namespace opensn

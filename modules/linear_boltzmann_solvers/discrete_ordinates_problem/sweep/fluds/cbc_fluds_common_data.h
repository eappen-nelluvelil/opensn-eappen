// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace opensn
{

class CBC_FLUDSCommonData : public FLUDSCommonData
{
public:
  struct DelayedLocalFaceInfo
  {
    size_t slot_address = 0;
    size_t num_face_nodes = 0;
  };

  struct DelayedNonlocalFaceInfo
  {
    size_t prelocI = 0;
    size_t slot_address = 0;
    size_t num_face_nodes = 0;
  };

  CBC_FLUDSCommonData(const SPDS& spds,
                      const std::vector<CellFaceNodalMapping>& grid_nodal_mappings);

  size_t GetNumIncomingNonlocalFaces() const { return num_incoming_nonlocal_faces_; }

  size_t GetNumDelayedNonlocalFaces() const { return delayed_nonlocal_face_info_by_slot_.size(); }

  size_t GetNumDelayedLocalFaceNodes() const { return num_delayed_local_face_nodes_; }

  size_t GetDelayedPrelocIFaceNodeCount(size_t prelocI) const;

  size_t GetDelayedNonlocalFaceNodeCount(size_t delayed_face_slot) const;

  bool IsDelayedLocalIncomingFace(std::uint32_t cell_local_id, unsigned int face_id) const;

  bool IsDelayedLocalOutgoingFace(std::uint32_t cell_local_id, unsigned int face_id) const;

  bool IsDelayedNonlocalIncomingFace(std::uint32_t cell_local_id, unsigned int face_id) const;

  bool IsDelayedNonlocalOutgoingFace(std::uint32_t cell_local_id, unsigned int face_id) const;

  const DelayedLocalFaceInfo& GetDelayedLocalFaceInfo(std::uint32_t cell_local_id,
                                                      unsigned int face_id) const;

  const DelayedNonlocalFaceInfo& GetDelayedNonlocalFaceInfoByLocalFace(std::uint32_t cell_local_id,
                                                                       unsigned int face_id) const;

  const DelayedNonlocalFaceInfo& GetDelayedNonlocalFaceInfoBySlot(size_t delayed_face_slot) const;

  /// Return the incoming nonlocal face slot for a local cell-face pair.
  size_t GetIncomingNonlocalFaceSlotByLocalFace(std::uint32_t cell_local_id,
                                                unsigned int face_id) const;

  /// Return the downstream incoming face slot for an outgoing local cell face.
  size_t GetOutgoingNonlocalFaceSlotByLocalFace(std::uint32_t cell_local_id,
                                                unsigned int face_id) const;

  /// Return the SPDS-successor peer index for an outgoing local cell face.
  size_t GetOutgoingNonlocalFacePeerIndexByLocalFace(std::uint32_t cell_local_id,
                                                     unsigned int face_id) const;

  /// Return the destination location for an outgoing nonlocal cell face.
  int GetOutgoingNonlocalFaceLocationByLocalFace(std::uint32_t cell_local_id,
                                                 unsigned int face_id) const;

  /// Return the receiver-side face-node count for an outgoing nonlocal cell face.
  size_t GetOutgoingNonlocalFaceNodeCountByLocalFace(std::uint32_t cell_local_id,
                                                     unsigned int face_id) const;

  /// Return the local cell whose task becomes ready by an incoming nonlocal face slot.
  std::uint32_t GetIncomingNonlocalFaceLocalCell(size_t incoming_face_slot) const;

  /// Marker for cell faces without a nonlocal slot.
  static constexpr size_t INVALID_FACE_SLOT = std::numeric_limits<size_t>::max();

  /// Marker for cell faces without an outgoing nonlocal peer.
  static constexpr size_t INVALID_PEER_INDEX = std::numeric_limits<size_t>::max();

private:
  size_t num_incoming_nonlocal_faces_;
  size_t num_outgoing_nonlocal_faces_;
  size_t num_delayed_local_face_nodes_;
  /// Prefix offsets into local-face-indexed slot arrays.
  std::vector<size_t> local_face_slot_offsets_;
  /// Local-face-indexed incoming slots.
  std::vector<size_t> incoming_nonlocal_face_slots_by_local_face_;
  /// Local-face-indexed delayed incoming nonlocal slots.
  std::vector<size_t> delayed_nonlocal_face_slots_by_local_face_;
  /// Slot-indexed local cells unlocked by received payloads.
  std::vector<std::uint32_t> incoming_nonlocal_face_local_cells_;
  /// Local-face-indexed downstream incoming slots.
  std::vector<size_t> outgoing_nonlocal_face_slots_by_local_face_;
  /// Local-face-indexed SPDS-successor peer indices.
  std::vector<size_t> outgoing_nonlocal_face_peer_indices_by_local_face_;
  /// Local-face-indexed outgoing destination locations.
  std::vector<int> outgoing_nonlocal_face_locations_by_local_face_;
  /// Local-face-indexed receiver-side outgoing face-node counts.
  std::vector<size_t> outgoing_nonlocal_face_node_counts_by_local_face_;
  /// Local-face-indexed delayed local incoming face metadata.
  std::vector<DelayedLocalFaceInfo> delayed_local_face_info_by_local_face_;
  /// Delayed incoming nonlocal metadata indexed by delayed face slot.
  std::vector<DelayedNonlocalFaceInfo> delayed_nonlocal_face_info_by_slot_;
  /// Local-face-indexed delayed incoming nonlocal metadata.
  std::vector<DelayedNonlocalFaceInfo> delayed_nonlocal_face_info_by_local_face_;
  /// Per-delayed-predecessor face-node counts.
  std::vector<size_t> delayed_prelocI_face_node_counts_;
  /// Local-face-indexed delayed local incoming flags.
  std::vector<unsigned char> delayed_local_incoming_by_local_face_;
  /// Local-face-indexed delayed local outgoing flags.
  std::vector<unsigned char> delayed_local_outgoing_by_local_face_;
  /// Local-face-indexed delayed nonlocal incoming flags.
  std::vector<unsigned char> delayed_nonlocal_incoming_by_local_face_;
  /// Local-face-indexed delayed nonlocal outgoing flags.
  std::vector<unsigned char> delayed_nonlocal_outgoing_by_local_face_;
};

} // namespace opensn

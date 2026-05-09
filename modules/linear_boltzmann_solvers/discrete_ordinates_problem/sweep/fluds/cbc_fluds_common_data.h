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

  size_t GetIncomingNonlocalFaceSlotByLocalFace(std::uint32_t cell_local_id,
                                                unsigned int face_id) const;

  size_t GetOutgoingNonlocalFaceSlotByLocalFace(std::uint32_t cell_local_id,
                                                unsigned int face_id) const;

  size_t GetOutgoingNonlocalFacePeerIndexByLocalFace(std::uint32_t cell_local_id,
                                                     unsigned int face_id) const;

  int GetOutgoingNonlocalFaceLocationByLocalFace(std::uint32_t cell_local_id,
                                                 unsigned int face_id) const;

  std::uint32_t GetIncomingNonlocalFaceLocalCell(size_t incoming_face_slot) const;

  static constexpr size_t INVALID_FACE_SLOT = std::numeric_limits<size_t>::max();

  static constexpr size_t INVALID_PEER_INDEX = std::numeric_limits<size_t>::max();

private:
  size_t num_incoming_nonlocal_faces_;
  size_t num_delayed_local_face_nodes_;
  std::vector<size_t> local_face_slot_offsets_;
  std::vector<size_t> incoming_nonlocal_face_slots_by_local_face_;
  std::vector<size_t> delayed_nonlocal_face_slots_by_local_face_;
  std::vector<std::uint32_t> incoming_nonlocal_face_local_cells_;
  std::vector<size_t> outgoing_nonlocal_face_slots_by_local_face_;
  std::vector<size_t> outgoing_nonlocal_face_peer_indices_by_local_face_;
  std::vector<int> outgoing_nonlocal_face_locations_by_local_face_;
  std::vector<DelayedLocalFaceInfo> delayed_local_face_info_by_local_face_;
  std::vector<DelayedNonlocalFaceInfo> delayed_nonlocal_face_info_by_slot_;
  std::vector<DelayedNonlocalFaceInfo> delayed_nonlocal_face_info_by_local_face_;
  std::vector<size_t> delayed_prelocI_face_node_counts_;
  std::vector<unsigned char> delayed_local_incoming_by_local_face_;
  std::vector<unsigned char> delayed_local_outgoing_by_local_face_;
  std::vector<unsigned char> delayed_nonlocal_incoming_by_local_face_;
  std::vector<unsigned char> delayed_nonlocal_outgoing_by_local_face_;
};

} // namespace opensn

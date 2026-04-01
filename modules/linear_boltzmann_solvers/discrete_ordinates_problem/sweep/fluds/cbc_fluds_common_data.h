// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include <cinttypes>
#include <cstddef>
#include <unordered_map>

namespace opensn
  {

  class CBC_FLUDSCommonData : public FLUDSCommonData
  {
  public:
    struct DelayedNonlocalFaceInfo
    {
      std::uint32_t prelocI = 0;
      std::uint64_t slot_address = 0;
      std::uint32_t num_face_nodes = 0;
    };

    CBC_FLUDSCommonData(const SPDS& spds,
                        const std::vector<CellFaceNodalMapping>& grid_nodal_mappings);

    size_t GetNumIncomingNonlocalFaces() const noexcept { return num_incoming_nonlocal_faces_; }
    size_t GetNumOutgoingNonlocalFaces() const noexcept { return num_outgoing_nonlocal_faces_; }

    bool HasDelayedLocalDependencies() const noexcept { return has_delayed_local_dependencies_; }

    bool IsDelayedLocalIncomingFace(std::uint32_t cell_local_id, unsigned int face_id) const noexcept;
    bool IsDelayedLocalOutgoingFace(std::uint32_t cell_local_id, unsigned int face_id) const noexcept;
    bool IsDelayedNonlocalIncomingFace(std::uint32_t cell_local_id, unsigned int face_id) const noexcept;

    const DelayedNonlocalFaceInfo&
    GetDelayedNonlocalFaceInfo(std::uint32_t cell_local_id, unsigned int face_id) const noexcept;

    bool TryGetDelayedNonlocalFaceInfo(std::uint64_t cell_global_id,
                                       unsigned int face_id,
                                       DelayedNonlocalFaceInfo& info) const noexcept;

    size_t GetDelayedPrelocIFaceNodeCount(std::size_t prelocI) const noexcept
    {
      return delayed_preloc_face_node_count_[prelocI];
    }

  private:
    bool has_delayed_local_dependencies_ = false;
    size_t num_incoming_nonlocal_faces_ = 0;
    size_t num_outgoing_nonlocal_faces_ = 0;

    std::vector<std::vector<std::uint8_t>> delayed_local_incoming_faces_;
    std::vector<std::vector<std::uint8_t>> delayed_local_outgoing_faces_;
    std::vector<std::vector<std::uint8_t>> delayed_nonlocal_incoming_faces_;
    std::vector<std::vector<DelayedNonlocalFaceInfo>> delayed_nonlocal_face_info_by_cell_;
    std::vector<size_t> delayed_preloc_face_node_count_;
    std::unordered_map<FLUDSCommonData::CellFaceKey, DelayedNonlocalFaceInfo, FLUDSCommonData::CellFaceKeyHash>
      delayed_nonlocal_face_info_;
  };

  } // namespace opensn

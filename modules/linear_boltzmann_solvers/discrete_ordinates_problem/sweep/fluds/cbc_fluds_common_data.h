// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include <cinttypes>
#include <cstddef>
#include <limits>
#include <unordered_map>

namespace opensn
{

class SpatialDiscretization;

class CBC_FLUDSCommonData : public FLUDSCommonData
{
public:
  using CellFaceKey = std::pair<std::uint64_t, unsigned int>;

  struct CellFaceKeyHash
  {
    size_t operator()(const CellFaceKey& key) const noexcept
    {
      size_t h = std::hash<std::uint64_t>{}(key.first);
      h ^= std::hash<unsigned int>{}(key.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  struct IncomingNonlocalFaceInfo
  {
    std::uint32_t face_node_offset = 0;
    std::uint32_t num_face_nodes = 0;
  };

  struct OutgoingNonlocalFaceInfo
  {
    int locality = 0;
    std::uint64_t cell_global_id = 0;
    unsigned int associated_face = 0;
    std::uint32_t num_face_nodes = 0;
  };

  CBC_FLUDSCommonData(const SPDS& spds,
                      const std::vector<CellFaceNodalMapping>& grid_nodal_mappings,
                      const SpatialDiscretization& sdm);

  size_t GetNumIncomingNonlocalFaces() const { return num_incoming_nonlocal_faces_; }

  size_t GetNumIncomingNonlocalFaceNodes() const { return num_incoming_nonlocal_face_nodes_; }

  size_t GetNumOutgoingNonlocalFaces() const { return num_outgoing_nonlocal_faces_; }
  size_t GetDeplocIFaceCount(std::size_t deplocI) const noexcept
  {
    return outgoing_nonlocal_face_counts_[deplocI];
  }

  size_t GetDeplocIFaceNodeCount(std::size_t deplocI) const noexcept
  {
    return outgoing_nonlocal_face_node_counts_[deplocI];
  }

  const IncomingNonlocalFaceInfo& GetIncomingNonlocalFaceInfo(std::uint32_t cell_local_id,
                                                              unsigned int face_id) const noexcept;
  const IncomingNonlocalFaceInfo&
  GetIncomingNonlocalFaceInfoByKey(std::uint64_t cell_global_id,
                                   unsigned int face_id) const noexcept;
  const IncomingNonlocalFaceInfo&
  GetIncomingNonlocalFaceInfoByStorageIndex(std::size_t storage_index) const noexcept;
  std::size_t GetIncomingNonlocalFaceStorageIndexByKey(std::uint64_t cell_global_id,
                                                       unsigned int face_id) const noexcept;
  std::size_t GetNumCellFaces() const noexcept { return cell_face_offsets_.back(); }

  const OutgoingNonlocalFaceInfo& GetOutgoingNonlocalFaceInfo(std::uint32_t cell_local_id,
                                                              unsigned int face_id) const noexcept;

  size_t GetCellFaceOffset(std::uint32_t cell_local_id) const noexcept
  {
    return cell_face_offsets_[cell_local_id];
  }

  size_t GetMaxLocalOutgoingNodeCount() const noexcept { return max_local_outgoing_node_count_; }

  std::uint32_t GetLocalOutgoingCompactNodeIndex(const std::uint32_t cell_local_id,
                                                 const std::uint32_t cell_node) const noexcept;

private:
  size_t num_incoming_nonlocal_faces_;
  size_t num_incoming_nonlocal_face_nodes_;
  size_t num_outgoing_nonlocal_faces_;
  size_t max_local_outgoing_node_count_ = 0;
  std::vector<size_t> cell_face_offsets_;
  std::vector<std::uint32_t> cell_node_offsets_;
  std::vector<std::uint32_t> local_outgoing_node_indices_;
  std::vector<IncomingNonlocalFaceInfo> incoming_nonlocal_face_info_;
  std::vector<OutgoingNonlocalFaceInfo> outgoing_nonlocal_face_info_;
  std::vector<size_t> outgoing_nonlocal_face_counts_;
  std::vector<size_t> outgoing_nonlocal_face_node_counts_;
  std::unordered_map<CellFaceKey, std::size_t, CellFaceKeyHash> incoming_nonlocal_face_info_by_key_;
};

} // namespace opensn

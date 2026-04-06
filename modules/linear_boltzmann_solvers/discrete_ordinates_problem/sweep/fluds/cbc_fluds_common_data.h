// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include <cinttypes>
#include <cstddef>
#include <unordered_map>

namespace opensn
{

/**
 * Shared face-level indexing metadata for host CBC FLUDS instances.
 *
 * Precomputes and stores the mapping between local cell faces and their
 * nonlocal incoming/outgoing counterparts. Multiple CBC_FLUDS instances
 * (one per angle set) reference the same CBC_FLUDSCommonData, amortizing
 * the cost of face enumeration and hash-map construction across all angle
 * sets that share the same SPDS.
 */
class CBC_FLUDSCommonData : public FLUDSCommonData
{
public:
  /// Composite key for identifying a nonlocal face: (cell_global_id, face_id).
  using CellFaceKey = std::pair<std::uint64_t, unsigned int>;

  /// Hash functor for CellFaceKey.
  struct CellFaceKeyHash
  {
    size_t operator()(const CellFaceKey& key) const noexcept
    {
      size_t h = std::hash<std::uint64_t>{}(key.first);
      h ^= std::hash<unsigned int>{}(key.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  /// Metadata for one incoming nonlocal face.
  struct IncomingNonlocalFaceInfo
  {
    /// Offset into the incoming nonlocal psi buffer for this face's node data.
    std::uint32_t face_node_offset = 0;
    /// Number of face nodes.
    std::uint32_t num_face_nodes = 0;
  };

  /// Metadata for one outgoing nonlocal face.
  struct OutgoingNonlocalFaceInfo
  {
    /// Destination MPI rank locality index.
    int locality = 0;
    /// Global ID of the destination cell.
    std::uint64_t cell_global_id = 0;
    /// Face index on the destination cell.
    unsigned int associated_face = 0;
    /// Number of face nodes.
    std::uint32_t num_face_nodes = 0;
  };

  /**
   * Construct common data from the SPDS and grid nodal mappings.
   *
   * \param spds sweep-plane data structure providing face orientations
   * \param grid_nodal_mappings per-cell-face nodal mapping data
   */
  CBC_FLUDSCommonData(const SPDS& spds,
                      const std::vector<CellFaceNodalMapping>& grid_nodal_mappings);

  /// Total number of incoming nonlocal faces across all local cells.
  size_t GetNumIncomingNonlocalFaces() const { return num_incoming_nonlocal_faces_; }

  /// Total number of incoming nonlocal face nodes across all local cells.
  size_t GetNumIncomingNonlocalFaceNodes() const { return num_incoming_nonlocal_face_nodes_; }

  /// Total number of outgoing nonlocal faces across all local cells.
  size_t GetNumOutgoingNonlocalFaces() const { return num_outgoing_nonlocal_faces_; }

  /// Number of outgoing nonlocal faces for dependent locality \p deplocI.
  size_t GetDeplocIFaceCount(std::size_t deplocI) const noexcept
  {
    return outgoing_nonlocal_face_counts_[deplocI];
  }

  /// Number of outgoing nonlocal face nodes for dependent locality \p deplocI.
  size_t GetDeplocIFaceNodeCount(std::size_t deplocI) const noexcept
  {
    return outgoing_nonlocal_face_node_counts_[deplocI];
  }

  /// Look up incoming nonlocal face info by cell local ID and face index.
  const IncomingNonlocalFaceInfo& GetIncomingNonlocalFaceInfo(std::uint32_t cell_local_id,
                                                              unsigned int face_id) const noexcept;

  /// Look up incoming nonlocal face info by global cell ID and face index.
  const IncomingNonlocalFaceInfo&
  GetIncomingNonlocalFaceInfoByKey(std::uint64_t cell_global_id,
                                   unsigned int face_id) const noexcept;

  /// Look up incoming nonlocal face info by flat storage index.
  const IncomingNonlocalFaceInfo&
  GetIncomingNonlocalFaceInfoByStorageIndex(std::size_t storage_index) const noexcept;

  /// Resolve a (cell_global_id, face_id) pair to a flat storage index.
  std::size_t GetIncomingNonlocalFaceStorageIndexByKey(std::uint64_t cell_global_id,
                                                       unsigned int face_id) const noexcept;

  /// Total number of cell-face entries in the flat face table.
  std::size_t GetNumCellFaces() const noexcept { return cell_face_offsets_.back(); }

  /// Look up outgoing nonlocal face info by cell local ID and face index.
  const OutgoingNonlocalFaceInfo& GetOutgoingNonlocalFaceInfo(std::uint32_t cell_local_id,
                                                              unsigned int face_id) const noexcept;

  /// Flat face-table offset for a given cell.
  size_t GetCellFaceOffset(std::uint32_t cell_local_id) const noexcept
  {
    return cell_face_offsets_[cell_local_id];
  }

private:
  /// Total incoming nonlocal faces.
  size_t num_incoming_nonlocal_faces_;
  /// Total incoming nonlocal face nodes.
  size_t num_incoming_nonlocal_face_nodes_;
  /// Total outgoing nonlocal faces.
  size_t num_outgoing_nonlocal_faces_;
  /// Prefix-sum offsets into the flat face tables, indexed by cell_local_id.
  std::vector<size_t> cell_face_offsets_;
  /// Flat incoming nonlocal face metadata, indexed by face storage index.
  std::vector<IncomingNonlocalFaceInfo> incoming_nonlocal_face_info_;
  /// Flat outgoing nonlocal face metadata, indexed by face storage index.
  std::vector<OutgoingNonlocalFaceInfo> outgoing_nonlocal_face_info_;
  /// Per-dependent-locality outgoing face counts.
  std::vector<size_t> outgoing_nonlocal_face_counts_;
  /// Per-dependent-locality outgoing face node counts.
  std::vector<size_t> outgoing_nonlocal_face_node_counts_;
  /// Hash map from (cell_global_id, face_id) to flat storage index.
  std::unordered_map<CellFaceKey, std::size_t, CellFaceKeyHash> incoming_nonlocal_face_info_by_key_;
};

} // namespace opensn

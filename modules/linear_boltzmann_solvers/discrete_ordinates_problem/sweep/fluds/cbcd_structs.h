// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_structs.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace opensn
{

/**
 * Bit-packed face-node index for CBCD FLUDS device kernels.
 *
 * Encodes the buffer identity and offset into a single 64-bit word so the GPU
 * kernel can resolve any face-node's angular flux pointer with one branch chain
 * (see GetIncomingFluxPointer / GetOutgoingFluxPointer).
 *
 * Layout (MSB → LSB):
 *   Bit 63   — direction  (0 = incoming, 1 = outgoing)
 *   Bit 62   — boundary   (1 = boundary buffer, 0 = local or non-local)
 *   Bit 61   — locality   (1 = local psi buffer, 0 = non-local buffer)
 *   Bits 0-60 — index into the selected buffer (capacity ~2.3 × 10^18)
 *
 * Does not support delayed nodes (reclaims the delayed bit used by AAHD).
 */
class CBCD_NodeIndex : public NodeIndex
{
public:
  constexpr CBCD_NodeIndex() = default;

  /// Construct from a raw 64-bit value (e.g. read back from device memory).
  constexpr CBCD_NodeIndex(const std::uint64_t& value) : NodeIndex(value) {}

  /// Construct a non-boundary node index (local or non-local).
  CBCD_NodeIndex(std::uint64_t index, bool is_outgoing, bool is_local)
  {
    if (index >= (std::uint64_t(1) << 61) - 1)
      throw std::runtime_error("Cannot hold an index greater than 2^61.");
    SetInOut(is_outgoing);
    SetLocal(is_local);
    SetBoundary(false);
    SetIndex(index);
  }

  /// Construct a boundary node index (always treated as local for buffer selection).
  CBCD_NodeIndex(std::uint64_t index, bool is_outgoing)
  {
    if (index >= (std::uint64_t(1) << 61) - 1)
      throw std::runtime_error("Cannot hold an index greater than 2^61.");
    SetInOut(is_outgoing);
    SetLocal(true);
    SetBoundary(true);
    SetIndex(index);
  }

  constexpr bool IsLocal() const noexcept { return (value_ & local_bit_mask) != 0; }
  constexpr std::uint64_t GetIndex() const noexcept { return value_ & index_bit_mask; }

private:
  static constexpr std::uint64_t local_bit_mask = std::uint64_t(1) << (64 - 3);
  static constexpr std::uint64_t index_bit_mask = (std::uint64_t(1) << (64 - 3)) - 1;

  constexpr void SetLocal(bool is_local) noexcept
  {
    if (is_local)
      value_ |= local_bit_mask;
    else
      value_ &= ~local_bit_mask;
  }

  constexpr void SetIndex(std::uint64_t index) noexcept
  {
    value_ &= ~index_bit_mask;
    value_ |= (index & index_bit_mask);
  }
};

/**
 * Device pointer set for CBCD FLUDS angular flux buffers.
 *
 * Extends FLUDSPointerSet with incoming/outgoing boundary pointers.  Passed
 * by value to GPU kernels; the GetIncomingFluxPointer / GetOutgoingFluxPointer
 * methods decode a CBCD_NodeIndex and return the corresponding buffer address.
 */
struct CBCD_FLUDSPointerSet : public FLUDSPointerSet
{
  double* __restrict__ incoming_boundary_psi = nullptr;
  double* __restrict__ outgoing_boundary_psi = nullptr;

  /// Resolve the incoming flux pointer for a face node (nullptr if outgoing or undefined).
  constexpr double* GetIncomingFluxPointer(const CBCD_NodeIndex& node_index) const noexcept
  {
    if (node_index.IsUndefined() or node_index.IsOutgoing())
      return nullptr;
    if (node_index.IsBoundary())
      return incoming_boundary_psi + node_index.GetIndex() * stride_size;
    if (node_index.IsLocal())
      return local_psi + node_index.GetIndex() * stride_size;
    return nonlocal_incoming_psi + node_index.GetIndex() * stride_size;
  }

  /// Resolve the outgoing flux pointer for a face node (nullptr if incoming or undefined).
  constexpr double* GetOutgoingFluxPointer(const CBCD_NodeIndex& node_index) const noexcept
  {
    if (node_index.IsUndefined() or not node_index.IsOutgoing())
      return nullptr;
    if (node_index.IsBoundary())
      return outgoing_boundary_psi + node_index.GetIndex() * stride_size;
    if (node_index.IsLocal())
      return local_psi + node_index.GetIndex() * stride_size;
    return nonlocal_outgoing_psi + node_index.GetIndex() * stride_size;
  }
};

/**
 * Host-side metadata for a single boundary face node.
 *
 * Used by CBCD_FLUDS to copy angular flux between host boundary buffers and
 * the sweep boundary conditions (reflecting or surface-source).
 *
 * Stored in flat vectors with per-cell span tables inside
 * CBCD_FLUDSCommonData. Sized to 24 bytes (down from 40) by narrowing fields
 * that never exceed 32- or 16-bit range.
 */
struct BoundaryNodeInfo
{
  std::uint64_t boundary_id;   ///< Boundary condition ID (key into the boundary map).
  std::uint32_t cell_local_id; ///< Local cell index (needed in flat incoming_boundary_node_map_).
  unsigned int face_id;        ///< Face index within the cell.
  std::uint32_t storage_index; ///< Offset into the boundary psi MappedHostVector.
  std::uint16_t face_node;     ///< Face-node index within the face.
};

/// Receive-side key for one incoming nonlocal face.
struct IncomingFaceKey
{
  std::uint64_t cell_global_id = 0;
  unsigned int face_id = 0;

  bool operator==(const IncomingFaceKey&) const = default;
};

/// Hash for `IncomingFaceKey`.
struct IncomingFaceKeyHash
{
  std::size_t operator()(const IncomingFaceKey& key) const noexcept
  {
    const auto h0 = std::hash<std::uint64_t>{}(key.cell_global_id);
    const auto h1 = std::hash<unsigned int>{}(key.face_id);
    return h0 ^ (h1 + 0x9e3779b97f4a7c15ULL + (h0 << 6) + (h0 >> 2));
  }
};

/// Grouped incoming nonlocal face.
struct GroupedIncomingNonlocalFace
{
  std::uint32_t cell_local_id = 0;      ///< Receiving local cell identifier.
  std::uint32_t base_storage_index = 0; ///< Base offset in the incoming nonlocal psi buffer.
  int source_partition = 0;             ///< Source partition for this incoming face.
  std::uint16_t num_nodes = 0;          ///< Number of nodes on this face.
};

/// Outgoing node-copy descriptor.
struct OutgoingNodeCopy
{
  std::uint32_t storage_index = 0; ///< Source offset in the outgoing nonlocal psi buffer.
  std::uint16_t face_node = 0;     ///< Destination face-node in the receiver-local payload layout.
};

/// Grouped outgoing nonlocal face.
struct GroupedOutgoingNonlocalFace
{
  std::array<std::byte, sizeof(std::uint64_t) + sizeof(unsigned int)> entry_header_prefix{};
  std::uint32_t pack_plan_index = 0;  ///< Stable index into angle-set-local outgoing pack plans.
  std::uint32_t dest_slot = 0;        ///< Destination slot in the outgoing locality table.
  std::uint16_t num_face_nodes = 0;   ///< Number of nodes on this face.
  std::uint32_t node_copy_offset = 0; ///< Offset into the flat outgoing-node-copy array.
  std::uint16_t num_node_copies = 0;  ///< Number of node-copy descriptors.
};

/// Reflecting-boundary face copy plan.
struct ReflectingBoundaryFacePlan
{
  std::uint64_t boundary_id = 0;     ///< Boundary identifier.
  std::uint32_t cell_local_id = 0;   ///< Local cell identifier.
  unsigned int face_id = 0;          ///< Face identifier on the local cell.
  std::uint16_t first_face_node = 0; ///< First face-node index on the reflecting face.
  std::size_t src_base_offset = 0;   ///< Base source offset in doubles from `outgoing_boundary_psi_`.
  std::uint16_t num_nodes = 0;       ///< Number of nodes on the reflecting face.
};

/// Outgoing node-copy plan entry.
struct OutgoingNodeMemcpy
{
  std::size_t src_offset = 0; ///< Source offset in doubles from `outgoing_nonlocal_psi_`.
  std::size_t dst_offset = 0; ///< Destination offset in doubles from the packed face payload base.
};

} // namespace opensn

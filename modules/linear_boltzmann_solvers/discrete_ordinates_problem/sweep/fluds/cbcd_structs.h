// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_structs.h"

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

} // namespace opensn

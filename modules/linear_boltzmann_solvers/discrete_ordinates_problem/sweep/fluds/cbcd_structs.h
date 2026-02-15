// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_structs.h"

namespace opensn
{

/**
 * Node index specific to CBCD FLUDS.
 * Does not support delayed nodes. Reclaims the delayed bit for indices.
 * - Bit 63: Incoming/outgoing bit.
 * - Bit 62: Boundary bit.
 * - Bit 61: Local bit.
 * - Bits 0-60: Index bits (capacity ~2.3e18).
 */
class CBCD_NodeIndex : public NodeIndex
{
public:
  /// Default constructor.
  constexpr CBCD_NodeIndex() = default;

  /// Direct assign core value.
  constexpr CBCD_NodeIndex(const std::uint64_t& value) : NodeIndex(value) {}

  /**
   * Construct a non-boundary node index.
   * \param index Index into the corresponding bank. Cannot exceed 2^61 - 1.
   * \param is_outgoing Flag indicating if the node corresponds to an outgoing face.
   * \param is_local Flag indicating if the index is in a local bank.
   */
  CBCD_NodeIndex(std::uint64_t index, bool is_outgoing, bool is_local)
  {
    if (index >= (std::uint64_t(1) << 61) - 1)
      throw std::runtime_error("Cannot hold an index greater than 2^61.");
    SetInOut(is_outgoing);
    SetLocal(is_local);
    SetBoundary(false);
    SetIndex(index);
  }

  /**
   * Construct a boundary node index.
   * \param index Index into the corresponding bank. Cannot exceed 2^61 - 1.
   * \param is_outgoing Flag indicating if the node corresponds to an outgoing face.
   */
  CBCD_NodeIndex(std::uint64_t index, bool is_outgoing)
  {
    if (index >= (std::uint64_t(1) << 61) - 1)
      throw std::runtime_error("Cannot hold an index greater than 2^61.");
    SetInOut(is_outgoing);
    SetLocal(true);
    SetBoundary(true);
    SetIndex(index);
  }

  /// Construct a local (non-boundary) node index from cell_local_id and node.
  CBCD_NodeIndex(std::uint64_t cell_local_id, std::uint32_t node, bool is_outgoing)
  {
    // cell_local_id must fit in 53 bits, node must fit in 8 bits
    if (cell_local_id >= (std::uint64_t(1) << 53))
      throw std::runtime_error("cell_local_id exceeds 53-bit capacity.");
    if (node >= 256)
      throw std::runtime_error("node index exceeds 8-bit capacity.");
    SetInOut(is_outgoing);
    SetLocal(true);
    SetBoundary(false);
    SetIndex((cell_local_id << 8) | static_cast<std::uint64_t>(node));
  }

  /// Extract cell_local_id from a local node index (bits 8-60 of index field).
  constexpr std::uint32_t GetCellLocalId() const noexcept
  {
    return static_cast<std::uint32_t>((value_ & index_bit_mask) >> 8);
  }

  /// Extract node index from a local node index (bits 0-7 of index field).
  constexpr std::uint32_t GetNode() const noexcept
  {
    return static_cast<std::uint32_t>((value_ & index_bit_mask) & 0xFF);
  }

  /// Check if the current index corresponds to a local bank.
  constexpr bool IsLocal() const noexcept { return (value_ & local_bit_mask) != 0; }

  /// Get the index into the bank.
  constexpr std::uint64_t GetIndex() const noexcept { return value_ & index_bit_mask; }

private:
  /// \name Local bit
  /// \{
  /// Third bit mask (``001`` followed by 61 zeros) - Bit 61.
  static constexpr std::uint64_t local_bit_mask = std::uint64_t(1) << (64 - 3);
  /// Encode the value as local.
  constexpr void SetLocal(bool is_local) noexcept
  {
    if (is_local)
      value_ |= local_bit_mask;
    else
      value_ &= ~local_bit_mask;
  }
  /// \}

  /// \name Index bits
  /// \{
  /// Index bit mask (``1`` at the last 61 bits).
  static constexpr std::uint64_t index_bit_mask = (std::uint64_t(1) << (64 - 3)) - 1;
  /// Encode the index.
  constexpr void SetIndex(std::uint64_t index) noexcept
  {
    value_ &= ~index_bit_mask;
    value_ |= (index & index_bit_mask);
  }
  /// \}
};

/**
 * Set of device pointers to local, boundary, and non-local buffers for CBCD FLUDS.
 */
struct CBCD_FLUDSPointerSet : public FLUDSPointerSet
{
  /// Pointer to incoming boundary angular fluxes.
  double* __restrict__ incoming_boundary_psi = nullptr;
  /// Pointer to outgoing boundary angular fluxes.
  double* __restrict__ outgoing_boundary_psi = nullptr;
  /// Device-visible slot indirection table: cell_local_id -> slot_id.
  /// Backed by crb::MappedHostVector on host (zero-copy).
  const std::uint32_t* __restrict__ slot_map = nullptr;
  /// Number of DOFs per slot.
  std::uint32_t dofs_per_slot = 0;

  /// Get pointer to the incoming angular flux (if the face is not incoming, a nullptr is returned).
  constexpr double* GetIncomingFluxPointer(const CBCD_NodeIndex& node_index) const noexcept
  {
    // Undefined case (corresponds to a parallel face)
    if (node_index.IsUndefined())
      return nullptr;

    // Outgoing case : nullptr
    if (node_index.IsOutgoing())
      return nullptr;

    // Incoming boundary case
    if (node_index.IsBoundary())
    {
      return incoming_boundary_psi + node_index.GetIndex() * stride_size;
    }
    // Incoming local case
    if (node_index.IsLocal())
    {
      // Slot indirection: look up which pool slot this cell currently occupies
      std::uint32_t cell_local_id = node_index.GetCellLocalId();
      std::uint32_t node = node_index.GetNode();
      std::uint32_t slot = slot_map[cell_local_id];
      return local_psi + (static_cast<std::uint64_t>(slot) * dofs_per_slot + node) * stride_size;
    }
    // Incoming non-local case
    else
    {
      return nonlocal_incoming_psi + node_index.GetIndex() * stride_size;
    }
  }

  /// Get pointer to the outgoing angular flux (if the face is not outgoing, a nullptr is returned).
  constexpr double* GetOutgoingFluxPointer(const CBCD_NodeIndex& node_index) const noexcept
  {
    // Undefined case (corresponds to a parallel face)
    if (node_index.IsUndefined())
      return nullptr;

    // Incoming case : nullptr
    if (!node_index.IsOutgoing())
      return nullptr;

    // Outgoing boundary case
    if (node_index.IsBoundary())
    {
      return outgoing_boundary_psi + node_index.GetIndex() * stride_size;
    }
    // Outgoing local case
    if (node_index.IsLocal())
    {
      std::uint32_t cell_local_id = node_index.GetCellLocalId();
      std::uint32_t node = node_index.GetNode();
      std::uint32_t slot = slot_map[cell_local_id];
      return local_psi + (static_cast<std::uint64_t>(slot) * dofs_per_slot + node) * stride_size;
    }
    // Outgoing non-local case
    else
    {
      return nonlocal_outgoing_psi + node_index.GetIndex() * stride_size;
    }
  }
};

/**
 * Metadata for boundary face nodes.
 */
struct BoundaryNodeInfo
{
  std::uint64_t cell_local_id;
  unsigned int face_id;
  size_t face_node;
  std::uint64_t storage_index;
  std::uint64_t boundary_id;
};

/**
 * Metadata for non-local face nodes.
 */
struct NonlocalNodeInfo
{
  std::uint64_t cell_local_id;
  std::uint64_t cell_global_id;
  unsigned int face_id;
  size_t face_node;
  short face_node_mapped;
  std::uint64_t storage_index;
};

} // namespace opensn
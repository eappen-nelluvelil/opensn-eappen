// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_structs.h"
#include <array>
#include <cstddef>
#include <functional>

namespace opensn
{

class SweepBoundary;

/**
 * Packed 64-bit angular flux buffer index for CBCD FLUDS.
 *
 * Encodes the buffer type (local/boundary/non-local, incoming/outgoing) and address into a
 * single 64-bit value. For boundary nodes, the local bit denotes reflecting boundaries and the
 * delayed bit denotes angle-dependent boundary storage.
 *
 * Bit layout:
 * - Bit 63: incoming (0) / outgoing (1).
 * - Bit 62: boundary (1) / non-boundary (0).
 * - Bit 61: local or reflecting boundary (1) / non-local or non-reflecting boundary (0).
 * - Bit 60: angle-dependent boundary (1) / angle-independent boundary (0).
 * - For local non-boundary nodes:
 *   - Bits 0-59: flat local-face-slot node bank index.
 * - For boundary or non-local nodes:
 *   - Bits 0-59: flat bank index.
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
   * \param index Index into the corresponding bank. Cannot exceed 2^60 - 1.
   * \param is_outgoing Flag indicating if the node corresponds to an outgoing face.
   * \param is_local Flag indicating if the index is in a local bank.
   */
  CBCD_NodeIndex(std::uint64_t index, bool is_outgoing, bool is_local)
  {
    if (index >= (std::uint64_t(1) << 60) - 1)
      throw std::runtime_error("Cannot hold an index greater than 2^60.");
    SetInOut(is_outgoing);
    SetLocal(is_local);
    SetAngleDependent(false);
    SetBoundary(false);
    SetIndex(index);
  }

  /**
   * Construct a boundary node index.
   * \param index Index into the corresponding bank. Cannot exceed 2^60 - 1.
   * \param is_outgoing Flag indicating if the node corresponds to an outgoing face.
   * \param is_reflecting Flag indicating if the boundary is reflecting.
   * \param is_angle_dependent Flag indicating if the boundary storage is angle dependent.
   */
  CBCD_NodeIndex(std::uint64_t index,
                 bool is_outgoing,
                 bool is_reflecting,
                 bool is_angle_dependent)
  {
    if (index >= (std::uint64_t(1) << 60) - 1)
      throw std::runtime_error("Cannot hold an index greater than 2^60.");
    SetInOut(is_outgoing);
    SetLocal(is_reflecting);
    SetAngleDependent(is_angle_dependent);
    SetBoundary(true);
    SetIndex(index);
  }

  /// Check if the current index corresponds to a local bank.
  constexpr bool IsLocal() const noexcept { return (value_ & local_bit_mask) != 0; }
  /// Check if the current boundary index corresponds to a reflecting boundary.
  constexpr bool IsReflecting() const noexcept { return IsLocal(); }
  /// Check if the current boundary index corresponds to angle-dependent storage.
  constexpr bool IsAngleDependent() const noexcept
  {
    return (value_ & angle_dependent_bit_mask) != 0;
  }

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

  /// \name Angle-dependent boundary bit
  /// \{
  /// Fourth bit mask (``0001`` followed by 60 zeros) - Bit 60.
  static constexpr std::uint64_t angle_dependent_bit_mask = std::uint64_t(1) << (64 - 4);
  /// Encode the value as angle-dependent boundary storage.
  constexpr void SetAngleDependent(bool is_angle_dependent) noexcept
  {
    if (is_angle_dependent)
      value_ |= angle_dependent_bit_mask;
    else
      value_ &= ~angle_dependent_bit_mask;
  }
  /// \}

  /// \name Index bits
  /// \{
  /// Index bit mask (``1`` at the last 60 bits).
  static constexpr std::uint64_t index_bit_mask = (std::uint64_t(1) << (64 - 4)) - 1;
  /// Encode the index.
  constexpr void SetIndex(std::uint64_t index) noexcept
  {
    value_ &= ~index_bit_mask;
    value_ |= (index & index_bit_mask);
  }
  /// \}
};

/**
 * Set of device pointers to local and non-local CBCD FLUDS buffers.
 */
struct CBCD_FLUDSPointerSet : public FLUDSPointerSet
{
  /// Get pointer to the incoming angular flux (if the face is not incoming, a nullptr is returned).
  constexpr double* GetIncomingFluxPointer(const CBCD_NodeIndex& node_index,
                                           unsigned int angle_group_idx,
                                           unsigned int group_idx,
                                           double* __restrict__ boundary,
                                           const std::uint64_t* __restrict__ boundary_offset,
                                           bool is_surface_source_active) const noexcept
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
      if (node_index.IsReflecting() or is_surface_source_active)
      {
        const unsigned int location =
          node_index.IsAngleDependent() ? angle_group_idx : group_idx;
        return boundary + boundary_offset[node_index.GetIndex()] + location;
      }
      return boundary + group_idx;
    }
    // Incoming local case
    if (node_index.IsLocal())
    {
      return local_psi + node_index.GetIndex() * stride_size + angle_group_idx;
    }
    // Incoming non-local case
    else
    {
      return nonlocal_incoming_psi + node_index.GetIndex() * stride_size + angle_group_idx;
    }
  }

  /// Get pointer to the outgoing angular flux (if the face is not outgoing, a nullptr is returned).
  constexpr double* GetOutgoingFluxPointer(const CBCD_NodeIndex& node_index,
                                           unsigned int angle_group_idx,
                                           double* __restrict__ boundary,
                                           const std::uint64_t* __restrict__ boundary_offset) const
    noexcept
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
      if (node_index.IsReflecting())
        return boundary + boundary_offset[node_index.GetIndex()] + angle_group_idx;
      return nullptr;
    }
    // Outgoing local case
    if (node_index.IsLocal())
    {
      return local_psi + node_index.GetIndex() * stride_size + angle_group_idx;
    }
    // Outgoing non-local case
    else
    {
      return nonlocal_outgoing_psi + node_index.GetIndex() * stride_size + angle_group_idx;
    }
  }
};

/**
 * Metadata for boundary face nodes.
 */
struct BoundaryNodeInfo
{
  std::uint64_t boundary_id = 0;
  std::uint32_t cell_local_id = 0;
  unsigned int face_id = 0;
  std::uint32_t storage_index = 0;
  std::uint16_t face_node = 0;
};

/// Grouped incoming-boundary face copy plan.
struct IncomingBoundaryFacePlan
{
  std::uint64_t boundary_id = 0;
  std::uint32_t cell_local_id = 0;
  unsigned int face_id = 0;
  std::uint16_t first_face_node = 0;
  std::uint32_t base_storage_index = 0;
  std::uint16_t num_nodes = 0;
};

/// Grouped incoming non-local face.
struct GroupedIncomingNonlocalFace
{
  std::uint32_t cell_local_id = 0;
  std::uint32_t base_storage_index = 0;
  std::uint32_t source_slot = 0;
  std::uint16_t num_nodes = 0;
};

/// Outgoing node-copy descriptor
struct OutgoingNodeCopy
{
  std::uint32_t storage_index = 0;
  std::uint16_t face_node = 0;
};

/// Grouped outgoing non-local face.
struct GroupedOutgoingNonlocalFace
{
  std::uint32_t dest_slot = 0;
  std::uint32_t remote_face_index = 0;
  std::uint32_t node_copy_offset = 0;
  std::uint16_t num_face_nodes = 0;
  std::uint16_t num_node_copies = 0;
};

/// Reflecting-boundary face copy plan.
struct ReflectingBoundaryFacePlan
{
  SweepBoundary* boundary = nullptr;
  std::uint32_t cell_local_id = 0;
  unsigned int face_id = 0;
  std::uint16_t first_face_node = 0;
  std::size_t src_base_offset = 0;
  std::uint16_t num_nodes = 0;
};

/// Outgoing node-copy plan entry.
struct OutgoingNodeMemcpy
{
  std::size_t src_offset = 0;
  std::size_t dest_offset = 0;
};

} // namespace opensn

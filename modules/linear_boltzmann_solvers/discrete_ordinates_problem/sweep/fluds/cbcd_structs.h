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
 * Encodes the buffer type (local/boundary/non-local, incoming/outgoing, delayed)
 * and address into a single 64-bit value.
 *
 * Bit layout:
 * - Bit 63: incoming (0) / outgoing (1).
 * - Bit 62: boundary (1) / non-boundary (0).
 * - Bit 61: delayed (1) / non-delayed (0).
 * - Bit 60: local (1) / non-local (0).
 * - Bits 0-59: bank index.
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
   * \param is_delayed Flag indicating if the index is in a delayed bank.
   */
  CBCD_NodeIndex(std::uint64_t index, bool is_outgoing, bool is_local, bool is_delayed)
  {
    if (index >= (std::uint64_t(1) << 60) - 1)
      throw std::runtime_error("Cannot hold an index greater than 2^60.");
    SetInOut(is_outgoing);
    SetDelayed(is_delayed);
    SetLocal(is_local);
    SetBoundary(false);
    SetIndex(index);
    if (is_delayed && is_outgoing && !is_local)
      throw std::runtime_error(
        "Non-local outgoing nodes cannot be stored in delayed banks for CBCD FLUDS.");
  }

  /**
   * Construct a boundary node index.
   * \param index Index into the corresponding bank. Cannot exceed 2^60 - 1.
   * \param is_outgoing Flag indicating if the node corresponds to an outgoing face.
   */
  CBCD_NodeIndex(std::uint64_t index, bool is_outgoing)
  {
    if (index >= (std::uint64_t(1) << 60) - 1)
      throw std::runtime_error("Cannot hold an index greater than 2^60.");
    SetInOut(is_outgoing);
    SetDelayed(false);
    SetLocal(true);
    SetBoundary(true);
    SetIndex(index);
  }

  /// Check if the current index corresponds to a delayed bank.
  constexpr bool IsDelayed() const noexcept { return (value_ & delayed_bit_mask) != 0; }

  /// Check if the current index corresponds to a local bank.
  constexpr bool IsLocal() const noexcept { return (value_ & local_bit_mask) != 0; }

  /// Get the index into the bank.
  constexpr std::uint64_t GetIndex() const noexcept { return value_ & index_bit_mask; }

private:
  /// \name Delayed bit
  /// \{
  static constexpr std::uint64_t delayed_bit_mask = std::uint64_t(1) << (64 - 3);
  constexpr void SetDelayed(bool is_delayed) noexcept
  {
    if (is_delayed)
      value_ |= delayed_bit_mask;
    else
      value_ &= ~delayed_bit_mask;
  }
  /// \}

  /// \name Local bit
  /// \{
  /// Fourth bit mask (``0001`` followed by 60 zeros) - Bit 60.
  static constexpr std::uint64_t local_bit_mask = std::uint64_t(1) << (64 - 4);
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
 * Set of device pointers to local, boundary, and non-local buffers for CBCD FLUDS.
 */
struct CBCD_FLUDSPointerSet : public FLUDSPointerSet
{
  /// Pointer to delayed local angular fluxes.
  double* __restrict__ delayed_local_psi = nullptr;
  /// Pointer to old delayed local angular fluxes.
  double* __restrict__ delayed_local_psi_old = nullptr;
  /// Pointer to incoming boundary angular fluxes.
  double* __restrict__ incoming_boundary_psi = nullptr;
  /// Pointer to outgoing boundary angular fluxes.
  double* __restrict__ outgoing_boundary_psi = nullptr;
  /// Pointer to old delayed incoming non-local angular fluxes.
  double* __restrict__ nonlocal_delayed_incoming_psi_old = nullptr;

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
      return (node_index.IsDelayed() ? delayed_local_psi_old : local_psi) +
             node_index.GetIndex() * stride_size;
    }
    // Incoming non-local case
    else
    {
      return (node_index.IsDelayed() ? nonlocal_delayed_incoming_psi_old : nonlocal_incoming_psi) +
             node_index.GetIndex() * stride_size;
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
      return (node_index.IsDelayed() ? delayed_local_psi : local_psi) +
             node_index.GetIndex() * stride_size;
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

/// Source-partition-local incoming face lookup record.
struct IncomingFaceLookup
{
  std::uint64_t cell_global_id = 0;
  unsigned int face_id = 0;
  std::uint32_t face_index = 0;
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
  std::uint64_t cell_global_id = 0;
  std::uint32_t dest_slot = 0;
  std::uint32_t node_copy_offset = 0;
  unsigned int face_id = 0;
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

// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_structs.h"
#include <array>
#include <cstddef>
#include <functional>

namespace opensn
{

/**
 * Node index specific to CBCD FLUDS.
 * Does not support delayed nodes. Reclaims the delayed bit for indices.
 * - Bit 63: Incoming/outgoing bit.
 * - Bit 62: Boundary bit.
 * - Bit 61: Local bit.
 * - For local nodes:
 *   - Bits 16-60: Cell local id.
 *   - Bits 0-15: Cell-local node index.
 * - For boundary/nonlocal nodes:
 *   - Bits 0-60: Bank index.
 */
class CBCD_NodeIndex : public NodeIndex
{
public:
  static constexpr std::uint32_t kLocalNodeBits = 16;
  static constexpr std::uint64_t kLocalNodeMask = (std::uint64_t(1) << kLocalNodeBits) - 1;

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
   * Construct a local node index.
   * \param cell_local_id Local cell id.
   * \param cell_node Local node id within the cell.
   * \param is_outgoing Flag indicating if the node corresponds to an outgoing face.
   */
  CBCD_NodeIndex(std::uint32_t cell_local_id, std::uint16_t cell_node, bool is_outgoing)
  {
    SetInOut(is_outgoing);
    SetLocal(true);
    SetBoundary(false);
    SetLocalCellNode(cell_local_id, cell_node);
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

  /// Check if the current index corresponds to a local bank.
  constexpr bool IsLocal() const noexcept { return (value_ & local_bit_mask) != 0; }

  /// Get the index into the bank.
  constexpr std::uint64_t GetIndex() const noexcept { return value_ & index_bit_mask; }
  /// Get local cell id for a local node index.
  constexpr std::uint32_t GetCellLocalID() const noexcept
  {
    return static_cast<std::uint32_t>((value_ & index_bit_mask) >> kLocalNodeBits);
  }
  /// Get local node id within a local cell.
  constexpr std::uint16_t GetCellNode() const noexcept
  {
    return static_cast<std::uint16_t>(value_ & kLocalNodeMask);
  }

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
  constexpr void SetLocalCellNode(std::uint32_t cell_local_id, std::uint16_t cell_node) noexcept
  {
    SetIndex((static_cast<std::uint64_t>(cell_local_id) << kLocalNodeBits) |
             static_cast<std::uint64_t>(cell_node));
  }
  /// \}
};

/**
 * Set of device pointers to local, boundary, and non-local buffers for CBCD FLUDS.
 */
struct CBCD_FLUDSPointerSet : public FLUDSPointerSet
{
  /// Pointer to local cell slot offsets, in node units.
  const std::uint32_t* __restrict__ local_slot_offsets = nullptr;
  /// Pointer to per-cell offsets into the compact local-node map.
  const std::uint32_t* __restrict__ local_cell_node_offsets = nullptr;
  /// Pointer to compact local-node indices for each cell node.
  const std::uint32_t* __restrict__ local_compact_node_indices = nullptr;
  /// Pointer to incoming boundary angular fluxes.
  double* __restrict__ incoming_boundary_psi = nullptr;
  /// Pointer to outgoing boundary angular fluxes.
  double* __restrict__ outgoing_boundary_psi = nullptr;

  constexpr double* GetLocalCellFluxBase(const std::uint32_t cell_local_id) const noexcept
  {
    return local_psi + static_cast<std::size_t>(local_slot_offsets[cell_local_id]) * stride_size;
  }

  constexpr std::uint32_t GetCompactLocalNodeIndex(const std::uint32_t cell_local_id,
                                                   const std::uint16_t cell_node) const noexcept
  {
    const auto index_offset = local_cell_node_offsets[cell_local_id] + static_cast<std::uint32_t>(cell_node);
    return local_compact_node_indices[index_offset];
  }

  constexpr double* GetLocalFluxPointer(double* local_cell_base,
                                        const std::uint32_t cell_local_id,
                                        const std::uint16_t cell_node) const noexcept
  {
    return local_cell_base +
           static_cast<std::size_t>(GetCompactLocalNodeIndex(cell_local_id, cell_node)) *
             stride_size;
  }

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
      return GetLocalFluxPointer(GetLocalCellFluxBase(node_index.GetCellLocalID()),
                                 node_index.GetCellLocalID(),
                                 node_index.GetCellNode());
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
      return GetLocalFluxPointer(GetLocalCellFluxBase(node_index.GetCellLocalID()),
                                 node_index.GetCellLocalID(),
                                 node_index.GetCellNode());
    }
    // Outgoing non-local case
    else
    {
      return nonlocal_outgoing_psi + node_index.GetIndex() * stride_size;
    }
  }
};

/**
 * Mutable device-visible CBCD task-state view.
 */
struct CBCD_TaskStateView
{
  int* __restrict__ remaining_dependencies = nullptr;
  std::uint32_t* __restrict__ remaining_successors_to_retire = nullptr;
  std::uint32_t* __restrict__ ready_task_indices = nullptr;
  std::uint32_t* __restrict__ ready_task_count = nullptr;
  std::uint32_t num_tasks = 0;
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
 * Receive-side key for one incoming nonlocal face.
 */
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
  std::uint32_t cell_local_id = 0;
  std::uint32_t base_storage_index = 0;
  int source_partition = 0;
  std::uint16_t num_nodes = 0;
};

/// Outgoing node-copy descriptor.
struct OutgoingNodeCopy
{
  std::uint32_t storage_index = 0;
  std::uint16_t face_node = 0;
};

/// Grouped outgoing nonlocal face.
struct GroupedOutgoingNonlocalFace
{
  std::array<std::byte, sizeof(std::uint64_t) + sizeof(unsigned int)> entry_header_prefix{};
  std::uint32_t pack_plan_index = 0;
  std::uint32_t dest_slot = 0;
  std::uint16_t num_face_nodes = 0;
  std::uint32_t node_copy_offset = 0;
  std::uint16_t num_node_copies = 0;
};

/// Reflecting-boundary face copy plan.
struct ReflectingBoundaryFacePlan
{
  std::uint64_t boundary_id = 0;
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
  std::size_t dst_offset = 0;
};

} // namespace opensn

// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_structs.h"
#include <cstddef>
#include <cstdint>

namespace opensn
{

class SweepBoundary;

/**
 * Packed 64-bit angular flux buffer index for CBCD FLUDS.
 *
 * Encodes the buffer type and address into a single 64-bit value.
 *
 * Bit layout:
 * - Bit 63: incoming (0) / outgoing (1).
 * - Bit 62: boundary (1) / non-boundary (0).
 * - Bit 61: local (1) / non-local (0).
 * - Bit 60: when non-boundary, delayed flux flag (0 normal, 1 delayed).
 *           When boundary, reserved (currently unused).
 * - Bits 0-59: flat bank index (capacity ~1.15e18).
 *
 * Delayed indices route through the lagged old/new banks managed by `CBCD_FLUDS`. The CBC
 * kernel uses the precomputed bit on each node index to select the lagged or normal bank;
 * it does not inspect graph-level cycle metadata.
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
   *
   * \param index Index into the corresponding bank. Cannot exceed 2^60 - 1.
   * \param is_outgoing Flag indicating if the node corresponds to an outgoing face.
   * \param is_local Flag indicating if the index is in a local bank.
   * \param is_delayed Flag indicating that the index routes through a lagged bank.
   */
  CBCD_NodeIndex(std::uint64_t index, bool is_outgoing, bool is_local, bool is_delayed = false)
  {
    if (index >= (std::uint64_t(1) << 60))
      throw std::runtime_error("CBCD_NodeIndex: cannot hold an index of 2^60 or greater.");
    SetInOut(is_outgoing);
    SetLocal(is_local);
    SetDelayed(is_delayed);
    SetBoundary(false);
    SetIndex(index);
  }

  /**
   * Construct a boundary node index.
   *
   * \param index Index into the corresponding bank. Cannot exceed 2^60 - 1.
   * \param is_outgoing Flag indicating if the node corresponds to an outgoing face.
   */
  CBCD_NodeIndex(std::uint64_t index, bool is_outgoing)
  {
    if (index >= (std::uint64_t(1) << 60))
      throw std::runtime_error("CBCD_NodeIndex: cannot hold an index of 2^60 or greater.");
    SetInOut(is_outgoing);
    SetLocal(true);
    SetDelayed(false);
    SetBoundary(true);
    SetIndex(index);
  }

  /// Build a normal-local non-boundary node index.
  [[nodiscard]] static CBCD_NodeIndex Local(std::uint64_t index, bool is_outgoing)
  {
    return {index, is_outgoing, /*is_local=*/true, /*is_delayed=*/false};
  }

  /// Build a delayed-local non-boundary node index.
  [[nodiscard]] static CBCD_NodeIndex DelayedLocal(std::uint64_t index, bool is_outgoing)
  {
    return {index, is_outgoing, /*is_local=*/true, /*is_delayed=*/true};
  }

  /// Build a normal-nonlocal non-boundary node index.
  [[nodiscard]] static CBCD_NodeIndex Nonlocal(std::uint64_t index, bool is_outgoing)
  {
    return {index, is_outgoing, /*is_local=*/false, /*is_delayed=*/false};
  }

  /// Build a delayed-nonlocal non-boundary node index.
  [[nodiscard]] static CBCD_NodeIndex DelayedNonlocal(std::uint64_t index, bool is_outgoing)
  {
    return {index, is_outgoing, /*is_local=*/false, /*is_delayed=*/true};
  }

  /// Build a boundary node index.
  [[nodiscard]] static CBCD_NodeIndex Boundary(std::uint64_t index, bool is_outgoing)
  {
    return {index, is_outgoing};
  }

  /// Check if the current index corresponds to a local bank.
  constexpr bool IsLocal() const noexcept { return (value_ & local_bit_mask) != 0; }

  /// Check if the current non-boundary index routes through a lagged bank.
  constexpr bool IsDelayed() const noexcept { return (value_ & delayed_bit_mask) != 0; }

  /// Get the index into the bank.
  constexpr std::uint64_t GetIndex() const noexcept { return value_ & index_bit_mask; }

private:
  /// Bit 61: local (1) / nonlocal (0).
  static constexpr std::uint64_t local_bit_mask = std::uint64_t(1) << (64 - 3);
  /// Bit 60: delayed flux flag (only meaningful for non-boundary indices).
  static constexpr std::uint64_t delayed_bit_mask = std::uint64_t(1) << (64 - 4);
  /// Bits 0-59: index payload.
  static constexpr std::uint64_t index_bit_mask = (std::uint64_t(1) << (64 - 4)) - 1;

  constexpr void SetLocal(bool is_local) noexcept
  {
    if (is_local)
      value_ |= local_bit_mask;
    else
      value_ &= ~local_bit_mask;
  }

  constexpr void SetDelayed(bool is_delayed) noexcept
  {
    if (is_delayed)
      value_ |= delayed_bit_mask;
    else
      value_ &= ~delayed_bit_mask;
  }

  constexpr void SetIndex(std::uint64_t index) noexcept
  {
    value_ &= ~index_bit_mask;
    value_ |= (index & index_bit_mask);
  }
};

/**
 * Set of device pointers to local, boundary, non-local, and lagged buffers for CBCD FLUDS.
 *
 * The lagged pointer fields (`delayed_local_psi_old`, `delayed_local_psi_new`,
 * `delayed_nonlocal_incoming_psi_old`, `delayed_nonlocal_outgoing_psi`) are populated only
 * when the owning angle set has at least one delayed-local or delayed-nonlocal face; they
 * are `nullptr` otherwise.  The flux-pointer accessors always check the delayed bit of the
 * supplied node index and dispatch to the appropriate bank.  For acyclic problems every
 * face's node index has the delayed bit cleared, so the runtime route resolves to the normal
 * bank. This mirrors AAHD's approach: the device kernel follows precomputed routes encoded
 * in the node index and never inspects graph-level cycle metadata.
 */
struct CBCD_FLUDSPointerSet : public FLUDSPointerSet
{
  /// Pointer to incoming boundary angular fluxes.
  double* __restrict__ incoming_boundary_psi = nullptr;
  /// Pointer to outgoing boundary angular fluxes.
  double* __restrict__ outgoing_boundary_psi = nullptr;
  /// Pointer to the old lagged local face-slot bank (downwind reads).
  double* __restrict__ delayed_local_psi_old = nullptr;
  /// Pointer to the new lagged local face-slot bank (upwind writes).
  double* __restrict__ delayed_local_psi_new = nullptr;
  /// Pointer to the old lagged incoming non-local bank (downwind reads).
  double* __restrict__ delayed_nonlocal_incoming_psi_old = nullptr;
  /// Pointer to the lagged outgoing non-local bank (upwind writes / sender stages).
  double* __restrict__ delayed_nonlocal_outgoing_psi = nullptr;

  /**
   * Return the angular-flux pointer for one face node on the incoming side.
   *
   * The caller supplies a defined incoming index. For boundary faces this returns the
   * incoming-boundary bank. Otherwise the delayed and local bits select the flux bank.
   */
  constexpr double* GetIncomingFluxPointer(const CBCD_NodeIndex& node_index,
                                           unsigned int angle_group_idx) const noexcept
  {
    const auto offset = node_index.GetIndex() * stride_size + angle_group_idx;

    if (node_index.IsBoundary())
      return incoming_boundary_psi + offset;

    if (node_index.IsDelayed())
      return node_index.IsLocal() ? delayed_local_psi_old + offset
                                  : delayed_nonlocal_incoming_psi_old + offset;

    return node_index.IsLocal() ? local_psi + offset : nonlocal_incoming_psi + offset;
  }

  /**
   * Return the angular-flux pointer for one face node on the outgoing side.
   *
   * The caller supplies a defined outgoing index. For boundary faces this returns the
   * outgoing-boundary bank. Otherwise the delayed and local bits select the flux bank.
   */
  constexpr double* GetOutgoingFluxPointer(const CBCD_NodeIndex& node_index,
                                           unsigned int angle_group_idx) const noexcept
  {
    const auto offset = node_index.GetIndex() * stride_size + angle_group_idx;

    if (node_index.IsBoundary())
      return outgoing_boundary_psi + offset;

    if (node_index.IsDelayed())
      return node_index.IsLocal() ? delayed_local_psi_new + offset
                                  : delayed_nonlocal_outgoing_psi + offset;

    return node_index.IsLocal() ? local_psi + offset : nonlocal_outgoing_psi + offset;
  }
};

/// Metadata for boundary face nodes.
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

/// Grouped outgoing non-local face.
struct GroupedOutgoingNonlocalFace
{
  std::uint32_t dest_slot = 0;
  std::uint32_t remote_face_index = 0;
  /// Base index of the contiguous receiver-node-ordered face payload.
  std::uint32_t base_storage_index = 0;
  std::uint16_t num_face_nodes = 0;
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

} // namespace opensn

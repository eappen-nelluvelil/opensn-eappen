// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace opensn
{

/**
 * Host-side flux data structures for the cell-by-cell (CBC) sweep algorithm.
 *
 * Manages angular-flux storage for local, nonlocal, and boundary face data
 * during a CBC sweep. The local angular-flux buffer is organized as
 * \f$s^*\f$ cache-line-aligned slots of uniform size, where \f$s^*\f$ is the
 * optimal slot count computed by CBC_SPDS::ComputeMaxNumLocalPsiSlots.
 * Each local cell is mapped to a slot via the static assignment
 * \f$\sigma(\text{cell}) \to \{0, \ldots, s^*{-}1\}\f$, enabling multiple
 * cells to share the same slot when DAG dependencies guarantee that the
 * previous occupant's data has been fully consumed.
 *
 * ## Buffer layout
 *
 * - **Local psi buffer:** \f$s^* \times \text{slot\_size}\f$ doubles, where
 *   each slot holds \f$N_{\text{dof}} \times N_{\text{angles}} \times G\f$
 *   values, rounded up to a cache-line boundary (64 bytes). Addressed via
 *   cell_slot_bases_, which maps each cell_local_id to its slot base pointer.
 * - **Incoming nonlocal psi buffer:** flat array indexed by face storage
 *   offset, holding angular fluxes received from remote MPI ranks.
 */
class CBC_FLUDS : public FLUDS
{
public:
  /**
   * Construct CBC FLUDS from the SPDS slot assignment and common face data.
   *
   * Allocates the local psi buffer with \f$s^*\f$ slots from the CBC_SPDS,
   * populates cell_slot_bases_ from the static slot assignment, and allocates
   * the incoming nonlocal psi buffer sized by face-node counts.
   *
   * \param num_groups number of energy groups in the groupset
   * \param num_angles number of angles in the angle set
   * \param common_data shared face-level indexing metadata
   * \param max_cell_dof_count maximum number of DOFs (nodes) per cell
   */
  CBC_FLUDS(unsigned int num_groups,
            size_t num_angles,
            const CBC_FLUDSCommonData& common_data,
            size_t max_cell_dof_count);

  /// Return the shared face-level indexing metadata.
  const FLUDSCommonData& GetCommonData() const noexcept { return common_data_; }

  /// Stride in doubles between consecutive angle slots (= num_groups).
  size_t GetStrideSize() const noexcept { return num_groups_and_angles_; }

  /**
   * Return a pointer to the upwind angular flux for a local neighbor cell.
   *
   * \param face_neighbor_local_id local ID of the upwind neighbor cell
   * \param adj_cell_node node index on the neighbor cell's face
   * \param as_ss_idx angle subset index within the angle set
   * \return pointer to the start of the group data for the specified node and angle
   */
  double* UpwindPsi(std::uint32_t face_neighbor_local_id,
                    unsigned int adj_cell_node,
                    size_t as_ss_idx) const noexcept
  {
    return LocalPsiBase(face_neighbor_local_id) +
           static_cast<size_t>(adj_cell_node) * num_groups_and_angles_ + as_ss_idx * num_groups_;
  }

  /**
   * Return a pointer to the outgoing angular flux slot for a local cell node.
   *
   * The caller writes the just-solved angular fluxes at this location so that
   * downwind neighbors can read them via UpwindPsi.
   *
   * \param cell_local_id local ID of the cell being swept
   * \param cell_node node index on the cell
   * \param as_ss_idx angle subset index within the angle set
   * \return pointer to the start of the group data for the specified node and angle
   */
  double*
  OutgoingPsi(std::uint32_t cell_local_id, unsigned int cell_node, size_t as_ss_idx) const noexcept
  {
    return LocalPsiBase(cell_local_id) + static_cast<size_t>(cell_node) * num_groups_and_angles_ +
           as_ss_idx * num_groups_;
  }

  /**
   * Return a pointer to received nonlocal upwind angular flux for a face node.
   *
   * \param cell_local_id local ID of the cell owning the face
   * \param face_id face index on the cell
   * \param face_node_mapped mapped face node index
   * \param as_ss_idx angle subset index within the angle set
   * \return pointer to the start of the group data for the specified face node and angle
   */
  double* NLUpwindPsi(std::uint32_t cell_local_id,
                      unsigned int face_id,
                      unsigned int face_node_mapped,
                      size_t as_ss_idx) noexcept
  {
    const size_t face_storage_index = common_data_.GetCellFaceOffset(cell_local_id) + face_id;
    const size_t dof_offset = incoming_nonlocal_face_dof_offsets_[face_storage_index] +
                              static_cast<size_t>(face_node_mapped) * num_groups_and_angles_ +
                              as_ss_idx * num_groups_;
    return incoming_nonlocal_psi_buffer_.get() + dof_offset;
  }

  /**
   * Return a pointer to the nonlocal outgoing angular flux for a face node.
   *
   * \param psi_nonlocal_outgoing base pointer to the face's outgoing psi buffer
   * \param face_node face node index
   * \param as_ss_idx angle subset index within the angle set
   * \return pointer to the start of the group data for the specified face node and angle
   */
  double* NLOutgoingPsi(double* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx) noexcept
  {
    assert(psi_nonlocal_outgoing != nullptr);
    return psi_nonlocal_outgoing + face_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  }

  /**
   * Store received nonlocal face angular flux into the incoming buffer.
   *
   * \param cell_global_id global ID of the neighbor cell that produced the data
   * \param face_id face index on the neighbor cell
   * \param psi_data pointer to the received angular flux payload
   * \param data_size number of doubles in the payload
   */
  void StoreIncomingFaceData(uint64_t cell_global_id,
                             unsigned int face_id,
                             const double* psi_data,
                             size_t data_size);

  /// Reset received nonlocal angular fluxes.
  void ClearLocalAndReceivePsi() override;
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocateDelayedLocalPsi() override {}
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override {}

protected:
  /// Custom deleter for 64-byte-aligned double arrays.
  struct AlignedDoubleDeleter
  {
    void operator()(double* ptr) const noexcept;
  };

  /// Owning pointer to a 64-byte-aligned double array.
  using AlignedDoubleBuffer = std::unique_ptr<double[], AlignedDoubleDeleter>;

  /// Allocate a zero-initialized, 64-byte-aligned double buffer.
  static AlignedDoubleBuffer AllocateAlignedBuffer(size_t num_values);

  /// Return the slot base pointer for a local cell.
  double* LocalPsiBase(std::uint32_t cell_local_id) const noexcept
  {
    auto* const slot_base = cell_slot_bases_[cell_local_id];
    assert(slot_base != nullptr);
    return slot_base;
  }

  /// Shared face-level indexing metadata.
  const CBC_FLUDSCommonData& common_data_;
  /// Number of angular-flux storage slots (\f$s^*\f$).
  size_t num_slots_;
  /// Size of each slot in doubles (cache-line-aligned).
  size_t slot_size_;
  /// Per-cell base pointer into the local psi buffer (indexed by cell_local_id).
  std::vector<double*> cell_slot_bases_;

  /**
   * Contiguous local angular-flux buffer with \f$s^*\f$ slots.
   *
   * Layout per slot: node-major, angle-in-angleset-major, group-in-groupset-major.
   */
  AlignedDoubleBuffer local_psi_buffer_;
  /// Per-face-storage-index DOF offset into the incoming nonlocal buffer.
  std::vector<size_t> incoming_nonlocal_face_dof_offsets_;
  /// Flat buffer holding received nonlocal angular fluxes.
  AlignedDoubleBuffer incoming_nonlocal_psi_buffer_;

  /// Per-boundary incoming angular flux storage.
  std::vector<std::vector<double>> boundryI_incoming_psi_;
};

} // namespace opensn

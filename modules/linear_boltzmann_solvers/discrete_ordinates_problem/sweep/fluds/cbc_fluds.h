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
 * optimal slot count computed by CBC_SPDS::ComputeMaxNumLocalPsiSlots for the
 * local directed-face reuse graph.
 *
 * Each local directed face is mapped to a slot via the static assignment
 * \f$\sigma(\text{face}) \to \{0, \ldots, s^*{-}1\}\f$, enabling multiple
 * producer/consumer face pairs to share the same slot when the corresponding
 * schedule-safe reuse relation guarantees that the previous occupant's data has
 * already been consumed under every valid CBC sweep order.
 *
 * ## Buffer layout
 *
 * - **Local psi buffer:** \f$s^* \times \text{slot\_size}\f$ doubles, where
 *   each slot holds \f$N_{\text{face,max}} \times N_{\text{angles}} \times G\f$
 *   values, rounded up to a cache-line boundary (64 bytes). Addressed via
 *   local_face_slot_bases_, which maps each local cell face to its slot base
 *   pointer when the face is local.
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
   * populates local_face_slot_bases_ from the static local-face assignment,
   * and allocates the incoming nonlocal psi buffer sized by face-node counts.
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

  /// Bytes in the local psi backing buffer for this FLUDS instance.
  size_t GetLocalPsiBytes() const noexcept { return num_slots_ * slot_size_ * sizeof(double); }

  /// Return the slot base pointer for a local cell face.
  double* GetLocalFacePsiBase(std::uint32_t cell_local_id, unsigned int face_id) const noexcept
  {
    auto* const slot_base = local_face_slot_bases_[cell_face_offsets_[cell_local_id] + face_id];
    assert(slot_base != nullptr);
    return slot_base;
  }

  /// Return the base pointer for an incoming nonlocal face.
  double* GetIncomingNonlocalFacePsiBase(std::uint32_t cell_local_id,
                                         unsigned int face_id) noexcept
  {
    auto* const face_base =
      incoming_nonlocal_face_bases_[cell_face_offsets_[cell_local_id] + face_id];
    assert(face_base != nullptr);
    return face_base;
  }

  /**
   * Return a pointer to the upwind angular flux for a local incoming face.
   *
   * \param cell_local_id local ID of the cell currently being swept
   * \param face_id local incoming face id on the current cell
   * \param face_node_mapped mapped node index on the producer's outgoing face
   * \param as_ss_idx angle subset index within the angle set
   * \return pointer to the start of the group data for the specified node and angle
   */
  double* UpwindPsi(std::uint32_t cell_local_id,
                    unsigned int face_id,
                    unsigned int face_node_mapped,
                    size_t as_ss_idx) const noexcept
  {
    return GetLocalFacePsiBase(cell_local_id, face_id) +
           static_cast<size_t>(face_node_mapped) * num_groups_and_angles_ + as_ss_idx * num_groups_;
  }

  /**
   * Return a pointer to the outgoing angular flux slot for a local outgoing face node.
   *
   * The caller writes the just-solved angular fluxes at this location so that
   * downwind neighbors can read them via UpwindPsi.
   *
   * \param cell_local_id local ID of the cell being swept
   * \param face_id outgoing face id on the current cell
   * \param face_node face-local node index on the outgoing face
   * \param as_ss_idx angle subset index within the angle set
   * \return pointer to the start of the group data for the specified node and angle
   */
  double*
  OutgoingPsi(std::uint32_t cell_local_id,
              unsigned int face_id,
              unsigned int face_node,
              size_t as_ss_idx) const noexcept
  {
    return GetLocalFacePsiBase(cell_local_id, face_id) +
           static_cast<size_t>(face_node) * num_groups_and_angles_ +
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
    return GetIncomingNonlocalFacePsiBase(cell_local_id, face_id) +
           static_cast<size_t>(face_node_mapped) * num_groups_and_angles_ + as_ss_idx * num_groups_;
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
   * \param psi_data_bytes pointer to the received angular flux payload bytes
   * \param data_size number of doubles in the payload
   */
  uint64_t StoreIncomingFaceData(uint64_t cell_global_id,
                                unsigned int face_id,
                                const std::byte* psi_data_bytes,
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

  /// Shared face-level indexing metadata.
  const CBC_FLUDSCommonData& common_data_;
  /// Flat face-table offsets cached locally for hot-path indexing.
  std::vector<size_t> cell_face_offsets_;
  /// Number of angular-flux storage slots (\f$s^*\f$).
  size_t num_slots_;
  /// Size of each slot in doubles (cache-line-aligned).
  size_t slot_size_;
  /// Per-face-storage base pointer into the local psi buffer.
  std::vector<double*> local_face_slot_bases_;

  /**
   * Contiguous local angular-flux buffer with \f$s^*\f$ slots.
   *
   * Layout per slot: node-major, angle-in-angleset-major, group-in-groupset-major.
   */
  AlignedDoubleBuffer local_psi_buffer_;
  /// Per-face-storage-index DOF offset into the incoming nonlocal buffer.
  std::vector<size_t> incoming_nonlocal_face_dof_offsets_;
  /// Per-face-storage-index base pointer into the incoming nonlocal buffer.
  std::vector<double*> incoming_nonlocal_face_bases_;
  /// Flat buffer holding received nonlocal angular fluxes.
  AlignedDoubleBuffer incoming_nonlocal_psi_buffer_;

  /// Per-boundary incoming angular flux storage.
  std::vector<std::vector<double>> boundryI_incoming_psi_;
};

} // namespace opensn

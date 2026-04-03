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
 * Flux data structures (FLUDS) specific to the cell-by-cell (CBC) sweep algorithm
 *
 * This class manages the storage and access of angular flux data during a CBC sweep
 *
 * It provides methods to access:
 * - Upwind angular flux data from local neighbor cells
 * - Storage locations for downwind angular flux data for the current cell
 * - Upwind angular flux data received from remote MPI ranks
 */
class CBC_FLUDS : public FLUDS
{
public:
  CBC_FLUDS(unsigned int num_groups,
            size_t num_angles,
            const CBC_FLUDSCommonData& common_data,
            size_t max_cell_dof_count);

  const FLUDSCommonData& GetCommonData() const noexcept { return common_data_; }
  size_t GetStrideSize() const noexcept { return num_groups_and_angles_; }

  /**
   * Given a local upwind neighbor cell, a node index on this cell, and an
   * angleset subset index, this function returns a pointer to
   * the start of the group data for the specified node and angle.
   */
  double* UpwindPsi(std::uint32_t face_neighbor_local_id,
                    unsigned int adj_cell_node,
                    size_t as_ss_idx) const noexcept
  {
    return LocalPsiBase(face_neighbor_local_id) +
           static_cast<size_t>(adj_cell_node) * num_groups_and_angles_ + as_ss_idx * num_groups_;
  }

  /**
   * Given a local cell, a node index on this cell, and an angleset subset index,
   * this function returns a pointer to the start of the group data for the specified
   * node and angle for writing its just solved angular fluxes.
   */
  double*
  OutgoingPsi(std::uint32_t cell_local_id, unsigned int cell_node, size_t as_ss_idx) const noexcept
  {
    return LocalPsiBase(cell_local_id) + static_cast<size_t>(cell_node) * num_groups_and_angles_ +
           as_ss_idx * num_groups_;
  }

  /**
   * Given a local cell id, a face ID on this cell,
   * a node index on this face, and an angleset subset index,
   * this function returns a pointer to the start of the group data for the specified
   * face node and angle.
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
   * Given a pointer to a vector holding the non-local outgoing psi data for a face,
   * a node index on this face, and an angleset subset index,
   * this function returns a pointer to the start of the group data for the specified
   * face node and angle.
   */
  double* NLOutgoingPsi(double* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx) noexcept
  {
    assert(psi_nonlocal_outgoing != nullptr);
    return psi_nonlocal_outgoing + face_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  }

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
  struct AlignedDoubleDeleter
  {
    void operator()(double* ptr) const noexcept;
  };

  using AlignedDoubleBuffer = std::unique_ptr<double[], AlignedDoubleDeleter>;

  static AlignedDoubleBuffer AllocateAlignedBuffer(size_t num_values);

  double* LocalPsiBase(std::uint32_t cell_local_id) const noexcept
  {
    auto* const slot_base = cell_slot_bases_[cell_local_id];
    assert(slot_base != nullptr);
    return slot_base;
  }

  const CBC_FLUDSCommonData& common_data_;
  size_t num_slots_;
  size_t slot_size_;
  std::vector<double*> cell_slot_bases_;

  /**
   * Layout for a single slot:
   * node major -> angle in angleset major -> group in groupset major.
   */
  AlignedDoubleBuffer local_psi_buffer_;
  std::vector<size_t> incoming_nonlocal_face_dof_offsets_;
  AlignedDoubleBuffer incoming_nonlocal_psi_buffer_;

  std::vector<std::vector<double>> boundryI_incoming_psi_;
};

} // namespace opensn

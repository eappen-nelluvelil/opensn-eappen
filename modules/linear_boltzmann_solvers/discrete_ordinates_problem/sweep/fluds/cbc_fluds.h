// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace opensn
{

/**
 * Host CBC FLUDS.
 *
 * Owns the compact normal-local face-slot bank, the compact incoming-nonlocal face-psi bank
 * with receive-epoch validity, and the lagged local/nonlocal banks used to break cycles.
 * Normal-local upwind/outgoing accessors are face-addressed and resolve the slot through
 * the precomputed `CBC_FLUDSCommonData::LocalFaceSlotNodeOffset` table.
 */
class CBC_FLUDS : public FLUDS
{
public:
  /// Incoming nonlocal face psi and owning local cell.
  struct IncomingNonlocalPsi
  {
    /// Face psi storage.
    std::span<double> psi;
    /// Local cell associated with the received face data.
    std::uint32_t cell_local_id = 0;
  };

  /**
   * Construct the host CBC FLUDS.
   *
   * \param num_groups Number of groups in the angle set.
   * \param num_angles Number of angles in the angle set.
   * \param common_data Shared CBC FLUDS metadata.
   */
  CBC_FLUDS(unsigned int num_groups,
            std::size_t num_angles,
            const CBC_FLUDSCommonData& common_data);

  const CBC_FLUDSCommonData& GetCommonData() const;

  /**
   * Return local upwind face psi for a mapped face node.
   *
   * \param cell_local_id Local downwind cell ID.
   * \param face_id Local incoming face ID on the downwind cell.
   * \param face_node_mapped Mapped face-node index on the producer's outgoing face.
   * \param as_ss_idx Angle-set subset index.
   * \return Pointer to the compact-bank slot for the specified face node and angle subset.
   */
  double* UpwindPsi(std::uint32_t cell_local_id,
                    unsigned int face_id,
                    unsigned int face_node_mapped,
                    std::size_t as_ss_idx) noexcept;

  /**
   * Return lagged local upwind face psi for a delayed local dependency.
   *
   * \param cell_local_id Local downwind cell ID.
   * \param face_id Local incoming face ID.
   * \param face_node_mapped Mapped upwind face node.
   * \param as_ss_idx Angle-set subset index.
   * \return Lagged local upwind face psi for the specified node and angle subset.
   */
  double* DelayedUpwindPsi(std::uint32_t cell_local_id,
                           unsigned int face_id,
                           unsigned int face_node_mapped,
                           std::size_t as_ss_idx);

  /**
   * Return writable local outgoing face psi for a face node.
   *
   * \param cell_local_id Local upwind cell ID.
   * \param face_id Local outgoing face ID on the upwind cell.
   * \param face_node Face-local node index on the outgoing face.
   * \param as_ss_idx Angle-set subset index.
   * \return Writable compact-bank slot for the specified face node and angle subset.
   */
  double* OutgoingPsi(std::uint32_t cell_local_id,
                      unsigned int face_id,
                      unsigned int face_node,
                      std::size_t as_ss_idx) noexcept;

  /**
   * Return writable delayed local outgoing face psi.
   *
   * \param cell_local_id Local upwind cell ID.
   * \param face_id Local outgoing face ID.
   * \param face_node Face node.
   * \param as_ss_idx Angle-set subset index.
   * \return Writable delayed local outgoing face psi for the specified node and angle subset.
   */
  double* DelayedLocalOutgoingPsi(std::uint32_t cell_local_id,
                                  unsigned int face_id,
                                  unsigned int face_node,
                                  std::size_t as_ss_idx);

  /**
   * Return received nonlocal upwind face psi for a mapped face node.
   *
   * \param incoming_face_slot Incoming nonlocal face slot.
   * \param face_node_mapped Mapped upwind face node.
   * \param as_ss_idx Angle-set subset index.
   * \return Received face psi, or nullptr if the slot has not been received this sweep epoch.
   */
  double*
  NLUpwindPsi(std::size_t incoming_face_slot, unsigned int face_node_mapped, std::size_t as_ss_idx);

  /**
   * Return lagged nonlocal upwind face psi for a delayed nonlocal dependency.
   *
   * \param info Delayed nonlocal face metadata.
   * \param face_node_mapped Mapped upwind face node.
   * \param as_ss_idx Angle-set subset index.
   * \return Lagged nonlocal upwind face psi for the specified node and angle subset.
   */
  double* DelayedNLUpwindPsi(const CBC_FLUDSCommonData::DelayedNonlocalFaceInfo& info,
                             unsigned int face_node_mapped,
                             std::size_t as_ss_idx);

  /**
   * Return writable nonlocal outgoing face psi for a face node.
   *
   * \param psi_nonlocal_outgoing Nonlocal outgoing face storage.
   * \param face_node Face node.
   * \param as_ss_idx Angle-set subset index.
   * \return Writable face psi for the specified face node and angle subset.
   */
  double* NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing,
                        std::size_t face_node,
                        std::size_t as_ss_idx);

  /// Advance the receive epoch for nonlocal face psi.
  void ClearLocalAndReceivePsi() override;

  /// Allocate old and new lagged local face-psi storage.
  void AllocateDelayedLocalPsi() override;

  /// Allocate old and new lagged outgoing nonlocal face-psi storage.
  void AllocateDelayedPrelocIOutgoingPsi() override;

  /// Copy old lagged local face psi into the new storage bank.
  void SetDelayedLocalPsiOldToNew() override;

  /// Copy new lagged local face psi into the old storage bank.
  void SetDelayedLocalPsiNewToOld() override;

  /// Copy old lagged outgoing nonlocal face psi into the new storage bank.
  void SetDelayedOutgoingPsiOldToNew() override;

  /// Copy new lagged outgoing nonlocal face psi into the old storage bank.
  void SetDelayedOutgoingPsiNewToOld() override;

  /**
   * Prepare incoming nonlocal face-psi storage and return the owning local cell.
   *
   * \param incoming_face_slot Incoming nonlocal face slot.
   * \param data_size Number of psi values to store for the face.
   * \return Prepared face-psi storage and associated local cell ID.
   */
  IncomingNonlocalPsi PrepareIncomingNonlocalPsiBySlot(std::size_t incoming_face_slot,
                                                       std::size_t data_size);

  /// Return the number of psi values for a delayed nonlocal face slot.
  std::size_t GetDelayedNonlocalPsiSize(std::size_t delayed_face_slot) const;

  /**
   * Prepare incoming delayed nonlocal face-psi storage.
   *
   * \param delayed_face_slot Delayed nonlocal face slot.
   * \param data_size Number of psi values to store for the face.
   * \return Writable delayed nonlocal face-psi storage.
   */
  std::span<double> PrepareIncomingDelayedNonlocalPsiBySlot(std::size_t delayed_face_slot,
                                                            std::size_t data_size);

protected:
  const CBC_FLUDSCommonData& common_data_;
  /**
   * Compact normal-local face-slot bank with `slot-node -> angleset subset -> group` layout.
   *
   * Sized as `common_data_.TotalLocalFaceSlotNodes() * num_groups_and_angles_`.  Slots are
   * the chains computed by the local-face slot planner, sized per-slot by the maximum
   * face-node extent over the chain.
   */
  std::vector<double> local_psi_data_;
  /// Contiguous storage for received nonlocal face psi.
  std::vector<double> incoming_nonlocal_psi_;
  /// Offsets into received nonlocal face-psi storage.
  std::vector<std::size_t> incoming_nonlocal_psi_offsets_;
  /// Receive epoch for each nonlocal face slot.
  std::vector<std::uint32_t> incoming_psi_epoch_;
  /// Current receive epoch.
  std::uint32_t current_psi_epoch_ = 1;
  /// New lagged local face-psi storage.
  std::vector<double> delayed_local_psi_;
  /// Old lagged local face-psi storage.
  std::vector<double> delayed_local_psi_old_;
  /// New lagged outgoing nonlocal face-psi storage by delayed upstream location.
  std::vector<std::vector<double>> delayed_prelocI_outgoing_psi_;
  /// Old lagged outgoing nonlocal face-psi storage by delayed upstream location.
  std::vector<std::vector<double>> delayed_prelocI_outgoing_psi_old_;
};

} // namespace opensn

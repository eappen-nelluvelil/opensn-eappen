// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include <cstddef>
#include <vector>
#include <climits>

namespace opensn
{

class UnknownManager;
class SpatialDiscretization;
class Cell;

/// Descriptor for one per-rank send or receive message.
struct NLMessageDescriptor
{
  int rank;          ///< MPI rank (destination for send, source for receive)
  size_t buf_start;  ///< Start offset in the contiguous buffer (in doubles)
  size_t buf_size;   ///< Number of doubles in this message
};

/// Per-cell contribution to send messages (for counter-based send triggering).
struct CellSendContrib
{
  size_t msg_index;  ///< Index into nl_send_messages_
  size_t face_count; ///< Number of faces this cell contributes to that message
};

class CBC_FLUDS : public FLUDS
{
public:
  CBC_FLUDS(unsigned int num_groups,
            size_t num_angles,
            const CBC_FLUDSCommonData& common_data,
            const UnknownManager& psi_uk_man,
            const SpatialDiscretization& sdm);

  const FLUDSCommonData& GetCommonData() const;

  /// Returns upwind psi for a local neighbor cell.
  double* UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx);

  /// Returns outgoing psi storage for a local cell.
  double* OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx);

  /// Returns upwind psi from a non-local (remote) face.
  /// @param cell_local_id Local id of the cell whose face receives the data.
  /// @param face_id Face index on that cell.
  double* NLUpwindPsi(uint64_t cell_local_id,
                      unsigned int face_id,
                      unsigned int face_node_mapped,
                      size_t as_ss_idx);

  /// Returns outgoing psi storage for a non-local face, writing into the send buffer.
  /// @param face_base_offset Pre-computed offset from GetNLSendFaceOffset().
  double* NLOutgoingPsi(size_t face_base_offset, size_t face_node, size_t as_ss_idx);

  /// Returns the send buffer offset for a non-local outgoing face.
  /// Returns SIZE_MAX if the face is not a non-local outgoing face.
  size_t GetNLSendFaceOffset(uint64_t cell_local_id, unsigned int face_id) const;

  void ClearLocalAndReceivePsi() override;
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocateDelayedLocalPsi() override {}
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override {}

  // --- Accessors for communicator ---

  const std::vector<NLMessageDescriptor>& GetNLSendMessages() const { return nl_send_messages_; }
  const std::vector<NLMessageDescriptor>& GetNLRecvMessages() const { return nl_recv_messages_; }

  double* GetNLSendBufferPtr() { return nl_send_buffer_.data(); }
  double* GetNLRecvBufferPtr() { return nl_receive_buffer_.data(); }

  const std::vector<size_t>& GetNLSendTotalFacesPerMsg() const
  {
    return nl_send_total_faces_per_msg_;
  }

  const std::vector<CellSendContrib>& GetCellSendContributions(uint64_t cell_local_id) const
  {
    return cell_send_contributions_[cell_local_id];
  }

  /// For each recv message, the list of local cell IDs whose dependency counters
  /// should be decremented (one entry per face from that source rank).
  const std::vector<std::vector<uint64_t>>& GetRecvMsgCellLocalIDs() const
  {
    return recv_msg_cell_local_ids_;
  }

private:
  const CBC_FLUDSCommonData& common_data_;
  const UnknownManager& psi_uk_man_;
  const SpatialDiscretization& sdm_;
  size_t num_angles_in_gs_quadrature_;
  size_t num_local_spatial_dofs_;
  size_t local_psi_data_size_;

  // --- Local angular flux storage ---
  std::vector<double> local_psi_data_;
  std::vector<size_t> cell_psi_data_start_;

  // --- Non-local send infrastructure ---
  std::vector<double> nl_send_buffer_;
  std::vector<NLMessageDescriptor> nl_send_messages_;
  std::vector<size_t> nl_send_total_faces_per_msg_;
  /// nl_send_face_offset_[cell_local_id][face_id] = offset into nl_send_buffer_,
  /// or SIZE_MAX if the face is not a non-local outgoing face.
  std::vector<std::vector<size_t>> nl_send_face_offset_;
  /// Per-cell contributions to send messages (for counter-based triggering).
  std::vector<std::vector<CellSendContrib>> cell_send_contributions_;

  // --- Non-local receive infrastructure ---
  std::vector<double> nl_receive_buffer_;
  std::vector<NLMessageDescriptor> nl_recv_messages_;
  /// nl_recv_face_offset_[cell_local_id][face_id] = offset into nl_receive_buffer_,
  /// or SIZE_MAX if the face is not a non-local incoming face.
  std::vector<std::vector<size_t>> nl_recv_face_offset_;
  /// For each recv message, list of cell local IDs served (one per face, may have duplicates).
  std::vector<std::vector<uint64_t>> recv_msg_cell_local_ids_;
};

} // namespace opensn

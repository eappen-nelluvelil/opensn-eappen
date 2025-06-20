// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "caliper/cali.h"
#include <map>
#include <functional>
#include <memory_resource>
#include <vector>
#include <cstddef>

namespace opensn
{

class UnknownManager;
class SpatialDiscretization;
class Cell;

class CBC_FLUDS : public FLUDS
{
public:
  CBC_FLUDS(size_t num_groups, // Number of groups in this AngleSet's LBSGroupset
            size_t num_angles, // Number of angles in THIS specific AngleSet
            const CBC_FLUDSCommonData& common_data,
            const SpatialDiscretization& sdm,
            size_t max_wavefront_size,
            size_t max_num_cell_dofs);

  const FLUDSCommonData& GetCommonData() const;

  // --- Methods for non-local (remote) angular flux data ---

  /**
   * Retrieves the pre-received angular flux data packet for a specific
   *        cell face that depends on a remote (off-processor) upwind cell.
   * @param cell_global_id Global ID of the local cell whose face data is requested.
   * @param face_id Local index of the face on `cell_global_id`.
   * @return Constant reference to a vector of doubles. This vector is a data packet
   *         containing angular fluxes for all angles in this AngleSet, for all
   *         nodes on the specified face.
   * @note The layout of the returned vector is expected to be:
   *       face spatial DOF major -> angle in set major -> group major
   */
  const std::vector<double>& GetNonLocalUpwindData(uint64_t cell_global_id,
                                                   unsigned int face_id) const;

  /**
   * Interprets a received multi-angle packet (for all angles in this AngleSet,
   *        for all nodes on a face) and returns a pointer to the data for a
   *        specific angle and face node.
   * @param psi_data The data packet received via MPI, obtained from
   *                 `GetNonLocalUpwindData`.
   * @param face_node_mapped The 0-indexed node on the face for which data is needed.
   * @param angle_set_index The local 0-indexed angular direction within this AngleSet.
   * @return Pointer to the start of the group data for the specified
   *         `face_node_mapped` and `angle_set_index`.
   */
  const double* GetNonLocalUpwindPsi(const std::vector<double>& psi_data,
                                     unsigned int face_node_mapped,
                                     unsigned int angle_set_index);

  double* AllocateForCell(uint64_t cell_local_id);
  void DeallocateForCell(uint64_t cell_local_id);
  const double* GetPsiForCell(uint64_t cell_local_id) const;

  void ClearLocalAndReceivePsi() override { deplocs_outgoing_messages_.clear(); }
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi(size_t num_grps, size_t num_angles) override {}
  void AllocateOutgoingPsi(size_t num_grps, size_t num_angles, size_t num_loc_sucs) override {}

  void AllocateDelayedLocalPsi(size_t num_grps, size_t num_angles) override {}
  void AllocatePrelocIOutgoingPsi(size_t num_grps, size_t num_angles, size_t num_loc_deps) override
  {
  }
  void AllocateDelayedPrelocIOutgoingPsi(size_t num_grps,
                                         size_t num_angles,
                                         size_t num_loc_deps) override
  {
  }

  std::vector<double>& DelayedLocalPsi() override { return delayed_local_psi_; }
  std::vector<double>& DelayedLocalPsiOld() override { return delayed_local_psi_old_; }

  std::vector<std::vector<double>>& DeplocIOutgoingPsi() override { return deplocI_outgoing_psi_; }

  std::vector<std::vector<double>>& PrelocIOutgoingPsi() override { return prelocI_outgoing_psi_; }

  std::vector<std::vector<double>>& DelayedPrelocIOutgoingPsi() override
  {
    return delayed_prelocI_outgoing_psi_;
  }
  std::vector<std::vector<double>>& DelayedPrelocIOutgoingPsiOld() override
  {
    return delayed_prelocI_outgoing_psi_old_;
  }

  // Key for messages from deploying locations: pairs a cell's global ID with a face index.
  using CellFaceKey = std::pair<uint64_t, unsigned int>;

  std::map<CellFaceKey, std::vector<double>>& GetDeplocsOutgoingMessages()
  {
    return deplocs_outgoing_messages_;
  }

private:
  const CBC_FLUDSCommonData& common_data_;

  const SpatialDiscretization& sdm_;

  // Size in doubles for a single cell's full psi data
  size_t single_cell_block_size_ = 0;

  // Raw memory buffer for fixed-size pool allocator memory pool
  std::vector<std::byte> memory_buffer_;

  // Monotonic buffer resource that manages the memory buffer
  std::pmr::monotonic_buffer_resource upstream_resource_;

  // Memory resource that manages pools of fixed-sized blocks
  std::pmr::unsynchronized_pool_resource memory_pool_;

  // Type-aware allocator that uses memory_pool_
  std::pmr::polymorphic_allocator<double> psi_allocator_;

  // Maps a cell local ID to its currently allocated memory block
  // std::map<uint64_t, double*> cell_memory_map_;
  std::vector<double*> cell_memory_map_;

  std::vector<double> delayed_local_psi_;
  std::vector<double> delayed_local_psi_old_;
  std::vector<std::vector<double>> deplocI_outgoing_psi_;
  std::vector<std::vector<double>> prelocI_outgoing_psi_;
  std::vector<std::vector<double>> boundryI_incoming_psi_;

  std::vector<std::vector<double>> delayed_prelocI_outgoing_psi_;
  std::vector<std::vector<double>> delayed_prelocI_outgoing_psi_old_;

  std::map<CellFaceKey, std::vector<double>> deplocs_outgoing_messages_;
};

} // namespace opensn

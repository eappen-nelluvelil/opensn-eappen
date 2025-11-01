// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "boost/pool/simple_segregated_storage.hpp"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_structs.h"
#include <cstddef>
#include <map>
#include <functional>
#include <tuple>

namespace opensn
{

class UnknownManager;
class SpatialDiscretization;
class Cell;
class LBSProblem;
class LBSGroupset;
class AngleSet;
class SweepChunk;
class CellLBSView;

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
  CBC_FLUDS(size_t num_groups,
            size_t num_angles,
            const CBC_FLUDSCommonData& common_data,
            const UnknownManager& psi_uk_man,
            const SpatialDiscretization& sdm,
            std::vector<CellLBSView>& cell_transport_views,
            size_t num_local_cells,
            size_t max_cell_dof_count,
            size_t min_num_pool_allocator_slots,
            bool use_gpus);

  ~CBC_FLUDS();

  const FLUDSCommonData& GetCommonData() const;

  /**
   * Given a cell's local ID, retrieve a free slot in the pool allocator for storing
   * the cell's angular flux data and map the cell's local ID to the slot
   */
  void Allocate(uint64_t cell_local_ID);

  /**
   * Given a cell's local ID, deallocate the memory used for the cell's angular flux data
   * and remove the mapping from the cell's local ID to the slot
   */
  void Deallocate(uint64_t cell_local_ID);

  /**
   * Given a local upwind neighbor cell local cell ID, a node index on this cell, and an
   * angleset subset index, this function returns a pointer to
   * the start of the group data for the specified node and angle.
   */
  double* UpwindPsi(uint64_t cell_local_id, unsigned int adj_cell_node, size_t as_ss_idx);

  double* GPUUpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx);

  /**
   * Given a local cell's local ID, a node index on this cell, and an angleset subset index,
   * this function returns a pointer to the start of the group data for the specified
   * node and angle for writing its just solved angular fluxes.
   */
  double* OutgoingPsi(uint64_t cell_local_id, unsigned int cell_node, size_t as_ss_idx);

  double* GPUOutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx);

  /**
   * Given a remote upwind cell's global ID, a face ID on this cell,
   * a node index on this face, and an angleset subset index,
   * this function returns a pointer to the start of the group data for the specified
   * face node and angle.
   */
  double* NLUpwindPsi(uint64_t cell_global_id,
                      unsigned int face_id,
                      unsigned int face_node_mapped,
                      size_t as_ss_idx);

  /**
   * Given a pointer to a vector holding the non-local outgoing psi data for a face,
   * a node index on this face, and an angleset subset index,
   * this function returns a pointer to the start of the group data for the specified
   * face node and angle.
   */
  double*
  NLOutgoingPsi(std::vector<double>* psi_nonlocal_outgoing, size_t face_node, size_t as_ss_idx);

  size_t GetGPULocalPsiDataSize() const { return gpu_local_psi_data_size_; }

  size_t GetNumLocalCells() const { return num_local_cells_; }

  std::tuple<size_t, size_t, std::vector<uint64_t>, std::vector<unsigned int>, std::vector<int>, std::vector<int>>
  Prepare_CBCD_FLUDS(AngleSet& angle_set);

  std::vector<double> GetBoundaryPsiData(SweepChunk& sweep_chunk,
                                         AngleSet& angle_set);

  void UpdateBoundaryPsiData(SweepChunk& sweep_chunk,
                             AngleSet& angle_set);

  void* Get_CBCD_FLUDS_Ptr() { return cbcd_fluds_; }

  void Create_CBCD_FLUDS(size_t num_total_faces,
                         size_t incoming_boundary_psi_buffer_size,
                         const std::vector<uint64_t>& face_neighbor_local_ids,
                         const std::vector<unsigned int>& face_neighbor_cell_node_map,
                         const std::vector<int>& cell_to_local_face_offset_map,
                         const std::vector<int>& boundary_psi_map);

  void SetBoundaryPsiData(const std::vector<double>& boundary_psi);

  void Destroy_CBCD_FLUDS();

  void BuildDeviceCellDOFMap();

  std::vector<size_t>& GetCellDOFMap() { return cell_dof_map_; }

  const SpatialDiscretization& GetSDM() const { return sdm_; }

  void ClearLocalAndReceivePsi() override { deplocs_outgoing_messages_.clear(); }
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocateDelayedLocalPsi() override {}
  void AllocatePrelocIOutgoingPsi() override {}
  void AllocateDelayedPrelocIOutgoingPsi() override {}

  // cell_global_id, face_id
  using CellFaceKey = std::pair<uint64_t, unsigned int>;

  std::map<CellFaceKey, std::vector<double>>& GetDeplocsOutgoingMessages()
  {
    return deplocs_outgoing_messages_;
  }

private:
  const CBC_FLUDSCommonData& common_data_;
  const UnknownManager& psi_uk_man_;
  const SpatialDiscretization& sdm_;
  std::vector<CellLBSView>& cell_transport_views_;
  size_t num_local_cells_;
  size_t num_angles_in_gs_quadrature_;
  size_t num_quadrature_local_dofs_;
  size_t num_local_spatial_dofs_;
  size_t gpu_local_psi_data_size_;
  bool use_gpus_ = false;
  std::vector<double> local_psi_data_gpu_buffer_;
  size_t slot_size_;
  std::vector<double*> cell_local_ID_to_psi_map_;
  std::vector<double> local_psi_data_backing_buffer_;
  boost::simple_segregated_storage<size_t> local_psi_data_;

  size_t num_total_faces_;
  size_t incoming_boundary_psi_buffer_size_;
  std::vector<int> cell_to_local_face_offset_map_;
  std::vector<int> boundary_psi_map_;
  std::vector<double> incoming_boundary_psi_buffer_;

  std::vector<size_t> cell_dof_map_;
  void* cbcd_fluds_ = nullptr;

  std::vector<std::vector<double>> boundryI_incoming_psi_;

  std::map<CellFaceKey, std::vector<double>> deplocs_outgoing_messages_;
};

} // namespace opensn
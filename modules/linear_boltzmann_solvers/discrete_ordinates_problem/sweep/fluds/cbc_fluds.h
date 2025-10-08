// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include <boost/pool/simple_segregated_storage.hpp>
#include <functional>
#include <map>
#include <memory_resource>
#include <unordered_map>

namespace opensn
{

class UnknownManager;
class SpatialDiscretization;
class Cell;

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
            const size_t num_local_cells,
            const size_t peak_number_alive_cells,
            const size_t max_cell_dof_count,
            bool use_gpus = false);

  ~CBC_FLUDS();

  const FLUDSCommonData& GetCommonData() const;

  /**
   * Given a remote upwind cell's global ID and local face index, this function
   * returns the pre-received angular flux data for the face on the upwind cell
   */
  const std::vector<double>& GetNonLocalUpwindData(uint64_t cell_global_id,
                                                   unsigned int face_id) const;

  /**
   * Given the angular flux data for a face on a remote upwind cell, a face node
   * index on the face, and an angle set index, this function returns a pointer
   * to start of the group data for the specified face node and angle
   */
  const double* GetNonLocalUpwindPsi(const std::vector<double>& psi_data,
                                     unsigned int face_node_mapped,
                                     unsigned int angle_set_index);

  void Allocate(const uint64_t cell_local_ID);

  void Deallocate(const uint64_t cell_local_ID);

  double* GetCellBlock(const uint64_t cell_local_ID);

  const double* GetCellBlock(const uint64_t cell_local_ID) const;

  size_t GetNumBlocks() const { return num_blocks_; }

  unsigned int GetNumAllocations() const { return num_allocations_; }

  unsigned int GetNumDeallocations() const { return num_deallocations_; }

  unsigned int GetNumCurrentAllocations() const { return num_current_allocations_; }

  unsigned int GetNumPeakAllocations() const { return num_peak_allocations_; }

  size_t GetPeakNumberAliveCells() const { return num_blocks_; }

  size_t GetNumberOfMemoryMapElements() const { return cell_local_ID_to_ptr_map_.size(); }

  void ResetCounters()
  {
    num_allocations_ = 0;
    num_deallocations_ = 0;
    num_current_allocations_ = 0;
    num_peak_allocations_ = 0;
  }

  size_t GetBufferSize() const { return backing_buffer_.size(); }

  void ResetPool()
  {
    std::fill(cell_local_ID_to_ptr_map_.begin(), cell_local_ID_to_ptr_map_.end(), nullptr);
  }

  /// Initialize GPU memory structures
  void InitializeGPUMemory();

  /// Cleanup GPU memory 
  void DestroyGPUMemory();

  /// Copy a local cell's angular fluxes from host to the GPU
  void CopyCellToDevice(uint64_t cell_local_id);

  /// Copy a cell's angular fluxes from GPU to host
  void CopyCellFromDevice(uint64_t cell_local_id);

  /// Copy non-local upwind data for a specific cell face to GPU
  void CopyNLUpwindToDevice(uint64_t cell_local_id, unsigned int face_id);

  /// Copy boundary data to GPU
  void CopyBoundaryDataToDevice();

  /// Get device pointer for accessing GPU memory
  void* GetDeviceMemoryPtr() { return gpu_cbc_fluds_; }

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
  bool use_gpus_;

  // ---------------------------------------------------------------
  // Required objects to implement a free-list memory pool allocator
  // ---------------------------------------------------------------  
  size_t num_blocks_;
  size_t block_size_;

  // Storage for local angular fluxes
  // Layout: spatial DOF major -> angle in set major -> group major
  std::vector<double> backing_buffer_;
  boost::simple_segregated_storage<size_t> storage_;
  std::vector<double*> cell_local_ID_to_ptr_map_;

  unsigned int num_allocations_ = 0;
  unsigned int num_deallocations_ = 0;
  unsigned int num_current_allocations_ = 0;
  unsigned int num_peak_allocations_ = 0;

  std::vector<double> delayed_local_psi_;
  std::vector<double> delayed_local_psi_old_;
  std::vector<std::vector<double>> deplocI_outgoing_psi_;
  std::vector<std::vector<double>> prelocI_outgoing_psi_;
  std::vector<std::vector<double>> boundryI_incoming_psi_;

  std::vector<std::vector<double>> delayed_prelocI_outgoing_psi_;
  std::vector<std::vector<double>> delayed_prelocI_outgoing_psi_old_;

  std::map<CellFaceKey, std::vector<double>> deplocs_outgoing_messages_;

  /// Pointer to GPU memory structures
  void* gpu_cbc_fluds_ = nullptr;
};

} // namespace opensn

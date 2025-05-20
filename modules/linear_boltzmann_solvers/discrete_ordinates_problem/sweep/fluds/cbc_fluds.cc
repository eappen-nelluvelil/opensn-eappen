// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/math/unknown_manager/unknown_manager.h"

namespace opensn
{

/**
 * @brief Constructs a CBC_FLUDS object.
 *
 * The `local_psi_data_` vector is sized based on the number of angles specific
 * to this `AngleSet` instance, the number of groups in the `LBSGroupset`, and
 * the number of local spatial degrees of freedom.
 * Verification logging is performed to compare the optimized size against the
 * previous, larger sizing approach.
 */
CBC_FLUDS::CBC_FLUDS(size_t num_groups_in_angle_set,
                     size_t num_angles_in_angle_set,
                     const CBC_FLUDSCommonData& common_data,
                     const UnknownManager& lbs_groupset_psi_uk_man,
                     const SpatialDiscretization& sdm,
                     const int peak_liveness_cell_count,
                     const size_t max_nodes_per_cell_for_sdm,
                     const std::vector<int>* store_map_ptr,
                     const std::vector<int>* discard_map_ptr)
  : FLUDS(num_groups_in_angle_set, num_angles_in_angle_set, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(lbs_groupset_psi_uk_man),
    sdm_(sdm),
    psi_store_timestep_map_ptr_(store_map_ptr),
    psi_discard_timestep_map_ptr_(discard_map_ptr)
{
  if (!psi_store_timestep_map_ptr_ || !psi_discard_timestep_map_ptr_)
  {
    throw std::invalid_argument("CBC_FLUDS: Liveness store/discard maps cannot be null.");
  }

  size_t slot_size_per_cell_doubles =
    max_nodes_per_cell_for_sdm * num_angles_in_angle_set * num_groups_in_angle_set;

  // Ensure peak_liveness_cell_count is non-negative. If 0, pool might be empty.
  if (peak_liveness_cell_count < 0)
  {
    throw std::invalid_argument("CBC_FLUDS: peak_liveness_cell_count cannot be negative.");
  }

  psi_pool_ = std::make_unique<AngularFluxMemoryPool>(static_cast<size_t>(peak_liveness_cell_count),
                                                      slot_size_per_cell_doubles);

  // Logging for verification (original logging for size comparison is less relevant now)
  opensn::log.Log() << "CBC_FLUDS initialized with AngularFluxMemoryPool:"
                    << " Max live cells (slots): " << peak_liveness_cell_count
                    << ", Slot size (doubles): " << slot_size_per_cell_doubles << " (for "
                    << max_nodes_per_cell_for_sdm << " nodes/cell, " << num_angles_in_angle_set
                    << " ang/set, " << num_groups_in_angle_set << " grp/set)";
  opensn::log.Log() << "  Pool capacity: "
                    << static_cast<double>(peak_liveness_cell_count * slot_size_per_cell_doubles *
                                           sizeof(double)) /
                         (1024.0 * 1024.0)
                    << " MB";
}

/**
 * @brief Gets the common data shared among FLUDS instances.
 * @return Constant reference to the `CBC_FLUDSCommonData` object.
 */
const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

// --- New Memory Management Methods ---
void
CBC_FLUDS::AllocatePsiForCell(uint64_t original_cell_local_id)
{
  if (live_cell_psi_pointers_.count(original_cell_local_id))
  {
    // This could happen if a cell is processed multiple times in error, or liveness map is off
    opensn::log.Log0Warning() << "CBC_FLUDS::AllocatePsiForCell: Cell " << original_cell_local_id
                              << " already has allocated PSI. Overwriting or ignoring.";
    // For now, let's assume it's an error to re-allocate without deallocating.
    // Or, if this is valid (e.g. recomputing), ensure old one is deallocated first if ptr changes.
    // Current pool model would give a *new* slot if old wasn't deallocated.
    // Safest is to throw if already present, means logic error elsewhere.
    throw std::logic_error("CBC_FLUDS: Attempting to allocate for already live cell " +
                           std::to_string(original_cell_local_id));
  }
  double* slot = psi_pool_->allocate_slot();
  if (!slot)
  {
    throw std::runtime_error("CBC_FLUDS: PSI Pool exhausted during AllocatePsiForCell for cell " +
                             std::to_string(original_cell_local_id));
  }
  live_cell_psi_pointers_[original_cell_local_id] = slot;
}

void
CBC_FLUDS::DeallocatePsiForCell(uint64_t original_cell_local_id)
{
  auto it = live_cell_psi_pointers_.find(original_cell_local_id);
  if (it != live_cell_psi_pointers_.end())
  {
    psi_pool_->deallocate_slot(it->second);
    live_cell_psi_pointers_.erase(it);
  }
  else
  {
    // Attempting to deallocate something not tracked as live.
    opensn::log.Log0Warning() << "CBC_FLUDS::DeallocatePsiForCell: Cell " << original_cell_local_id
                              << " not found in live pointers map, or already deallocated.";
  }
}

const double*
CBC_FLUDS::GetUpwindPsiData(uint64_t original_upwind_cell_local_id) const
{
  auto it = live_cell_psi_pointers_.find(original_upwind_cell_local_id);
  if (it != live_cell_psi_pointers_.end())
  {
    return it->second;
  }
  // This is a critical error: means required upwind data was expected but not allocated/live.
  throw std::runtime_error("CBC_FLUDS::GetUpwindPsiData: Upwind cell " +
                           std::to_string(original_upwind_cell_local_id) +
                           " PSI data not found in live pool. Liveness logic error.");
  return nullptr;
}

double*
CBC_FLUDS::GetDownwindPsiWritePtr(uint64_t original_current_cell_local_id)
{
  auto it = live_cell_psi_pointers_.find(original_current_cell_local_id);
  if (it != live_cell_psi_pointers_.end())
  {
    return it->second;
  }
  // This implies AllocatePsiForCell was not called for the current cell before sweep_chunk needs to
  // write to it.
  throw std::runtime_error("CBC_FLUDS::GetDownwindPsiWritePtr: Current cell " +
                           std::to_string(original_current_cell_local_id) +
                           " PSI data not allocated. Call AllocatePsiForCell first.");
  return nullptr;
}

bool
CBC_FLUDS::IsCellPsiAllocated(uint64_t original_cell_local_id) const
{
  return live_cell_psi_pointers_.count(original_cell_local_id) > 0;
}

int
CBC_FLUDS::GetCellStoreStep(uint64_t original_cell_local_id) const
{
  if (!psi_store_timestep_map_ptr_)
    return -1; // Or throw
  // Add bounds check if psi_store_timestep_map_ptr_ is a vector indexed by original_cell_local_id
  if (original_cell_local_id < psi_store_timestep_map_ptr_->size())
  {
    return (*psi_store_timestep_map_ptr_)[original_cell_local_id];
  }
  throw std::out_of_range(
    "CBC_FLUDS::GetCellStoreStep: original_cell_local_id out of bounds for store map.");
}

int
CBC_FLUDS::GetCellDiscardStep(uint64_t original_cell_local_id) const
{
  if (!psi_discard_timestep_map_ptr_)
    return -1; // Or throw
  if (original_cell_local_id < psi_discard_timestep_map_ptr_->size())
  {
    return (*psi_discard_timestep_map_ptr_)[original_cell_local_id];
  }
  throw std::out_of_range(
    "CBC_FLUDS::GetCellDiscardStep: original_cell_local_id out of bounds for discard map.");
}

/**
 * @brief Retrieves a data packet received via MPI for a specific face of a local cell.
 *
 * This packet contains angular fluxes from a remote upwind cell, covering all angles
 * managed by this `AngleSet` and all nodes on the specified face.
 *
 * @param cell_global_id Global ID of the local cell.
 * @param face_id Local index of the face on the cell.
 * @return Constant reference to the `std::vector<double>` data packet.
 */
const std::vector<double>&
CBC_FLUDS::GetNonLocalUpwindData(uint64_t cell_global_id, unsigned int face_id) const
{
  return deplocs_outgoing_messages_.at({cell_global_id, face_id});
}

/**
 * @brief Extracts a pointer to angular flux data for a specific angle and face node
 *        from a raw MPI data packet.
 *
 * The input `psi_data` is assumed to be a packet containing data for all angles
 * in this `AngleSet`, for all nodes on a face, received via MPI.
 * The layout within `psi_data` is face spatial DOF major -> angle in set major -> group major.
 *
 * @param psi_data The MPI data packet (vector of doubles).
 * @param face_node_mapped The 0-indexed node on the face.
 * @param angle_set_index The local 0-indexed angular direction within this `AngleSet`.
 * @return `const double*` pointing to the start of group data for the specified
 *         `face_node_mapped` and `angle_set_index` within the `psi_data` packet.
 * @throws std::runtime_error if calculated offsets are out of bounds or `psi_data` is empty.
 */
const double*
CBC_FLUDS::GetNonLocalUpwindPsi(const std::vector<double>& psi_data,
                                unsigned int face_node_mapped,
                                unsigned int angle_set_index) const
{
  /// Stride to jump from one face node's data block to the next within `psi_data`.
  /// Each face node block contains data for all angles in this AngleSet and all groups.
  const size_t num_psi_per_face_node_for_set = this->num_angles_ * this->num_groups_;

  /// Stride to jump from one angle's data block to the next (for the same face node) within
  /// `psi_data`. Each angle block contains data for all groups.
  const size_t num_groups_stride = this->num_groups_;

  /// Calculate the 1D offset into psi_data.
  const size_t dof_map =
    face_node_mapped *
      num_psi_per_face_node_for_set +    /// Offset to the start of data for this face_node_mapped
    angle_set_index * num_groups_stride; /// Further offset to the start of data for this specific
                                         /// angle_set_index (for group 0)

  if (dof_map + num_groups_stride > psi_data.size() && num_groups_stride > 0)
  {
    std::ostringstream err_stream;
    err_stream << "CBC_FLUDS::GetNonLocalUpwindPsi: Offset out of bounds. "
               << "Calculated_dof_map=" << dof_map << ", num_groups_stride=" << num_groups_stride
               << ", psi_data.size()=" << psi_data.size()
               << ", face_node_mapped=" << face_node_mapped
               << ", angle_set_index=" << angle_set_index
               << ", this->num_angles_=" << this->num_angles_
               << ", this->num_groups_=" << this->num_groups_;
    throw std::runtime_error(err_stream.str());
  }
  if (psi_data.empty() && (face_node_mapped > 0 || angle_set_index > 0))
  {
    throw std::runtime_error(
      "CBC_FLUDS::GetNonLocalUpwindPsi: Accessing non-empty psi_data with non-zero indices.");
  }

  return &psi_data[dof_map];
}

// This used to clear local_psi_data_. Now it means ensuring the pool is ready for a new sweep.
// If the sweep scheduler guarantees all allocated slots are deallocated by the end of a sweep,
// then the pool should be all free. This method then only needs to clear MPI message maps.
void
CBC_FLUDS::ClearLocalAndReceivePsi()
{
  if (psi_pool_ && psi_pool_->get_num_allocated_slots() > 0)
  {
    opensn::log.Log0Warning()
      << "CBC_FLUDS::ClearLocalAndReceivePsi: " << psi_pool_->get_num_allocated_slots()
      << " PSI slots still allocated. Potential logic error in sweep completion.";
    // Forcing a reset of the pool state might hide bugs.
    // One option: recreate the pool if this is truly a "hard reset" point.
    // For now, just clear the MPI message map.
  }

  live_cell_psi_pointers_
    .clear(); // Ensure map is clear for next round, assuming pool itself is reset/managed
  deplocs_outgoing_messages_.clear();
}

void
CBC_FLUDS::ForceDeallocateAllTrackedPsi()
{
  std::vector<uint64_t> keys_to_dealloc;
  for (const auto& pair : live_cell_psi_pointers_)
  {
    keys_to_dealloc.push_back(pair.first);
  }
  for (uint64_t key : keys_to_dealloc)
  {                            // Must iterate copy of keys
    DeallocatePsiForCell(key); // Uses existing public deallocate
  }
}
size_t
CBC_FLUDS::GetNumAllocatedPoolSlots() const
{
  return psi_pool_ ? psi_pool_->get_num_allocated_slots() : 0;
}

} // namespace opensn

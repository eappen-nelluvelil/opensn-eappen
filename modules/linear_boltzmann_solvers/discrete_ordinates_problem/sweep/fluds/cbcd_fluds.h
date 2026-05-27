// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_structs.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "caribou/main.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace crb = caribou;

namespace opensn
{

class CBCD_AngleSet;
class CBCD_AsynchronousCommunicator;
class CBCDSweepChunk;
class DiscreteOrdinatesProblem;
class LBSGroupset;
class AngleSet;
class UnknownManager;
class SpatialDiscretization;
class SweepBoundary;

/// Device CBC FLUDS.
class CBCD_FLUDS : public FLUDS
{
public:
  /**
   * Construct per-angle-set host and device angular-flux storage.
   *
   * \param num_groups Number of energy groups in the groupset.
   * \param num_angles Number of angles in the angle set.
   * \param num_local_cells Number of local cell-batch entries to reserve.
   * \param common_data Immutable indexing and communication metadata.
   * \param psi_uk_man Angular-flux unknown layout.
   * \param sdm Spatial discretization for saved angular flux.
   * \param save_angular_flux Whether angular flux must be retained after the sweep.
   */
  CBCD_FLUDS(std::size_t num_groups,
             std::size_t num_angles,
             std::size_t num_local_cells,
             const CBCD_FLUDSCommonData& common_data,
             const UnknownManager& psi_uk_man,
             const SpatialDiscretization& sdm,
             bool save_angular_flux);

  ~CBCD_FLUDS() override;

  const CBCD_FLUDSCommonData& GetCommonData() const { return common_data_; }

  crb::Stream& GetStream() { return stream_; }

  void AllocateLocalAndSavedPsi();

  /// Refresh reflecting copy plans and device store flags after boundary rebuilds.
  void BuildReflectingBoundaryPlans(
    const std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries);

  std::size_t GetStrideSize() const { return num_groups_and_angles_; }

  /// Return one of the three mapped cell-batch buffers.
  crb::MappedHostVector<std::uint32_t>& GetCellBatchBuffer(const std::size_t buffer_index)
  {
    return cell_batch_buffers_[buffer_index];
  }

  double* GetSavedAngularFluxDevicePointer() { return device_saved_psi_.get(); }

  /// Begin an asynchronous copy of saved angular flux to the host.
  void CopySaveAngularFluxFromDevice();

  /// Scatter saved angular flux from this angle set into the problem vector.
  void CopySaveAngularFluxToDestinationPsi(DiscreteOrdinatesProblem& problem,
                                           const LBSGroupset& groupset,
                                           AngleSet& angle_set);

  CBCD_FLUDSPointerSet GetDevicePointerSet() const { return pointer_set_; }

  /// Gather incoming boundary psi into mapped CBCD storage.
  void LoadIncomingBoundaryPsi(CBCDSweepChunk& sweep_chunk, CBCD_AngleSet& angle_set);

  /// Publish a completed cell batch to reflecting boundaries and MPI queues.
  void PublishOutgoingPsi(CBCDSweepChunk& sweep_chunk,
                          CBCD_AsynchronousCommunicator& async_comm,
                          std::size_t worker_id,
                          std::size_t angle_set_id,
                          const std::vector<std::uint32_t>& angle_indices,
                          std::span<const std::uint32_t> cell_local_ids);

  /// Store one received nonlocal face and return its downwind local cell ID.
  std::uint32_t StoreIncomingFace(std::uint32_t source_partition_index,
                                  std::uint32_t incoming_face_index,
                                  const std::byte* psi_values);

  void ClearLocalAndReceivePsi() override {}
  void ClearSendPsi() override {}
  void AllocateInternalLocalPsi() override {}
  void AllocateOutgoingPsi() override {}

  void AllocatePrelocIOutgoingPsi() override {}

private:
  const CBCD_FLUDSCommonData& common_data_;
  std::size_t num_local_psi_values_;
  std::size_t num_saved_psi_values_;
  /// Mapped host vectors for boundary and non-local angular fluxes.
  crb::MappedHostVector<double> incoming_boundary_psi_;
  crb::MappedHostVector<double> outgoing_boundary_psi_;
  crb::MappedHostVector<double> incoming_nonlocal_psi_;
  crb::MappedHostVector<double> outgoing_nonlocal_psi_;
  crb::Stream stream_;
  /// Triple-buffered ready, launched, and completed cell batches.
  std::array<crb::MappedHostVector<std::uint32_t>, 3> cell_batch_buffers_;
  bool save_angular_flux_;
  crb::DeviceMemory<double> local_psi_;
  crb::DeviceMemory<double> device_saved_psi_;
  crb::HostVector<double> host_saved_psi_;
  CBCD_FLUDSPointerSet pointer_set_;
  /// CSR offsets for reflecting face plans indexed by local cell ID.
  std::vector<std::uint32_t> reflecting_outgoing_boundary_face_offsets_;
  std::vector<ReflectingBoundaryFacePlan> reflecting_boundary_face_plans_;
  /// Device store flags; absent when no outgoing physical boundary reflects.
  crb::DeviceMemory<std::uint8_t> outgoing_boundary_reflecting_;
  /// Cells with reflecting or nonlocal output, refreshed with boundary plans.
  std::vector<std::uint8_t> outgoing_publication_;

  void CreatePointerSet();

  /// Return reflecting face plans for one local cell.
  std::span<const ReflectingBoundaryFacePlan>
  GetReflectingOutgoingBoundaryFaces(std::uint64_t cell_local_id) const
  {
    const auto begin = reflecting_outgoing_boundary_face_offsets_[cell_local_id];
    const auto end = reflecting_outgoing_boundary_face_offsets_[cell_local_id + 1];
    return {reflecting_boundary_face_plans_.data() + begin, end - begin};
  }
};

} // namespace opensn

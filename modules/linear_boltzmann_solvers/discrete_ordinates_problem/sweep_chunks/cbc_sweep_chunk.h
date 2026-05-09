// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/avx_sweep_chunk_utils.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_kernels.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"

namespace opensn
{

class CellMapping;
class DiscreteOrdinatesProblem;

/// CBC sweep chunk.
class CBCSweepChunk : public SweepChunk
{
public:
  CBCSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  void SetAngleSet(AngleSet& angle_set) override;

  void SetCell(const Cell* cell_ptr, AngleSet&) override;

  void Sweep(AngleSet& angle_set) override;

protected:
  template <class SweepChunkT>
  friend void
  PrepareOutgoingNonlocalFaceBuffers(SweepChunkT& sweep_chunk,
                                     const std::vector<FaceOrientation>& face_orientations);

  template <class SweepChunkT>
  friend void QueueOutgoingNonlocalFaceBuffers(SweepChunkT& sweep_chunk);

  template <bool time_dependent, class SweepChunkT>
  friend void CBC_Sweep_Generic(SweepChunkT& sweep_chunk, AngleSet& angle_set);

  template <unsigned int NumNodes, bool time_dependent, class SweepChunkT>
  friend void CBC_Sweep_FixedN(SweepChunkT& sweep_chunk, AngleSet& angle_set);

  CBC_FLUDS* fluds_ = nullptr;
  CBC_AsynchronousCommunicator* async_comm_ = nullptr;
  size_t gs_size_ = 0;
  unsigned int gs_gi_ = 0;
  size_t num_angles_in_as_ = 0;
  unsigned int group_stride_ = 0;
  size_t group_angle_stride_ = 0;

  const Cell* cell_ = nullptr;
  std::uint32_t cell_local_id_ = 0;
  const CellMapping* cell_mapping_ = nullptr;
  const CellLBSView* cell_transport_view_ = nullptr;
  CellOutflowView* cell_outflow_view_ = nullptr;
  size_t cell_num_faces_ = 0;
  size_t cell_num_nodes_ = 0;

  const DenseMatrix<Vector3>* G_ = nullptr;
  const DenseMatrix<double>* M_ = nullptr;
  const std::vector<DenseMatrix<double>>* M_surf_ = nullptr;
  const std::vector<Vector<double>>* IntS_shapeI_ = nullptr;

  unsigned int group_block_size_ = 0;
  CBCSweepScratch scratch_;

private:
  using SweepFunc = void (CBCSweepChunk::*)(AngleSet&);

  SweepFunc sweep_impl_ = nullptr;

  void Sweep_Generic(AngleSet& angle_set);

  template <unsigned int NumNodes>
  void Sweep_FixedN(AngleSet& angle_set);
};

} // namespace opensn

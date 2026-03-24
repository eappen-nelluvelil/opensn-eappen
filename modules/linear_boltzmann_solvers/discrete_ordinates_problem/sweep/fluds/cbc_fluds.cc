// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"
#include <algorithm>
#include <map>
#include <cassert>

namespace opensn
{

CBC_FLUDS::CBC_FLUDS(unsigned int num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     const UnknownManager& psi_uk_man,
                     const SpatialDiscretization& sdm)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    psi_uk_man_(psi_uk_man),
    sdm_(sdm),
    num_angles_in_gs_quadrature_(psi_uk_man_.GetNumberOfUnknowns()),
    num_local_spatial_dofs_(sdm_.GetNumLocalDOFs(psi_uk_man_) / num_angles_in_gs_quadrature_ /
                            num_groups_),
    local_psi_data_size_(num_local_spatial_dofs_ * num_groups_and_angles_),
    local_psi_data_(local_psi_data_size_)
{
  CALI_CXX_MARK_SCOPE("CBC_FLUDS::CBC_FLUDS");

  const auto& spds = common_data.GetSPDS();
  const auto& grid = *spds.GetGrid();
  const auto& cell_face_orientations = spds.GetCellFaceOrientations();
  const size_t num_local_cells = grid.local_cells.size();

  // --- Pre-compute cell_local_id -> psi data start offset ---
  cell_psi_data_start_.resize(num_local_cells);
  for (size_t c = 0; c < num_local_cells; ++c)
  {
    const auto& cell = grid.local_cells[c];
    const size_t spatial_dof_0 =
      sdm_.MapDOFLocal(cell, 0, psi_uk_man_, 0, 0) / num_angles_in_gs_quadrature_ / num_groups_;
    cell_psi_data_start_[c] = spatial_dof_0 * num_groups_and_angles_;
  }

  // ===================================================================
  // Build non-local send/receive buffer layouts
  // ===================================================================

  // Temporary structures to collect per-rank face information.
  // Key: rank. Value: list of {cell_global_id, face_id, data_size, cell_local_id}
  struct FaceRecord
  {
    uint64_t sort_key_global_id; // neighbor_global_id for send, cell_global_id for recv
    unsigned int sort_key_face;  // associated_face for send, face_id for recv
    size_t data_size;            // num_face_nodes * num_groups_and_angles_
    uint64_t cell_local_id;      // local cell owning this face
    unsigned int face_id;        // face index on local cell
  };

  std::map<int, std::vector<FaceRecord>> send_faces_by_rank;
  std::map<int, std::vector<FaceRecord>> recv_faces_by_rank;

  // Initialize offset tables
  nl_send_face_offset_.resize(num_local_cells);
  nl_recv_face_offset_.resize(num_local_cells);
  cell_send_contributions_.resize(num_local_cells);

  for (size_t c = 0; c < num_local_cells; ++c)
  {
    const auto& cell = grid.local_cells[c];
    const size_t num_faces = cell.faces.size();
    nl_send_face_offset_[c].assign(num_faces, SIZE_MAX);
    nl_recv_face_offset_[c].assign(num_faces, SIZE_MAX);
  }

  // Scan all local cells to identify non-local send/recv faces
  for (const auto& cell : grid.local_cells)
  {
    const auto& cell_mapping = sdm_.GetCellMapping(cell);
    const size_t num_faces = cell.faces.size();

    for (size_t f = 0; f < num_faces; ++f)
    {
      const auto& face = cell.faces[f];
      if (not face.has_neighbor)
        continue;
      if (grid.IsCellLocal(face.neighbor_id))
        continue;

      const auto orientation = cell_face_orientations[cell.local_id][f];
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const size_t data_size = num_face_nodes * num_groups_and_angles_;

      if (orientation == FaceOrientation::OUTGOING)
      {
        const int dest_rank = grid.cells[face.neighbor_id].partition_id;
        const auto& fnm = common_data_.GetFaceNodalMapping(cell.local_id, f);
        send_faces_by_rank[dest_rank].push_back(
          {face.neighbor_id,
           static_cast<unsigned int>(fnm.associated_face_),
           data_size,
           cell.local_id,
           static_cast<unsigned int>(f)});
      }
      else if (orientation == FaceOrientation::INCOMING)
      {
        const int source_rank = grid.cells[face.neighbor_id].partition_id;
        recv_faces_by_rank[source_rank].push_back(
          {cell.global_id,
           static_cast<unsigned int>(f),
           data_size,
           cell.local_id,
           static_cast<unsigned int>(f)});
      }
    }
  }

  // --- Build send layout ---
  // Sort faces within each rank by (sort_key_global_id, sort_key_face) for deterministic ordering
  size_t total_send_size = 0;
  for (auto& [rank, faces] : send_faces_by_rank)
  {
    std::sort(faces.begin(), faces.end(), [](const FaceRecord& a, const FaceRecord& b) {
      if (a.sort_key_global_id != b.sort_key_global_id)
        return a.sort_key_global_id < b.sort_key_global_id;
      return a.sort_key_face < b.sort_key_face;
    });
    for (const auto& fr : faces)
      total_send_size += fr.data_size;
  }

  nl_send_buffer_.assign(total_send_size, 0.0);
  nl_send_messages_.reserve(send_faces_by_rank.size());
  nl_send_total_faces_per_msg_.reserve(send_faces_by_rank.size());

  size_t send_offset = 0;
  for (auto& [rank, faces] : send_faces_by_rank)
  {
    const size_t msg_start = send_offset;
    const size_t msg_idx = nl_send_messages_.size();

    // Track per-cell contributions to this message
    std::map<uint64_t, size_t> cell_face_count;

    for (const auto& fr : faces)
    {
      nl_send_face_offset_[fr.cell_local_id][fr.face_id] = send_offset;
      send_offset += fr.data_size;
      cell_face_count[fr.cell_local_id]++;
    }

    nl_send_messages_.push_back({rank, msg_start, send_offset - msg_start});
    nl_send_total_faces_per_msg_.push_back(faces.size());

    for (const auto& [cell_lid, count] : cell_face_count)
      cell_send_contributions_[cell_lid].push_back({msg_idx, count});
  }

  // --- Build receive layout ---
  size_t total_recv_size = 0;
  for (auto& [rank, faces] : recv_faces_by_rank)
  {
    std::sort(faces.begin(), faces.end(), [](const FaceRecord& a, const FaceRecord& b) {
      if (a.sort_key_global_id != b.sort_key_global_id)
        return a.sort_key_global_id < b.sort_key_global_id;
      return a.sort_key_face < b.sort_key_face;
    });
    for (const auto& fr : faces)
      total_recv_size += fr.data_size;
  }

  nl_receive_buffer_.assign(total_recv_size, 0.0);
  nl_recv_messages_.reserve(recv_faces_by_rank.size());
  recv_msg_cell_local_ids_.reserve(recv_faces_by_rank.size());

  size_t recv_offset = 0;
  for (auto& [rank, faces] : recv_faces_by_rank)
  {
    const size_t msg_start = recv_offset;
    std::vector<uint64_t> cell_ids;

    for (const auto& fr : faces)
    {
      nl_recv_face_offset_[fr.cell_local_id][fr.face_id] = recv_offset;
      recv_offset += fr.data_size;
      cell_ids.push_back(fr.cell_local_id);
    }

    nl_recv_messages_.push_back({rank, msg_start, recv_offset - msg_start});
    recv_msg_cell_local_ids_.push_back(std::move(cell_ids));
  }
}

const FLUDSCommonData&
CBC_FLUDS::GetCommonData() const
{
  return common_data_;
}

double*
CBC_FLUDS::UpwindPsi(const Cell& face_neighbor, unsigned int adj_cell_node, size_t as_ss_idx)
{
  const size_t start = cell_psi_data_start_[face_neighbor.local_id];
  const size_t addr_offset = adj_cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;

  assert((start + addr_offset) < local_psi_data_.size());
  return &local_psi_data_[start + addr_offset];
}

double*
CBC_FLUDS::OutgoingPsi(const Cell& cell, unsigned int cell_node, size_t as_ss_idx)
{
  const size_t start = cell_psi_data_start_[cell.local_id];
  const size_t addr_offset = cell_node * num_groups_and_angles_ + as_ss_idx * num_groups_;

  assert((start + addr_offset) < local_psi_data_.size());
  return &local_psi_data_[start + addr_offset];
}

double*
CBC_FLUDS::NLUpwindPsi(uint64_t cell_local_id,
                       unsigned int face_id,
                       unsigned int face_node_mapped,
                       size_t as_ss_idx)
{
  const size_t base = nl_recv_face_offset_[cell_local_id][face_id];
  assert(base != SIZE_MAX);
  const size_t dof_map = face_node_mapped * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return &nl_receive_buffer_[base + dof_map];
}

double*
CBC_FLUDS::NLOutgoingPsi(size_t face_base_offset, size_t face_node, size_t as_ss_idx)
{
  const size_t addr_offset = face_node * num_groups_and_angles_ + as_ss_idx * num_groups_;
  return &nl_send_buffer_[face_base_offset + addr_offset];
}

size_t
CBC_FLUDS::GetNLSendFaceOffset(uint64_t cell_local_id, unsigned int face_id) const
{
  return nl_send_face_offset_[cell_local_id][face_id];
}

void
CBC_FLUDS::ClearLocalAndReceivePsi()
{
  std::fill(nl_receive_buffer_.begin(), nl_receive_buffer_.end(), 0.0);
}

} // namespace opensn

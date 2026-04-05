// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/mesh/cell/cell.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <new>

namespace opensn
{

namespace
{
constexpr std::size_t kLocalPsiAlignment = 64;
constexpr std::size_t kDoublesPerCacheLine = kLocalPsiAlignment / sizeof(double);

size_t
RoundUpToCacheLineMultiple(const size_t value)
{
  return ((value + kDoublesPerCacheLine - 1) / kDoublesPerCacheLine) * kDoublesPerCacheLine;
}
} // namespace

void
CBC_FLUDS::AlignedDoubleDeleter::operator()(double* ptr) const noexcept
{
  ::operator delete[](ptr, std::align_val_t{kLocalPsiAlignment});
}

CBC_FLUDS::AlignedDoubleBuffer
CBC_FLUDS::AllocateAlignedBuffer(const size_t num_values)
{
  auto* const ptr = static_cast<double*>(
    ::operator new[](num_values * sizeof(double), std::align_val_t{kLocalPsiAlignment}));
  std::fill_n(ptr, num_values, 0.0);
  return AlignedDoubleBuffer(ptr);
}

CBC_FLUDS::CBC_FLUDS(unsigned int num_groups,
                     size_t num_angles,
                     const CBC_FLUDSCommonData& common_data,
                     size_t max_cell_dof_count)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    num_slots_(static_cast<const CBC_SPDS&>(common_data.GetSPDS()).GetNumStaticLocalPsiSlots()),
    slot_size_(RoundUpToCacheLineMultiple(common_data.GetMaxLocalOutgoingNodeCount() *
                                          num_groups_and_angles_)),
    cell_slot_bases_(common_data.GetSPDS().GetGrid()->local_cells.size(), nullptr),
    local_psi_buffer_(AllocateAlignedBuffer(num_slots_ * slot_size_)),
    incoming_nonlocal_face_dof_offsets_(common_data.GetNumCellFaces(), 0),
    incoming_nonlocal_psi_buffer_(
      [&]()
      {
        size_t incoming_nonlocal_dof_count = 0;
        for (size_t face_storage_index = 0; face_storage_index < common_data.GetNumCellFaces();
             ++face_storage_index)
        {
          const auto& face_info =
            common_data.GetIncomingNonlocalFaceInfoByStorageIndex(face_storage_index);
          if (face_info.num_face_nodes == 0)
            continue;

          incoming_nonlocal_face_dof_offsets_[face_storage_index] = incoming_nonlocal_dof_count;
          incoming_nonlocal_dof_count += RoundUpToCacheLineMultiple(
            static_cast<size_t>(face_info.num_face_nodes) * num_groups_and_angles_);
        }
        return AllocateAlignedBuffer(incoming_nonlocal_dof_count);
      }())
{
  static_cast<void>(max_cell_dof_count);

  const auto& cbc_spds = static_cast<const CBC_SPDS&>(common_data.GetSPDS());
  const auto& task_list = cbc_spds.GetTaskList();
  const auto& task_slot_ids = cbc_spds.GetTaskSlotIDs();
  assert(task_list.size() == task_slot_ids.size());

  for (std::size_t task_idx = 0; task_idx < task_list.size(); ++task_idx)
  {
    const auto cell_local_id = task_list[task_idx].reference_id;
    const auto slot_id = task_slot_ids[task_idx];
    assert(slot_id < num_slots_);
    cell_slot_bases_[cell_local_id] =
      local_psi_buffer_.get() + static_cast<size_t>(slot_id) * slot_size_;
  }
}

void
CBC_FLUDS::StoreIncomingFaceData(uint64_t cell_global_id,
                                 unsigned int face_id,
                                 const double* psi_data,
                                 size_t data_size)
{
  const auto face_storage_index =
    common_data_.GetIncomingNonlocalFaceStorageIndexByKey(cell_global_id, face_id);
  const auto& face_info =
    common_data_.GetIncomingNonlocalFaceInfoByStorageIndex(face_storage_index);

  const auto expected_size = static_cast<size_t>(face_info.num_face_nodes) * num_groups_and_angles_;
  assert(data_size == expected_size);

  const size_t base = incoming_nonlocal_face_dof_offsets_[face_storage_index];
  std::memcpy(incoming_nonlocal_psi_buffer_.get() + base, psi_data, data_size * sizeof(double));
}

void
CBC_FLUDS::ClearLocalAndReceivePsi()
{
}

} // namespace opensn

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace opensn::detail
{

/// Result of an exact local-face slot assignment.
struct SlotSolveResult
{
  std::size_t slot_count = 0;
  bool verifier_rejected = false;
};

/**
 * Compute the exact minimum safe local-face slot assignment.
 *
 * A face may reuse another face's slot when its producer follows the earlier face's
 * consumer in the reduced task DAG. The minimum chain cover of this face poset equals
 * its width by Dilworth's theorem. The implementation obtains that cover with sparse
 * maximum flow and returns the identity assignment if final verification fails.
 */
SlotSolveResult
ComputeLocalFaceSlotPlan(const std::vector<std::uint32_t>& successor_rank_offsets,
                         const std::vector<std::uint32_t>& successor_ranks,
                         const std::vector<std::uint32_t>& face_producer_ranks,
                         const std::vector<std::uint32_t>& face_consumer_ranks,
                         const std::vector<std::uint32_t>& producer_cell_face_offsets,
                         std::vector<std::uint32_t>& face_slot_ids);

} // namespace opensn::detail

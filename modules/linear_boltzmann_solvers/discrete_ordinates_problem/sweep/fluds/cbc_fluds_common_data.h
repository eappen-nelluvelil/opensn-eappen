// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include <cinttypes>
#include <cstddef>

namespace opensn
{

class CBC_FLUDSCommonData : public FLUDSCommonData
{
public:
  CBC_FLUDSCommonData(const SPDS& spds,
                      const std::vector<CellFaceNodalMapping>& grid_nodal_mappings);

  /// Number of non-local outgoing faces (for pre-reserving outgoing message queues).
  size_t GetNumNonLocalOutgoingFaces() const { return num_nonlocal_outgoing_faces_; }

  /// Number of non-local incoming faces (for pre-reserving incoming message storage).
  size_t GetNumNonLocalIncomingFaces() const { return num_nonlocal_incoming_faces_; }

private:
  size_t num_nonlocal_outgoing_faces_ = 0;
  size_t num_nonlocal_incoming_faces_ = 0;
};

} // namespace opensn

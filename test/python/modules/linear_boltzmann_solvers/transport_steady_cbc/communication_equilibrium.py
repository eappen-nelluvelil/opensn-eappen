# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Uniform reflecting absorber: phi_g = q_g / Sigma_a,g = g+1.

The constant solution is exactly represented by PWLD and the quadrature. Check
every group's extrema, integrated absorption/production, and zero net leakage.
The 1e-8 relative acceptance tolerance allows accumulated floating-point and
iterative errors with an absolute linear-solve tolerance of 1e-12.

The first groupset has 32 groups: one four-node, single-angle face holds 1024
payload bytes, so a 1024-byte message limit forces fragmentation once headers
are included. The second groupset tests a nonzero group offset. Both executions
reuse the same solver/communicators. The cyclic mesh also exercises delayed
faces and completion markers; the orthogonal mesh exercises acyclic faces.
"""

import math
from pathlib import Path

if "opensn_console" not in globals():
    from mpi4py import MPI
    from pyopensn.mesh import (
        OrthogonalMeshGenerator, ExtruderMeshGenerator, FromFileMeshGenerator,
        KBAGraphPartitioner,
    )
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size

assert size == globals().get("expected_ranks", 4)
assets = Path(__file__).resolve().parents[4] / "assets"
if globals().get("cyclic", False):
    grid = ExtruderMeshGenerator(
        inputs=[FromFileMeshGenerator(
            filename=str(assets / "mesh/square2x2_partition_cyclic3.obj"))],
        layers=[{"z": 1.6, "n": 4}],
        partitioner=KBAGraphPartitioner(nx=2, ny=2, xcuts=[0.0], ycuts=[0.0]),
    ).Execute()
    volume = 4.0 * 1.6
else:
    nodes = [float(i) for i in range(5)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()
    volume = 64.0
grid.SetUniformBlockID(0)
grid.SetOrthogonalBoundaries()
xs = MultiGroupXS()
xs.LoadFromOpenSn(str(assets / "xs/absorber_33g.xs"))
quad = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)
problem = DiscreteOrdinatesProblem(
    mesh=grid, num_groups=33, sweep_type="CBC",
    groupsets=[{
        "groups_from_to": pair, "angular_quadrature": quad,
        "angle_aggregation_type": "single", "allow_cycles": True,
        "inner_linear_method": "petsc_gmres", "l_abs_tol": 1e-12, "l_max_its": 300,
    } for pair in ((0, 31), (32, 32))],
    xs_map=[{"block_ids": [0], "xs": xs}],
    volumetric_sources=[VolumetricSource(
        block_ids=[0], group_strength=[float(g + 1) for g in range(33)])],
    boundary_conditions=[{"name": name, "type": "reflecting"}
                         for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")],
    options={"max_mpi_message_size": globals().get("message_size", 1024),
             "save_angular_flux": True},
)
solver = SteadyStateSourceSolver(problem=problem, compute_balance=True)
solver.Initialize()
error = 0.0
production = volume * sum(range(1, 34))
for execution in range(2):
    solver.Execute()
    fields = problem.GetScalarFluxFieldFunction(only_scalar_flux=False)
    assert len(fields) == 33 and all(len(moments) == 1 for moments in fields)
    for group, moments in enumerate(fields):
        for operation in ("min", "max"):
            interpolation = FieldFunctionInterpolationVolume()
            interpolation.SetOperationType(operation)
            interpolation.SetLogicalVolume(RPPLogicalVolume(infx=True, infy=True, infz=True))
            interpolation.AddFieldFunction(moments[0])
            interpolation.Execute()
            value = interpolation.GetValue()
            assert math.isfinite(value)
            error = max(error, abs(value / (group + 1) - 1.0))
    balance = solver.ComputeBalanceTable()
    for name in ("absorption_rate", "production_rate", "outflow_rate", "inflow_rate"):
        assert math.isfinite(balance[name])
    error = max(error, abs(balance["absorption_rate"] / production - 1.0),
                abs(balance["production_rate"] / production - 1.0),
                abs(balance["outflow_rate"] - balance["inflow_rate"]) / production)
if rank == 0:
    print(f"Communication equilibrium error={error:.12e}")

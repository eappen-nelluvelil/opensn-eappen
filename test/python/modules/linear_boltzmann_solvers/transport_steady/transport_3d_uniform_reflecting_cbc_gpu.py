#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D PWLD uniform-source transport with reflecting boundaries (CBC)
The constant solution is phi = q / Sigma_a = 1.25. Absorption equals the
volume source of 64, with zero net leakage. Both are represented exactly
by the spatial and angular discretizations. Test: Max-difference=0.0
"""

import os
import sys
import numpy as np
from mpi4py import MPI

if "opensn_console" not in globals():
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver


if __name__ == "__main__":
    nodes = [float(i) for i in range(5)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()
    grid.SetUniformBlockID(0)
    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(1.0, 0.2)
    quadrature = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)

    def run_problem():
        problem = DiscreteOrdinatesProblem(
            mesh=grid,
            num_groups=1,
            sweep_type="CBC",
            use_gpus=True,
            groupsets=[{
                "groups_from_to": (0, 0),
                "angular_quadrature": quadrature,
                "angle_aggregation_type": "single",
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-12,
                "l_max_its": 300,
                "gmres_restart_interval": 30,
            }],
            xs_map=[{"block_ids": [0], "xs": xs}],
            volumetric_sources=[VolumetricSource(block_ids=[0], group_strength=[1.0])],
            boundary_conditions=[
                {"name": face, "type": "reflecting"}
                for face in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
            ],
        )
        solver = SteadyStateSourceSolver(problem=problem, compute_balance=True)
        solver.Initialize()
        errors = []
        for _ in range(2):
            solver.Execute()
            errors.append(float(np.max(np.abs(np.array(problem.GetPhiOldLocal()) - 1.25))))
            balance = solver.ComputeBalanceTable()
            errors.append(abs(balance["absorption_rate"] - 64.0) / 64.0)
            errors.append(abs(balance["outflow_rate"] - balance["inflow_rate"]) / 64.0)
        return max(errors) if np.all(np.isfinite(errors)) else float("inf")

    difference = max(run_problem(), run_problem())
    difference = MPI.COMM_WORLD.allreduce(difference, op=MPI.MAX)
    if rank == 0:
        print(f"Max-difference={difference:.12e}")

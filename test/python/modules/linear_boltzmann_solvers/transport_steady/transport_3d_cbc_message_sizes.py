#!/usr/bin/env python3
"""Compare fragmented and unfragmented CBC face fluxes on a cyclic partitioning."""

import os
import sys
import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import ExtruderMeshGenerator, FromFileMeshGenerator, KBAGraphPartitioner
    from pyopensn.xs import MultiGroupXS
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.logvol import RPPLogicalVolume

    def MPIAllReduce(value, op):
        return MPI.COMM_WORLD.allreduce(value, op=MPI.MAX)


if __name__ == "__main__":
    if size != 4:
        sys.exit(f"Incorrect number of processors. Expected 4 but got {size}.")

    meshgen = ExtruderMeshGenerator(
        inputs=[FromFileMeshGenerator(
            filename="../../../../assets/mesh/square2x2_partition_cyclic3.obj")],
        layers=[{"z": 1.6, "n": 2}],
        partitioner=KBAGraphPartitioner(nx=2, ny=2, xcuts=[0.0], ycuts=[0.0]),
    )
    grid = meshgen.Execute()
    grid.SetOrthogonalBoundaries()
    grid.SetBlockIDFromLogicalVolume(RPPLogicalVolume(infx=True, infy=True, infz=True), 0, True)
    xs = MultiGroupXS()
    xs.LoadFromOpenSn("../../../../assets/xs/xs_graphite_pure.xs")
    quad = GLCProductQuadrature3DXYZ(n_polar=4, n_azimuthal=4, scattering_order=1)
    source = [1.0] + [0.0] * 20
    reference = None

    # Polar aggregation makes each nonlocal face exceed the 1024-byte limit.
    for message_size in (32768, 1024, 2048):
        problem = DiscreteOrdinatesProblem(
            mesh=grid,
            num_groups=21,
            groupsets=[{
                "groups_from_to": [0, 20],
                "angular_quadrature": quad,
                "angle_aggregation_type": "polar",
                "allow_cycles": True,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-10,
                "l_max_its": 300,
                "gmres_restart_interval": 30,
            }],
            xs_map=[{"block_ids": [0], "xs": xs}],
            boundary_conditions=[{"name": "zmax", "type": "isotropic",
                                  "group_strength": source}],
            options={"max_mpi_message_size": message_size, "save_angular_flux": False},
            sweep_type="CBC",
        )
        solver = SteadyStateSourceSolver(problem=problem)
        solver.Initialize()
        solver.Execute()
        phi = np.array(problem.GetPhiNewLocal(), copy=True)
        if reference is None:
            reference = phi
            nonfinite = MPIAllReduce(int(not np.all(np.isfinite(reference))), "max")
            magnitude = MPIAllReduce(float(np.max(np.abs(reference))), "max")
            if rank == 0:
                print(f"Nonzero reference={int(nonfinite == 0 and magnitude > 0.0)}")
        else:
            local_error = (float(np.max(np.abs(phi - reference)))
                           if np.all(np.isfinite(phi)) else float("inf"))
            error = MPIAllReduce(local_error, "max")
            if rank == 0:
                print(f"Message-size error {message_size}={error:.12e}")

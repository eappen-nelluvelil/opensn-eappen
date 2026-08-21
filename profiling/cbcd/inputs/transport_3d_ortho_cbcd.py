#!/usr/bin/env python3
"""Rank-parameterized 3-D orthogonal CBCD workload for local profiling."""

import os
import sys


def get_int_option(name, default):
    """Return a positive integer injected through OpenSn's ``--py`` option."""
    value = int(globals().get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../test")))
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.source import VolumetricSource
    from pyopensn.xs import MultiGroupXS


if __name__ == "__main__":
    cells_per_axis = get_int_option("profile_cells_per_axis", 18)
    max_iterations = get_int_option("profile_max_iterations", 100)
    num_groups = 21
    num_polar = 4
    num_azimuthal = 8

    length = 5.0
    minimum = -0.5 * length
    spacing = length / cells_per_axis
    nodes = [minimum + i * spacing for i in range(cells_per_axis + 1)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()

    domain = RPPLogicalVolume(infx=True, infy=True, infz=True)
    grid.SetBlockIDFromLogicalVolume(domain, 0, True)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    xs_graphite = MultiGroupXS()
    xs_graphite.LoadFromOpenSn(
        os.path.join(repo_root, "test/assets/xs/xs_graphite_pure.xs")
    )

    zero_source = VolumetricSource(
        block_ids=[0], group_strength=[0.0 for _ in range(num_groups)]
    )
    boundary_source = [0.0 for _ in range(num_groups)]
    boundary_source[0] = 1.0
    quadrature = GLCProductQuadrature3DXYZ(
        n_polar=num_polar,
        n_azimuthal=num_azimuthal,
        scattering_order=1,
    )

    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        sweep_type="CBC",
        use_gpus=True,
        groupsets=[
            {
                "groups_from_to": [0, num_groups - 1],
                "angular_quadrature": quadrature,
                "angle_aggregation_type": "single",
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-6,
                "l_max_its": max_iterations,
                "gmres_restart_interval": max_iterations,
            }
        ],
        xs_map=[{"block_ids": [0], "xs": xs_graphite}],
        volumetric_sources=[zero_source],
        boundary_conditions=[
            {"name": "xmin", "type": "isotropic", "group_strength": boundary_source}
        ],
        options={"save_angular_flux": False},
    )

    if rank == 0:
        print(
            "CBCD_PROFILE_CONFIG "
            f"ranks={size} cells_per_axis={cells_per_axis} "
            f"cells={cells_per_axis ** 3} groups={num_groups} "
            f"directions={num_polar * num_azimuthal} save_angular_flux=false"
        )

    solver = SteadyStateSourceSolver(problem=problem)
    solver.Initialize()
    solver.Execute()

    scalar_flux = problem.GetScalarFluxFieldFunction()
    for group in (0, num_groups - 2):
        interpolator = FieldFunctionInterpolationVolume()
        interpolator.SetOperationType("max")
        interpolator.SetLogicalVolume(domain)
        interpolator.AddFieldFunction(scalar_flux[group])
        interpolator.Execute()
        if rank == 0:
            print(f"CBCD_PROFILE_MAX group={group} value={interpolator.GetValue():.12e}")

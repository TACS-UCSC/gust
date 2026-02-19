"""Dedalus Rayleigh-Bénard convection solver.

2D Boussinesq convection in a rectangular domain with:
- Periodic (Fourier) in x, wall-bounded (Chebyshev) in z
- No-slip walls, fixed temperature BCs
- CFL-adaptive RK222 timestepping

Based on RudyMorel/the-well-rbc-sf.
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
import dedalus.public as d3

from .global_constants import filename_rbc


def generate_rayleigh_benard(
    resolution,
    rayleigh, prandtl,
    init, seed, dT,
    dpath, safety_factor=32, min_dt=1e-8,
    stop_sim_time=50, snapshot_dt=0.25, max_writes=200,
):
    """Run a single Rayleigh-Bénard convection simulation.

    Args:
        resolution: (Nx, Nz) grid points
        rayleigh: Rayleigh number
        prandtl: Prandtl number
        init: initial condition type ("default")
        seed: random seed for IC noise
        dT: temperature difference for linear background
        dpath: output directory (Path)
        safety_factor: CFL safety divisor (higher = smaller dt)
        min_dt: minimum timestep
        stop_sim_time: total simulation time
        snapshot_dt: time between saved snapshots
        max_writes: max snapshots per HDF5 file
    """
    save_name = filename_rbc.format(
        resolution[0], resolution[1], rayleigh, prandtl, dT, seed
    ).replace('.', '_')

    # Parameters
    Lx, Lz = 4, 1
    Nx, Nz = resolution[0], resolution[1]
    Rayleigh = rayleigh
    Prandtl = prandtl
    dealias = 3 / 2
    timestepper = d3.RK222
    max_timestep = 0.125
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields
    p = dist.Field(name='p', bases=(xbasis, zbasis))
    b = dist.Field(name='b', bases=(xbasis, zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl) ** (-1 / 2)
    nu = (Rayleigh / Prandtl) ** (-1 / 2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez * lift(tau_u1)
    grad_b = d3.grad(b) + ez * lift(tau_b1)

    # Problem
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
    problem.add_equation("b(z=0) = Lz")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0")

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    if init == "default":
        b.fill_random('g', seed=seed, distribution='normal', scale=1e-3)
        b['g'] *= z * (Lz - z)
        b['g'] += dT * (Lz - z)
    else:
        raise ValueError(f"Unknown initial condition: {init}")

    # Analysis
    snapshots = solver.evaluator.add_file_handler(
        str(dpath / save_name), sim_dt=snapshot_dt, max_writes=max_writes)
    snapshots.add_task(b, name='buoyancy')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
    snapshots.add_task(p, name='pressure')
    snapshots.add_task(u, name='velocity')

    # CFL
    CFL = d3.CFL(
        solver, initial_dt=max_timestep, cadence=10,
        safety=0.5 / 4 / safety_factor, threshold=0.05,
        max_change=1.5, min_change=0.5, max_dt=max_timestep, min_dt=min_dt,
    )
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u @ u) / nu, name='Re')

    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration - 1) % 10 == 0:
                max_Re = flow.max('Re')
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' % (
                    solver.iteration, solver.sim_time, timestep, max_Re))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

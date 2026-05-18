"""Correctness tests for `TimeDependentPDEVariationalProblem` (fenicsx env).

Checks:
- Forward solve preserves zero state when initial condition is zero and
  source is zero (sanity).
- Gradient FD slope is 1 across many decades of eps (i.e. analytic
  gradient is correct).
- Hessian symmetry to ~1e-12.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
import dolfinx as dlx
from mpi4py import MPI

import hippylibX as hpx
from hippylibX import time_dependent as td

STATE, PARAMETER, ADJOINT = hpx.STATE, hpx.PARAMETER, hpx.ADJOINT


@pytest.fixture(scope="module")
def setup():
    nx, ny, nt = 8, 8, 3
    T_INIT, T_FINAL = 0.0, 1.0

    comm = MPI.COMM_WORLD
    msh = dlx.mesh.create_unit_square(comm, nx, ny)
    Vh2 = dlx.fem.functionspace(msh, ("Lagrange", 2))
    Vh1 = dlx.fem.functionspace(msh, ("Lagrange", 1))
    Vh = [Vh2, Vh1, Vh2]

    def boundary(x):
        return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

    fdim = msh.topology.dim - 1
    facets = dlx.mesh.locate_entities_boundary(msh, fdim, boundary)
    dofs = dlx.fem.locate_dofs_topological(Vh[STATE], fdim, facets)
    uD = dlx.fem.Function(Vh[STATE])
    uD.interpolate(lambda x: 0.0 * x[0])
    uD.x.scatter_forward()
    bc = dlx.fem.dirichletbc(uD, dofs)

    u0 = dlx.fem.Function(Vh[STATE])
    u0.interpolate(lambda x: x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]))
    u0.x.scatter_forward()

    class HeatVarf:
        def __init__(self, dt):
            self._dt = float(dt)
            self.dt_inv = dlx.fem.Constant(msh, dlx.default_scalar_type(1.0 / dt))

        @property
        def dt(self):
            return self._dt

        def __call__(self, u, u_old, m, p, t):
            return (
                (u - u_old) * p * self.dt_inv * ufl.dx
                + ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx
            )

    pde = td.TimeDependentPDEVariationalProblem(
        Vh, HeatVarf((T_FINAL - T_INIT) / nt), bc=[bc], bc0=[bc], u0=u0,
        t_init=T_INIT, t_final=T_FINAL, is_fwd_linear=True,
    )

    prior_mean = dlx.fem.Function(Vh[PARAMETER])
    prior_mean.x.array[:] = 0.0
    prior_mean = prior_mean.x
    prior = hpx.BiLaplacianPrior(Vh[PARAMETER], 0.1, 0.5, mean=prior_mean)

    return msh, Vh, bc, u0, pde, prior


def test_forward_zero_initial_condition_zero_solution(setup):
    msh, Vh, bc, u0, pde, prior = setup
    # Replace u0 with zero
    u0_zero = dlx.fem.Function(Vh[STATE])
    u0_zero.x.array[:] = 0.0
    u0_zero.x.scatter_forward()
    pde.init_cond = u0_zero

    m = prior.generate_parameter(0)
    m.array[:] = 0.0
    m.scatter_forward()

    u = pde.generate_state()
    pde.solveFwd(u, [u, m, None])
    for t in pde.times:
        assert float(u.view(t).petsc_vec.norm()) < 1e-10
    # restore IC
    pde.init_cond = u0


def test_forward_with_nontrivial_ic_decays_in_time(setup):
    """With m=0 (so diffusivity = 1), heat eq decays IC monotonically in L2."""
    msh, Vh, bc, u0, pde, prior = setup
    m = prior.generate_parameter(0)
    m.array[:] = 0.0
    m.scatter_forward()

    u = pde.generate_state()
    pde.solveFwd(u, [u, m, None])
    norms = [float(u.view(t).petsc_vec.norm()) for t in pde.times]
    # IC is positive bump; solution should decay strictly (backward Euler, theta=1)
    for k in range(1, len(norms)):
        assert norms[k] < norms[k - 1], f"step {k}: {norms}"


def test_fd_gradient_slope_one(setup):
    """For misfit_only, the FD gradient error must scale ~O(eps) at small eps."""
    msh, Vh, bc, u0, pde, prior = setup

    # build a simple misfit on the final state only via a TD problem with
    # zero-noise data set to u_true at every frame
    m_true_fun = dlx.fem.Function(Vh[PARAMETER])
    m_true_fun.interpolate(lambda x: np.log(2.0 + np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])))
    m_true_fun.x.scatter_forward()
    m_true = m_true_fun.x

    u_true = pde.generate_state()
    pde.solveFwd(u_true, [u_true, m_true, None])

    misfits = []
    for t in pde.times:
        obs = td.ContinuousStateObservation(Vh[STATE], ufl.dx, bcs=[bc])
        obs.d.array[:] = u_true.view(t).array[:]
        obs.d.scatter_forward()
        obs.noise_variance = 1e-4
        misfits.append(obs)
    misfit = td.MisfitTD(misfits, pde.times)

    model = hpx.Model(pde, prior, misfit)

    m0 = prior.generate_parameter(0)
    m0.array[:] = 0.1
    m0.scatter_forward()

    eps = np.power(0.5, np.arange(10, 22))[::-1]  # 12 values in a small-eps band
    res = hpx.modelVerify(
        model, m0, is_quadratic=False, misfit_only=True, verbose=False, eps=eps,
    )
    # ratio err_grad[i+1] / err_grad[i] should be ~2 (eps doubles)
    err_grad = res["err_grad"]
    ratios = err_grad[1:] / np.maximum(err_grad[:-1], 1e-30)
    # accept band [1.5, 2.5] for "slope ~ 1"
    assert np.median(ratios) > 1.5 and np.median(ratios) < 2.5, (
        f"median FD-grad ratio {np.median(ratios):.3f} (ratios={ratios})"
    )
    # symmetry
    assert abs(res["sym_Hessian_value"]) < 1e-10


def test_hessian_symmetry_with_prior(setup):
    msh, Vh, bc, u0, pde, prior = setup

    m_true_fun = dlx.fem.Function(Vh[PARAMETER])
    m_true_fun.interpolate(lambda x: np.log(2.0) + 0.0 * x[0])
    m_true_fun.x.scatter_forward()
    m_true = m_true_fun.x

    u_true = pde.generate_state()
    pde.solveFwd(u_true, [u_true, m_true, None])

    misfits = []
    for t in pde.times:
        obs = td.ContinuousStateObservation(Vh[STATE], ufl.dx, bcs=[bc])
        obs.d.array[:] = u_true.view(t).array[:]
        obs.d.scatter_forward()
        obs.noise_variance = 1e-3
        misfits.append(obs)
    misfit = td.MisfitTD(misfits, pde.times)
    model = hpx.Model(pde, prior, misfit)

    m0 = prior.generate_parameter(0)
    m0.array[:] = 0.1
    m0.scatter_forward()
    res = hpx.modelVerify(
        model, m0, is_quadratic=False, misfit_only=False, verbose=False,
        eps=np.array([1e-5, 1e-6]),
    )
    assert abs(res["sym_Hessian_value"]) < 1e-10

# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Direct unit tests for KKT-block primitives on
:class:`TimeDependentPDEVariationalProblem`.

Most of the KKT methods (`applyC`, `applyCt`, `applyWuu`, `applyWum`,
`applyWmu`, `applyWmm`) are exercised indirectly via :func:`modelVerify`
and the parity tests. These tests assert the algebraic identities that
those operators must satisfy, directly on small problems.

Identities checked:

1. **applyCt is the transpose of applyC** in the inner product
   ``<applyCt(dp), dm>_M = <applyC(dm), dp>_M`` (across time).
2. **applyWmu is the transpose of applyWum** for any directions.
3. **applyWuu, applyWmm symmetry** (matrix-Hessian symmetry).
4. **gauss_newton_approx zeroes the W blocks** that come from the PDE.
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
def small_heat():
    """Build a small heat-equation problem and a linearization point."""
    comm = MPI.COMM_WORLD
    msh = dlx.mesh.create_unit_square(comm, 6, 6)
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
            return ((u - u_old) * p * self.dt_inv * ufl.dx
                    + ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx)

    pde = td.TimeDependentPDEVariationalProblem(
        Vh, HeatVarf(1.0 / 3), bc=[bc], bc0=[bc], u0=u0,
        t_init=0.0, t_final=1.0, is_fwd_linear=True,
    )

    m_fun = dlx.fem.Function(Vh[PARAMETER])
    m_fun.interpolate(lambda x: 0.2 + 0.1 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    m_fun.x.scatter_forward()
    m = m_fun.x

    # forward + adjoint at this m
    u = pde.generate_state()
    pde.solveFwd(u, [u, m, None])

    p = pde.generate_state()  # zero adjoint is fine for linearization
    return pde, Vh, m, u, p, bc


def _td_inner(a, b):
    """Sum of per-snapshot petsc dot products."""
    return sum(float(a.data[k].petsc_vec.dot(b.data[k].petsc_vec))
               for k in range(a.nsteps))


def test_applyC_is_transpose_of_applyCt(small_heat):
    """The KKT operators implicitly act on the homogeneous-BC subspace
    of state space. The duality identity holds only when ``dp`` is zeroed
    on BC dofs; otherwise applyC's BC zeroing breaks the symmetry."""
    pde, Vh, m, u, p, bc = small_heat
    pde.setLinearizationPoint([u, m, p], gauss_newton_approx=False)

    rng = np.random.default_rng(11)
    dm = pde.generate_parameter()
    dm.array[:] = rng.normal(size=dm.array.shape)
    dm.scatter_forward()

    # dp ∈ homogeneous-BC subspace: random, then zero BC dofs
    bc_dofs = bc._cpp_object.dof_indices()[0]
    dp = pde.generate_state()
    for k in range(dp.nsteps):
        dp.data[k].array[:] = rng.normal(size=dp.data[k].array.shape)
        dp.data[k].array[bc_dofs] = 0.0
        dp.data[k].scatter_forward()

    Cdm = pde.generate_state()
    pde.applyC(dm, Cdm)

    Ctdp = pde.generate_parameter()
    pde.applyCt(dp, Ctdp)

    lhs = _td_inner(Cdm, dp)
    rhs = float(dm.petsc_vec.dot(Ctdp.petsc_vec))
    rel = abs(lhs - rhs) / max(abs(rhs), 1.0)
    assert rel < 1e-10, f"<C dm, dp>={lhs} vs <dm, C^T dp>={rhs} (rel={rel})"


def test_applyWum_is_transpose_of_applyWmu(small_heat):
    pde, Vh, m, u, p, bc = small_heat
    pde.setLinearizationPoint([u, m, p], gauss_newton_approx=False)

    dm = pde.generate_parameter(); dm.array[:] = 0.5; dm.scatter_forward()
    du = pde.generate_state()
    for k in range(du.nsteps):
        du.data[k].array[:] = (k + 1) * 0.4
        du.data[k].scatter_forward()

    Wum_dm = pde.generate_state()
    pde.applyWum(dm, Wum_dm)
    Wmu_du = pde.generate_parameter()
    pde.applyWmu(du, Wmu_du)

    lhs = _td_inner(Wum_dm, du)
    rhs = float(dm.petsc_vec.dot(Wmu_du.petsc_vec))
    rel = abs(lhs - rhs) / max(abs(rhs), 1.0)
    assert rel < 1e-10, f"<Wum dm, du>={lhs} vs <dm, Wmu du>={rhs} (rel={rel})"


def test_applyWmm_is_symmetric(small_heat):
    pde, Vh, m, u, p, bc = small_heat
    pde.setLinearizationPoint([u, m, p], gauss_newton_approx=False)

    rng = np.random.default_rng(7)
    a = pde.generate_parameter(); a.array[:] = rng.normal(size=a.array.shape); a.scatter_forward()
    b = pde.generate_parameter(); b.array[:] = rng.normal(size=b.array.shape); b.scatter_forward()

    Wmm_a = pde.generate_parameter()
    Wmm_b = pde.generate_parameter()
    pde.applyWmm(a, Wmm_a)
    pde.applyWmm(b, Wmm_b)

    lhs = float(b.petsc_vec.dot(Wmm_a.petsc_vec))
    rhs = float(a.petsc_vec.dot(Wmm_b.petsc_vec))
    rel = abs(lhs - rhs) / max(abs(rhs), 1.0)
    assert rel < 1e-10, f"Wmm asymmetric: <b, Wmm a>={lhs} vs <a, Wmm b>={rhs}"


def test_gauss_newton_approx_zeroes_pde_W_blocks(small_heat):
    pde, Vh, m, u, p, bc = small_heat
    pde.setLinearizationPoint([u, m, p], gauss_newton_approx=True)

    dm = pde.generate_parameter(); dm.array[:] = 1.0; dm.scatter_forward()
    du = pde.generate_state()
    for k in range(du.nsteps):
        du.data[k].array[:] = 1.0
        du.data[k].scatter_forward()

    Wuu_du = pde.generate_state()
    pde.applyWuu(du, Wuu_du)
    Wum_dm = pde.generate_state()
    pde.applyWum(dm, Wum_dm)
    Wmu_du = pde.generate_parameter()
    pde.applyWmu(du, Wmu_du)
    Wmm_dm = pde.generate_parameter()
    pde.applyWmm(dm, Wmm_dm)

    for tdv in (Wuu_du, Wum_dm):
        for k in range(tdv.nsteps):
            assert np.allclose(tdv.data[k].array, 0.0)
    assert np.allclose(Wmu_du.array, 0.0)
    assert np.allclose(Wmm_dm.array, 0.0)

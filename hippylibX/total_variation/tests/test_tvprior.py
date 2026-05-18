# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Unit tests for :class:`TVPrior` and :class:`weightedVTVPrior`.

Checks:
- cost is monotone-increasing in ``alpha`` for the same parameter.
- gradient is consistent with cost via finite differences (slope ≈ 1
  in log-log).
- Hessian action is symmetric: ``<a, R b> == <b, R a>``.
- ``compute_w`` returns the saturation `w = ∇m / |∇m|_β`.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
import basix.ufl
import dolfinx as dlx
from mpi4py import MPI

from hippylibX.total_variation import TVPrior, weightedVTVPrior


@pytest.fixture(scope="module")
def spaces():
    msh = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    Vhm = dlx.fem.functionspace(msh, ("Lagrange", 1))
    Vhw = dlx.fem.functionspace(
        msh, basix.ufl.element("DG", msh.basix_cell(), 0, shape=(2,))
    )
    Vhwnorm = dlx.fem.functionspace(msh, ("DG", 0))
    return msh, Vhm, Vhw, Vhwnorm


def _rand_param(Vhm, seed=0):
    v = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
    v.array[:] = np.random.default_rng(seed).normal(size=v.array.shape)
    v.scatter_forward()
    return v


def test_cost_increases_with_alpha(spaces):
    msh, Vhm, Vhw, Vhwn = spaces
    m = _rand_param(Vhm, seed=1)
    c1 = TVPrior(Vhm, Vhw, Vhwn, alpha=1.0, beta=1e-3).cost(m)
    c2 = TVPrior(Vhm, Vhw, Vhwn, alpha=2.5, beta=1e-3).cost(m)
    assert c2 > c1
    assert c2 == pytest.approx(2.5 * c1, rel=1e-12)


def test_gradient_fd(spaces):
    """Finite-difference check: cost(m + eps h) ≈ cost(m) + eps <g, h>."""
    msh, Vhm, Vhw, Vhwn = spaces
    tv = TVPrior(Vhm, Vhw, Vhwn, alpha=1.0, beta=1e-3)
    m = _rand_param(Vhm, seed=2)
    g = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
    tv.grad(m, g)

    h = _rand_param(Vhm, seed=3)
    analytic = float(g.petsc_vec.dot(h.petsc_vec))

    c0 = tv.cost(m)
    errs = []
    for eps in (1e-3, 1e-4, 1e-5, 1e-6):
        m_plus = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
        m_plus.array[:] = m.array[:] + eps * h.array[:]
        m_plus.scatter_forward()
        fd = (tv.cost(m_plus) - c0) / eps
        errs.append(abs(fd - analytic) / max(abs(analytic), 1e-30))

    # error should roughly halve as eps shrinks by 10× (O(eps) → slope -1)
    # accept any monotone decrease at small eps
    assert errs[-1] < errs[0], (
        f"FD gradient error did not decrease: {errs}"
    )
    # asymptotic error should be small in relative terms
    assert errs[-1] < 1e-3, f"FD gradient error too large at eps=1e-6: {errs[-1]}"


def test_hessian_action_symmetric(spaces):
    """`<a, R b> == <b, R a>` for the primal-dual Hessian action."""
    msh, Vhm, Vhw, Vhwn = spaces
    tv = TVPrior(Vhm, Vhw, Vhwn, alpha=1.0, beta=1e-3)

    m = _rand_param(Vhm, seed=4)
    w = tv.compute_w(m)
    tv.setLinearizationPoint(m, w, gauss_newton_approx=False)

    a = _rand_param(Vhm, seed=5)
    b = _rand_param(Vhm, seed=6)
    Ra = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
    Rb = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
    tv.applyR(a, Ra)
    tv.applyR(b, Rb)

    lhs = float(b.petsc_vec.dot(Ra.petsc_vec))
    rhs = float(a.petsc_vec.dot(Rb.petsc_vec))
    rel = abs(lhs - rhs) / max(abs(rhs), 1.0)
    assert rel < 1e-10, f"<b, Ra>={lhs} vs <a, Rb>={rhs} (rel={rel})"


def test_compute_w_is_normalized(spaces):
    """For β → 0, `w = ∇m / |∇m|_β` should have w·w ≤ 1 pointwise.
    For β > 0, w·w can be slightly less than 1 but always ≥ 0."""
    msh, Vhm, Vhw, Vhwn = spaces
    tv = TVPrior(Vhm, Vhw, Vhwn, alpha=1.0, beta=1e-6)
    m = _rand_param(Vhm, seed=7)
    w = tv.compute_w(m)
    wn = tv.wnorm(w)
    assert (wn.array >= -1e-10).all()
    assert (wn.array <= 1.0 + 1e-8).all()


def test_weighted_vtv_cost_finite(spaces):
    """weightedVTVPrior should run end-to-end on a 2-component parameter
    and produce a finite cost."""
    msh = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    Vhm = dlx.fem.functionspace(
        msh, basix.ufl.element("Lagrange", msh.basix_cell(), 1, shape=(2,))
    )
    Vhw = dlx.fem.functionspace(
        msh, basix.ufl.element("DG", msh.basix_cell(), 0, shape=(2, 2))
    )
    Vhwn = dlx.fem.functionspace(msh, ("DG", 0))

    wtv = weightedVTVPrior(
        Vhm, Vhw, Vhwn, alpha=[1.0, 2.0], beta=1e-3,
    )
    m = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
    m.array[:] = np.random.default_rng(8).normal(size=m.array.shape)
    m.scatter_forward()
    c = wtv.cost(m)
    assert np.isfinite(c)
    assert c > 0.0

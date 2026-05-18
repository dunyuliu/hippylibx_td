# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""End-to-end TV-denoising smoke test.

Drives :class:`ModelNS` + :class:`ReducedSpacePDNewtonCG` on a tiny
problem and asserts that the solver decreases the cost monotonically
and lands on a smoothed denoised image.

This exercises the full chain: TVPrior, NSReducedHessian,
ReducedSpacePDNewtonCG, ModelNS, plus the upstream PDE/misfit pieces.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
import basix.ufl
import dolfinx as dlx
from mpi4py import MPI

import hippylibX as hpx
import hippylibX.total_variation as tv


def _build_small_denoising(seed: int = 0):
    msh = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 12, 12)
    Vhm = dlx.fem.functionspace(msh, ("Lagrange", 1))
    Vhw = dlx.fem.functionspace(
        msh, basix.ufl.element("DG", msh.basix_cell(), 0, shape=(2,))
    )
    Vhwn = dlx.fem.functionspace(msh, ("DG", 0))
    Vh = [Vhm, Vhm, Vhm]

    # zero Dirichlet on boundary
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = dlx.mesh.exterior_facet_indices(msh.topology)
    dofs = dlx.fem.locate_dofs_topological(Vhm, fdim, facets)
    uD = dlx.fem.Function(Vhm); uD.x.array[:] = 0.0; uD.x.scatter_forward()
    bc = dlx.fem.dirichletbc(uD, dofs)

    # piecewise-constant "true" image (single centered disk)
    m_true = dlx.fem.Function(Vhm)
    m_true.interpolate(lambda x: np.where(
        (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 < 0.2 ** 2, 1.0, 0.0,
    ))
    m_true.x.scatter_forward()

    rng = np.random.default_rng(seed)
    d = dlx.fem.Function(Vhm)
    d.x.array[:] = m_true.x.array[:] + rng.normal(0.0, 0.2, size=m_true.x.array.shape)
    d.x.scatter_forward()

    class IdentityVarf:
        def __call__(self, u, m, p):
            return u * p * ufl.dx - m * p * ufl.dx

    pde = hpx.PDEVariationalProblem(
        Vh, IdentityVarf(), [bc], [bc], is_fwd_linear=True,
    )
    misfit = hpx.NonGaussianContinuousMisfit(
        Vh,
        lambda u, m: 0.5 * ufl.inner(u - d, u - d) * ufl.dx,
        bc0=[bc],
    )
    tvprior = tv.TVPrior(
        Vhm, Vhw, Vhwn, alpha=1e-2, beta=1e-3, peps=5e-3,
    )
    model = tv.ModelNS(
        pde, misfit, prior=None, nsprior=tvprior, which=[True, False, True],
    )
    return msh, Vhm, model, m_true, d


def test_pdnewton_decreases_cost():
    """Cost should strictly decrease across PDNewtonCG iterations."""
    msh, Vhm, model, m_true, d = _build_small_denoising(seed=0)

    costs: list[float] = []

    def callback(it, x):
        costs.append(model.cost(x)[0])

    m0 = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
    m0.array[:] = 0.0
    m0.scatter_forward()
    params = tv.ReducedSpacePDNewtonCG_ParameterList()
    params["max_iter"] = 8
    params["cg_max_iter"] = 30
    params["print_level"] = -1
    solver = tv.ReducedSpacePDNewtonCG(model, params, callback=callback)
    solver.solve([None, m0, None, None])

    assert len(costs) >= 2, "solver should record at least 2 iterations"
    # monotone (allow tiny numerical noise)
    for k in range(1, len(costs)):
        assert costs[k] <= costs[k - 1] + 1e-12, (
            f"cost did not decrease at iter {k}: {costs[k - 1]} -> {costs[k]}"
        )


def test_pdnewton_recovers_image_better_than_noisy():
    """The denoised image should be closer to the truth than the raw noisy."""
    msh, Vhm, model, m_true, d = _build_small_denoising(seed=1)

    m0 = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
    m0.array[:] = 0.0
    m0.scatter_forward()
    params = tv.ReducedSpacePDNewtonCG_ParameterList()
    params["max_iter"] = 15
    params["cg_max_iter"] = 30
    params["print_level"] = -1
    solver = tv.ReducedSpacePDNewtonCG(model, params)
    from hippylibX.modeling.variables import PARAMETER
    x = solver.solve([None, m0, None, None])
    m_map = x[PARAMETER]

    # L2 distance to truth
    err_denoised = float(np.linalg.norm(m_map.array - m_true.x.array))
    err_noisy = float(np.linalg.norm(d.x.array - m_true.x.array))
    assert err_denoised < err_noisy, (
        f"denoised L2 error {err_denoised} >= noisy {err_noisy}"
    )

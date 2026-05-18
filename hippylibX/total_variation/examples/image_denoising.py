# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Image denoising via Total Variation regularization.

Faithful dolfinx port of the legacy example originally authored by
**Xindi Gong** at
``applications/total_variation/image_denoising/tv_image.py`` in
``xindigong/hippylib:tv-enhanced``.

The legacy example loads a 2D ``circles.mat`` raster; this dolfinx port
synthesizes the same kind of two-disk image analytically so the example
runs without external data.

The problem:
    Given a noisy image ``d``, find ``m`` that minimizes

        ``J(m) = 0.5 * ∫ (m - d)² dx + α * ∫ sqrt(|∇m|² + β) dx``

Implemented as an inverse problem with:
    - "Forward" PDE: ``u = m`` (the parameter is the state, identity map)
    - Misfit: ``ContinuousStateObservation`` against the noisy data
    - Non-smooth prior: :class:`TVPrior`
    - Solver: :class:`ReducedSpacePDNewtonCG`
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import ufl
import basix.ufl
import dolfinx as dlx
import dolfinx.fem.petsc
from mpi4py import MPI

# make the repo root importable
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import hippylibX as hpx
import hippylibX.total_variation as tv

STATE, PARAMETER, ADJOINT = hpx.STATE, hpx.PARAMETER, hpx.ADJOINT


# ---- problem setup ------------------------------------------------------
NX = NY = 32
ALPHA = 1e-3
BETA = 1e-4
NOISE_STD = 0.30


def _two_disk_image(x):
    """Synthetic 'cartoon' image — two overlapping disks at fixed centres."""
    in_disk_a = (x[0] - 0.35) ** 2 + (x[1] - 0.5) ** 2 < 0.18 ** 2
    in_disk_b = (x[0] - 0.65) ** 2 + (x[1] - 0.5) ** 2 < 0.18 ** 2
    return np.where(in_disk_a | in_disk_b, 1.0, 0.0)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    msh = dlx.mesh.create_unit_square(comm, NX, NY)

    Vhm = dlx.fem.functionspace(msh, ("Lagrange", 1))
    Vhw = dlx.fem.functionspace(
        msh, basix.ufl.element("DG", msh.basix_cell(), 0, shape=(2,))
    )
    Vhwnorm = dlx.fem.functionspace(msh, ("DG", 0))
    Vh = [Vhm, Vhm, Vhm]
    if rank == 0:
        n_state = Vhm.dofmap.index_map.size_global * Vhm.dofmap.index_map_bs
        print(f"# DOFs: STATE={n_state}, PARAMETER={n_state}, ADJOINT={n_state}")

    # Homogeneous Dirichlet on the boundary (so the identity "PDE" lives in
    # the H¹₀ subspace, matching legacy).
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    boundary_facets = dlx.mesh.exterior_facet_indices(msh.topology)
    dofs = dlx.fem.locate_dofs_topological(Vhm, fdim, boundary_facets)
    uD = dlx.fem.Function(Vhm)
    uD.x.array[:] = 0.0
    uD.x.scatter_forward()
    bc = dlx.fem.dirichletbc(uD, dofs)

    # True image
    m_true_fun = dlx.fem.Function(Vhm)
    m_true_fun.interpolate(_two_disk_image)
    m_true_fun.x.scatter_forward()
    m_true = m_true_fun.x

    # Noisy data
    rng = np.random.default_rng(1)
    d_fun = dlx.fem.Function(Vhm)
    d_fun.x.array[:] = m_true.array[:] + rng.normal(0.0, NOISE_STD, size=m_true.array.shape)
    d_fun.x.scatter_forward()

    # Forward "PDE": u - m = 0  (identity map)
    class IdentityVarf:
        def __call__(self, u, m, p):
            return u * p * ufl.dx - m * p * ufl.dx

    pde = hpx.PDEVariationalProblem(
        Vh, IdentityVarf(), [bc], [bc], is_fwd_linear=True,
    )

    # Misfit
    misfit = hpx.NonGaussianContinuousMisfit(
        Vh,
        lambda u, m: 0.5 * ufl.inner(u - d_fun, u - d_fun) * ufl.dx,
        bc0=[bc],
    )

    # TV prior
    tvprior = tv.TVPrior(
        Vhm, Vhw, Vhwnorm, alpha=ALPHA, beta=BETA, peps=0.5 * ALPHA,
    )

    # ModelNS: misfit + TV only (no smooth prior)
    TVonly = [True, False, True]
    model = tv.ModelNS(pde, misfit, prior=None, nsprior=tvprior, which=TVonly)

    # Solve via primal-dual Newton-CG
    m0 = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
    m0.array[:] = 0.0
    m0.scatter_forward()
    x0 = [None, m0, None, None]

    params = tv.ReducedSpacePDNewtonCG_ParameterList()
    params["max_iter"] = 30
    params["cg_max_iter"] = 75
    if rank != 0:
        params["print_level"] = -1

    solver = tv.ReducedSpacePDNewtonCG(model, parameters=params)
    t0 = time.perf_counter()
    x = solver.solve(x0)
    dt = time.perf_counter() - t0
    if rank == 0:
        print(f"\nSolve took {dt:.2f}s; converged={solver.converged}")
        print(f"  termination: {solver.termination_reasons[solver.reason]}")
        print(f"  iters: {solver.it}, total CG: {solver.total_cg_iter}")
        print(f"  final cost: {solver.final_cost:.4e}")

    return solver, x, m_true, d_fun


if __name__ == "__main__":
    main()

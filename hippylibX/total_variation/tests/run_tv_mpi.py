# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""MPI consistency runner — solve the TV denoising problem under arbitrary np
and gather the globally-ordered MAP parameter vector.

The output is a JSON dump suitable for comparing serial vs parallel.
``$RESULTS_JSON`` controls the dump path.

Run via::

    mpirun -n N python3 run_tv_mpi.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import ufl
import basix.ufl
import dolfinx as dlx
import dolfinx.fem.petsc
from mpi4py import MPI

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import hippylibX as hpx
import hippylibX.total_variation as tv

from refproblem import (
    NX, NY, ALPHA, BETA, PEPS_FACTOR,
    NOISE_VARIANCE,
    REL_TOL, ABS_TOL, MAX_ITER, CG_MAX_ITER, GN_ITER,
    two_disk_image_numpy,
)

STATE, PARAMETER, ADJOINT = hpx.STATE, hpx.PARAMETER, hpx.ADJOINT
comm = MPI.COMM_WORLD


def _gather_global(vec: dlx.la.Vector, V) -> np.ndarray | None:
    """Return the globally-ordered scalar array on rank 0; None elsewhere."""
    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    nlocal = imap.size_local
    local_arr = np.asarray(vec.array)[: nlocal * bs].copy()
    l2g = imap.local_to_global(np.arange(nlocal, dtype=np.int32))
    if bs == 1:
        global_idx = l2g.astype(np.int64)
    else:
        glob = np.empty(nlocal * bs, dtype=np.int64)
        for b in range(bs):
            glob[b::bs] = l2g * bs + b
        global_idx = glob

    counts = comm.gather(local_arr.size, root=0)
    idxs = comm.gather(global_idx, root=0)
    vals = comm.gather(local_arr.astype(np.float64), root=0)
    if comm.rank != 0:
        return None
    n_total = imap.size_global * bs
    out = np.full(n_total, np.nan, dtype=np.float64)
    for ids, vs in zip(idxs, vals):
        out[ids] = vs
    if np.isnan(out).any():
        raise RuntimeError("global gather missed some entries")
    return out


def main(out_path: str) -> None:
    msh = dlx.mesh.create_unit_square(comm, NX, NY)
    Vhm = dlx.fem.functionspace(msh, ("Lagrange", 1))
    Vhw = dlx.fem.functionspace(
        msh, basix.ufl.element("DG", msh.basix_cell(), 0, shape=(2,))
    )
    Vhwnorm = dlx.fem.functionspace(msh, ("DG", 0))
    Vh = [Vhm, Vhm, Vhm]

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = dlx.mesh.exterior_facet_indices(msh.topology)
    dofs = dlx.fem.locate_dofs_topological(Vhm, fdim, facets)
    uD = dlx.fem.Function(Vhm)
    uD.x.array[:] = 0.0
    uD.x.scatter_forward()
    bc = dlx.fem.dirichletbc(uD, dofs)

    m_true_fun = dlx.fem.Function(Vhm)
    m_true_fun.interpolate(two_disk_image_numpy)
    m_true_fun.x.scatter_forward()

    d = dlx.fem.Function(Vhm)
    d.x.array[:] = m_true_fun.x.array[:]
    d.x.scatter_forward()

    class IdentityVarf:
        def __call__(self, u, m, p):
            return u * p * ufl.dx - m * p * ufl.dx

    pde = hpx.PDEVariationalProblem(Vh, IdentityVarf(), [bc], [bc], is_fwd_linear=True)
    misfit = hpx.NonGaussianContinuousMisfit(
        Vh,
        lambda u, m: 0.5 / NOISE_VARIANCE * ufl.inner(u - d, u - d) * ufl.dx,
        bc0=[bc],
    )
    tvprior = tv.TVPrior(
        Vhm, Vhw, Vhwnorm,
        alpha=ALPHA, beta=BETA, peps=PEPS_FACTOR * ALPHA,
    )
    model = tv.ModelNS(pde, misfit, prior=None, nsprior=tvprior, which=[True, False, True])

    m0 = dlx.la.vector(Vhm.dofmap.index_map, Vhm.dofmap.index_map_bs)
    m0.array[:] = 0.0
    m0.scatter_forward()

    params = tv.ReducedSpacePDNewtonCG_ParameterList()
    params["rel_tolerance"] = REL_TOL
    params["abs_tolerance"] = ABS_TOL
    params["max_iter"] = MAX_ITER
    params["cg_max_iter"] = CG_MAX_ITER
    params["GN_iter"] = GN_ITER
    params["print_level"] = -1

    solver = tv.ReducedSpacePDNewtonCG(model, params)
    x = solver.solve([None, m0, None, None])
    m_map = x[PARAMETER]

    m_map_global = _gather_global(m_map, Vhm)
    m_map_l2 = float(m_map.petsc_vec.norm())

    if comm.rank == 0:
        out = {
            "np": comm.size,
            "ndofs_param": int(Vhm.dofmap.index_map.size_global * Vhm.dofmap.index_map_bs),
            "newton_iters": int(solver.it),
            "total_cg_iter": int(solver.total_cg_iter),
            "converged": bool(solver.converged),
            "final_cost": float(solver.final_cost),
            "m_map_l2": m_map_l2,
            "m_map_global": m_map_global.tolist(),
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {out_path} (np={comm.size})")


if __name__ == "__main__":
    main(os.environ.get("RESULTS_JSON", f"results_tv_mpi_np{comm.size}.json"))

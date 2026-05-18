# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""TV parity runner — fenicsx + hippylibX side.

Dumps results to ``$RESULTS_JSON``.
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
    NOISE_STD, NOISE_VARIANCE,
    REL_TOL, ABS_TOL, MAX_ITER, CG_MAX_ITER, GN_ITER, PRINT_LEVEL,
    two_disk_image_numpy,
)

STATE, PARAMETER, ADJOINT = hpx.STATE, hpx.PARAMETER, hpx.ADJOINT


def main(out_path: str) -> None:
    comm = MPI.COMM_WORLD
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
    uD = dlx.fem.Function(Vhm); uD.x.array[:] = 0.0; uD.x.scatter_forward()
    bc = dlx.fem.dirichletbc(uD, dofs)

    m_true = dlx.fem.Function(Vhm)
    m_true.interpolate(two_disk_image_numpy)
    m_true.x.scatter_forward()

    # zero noise → d = m_true exactly
    d = dlx.fem.Function(Vhm)
    d.x.array[:] = m_true.x.array[:]
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
    params["print_level"] = PRINT_LEVEL
    solver = tv.ReducedSpacePDNewtonCG(model, params)

    # cost at m=0 (informational)
    x0 = [model.generate_vector(STATE), m0, None,
          model.generate_vector(tv.SLACK)]
    model.solveFwd(x0[STATE], x0)
    cost_at_m0 = model.cost(x0)

    x = solver.solve([None, m0, None, None])
    m_map = x[PARAMETER]

    # L2 distance to truth (parameter space)
    m_map_l2 = float(m_map.petsc_vec.norm())
    err_to_true = float(
        np.linalg.norm(np.asarray(m_map.array) - np.asarray(m_true.x.array))
    )

    out = {
        "problem": "tv_denoise",
        "ndofs_param": int(Vhm.dofmap.index_map.size_global * Vhm.dofmap.index_map_bs),
        "cost_at_m0": [float(c) for c in cost_at_m0],
        "newton_iters": int(solver.it),
        "total_cg_iter": int(solver.total_cg_iter),
        "converged": bool(solver.converged),
        "termination": solver.termination_reasons[solver.reason],
        "final_cost": float(solver.final_cost),
        "final_grad_norm": float(solver.final_grad_norm),
        "m_map_l2": m_map_l2,
        "err_to_true": err_to_true,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main(os.environ.get("RESULTS_JSON", "results_x_tv.json"))

# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""TV parity runner — legacy dolfin + xindigong/hippylib:tv-enhanced side.

Requires the `fenicsproject` conda env and the env var
``HIPPYLIB_TV_PATH`` pointing at a clone of
``xindigong/hippylib`` checked out on the ``tv-enhanced`` branch.

Dumps results to ``$RESULTS_JSON``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import dolfin as dl
import ufl

_TV_PATH = os.environ.get("HIPPYLIB_TV_PATH", "").strip()
if _TV_PATH:
    sys.path.insert(0, _TV_PATH)
import hippylib as hp

from refproblem import (
    NX, NY, ALPHA, BETA, PEPS_FACTOR,
    NOISE_STD, NOISE_VARIANCE,
    REL_TOL, ABS_TOL, MAX_ITER, CG_MAX_ITER, GN_ITER, PRINT_LEVEL,
    LEGACY_M_TRUE_EXPR,
)


STATE, PARAMETER, ADJOINT, SLACK = 0, 1, 2, 3


def main(out_path: str) -> None:
    mesh = dl.UnitSquareMesh(NX, NY)
    Vhm = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vhw = dl.VectorFunctionSpace(mesh, "DG", 0)
    Vhwnorm = dl.FunctionSpace(mesh, "DG", 0)
    Vh = [Vhm, Vhm, Vhm]

    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), u0_boundary)
    bc0 = bc

    # true image (single disk)
    m_true_expr = dl.Expression(
        LEGACY_M_TRUE_EXPR, element=Vhm.ufl_element()
    )
    m_true = dl.interpolate(m_true_expr, Vhm)

    # zero noise → d = m_true exactly
    d = dl.Function(Vhm)
    d.vector().set_local(m_true.vector().get_local().copy())
    d.vector().apply("")

    def pde_varf(u, m, p):
        return u * p * ufl.dx - m * p * ufl.dx

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    misfit = hp.ContinuousStateObservation(
        Vh=Vh[STATE], dX=ufl.dx,
        data=d.vector(), noise_variance=NOISE_VARIANCE, bcs=[bc0],
    )
    tvprior = hp.TVPrior(
        Vhm, Vhw, Vhwnorm,
        ALPHA, BETA, peps=PEPS_FACTOR * ALPHA,
    )
    TVonly = [True, False, True]
    model = hp.ModelNS(pde, misfit, None, tvprior, which=TVonly)

    # cost at m=0
    m0 = dl.Function(Vhm).vector()
    m0.zero()
    x0 = [model.generate_vector(STATE), m0, model.generate_vector(ADJOINT),
          tvprior.generate_slack()]
    model.solveFwd(x0[STATE], x0)
    cost_at_m0 = model.cost(x0)

    params = hp.ReducedSpacePDNewtonCG_ParameterList()
    params["rel_tolerance"] = REL_TOL
    params["abs_tolerance"] = ABS_TOL
    params["max_iter"] = MAX_ITER
    params["cg_max_iter"] = CG_MAX_ITER
    params["GN_iter"] = GN_ITER
    params["print_level"] = PRINT_LEVEL
    solver = hp.ReducedSpacePDNewtonCG(model, parameters=params)

    m_init = dl.Function(Vhm).vector()
    m_init.zero()
    x = solver.solve([None, m_init, None, None])
    m_map = x[PARAMETER]

    m_map_l2 = float(m_map.norm("l2"))
    err_to_true = float(
        np.linalg.norm(m_map.get_local() - m_true.vector().get_local())
    )

    out = {
        "problem": "tv_denoise",
        "ndofs_param": Vhm.dim(),
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
    main(os.environ.get("RESULTS_JSON", "results_legacy_tv.json"))

#!/usr/bin/env python
"""Run a parity test problem with legacy dolfin + hippylib.

Selects the problem via the ``PROBLEM`` env var (default ``heat``).
Dumps results to JSON at ``$RESULTS_JSON``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import dolfin as dl
import ufl

sys.path.append(os.environ.get(
    "HIPPYLIB_PATH",
    "/Users/dliu/scratch/visco_inversion/src/hippylib",
))
import hippylib as hp

from refproblem import (
    NX, NY, NT, T_INIT, T_FINAL, GAMMA, DELTA,
    NOISE_STD, NOISE_VARIANCE, ROBIN_BC,
    REL_TOL, ABS_TOL, MAX_ITER, CG_COARSE_TOL, GLOB, GN_ITER,
    AD_DIFF_M_CX, AD_DIFF_M_CY, AD_DIFF_M_AMP, AD_DIFF_M_SIGMA,
    AD_DIFF_N_SENS_X, AD_DIFF_N_SENS_Y, TUMOR_LAMBDA,
)

STATE, PARAMETER, ADJOINT = 0, 1, 2


def _mesh_and_spaces(state_param_same_space: bool = False):
    mesh = dl.UnitSquareMesh(NX, NY)
    if state_param_same_space:
        Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
        Vh = [Vh1, Vh1, Vh1]
    else:
        Vh2 = dl.FunctionSpace(mesh, "Lagrange", 2)
        Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
        Vh = [Vh2, Vh1, Vh2]

    def boundary(x, on_boundary):
        return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

    u_bdr = dl.Expression("0.0", element=Vh[STATE].ufl_element())
    bc = dl.DirichletBC(Vh[STATE], u_bdr, boundary)
    bc0 = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), boundary)
    return mesh, Vh, bc, bc0


def setup_heat():
    mesh, Vh, bc, bc0 = _mesh_and_spaces()
    u0_expr = dl.Expression(
        "x[0]*(1. - x[0])*x[1]*(1. - x[1])",
        element=Vh[STATE].ufl_element(),
    )
    u0 = dl.interpolate(u0_expr, Vh[STATE])

    class HeatVarf:
        def __init__(self, dt):
            self._dt = float(dt)
            self.dt_inv = dl.Constant(1.0 / dt)

        @property
        def dt(self):
            return self._dt

        def __call__(self, u, u_old, m, p, t):
            return ((u - u_old) * p * self.dt_inv * ufl.dx
                    + ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx)

    pde = hp.TimeDependentPDEVariationalProblem(
        Vh, HeatVarf((T_FINAL - T_INIT) / NT),
        bc, bc0, u0, T_INIT, T_FINAL, is_fwd_linear=True,
    )

    m_true_expr = dl.Expression(
        "0.5 + 0.5*sin(pi*x[0])*sin(pi*x[1])",
        element=Vh[PARAMETER].ufl_element(),
    )
    m_true = dl.interpolate(m_true_expr, Vh[PARAMETER]).vector()

    prior = hp.BiLaplacianPrior(
        Vh[PARAMETER], GAMMA, DELTA, mean=None, robin_bc=ROBIN_BC,
    )

    return mesh, Vh, bc, bc0, pde, m_true, prior, "continuous"


def setup_tumor():
    mesh, Vh, bc, bc0 = _mesh_and_spaces()
    u0_expr = dl.Expression(
        "std::exp(-100.*(x[0]-.5)*(x[0]-.5) - 100.*(x[1]-.75)*(x[1]-.75))",
        element=Vh[STATE].ufl_element(),
    )
    u0 = dl.interpolate(u0_expr, Vh[STATE])

    class TumorVarf:
        def __init__(self, dt, lmbda):
            self._dt = float(dt)
            self.dt_inv = dl.Constant(1.0 / dt)
            self.lmbda = dl.Constant(float(lmbda))

        @property
        def dt(self):
            return self._dt

        def __call__(self, u, u_old, m, p, t):
            return ((u - u_old) * p * self.dt_inv * ufl.dx
                    + ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx
                    - self.lmbda * u * (dl.Constant(1.) - u) * p * ufl.dx)

    # Tumor: no Dirichlet BCs (legacy uses empty lists for both bc and bc0)
    pde = hp.TimeDependentPDEVariationalProblem(
        Vh, TumorVarf((T_FINAL - T_INIT) / NT, TUMOR_LAMBDA),
        [], [], u0, T_INIT, T_FINAL, is_fwd_linear=False,
    )

    m_true_expr = dl.Expression(
        "0.5 + 0.3*sin(pi*x[0])*sin(pi*x[1])",
        element=Vh[PARAMETER].ufl_element(),
    )
    m_true = dl.interpolate(m_true_expr, Vh[PARAMETER]).vector()

    prior = hp.BiLaplacianPrior(
        Vh[PARAMETER], GAMMA, DELTA, mean=None, robin_bc=ROBIN_BC,
    )

    return mesh, Vh, bc, bc0, pde, m_true, prior, "continuous_no_bc"


def _build_misfit_continuous(Vh, bc0, u_true, pde, with_bc: bool):
    misfits = []
    for t in pde.times:
        bcs = bc0 if with_bc else None
        m_obs = hp.ContinuousStateObservation(Vh[STATE], ufl.dx, bcs)
        d_local = u_true.view(t).get_local().copy()
        m_obs.d.set_local(d_local)
        m_obs.d.apply("")
        m_obs.noise_variance = NOISE_VARIANCE
        misfits.append(m_obs)
    return hp.MisfitTD(misfits, pde.times)


def main(out_path: str, problem: str) -> None:
    if problem == "heat":
        mesh, Vh, bc, bc0, pde, m_true, prior, misfit_kind = setup_heat()
    elif problem == "tumor":
        mesh, Vh, bc, bc0, pde, m_true, prior, misfit_kind = setup_tumor()
    else:
        raise ValueError(f"unknown PROBLEM: {problem}")

    u_true = pde.generate_state()
    pde.solveFwd(u_true, [u_true, m_true, None])
    state_norms = [float(u_true.view(t).norm("l2")) for t in pde.times]

    if misfit_kind == "continuous":
        misfit = _build_misfit_continuous(Vh, bc0, u_true, pde, with_bc=True)
    elif misfit_kind == "continuous_no_bc":
        misfit = _build_misfit_continuous(Vh, bc0, u_true, pde, with_bc=False)
    else:
        raise ValueError(f"unknown misfit_kind: {misfit_kind}")

    model = hp.Model(pde, prior, misfit)

    x_true_eval = [u_true, m_true, pde.generate_adjoint()]
    model.solveAdj(x_true_eval[ADJOINT], x_true_eval)
    cost_at_mtrue = model.cost(x_true_eval)

    m0 = dl.Function(Vh[PARAMETER]).vector()
    m0.zero()
    x0 = [pde.generate_state(), m0, pde.generate_adjoint()]
    model.solveFwd(x0[STATE], x0)
    cost_at_m0 = model.cost(x0)

    x = [pde.generate_state(), m0.copy(), pde.generate_adjoint()]
    pp = hp.ReducedSpaceNewtonCG_ParameterList()
    pp["rel_tolerance"] = REL_TOL
    pp["abs_tolerance"] = ABS_TOL
    pp["max_iter"] = MAX_ITER
    pp["cg_coarse_tolerance"] = CG_COARSE_TOL
    pp["globalization"] = GLOB
    pp["GN_iter"] = GN_ITER
    pp["print_level"] = -1
    solver = hp.ReducedSpaceNewtonCG(model, pp)
    x = solver.solve(x)

    map_state_norms = [float(x[STATE].view(t).norm("l2")) for t in pde.times]
    m_map_l2 = float(x[PARAMETER].norm("l2"))

    out = {
        "problem": problem,
        "ndofs_state": Vh[STATE].dim(),
        "ndofs_param": Vh[PARAMETER].dim(),
        "times": [float(t) for t in pde.times],
        "state_norms_at_mtrue": state_norms,
        "cost_at_mtrue": [float(c) for c in cost_at_mtrue],
        "cost_at_m0": [float(c) for c in cost_at_m0],
        "newton_iters": int(solver.it),
        "final_cost": float(solver.final_cost),
        "final_grad_norm": float(solver.final_grad_norm),
        "converged": bool(solver.converged),
        "termination": solver.termination_reasons[solver.reason],
        "map_state_norms": map_state_norms,
        "m_map_l2": m_map_l2,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = os.environ.get("RESULTS_JSON", "results_legacy.json")
    problem = os.environ.get("PROBLEM", "heat")
    main(out, problem)

#!/usr/bin/env python
"""Run a parity test problem with fenicsx + hippylibX (+ time_dependent).

Selects the problem via the ``PROBLEM`` env var (default ``heat``).
Dumps results to JSON at ``$RESULTS_JSON``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import ufl
import dolfinx as dlx
import dolfinx.fem.petsc
from mpi4py import MPI

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import hippylibX as hpx
from hippylibX import time_dependent as td

from refproblem import (
    NX, NY, NT, T_INIT, T_FINAL, GAMMA, DELTA,
    NOISE_STD, NOISE_VARIANCE, ROBIN_BC,
    REL_TOL, ABS_TOL, MAX_ITER, CG_COARSE_TOL, GLOB, GN_ITER,
    M_TRUE_EXPR_STR_HEAT, M_TRUE_EXPR_STR_TUMOR,
    AD_DIFF_M_CX, AD_DIFF_M_CY, AD_DIFF_M_AMP, AD_DIFF_M_SIGMA,
    AD_DIFF_N_SENS_X, AD_DIFF_N_SENS_Y, TUMOR_LAMBDA,
)

STATE, PARAMETER, ADJOINT = hpx.STATE, hpx.PARAMETER, hpx.ADJOINT
comm = MPI.COMM_WORLD


# ---------------------------------------------------------------------------
# Mesh + function spaces (shared across heat/tumor; AD uses Vh,Vh,Vh)
# ---------------------------------------------------------------------------
def _mesh_and_spaces(state_param_same_space: bool = False):
    msh = dlx.mesh.create_unit_square(comm, NX, NY, dlx.mesh.CellType.triangle)
    if state_param_same_space:
        Vh1 = dlx.fem.functionspace(msh, ("Lagrange", 1))
        Vh = [Vh1, Vh1, Vh1]
    else:
        Vh2 = dlx.fem.functionspace(msh, ("Lagrange", 2))
        Vh1 = dlx.fem.functionspace(msh, ("Lagrange", 1))
        Vh = [Vh2, Vh1, Vh2]

    def top_bottom(x):
        return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

    fdim = msh.topology.dim - 1
    tb_facets = dlx.mesh.locate_entities_boundary(msh, fdim, top_bottom)
    tb_dofs = dlx.fem.locate_dofs_topological(Vh[STATE], fdim, tb_facets)
    uD = dlx.fem.Function(Vh[STATE])
    uD.interpolate(lambda x: 0.0 * x[0])
    uD.x.scatter_forward()
    bc = dlx.fem.dirichletbc(uD, tb_dofs)
    return msh, Vh, bc


# ---------------------------------------------------------------------------
# Heat: u_t - div(exp(m) grad u) = 0
# ---------------------------------------------------------------------------
def setup_heat():
    msh, Vh, bc = _mesh_and_spaces()
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
        Vh, HeatVarf((T_FINAL - T_INIT) / NT),
        bc=[bc], bc0=[bc], u0=u0, t_init=T_INIT, t_final=T_FINAL, is_fwd_linear=True,
    )

    m_true_fun = dlx.fem.Function(Vh[PARAMETER])
    m_true_fun.interpolate(lambda x: 0.5 + 0.5 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    m_true_fun.x.scatter_forward()
    m_true = m_true_fun.x

    prior_mean = dlx.fem.Function(Vh[PARAMETER])
    prior_mean.x.array[:] = 0.0
    prior = hpx.BiLaplacianPrior(
        Vh[PARAMETER], GAMMA, DELTA, mean=prior_mean.x, robin_bc=ROBIN_BC,
    )

    return msh, Vh, bc, pde, m_true, prior, "continuous"


# ---------------------------------------------------------------------------
# Tumor: u_t - div(exp(m) grad u) - lambda*u(1-u) = 0    (nonlinear forward)
# ---------------------------------------------------------------------------
def setup_tumor():
    msh, Vh, bc = _mesh_and_spaces()
    # IC: localised bump near (0.5, 0.75), as in legacy
    u0 = dlx.fem.Function(Vh[STATE])
    u0.interpolate(lambda x: np.exp(-100.0 * (x[0] - 0.5) ** 2 - 100.0 * (x[1] - 0.75) ** 2))
    u0.x.scatter_forward()

    class TumorVarf:
        def __init__(self, dt, lmbda):
            self._dt = float(dt)
            self.dt_inv = dlx.fem.Constant(msh, dlx.default_scalar_type(1.0 / dt))
            self.lmbda = dlx.fem.Constant(msh, dlx.default_scalar_type(float(lmbda)))
            self.one = dlx.fem.Constant(msh, dlx.default_scalar_type(1.0))

        @property
        def dt(self):
            return self._dt

        def __call__(self, u, u_old, m, p, t):
            return ((u - u_old) * p * self.dt_inv * ufl.dx
                    + ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx
                    - self.lmbda * u * (self.one - u) * p * ufl.dx)

    pde = td.TimeDependentPDEVariationalProblem(
        Vh, TumorVarf((T_FINAL - T_INIT) / NT, TUMOR_LAMBDA),
        bc=[], bc0=[], u0=u0, t_init=T_INIT, t_final=T_FINAL, is_fwd_linear=False,
    )

    m_true_fun = dlx.fem.Function(Vh[PARAMETER])
    m_true_fun.interpolate(lambda x: 0.5 + 0.3 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    m_true_fun.x.scatter_forward()
    m_true = m_true_fun.x

    prior_mean = dlx.fem.Function(Vh[PARAMETER])
    prior_mean.x.array[:] = 0.0
    prior = hpx.BiLaplacianPrior(
        Vh[PARAMETER], GAMMA, DELTA, mean=prior_mean.x, robin_bc=ROBIN_BC,
    )

    return msh, Vh, bc, pde, m_true, prior, "continuous_no_bc"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _build_misfit_continuous(Vh, bc, u_true, pde, with_bc: bool):
    misfits = []
    for t in pde.times:
        obs = td.ContinuousStateObservation(
            Vh[STATE], ufl.dx, bcs=[bc] if with_bc else [],
        )
        obs.d.array[:] = u_true.view(t).array[:]
        obs.d.scatter_forward()
        obs.noise_variance = NOISE_VARIANCE
        misfits.append(obs)
    return td.MisfitTD(misfits, pde.times)


def main(out_path: str, problem: str) -> None:
    if problem == "heat":
        msh, Vh, bc, pde, m_true, prior, misfit_kind = setup_heat()
    elif problem == "tumor":
        msh, Vh, bc, pde, m_true, prior, misfit_kind = setup_tumor()
    else:
        raise ValueError(f"unknown PROBLEM: {problem}")

    # Forward at m_true
    u_true = pde.generate_state()
    pde.solveFwd(u_true, [u_true, m_true, None])
    state_norms = [float(u_true.view(t).petsc_vec.norm()) for t in pde.times]

    # Synthetic data (zero noise)
    if misfit_kind == "continuous":
        misfit = _build_misfit_continuous(Vh, bc, u_true, pde, with_bc=True)
    elif misfit_kind == "continuous_no_bc":
        misfit = _build_misfit_continuous(Vh, bc, u_true, pde, with_bc=False)
    else:
        raise ValueError(f"unknown misfit_kind: {misfit_kind}")

    model = hpx.Model(pde, prior, misfit)

    # Cost at m_true
    x_true_eval = [u_true, m_true, model.generate_vector(ADJOINT)]
    model.solveAdj(x_true_eval[ADJOINT], x_true_eval)
    cost_at_mtrue = model.cost(x_true_eval)

    # Cost at m=0
    m0 = prior.generate_parameter(0)
    m0.array[:] = 0.0
    m0.scatter_forward()
    x0 = [model.generate_vector(STATE), m0, model.generate_vector(ADJOINT)]
    model.solveFwd(x0[STATE], x0)
    cost_at_m0 = model.cost(x0)

    # Newton-CG
    x = [model.generate_vector(STATE), m0, model.generate_vector(ADJOINT)]
    p = hpx.ReducedSpaceNewtonCG_ParameterList()
    p["rel_tolerance"] = REL_TOL
    p["abs_tolerance"] = ABS_TOL
    p["max_iter"] = MAX_ITER
    p["cg_coarse_tolerance"] = CG_COARSE_TOL
    p["globalization"] = GLOB
    p["GN_iter"] = GN_ITER
    p["print_level"] = -1
    solver = hpx.ReducedSpaceNewtonCG(model, p)
    x = solver.solve(x)

    map_state_norms = [float(x[STATE].view(t).petsc_vec.norm()) for t in pde.times]
    m_map_l2 = float(x[PARAMETER].petsc_vec.norm())

    out = {
        "problem": problem,
        "ndofs_state": int(Vh[STATE].dofmap.index_map.size_global * Vh[STATE].dofmap.index_map_bs),
        "ndofs_param": int(Vh[PARAMETER].dofmap.index_map.size_global * Vh[PARAMETER].dofmap.index_map_bs),
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
    out = os.environ.get("RESULTS_JSON", "results_x.json")
    problem = os.environ.get("PROBLEM", "heat")
    main(out, problem)

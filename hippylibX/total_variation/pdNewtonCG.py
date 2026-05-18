# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Primal-Dual Newton-CG solver for TV-regularized inverse problems.

Faithful dolfinx port of the legacy hippylib class authored by
**Xindi Gong** (see ``hippylib/algorithms/PDNewtonCG.py`` in
``xindigong/hippylib:tv-enhanced``).

References:
    [1] Chan, Tony F., Gene H. Golub, and Pep Mulet.
        "A nonlinear primal-dual method for total variation-based
        image restoration."
        SIAM J. Sci. Comput. 20(6) (1999): 1964-1977.
"""

from __future__ import annotations

import math

import petsc4py
from petsc4py import PETSc

from ._cgsolverSteihaug import (
    CGSolverSteihaug, CGSolverSteihaug_ParameterList,
)
from ._variables import STATE, PARAMETER, ADJOINT
from . import SLACK
from .nsReducedHessian import NSReducedHessian


# ---- parameter dicts ----------------------------------------------------
def LS_ParameterList() -> dict:
    return {
        "c_armijo": 1e-4,
        "max_backtracking_iter": 10,
    }


def ReducedSpacePDNewtonCG_ParameterList() -> dict:
    return {
        "rel_tolerance": 1e-6,
        "abs_tolerance": 1e-12,
        "gdm_tolerance": 1e-18,
        "max_iter": 20,
        "globalization": "LS",
        "print_level": 0,
        "GN_iter": 5,
        "cg_coarse_tolerance": 0.5,
        "cg_max_iter": 100,
        "LS": LS_ParameterList(),
    }


# ---- vector helpers (work on dlx.la.Vector + petsc.Vec) ----------------
def _zero(v) -> None:
    if hasattr(v, "zero"):
        v.zero()
    else:
        v.array[:] = 0.0
        if hasattr(v, "scatter_forward"):
            v.scatter_forward()


def _axpy(dst, a: float, src) -> None:
    if hasattr(dst, "petsc_vec") and hasattr(src, "petsc_vec"):
        dst.petsc_vec.axpy(a, src.petsc_vec)
        if hasattr(dst, "scatter_forward"):
            dst.scatter_forward()
    elif hasattr(dst, "axpy"):
        dst.axpy(a, src)
    else:
        dst.array[:] += a * src.array[:]


def _inner(a, b) -> float:
    if hasattr(a, "petsc_vec") and hasattr(b, "petsc_vec"):
        return float(a.petsc_vec.dot(b.petsc_vec))
    if hasattr(a, "inner"):
        return float(a.inner(b))
    import numpy as np
    return float(np.dot(a.array, b.array))


def _linf_norm(v) -> float:
    if hasattr(v, "petsc_vec"):
        return float(v.petsc_vec.norm(PETSc.NormType.NORM_INFINITY))
    if hasattr(v, "norm"):
        return float(v.norm("linf"))
    import numpy as np
    return float(np.max(np.abs(v.array)))


# ---- solver ------------------------------------------------------------
class ReducedSpacePDNewtonCG:
    """Primal-Dual Inexact Newton-CG (Chan-Golub-Mulet)."""

    termination_reasons = [
        "Maximum number of Iterations reached",
        "Norm of the gradient less than tolerance",
        "Maximum number of (parameter) backtracking reached",
        "Maximum number of (slack) backtracking reached",
        "Norm of (g, dm) less than tolerance",
    ]

    def __init__(self, model, parameters: dict | None = None, callback=None):
        self.model = model
        self.parameters = parameters or ReducedSpacePDNewtonCG_ParameterList()
        self.it = 0
        self.converged = False
        self.total_cg_iter = 0
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0.0
        self.final_cost = 0.0
        self.callback = callback

    def solve(self, x):
        if self.model is None:
            raise TypeError("model must not be None.")
        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
        if x[SLACK] is None:
            x[SLACK] = self.model.generate_vector(SLACK)
        if self.parameters["globalization"] != "LS":
            raise ValueError(
                f"unsupported globalization: {self.parameters['globalization']!r}"
            )
        return self._solve_dls(x)

    def _solve_dls(self, x):
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tol = self.parameters["cg_coarse_tolerance"]
        cg_max_iter = self.parameters["cg_max_iter"]
        c_armijo = self.parameters["LS"]["c_armijo"]
        max_bt = self.parameters["LS"]["max_backtracking_iter"]

        self.model.solveFwd(x[STATE], x)
        self.it = 0
        self.converged = False
        self.ncalls += 1

        mhat = self.model.generate_vector(PARAMETER)
        what = self.model.generate_vector(SLACK)
        mg = self.model.generate_vector(PARAMETER)

        x_star = [
            self.model.generate_vector(STATE),
            self.model.generate_vector(PARAMETER),
            None,
            self.model.generate_vector(SLACK),
        ]

        cost_old = self.model.cost(x)[0]
        gradnorm_ini = 0.0
        tol = 0.0
        gradnorm = 0.0
        cost_new = cost_old
        smooth_reg_new = 0.0
        nonsmooth_reg_new = 0.0
        misfit_new = 0.0
        cg_it_last = 0
        alpha_m = 1.0
        tolcg = cg_coarse_tol
        mg_mhat = 0.0

        while (self.it < max_iter) and not self.converged:
            self.model.solveAdj(x[ADJOINT], x)
            self.model.setPointForHessianEvaluations(
                x, gauss_newton_approx=(self.it < GN_iter)
            )
            gradnorm = self.model.evalGradientParameter(x, mg)

            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini * rel_tol)

            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break

            self.it += 1
            tolcg = min(
                cg_coarse_tol,
                math.sqrt(gradnorm / gradnorm_ini) if gradnorm_ini > 0 else cg_coarse_tol,
            )

            # Solve for m_hat using the reduced Hessian
            HessApply = NSReducedHessian(self.model)
            cg_params = CGSolverSteihaug_ParameterList()
            cg_params["rel_tolerance"] = tolcg
            cg_params["max_iter"] = cg_max_iter
            cg_params["zero_initial_guess"] = True
            cg_params["print_level"] = print_level - 1
            solver = CGSolverSteihaug(
                parameters=cg_params, comm=self.model.nsprior.mpi_comm()
            )
            solver.set_operator(HessApply.mat)
            solver.set_preconditioner(self.model.Psolver())

            neg_mg = self.model.generate_vector(PARAMETER)
            neg_mg.array[:] = -mg.array[:]
            neg_mg.scatter_forward()
            _zero(mhat)
            solver.solve(neg_mg, mhat)
            cg_it_last = HessApply.ncalls
            self.total_cg_iter += cg_it_last

            # Compute incremental slack via (3.6)
            self.model.nsprior.compute_w_hat(
                x[PARAMETER], x[SLACK], mhat, what
            )

            # Line search for m
            alpha_m = 1.0
            descent_m = False
            n_bt_m = 0
            mg_mhat = _inner(mg, mhat)

            while not descent_m and n_bt_m < max_bt:
                _zero(x_star[PARAMETER])
                _axpy(x_star[PARAMETER], 1.0, x[PARAMETER])
                _axpy(x_star[PARAMETER], alpha_m, mhat)
                _zero(x_star[STATE])
                _axpy(x_star[STATE], 1.0, x[STATE])
                self.model.solveFwd(x_star[STATE], x_star)
                cost_new, smooth_reg_new, nonsmooth_reg_new, misfit_new = (
                    self.model.cost(x_star)
                )
                if (
                    cost_new < cost_old + alpha_m * c_armijo * mg_mhat
                    or -mg_mhat <= self.parameters["gdm_tolerance"]
                ):
                    cost_old = cost_new
                    descent_m = True
                    _zero(x[PARAMETER])
                    _axpy(x[PARAMETER], 1.0, x_star[PARAMETER])
                    _zero(x[STATE])
                    _axpy(x[STATE], 1.0, x_star[STATE])
                else:
                    n_bt_m += 1
                    alpha_m *= 0.5

            if n_bt_m == max_bt:
                self.converged = False
                self.reason = 2
                break

            # Line search for w (keep ||w||_∞ ≤ 1)
            alpha_w = 1.0
            descent_w = False
            n_bt_w = 0
            while not descent_w and n_bt_w < max_bt:
                _zero(x_star[SLACK])
                _axpy(x_star[SLACK], 1.0, x[SLACK])
                _axpy(x_star[SLACK], alpha_w, what)
                norm_w = self.model.nsprior.wnorm(x_star[SLACK])
                if _linf_norm(norm_w) <= 1.0:
                    descent_w = True
                    _zero(x[SLACK])
                    _axpy(x[SLACK], 1.0, x_star[SLACK])
                else:
                    n_bt_w += 1
                    alpha_w *= 0.5

            if n_bt_w == max_bt:
                self.converged = False
                self.reason = 3
                break

            if print_level >= 0:
                if self.it == 1:
                    print(
                        "\n{0:3} {1:5} {2:15} {3:15} {4:15} {5:15} {6:15} {7:14} {8:14} {9:14}".format(
                            "It", "cg_it", "cost", "misfit", "s_reg", "ns_reg",
                            "(g,dm)", "||g||L2", "alpha", "tolcg",
                        )
                    )
                print(
                    "{0:3d} {1:5d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e} {9:14e}".format(
                        self.it, cg_it_last, cost_new, misfit_new,
                        smooth_reg_new, nonsmooth_reg_new,
                        mg_mhat, gradnorm, alpha_m, tolcg,
                    )
                )

            if self.callback:
                self.callback(self.it, x)

            if -mg_mhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 4
                break

        self.final_grad_norm = gradnorm
        self.final_cost = cost_new
        return x

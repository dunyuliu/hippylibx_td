# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Model class for non-smooth (TV-regularized) inverse problems.

Faithful dolfinx port of the legacy hippylib class authored by
**Xindi Gong** (see ``hippylib/modeling/nonsmoothModel.py`` in
``xindigong/hippylib:tv-enhanced``).

Wraps a PDE problem, a smooth prior (e.g. BiLaplacian), a non-smooth
prior (e.g. TVPrior), and a misfit into one Model with the same surface
as :class:`hippylibX.Model` plus a fourth ``SLACK`` slot in the state
list.
"""

from __future__ import annotations

import math
from typing import List

from ._variables import STATE, PARAMETER, ADJOINT
from . import SLACK


class ModelNS:
    """Inverse-problem model with smooth + non-smooth regularization.

    The variable list is ``x = [u, m, p, w]`` where ``w`` is the TV
    slack variable.

    Args:
        problem: a :class:`hippylibX.modeling.PDEVariationalProblem`-style
            object (must implement ``solveFwd``, ``solveAdj``,
            ``evalGradientParameter``, ``setLinearizationPoint``,
            ``solveIncremental``, ``apply_ij``, ``generate_state``,
            ``generate_parameter``).
        misfit: the data-fidelity term.
        prior: smooth prior (optional).
        nsprior: non-smooth (TV) prior (optional).
        which: 3-tuple of bools — enables [misfit, smooth, nonsmooth].
    """

    def __init__(
        self,
        problem,
        misfit,
        prior=None,
        nsprior=None,
        which: List[bool] | None = None,
    ):
        if which is None:
            which = [True, True, True]
        if prior is None and nsprior is None:
            raise ValueError("ModelNS requires at least one of prior/nsprior")
        self.problem = problem
        self.misfit = misfit
        self.prior = prior
        self.nsprior = nsprior
        self.which = list(which)
        self.gauss_newton_approx = False

        self.n_fwd_solve = 0
        self.n_adj_solve = 0
        self.n_inc_solve = 0

    # ---- factories ----------------------------------------------------
    def generate_vector(self, component="ALL"):
        if component == "ALL":
            return [
                self.problem.generate_state(),
                self.problem.generate_parameter(),
                self.problem.generate_state(),
                self.nsprior.generate_slack() if self.nsprior is not None else None,
            ]
        if component == STATE:
            return self.problem.generate_state()
        if component == PARAMETER:
            return self.problem.generate_parameter()
        if component == ADJOINT:
            return self.problem.generate_state()
        if component == SLACK:
            if self.nsprior is None:
                raise ValueError("generate_vector(SLACK) needs a non-smooth prior")
            return self.nsprior.generate_slack()
        raise ValueError(f"unknown component: {component!r}")

    def init_parameter(self, m):
        # the prior owns the parameter shape
        if self.prior is not None:
            return self.prior.generate_parameter()
        return self.nsprior.init_vector(0)

    # ---- cost ---------------------------------------------------------
    def cost(self, x) -> list:
        """Return ``[total, smooth_reg, nonsmooth_reg, misfit]``."""
        misfit_cost = self.misfit.cost(x) if self.which[0] else 0.0
        smooth_reg_cost = (
            self.prior.cost(x[PARAMETER])
            if self.prior is not None and self.which[1]
            else 0.0
        )
        nonsmooth_reg_cost = (
            self.nsprior.cost(x[PARAMETER])
            if self.nsprior is not None and self.which[2]
            else 0.0
        )
        return [
            misfit_cost + smooth_reg_cost + nonsmooth_reg_cost,
            smooth_reg_cost,
            nonsmooth_reg_cost,
            misfit_cost,
        ]

    # ---- forward / adjoint -------------------------------------------
    def solveFwd(self, out, x) -> None:
        self.n_fwd_solve += 1
        self.problem.solveFwd(out, x)

    def solveAdj(self, out, x) -> None:
        self.n_adj_solve += 1
        rhs = self.problem.generate_state()
        self.misfit.grad(STATE, x, rhs)
        if hasattr(rhs, "petsc_vec"):
            rhs.petsc_vec.scale(-1.0)
            rhs.scatter_forward()
        else:
            rhs *= -1.0
        self.problem.solveAdj(out, x, rhs)

    # ---- gradient -----------------------------------------------------
    def evalGradientParameter(self, x, mg, misfit_only: bool = False) -> float:
        tmp = self.generate_vector(PARAMETER)

        if self.which[0]:
            self.problem.evalGradientParameter(x, mg)
            self.misfit.grad(PARAMETER, x, tmp)
            self._axpy(mg, 1.0, tmp)
        else:
            self._zero(mg)

        if not misfit_only:
            if self.prior is not None and self.which[1]:
                self.prior.grad(x[PARAMETER], tmp)
                self._axpy(mg, 1.0, tmp)
            if self.nsprior is not None and self.which[2]:
                self.nsprior.grad(x[PARAMETER], tmp)
                self._axpy(mg, 1.0, tmp)

        # M-norm grad (matches legacy: sqrt(mg . Msolver * mg))
        if self.prior is not None and hasattr(self.prior, "Msolver"):
            Msolver = self.prior.Msolver
        elif self.nsprior is not None:
            Msolver = self.nsprior.Msolver
        else:
            raise ValueError("No mass-matrix solver on either prior")
        Msolver.solve(mg.petsc_vec, tmp.petsc_vec)
        return math.sqrt(float(mg.petsc_vec.dot(tmp.petsc_vec)))

    # ---- linearization ------------------------------------------------
    def setPointForHessianEvaluations(self, x, gauss_newton_approx: bool = False) -> None:
        self.gauss_newton_approx = bool(gauss_newton_approx)
        self.problem.setLinearizationPoint(x, self.gauss_newton_approx)
        self.misfit.setLinearizationPoint(x, self.gauss_newton_approx)
        if self.prior is not None and hasattr(self.prior, "setLinearizationPoint"):
            self.prior.setLinearizationPoint(x[PARAMETER], self.gauss_newton_approx)
        if self.nsprior is not None:
            self.nsprior.setLinearizationPoint(
                x[PARAMETER], x[SLACK], self.gauss_newton_approx
            )

    # ---- incremental solves ------------------------------------------
    def solveFwdIncremental(self, sol, rhs) -> None:
        self.n_inc_solve += 1
        self.problem.solveIncremental(sol, rhs, False)

    def solveAdjIncremental(self, sol, rhs) -> None:
        self.n_inc_solve += 1
        self.problem.solveIncremental(sol, rhs, True)

    # ---- KKT blocks ---------------------------------------------------
    def applyC(self, dm, out) -> None:
        self.problem.apply_ij(ADJOINT, PARAMETER, dm, out)

    def applyCt(self, dp, out) -> None:
        self.problem.apply_ij(PARAMETER, ADJOINT, dp, out)

    def applyWuu(self, du, out) -> None:
        self.misfit.apply_ij(STATE, STATE, du, out)
        if not self.gauss_newton_approx:
            tmp = self.generate_vector(STATE)
            self.problem.apply_ij(STATE, STATE, du, tmp)
            self._axpy(out, 1.0, tmp)

    def applyWum(self, dm, out) -> None:
        if self.gauss_newton_approx:
            self._zero(out)
        else:
            self.problem.apply_ij(STATE, PARAMETER, dm, out)
            tmp = self.generate_vector(STATE)
            self.misfit.apply_ij(STATE, PARAMETER, dm, tmp)
            self._axpy(out, 1.0, tmp)

    def applyWmu(self, du, out) -> None:
        if self.gauss_newton_approx:
            self._zero(out)
        else:
            self.problem.apply_ij(PARAMETER, STATE, du, out)
            tmp = self.generate_vector(PARAMETER)
            self.misfit.apply_ij(PARAMETER, STATE, du, tmp)
            self._axpy(out, 1.0, tmp)

    def applyR(self, dm, out) -> None:
        if self.prior is None:
            raise ValueError("applyR called but no smooth prior is set")
        self.prior.R.mult(dm.petsc_vec, out.petsc_vec)
        if hasattr(out, "scatter_forward"):
            out.scatter_forward()

    def applyRNS(self, dm, out) -> None:
        if self.nsprior is None:
            raise ValueError("applyRNS called but no non-smooth prior is set")
        self.nsprior.applyR(dm, out)

    def Psolver(self):
        if self.nsprior is not None and self.which[2]:
            return self.nsprior.Psolver()
        if self.prior is not None and self.which[1]:
            return self.prior.Rsolver
        raise ValueError("No preconditioner available")

    def Rsolver(self):
        if self.prior is not None and self.which[1]:
            return self.prior.Rsolver
        return None

    def applyWmm(self, dm, out) -> None:
        if self.gauss_newton_approx:
            self._zero(out)
        else:
            self.problem.apply_ij(PARAMETER, PARAMETER, dm, out)
            tmp = self.generate_vector(PARAMETER)
            self.misfit.apply_ij(PARAMETER, PARAMETER, dm, tmp)
            self._axpy(out, 1.0, tmp)

    def apply_ij(self, i, j, d, out) -> None:
        if i == STATE and j == STATE:
            self.applyWuu(d, out)
        elif i == STATE and j == PARAMETER:
            self.applyWum(d, out)
        elif i == PARAMETER and j == STATE:
            self.applyWmu(d, out)
        elif i == PARAMETER and j == PARAMETER:
            self.applyWmm(d, out)
        elif i == PARAMETER and j == ADJOINT:
            self.applyCt(d, out)
        elif i == ADJOINT and j == PARAMETER:
            self.applyC(d, out)
        else:
            raise IndexError(f"apply_ij not allowed for i={i}, j={j}")

    # ---- helpers ------------------------------------------------------
    @staticmethod
    def _axpy(dst, a: float, src) -> None:
        if hasattr(dst, "petsc_vec") and hasattr(src, "petsc_vec"):
            dst.petsc_vec.axpy(a, src.petsc_vec)
            if hasattr(dst, "scatter_forward"):
                dst.scatter_forward()
        elif hasattr(dst, "axpy"):
            dst.axpy(a, src)
        else:
            dst.array[:] += a * src.array[:]

    @staticmethod
    def _zero(v) -> None:
        if hasattr(v, "zero"):
            v.zero()
        else:
            v.array[:] = 0.0
            if hasattr(v, "scatter_forward"):
                v.scatter_forward()

# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""MultiPDEProblem — N independent forward problems sharing a parameter.

Faithful dolfinx port of the legacy hippylib class authored by
**Xindi Gong** (see ``hippylib/modeling/multiPDEProblem.py`` in
``xindigong/hippylib:tv-enhanced``).

State, adjoint, and incremental quantities are stored as
:class:`BlockVector` (one slot per PDE instance). The parameter is
shared across all instances.
"""

from __future__ import annotations

from ._variables import STATE, PARAMETER, ADJOINT
from .blockVector import BlockVector


class MultiPDEProblem:
    """A coupled forward problem composed of ``N`` independent PDE problems
    sharing a single parameter.

    Each constituent must satisfy the :class:`hippylibX.modeling.
    PDEVariationalProblem`-style interface (``solveFwd``, ``solveAdj``,
    ``evalGradientParameter``, ``setLinearizationPoint``,
    ``solveIncremental``, ``apply_ij``, ``generate_state``,
    ``generate_parameter``).
    """

    def __init__(self, problems: list):
        if not problems:
            raise ValueError("MultiPDEProblem requires at least one problem")
        self.Vh = problems[0].Vh
        self.problems = list(problems)
        self.n_problems = len(self.problems)

    # ---- factories ----------------------------------------------------
    def generate_state(self) -> BlockVector:
        """A :class:`BlockVector` with one state-shaped slot per problem."""
        return BlockVector.fromFunctionSpace(self.Vh[STATE], self.n_problems)

    def generate_parameter(self):
        return self.problems[0].generate_parameter()

    # ---- forward / adjoint --------------------------------------------
    def solveFwd(self, state: BlockVector, x: list) -> None:
        u, m, p = x
        for k, problem in enumerate(self.problems):
            problem.solveFwd(state.data[k], [u.data[k], m, None])

    def solveAdj(self, adj: BlockVector, x: list, adj_rhs: BlockVector) -> None:
        u, m, p = x
        for k, problem in enumerate(self.problems):
            problem.solveAdj(
                adj.data[k], [u.data[k], m, p.data[k]], adj_rhs.data[k]
            )

    def evalGradientParameter(self, x: list, out) -> None:
        tmp = self.generate_parameter()
        u, m, p = x
        if hasattr(out, "zero"):
            out.zero()
        else:
            out.array[:] = 0.0
        for k, problem in enumerate(self.problems):
            if hasattr(tmp, "zero"):
                tmp.zero()
            else:
                tmp.array[:] = 0.0
            problem.evalGradientParameter([u.data[k], m, p.data[k]], tmp)
            _axpy_param(out, 1.0, tmp)

    # ---- Hessian -------------------------------------------------------
    def setLinearizationPoint(self, x: list, gauss_newton_approx: bool) -> None:
        u, m, p = x
        for k, problem in enumerate(self.problems):
            problem.setLinearizationPoint(
                [u.data[k], m, p.data[k]], gauss_newton_approx
            )

    def solveIncremental(self, out: BlockVector, rhs: BlockVector, is_adj: bool) -> None:
        for k, problem in enumerate(self.problems):
            problem.solveIncremental(out.data[k], rhs.data[k], is_adj)

    def apply_ij(self, i: int, j: int, dir, out) -> None:
        if hasattr(out, "zero"):
            out.zero()
        else:
            out.array[:] = 0.0

        if i == PARAMETER:
            tmp = self.generate_parameter()
            if j == PARAMETER:
                for problem in self.problems:
                    problem.apply_ij(i, j, dir, tmp)
                    _axpy_param(out, 1.0, tmp)
            else:
                for k, problem in enumerate(self.problems):
                    problem.apply_ij(i, j, dir.data[k], tmp)
                    _axpy_param(out, 1.0, tmp)
        else:
            if not isinstance(out, BlockVector):
                raise TypeError(
                    "apply_ij with i != PARAMETER expects a BlockVector `out`"
                )
            if j == PARAMETER:
                for k, problem in enumerate(self.problems):
                    problem.apply_ij(i, j, dir, out.data[k])
            else:
                for k, problem in enumerate(self.problems):
                    problem.apply_ij(i, j, dir.data[k], out.data[k])


# ---- helpers ----------------------------------------------------------
def _axpy_param(dst, a: float, src) -> None:
    """Generic axpy that works on either `dlx.la.Vector` or PETSc Vec-like."""
    if hasattr(dst, "axpy") and hasattr(src, "axpy"):
        dst.axpy(a, src)
    elif hasattr(dst, "petsc_vec"):
        dst.petsc_vec.axpy(a, src.petsc_vec)
        dst.scatter_forward()
    else:
        dst.array[:] += a * src.array[:]
        if hasattr(dst, "scatter_forward"):
            dst.scatter_forward()

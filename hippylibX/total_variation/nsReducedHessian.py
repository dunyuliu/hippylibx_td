# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Reduced Hessian aware of the non-smooth (TV) regularization term.

Faithful dolfinx port of the legacy hippylib class authored by
**Graham Pash** (primary implementer; hosted on the fork of **Xindi Gong** — see ``hippylib/modeling/reducedHessian.py:NSReducedHessian``
in ``xindigong/hippylib:tv-enhanced``).
"""

from __future__ import annotations

import petsc4py
from petsc4py import PETSc

from ._variables import STATE, PARAMETER, ADJOINT


class NSReducedHessian:
    """Matrix-free reduced Hessian for non-smooth (TV) inverse problems.

    The Hessian decomposes into misfit / smooth regularization /
    non-smooth regularization contributions, gated by the corresponding
    flags in ``model.which``.
    """

    def __init__(self, model, misfit_only: bool = False):
        self.model = model
        self.gauss_newton_approx = model.gauss_newton_approx
        self.which = list(model.which)
        self.misfit_only = bool(misfit_only)
        self.ncalls = 0

        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(ADJOINT)
        self.rhs_adj2 = model.generate_vector(ADJOINT)
        self.uhat = model.generate_vector(STATE)
        self.phat = model.generate_vector(ADJOINT)
        self.yhelp = model.generate_vector(PARAMETER)

        # PETSc python-mat wrapper so we can plug into CGSolverSteihaug
        m = model.generate_vector(PARAMETER)
        self.petsc_wrapper = PETSc.Mat().createPython(
            m.petsc_vec.getSizes(), comm=m.petsc_vec.getComm()
        )
        self.petsc_wrapper.setPythonContext(self)
        self.petsc_wrapper.setUp()

    def __del__(self):
        if getattr(self, "petsc_wrapper", None) is not None:
            try:
                self.petsc_wrapper.destroy()
            except Exception:
                pass

    @property
    def mat(self) -> PETSc.Mat:
        return self.petsc_wrapper

    def init_vector(self, x, dim: int) -> None:
        # parameter shape — nothing to do for dolfinx vectors; kept for
        # API compatibility with the legacy class.
        return

    # ---- PETSc python-mat callback ------------------------------------
    def mult(self, mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        x_dlx = self.model.generate_vector(PARAMETER)
        y_dlx = self.model.generate_vector(PARAMETER)
        x_dlx.petsc_vec.axpy(1.0, x)
        if self.gauss_newton_approx:
            self.GNHessian(x_dlx, y_dlx)
        else:
            self.TrueHessian(x_dlx, y_dlx)
        y.axpby(1.0, 0.0, y_dlx.petsc_vec)
        self.ncalls += 1

    def inner(self, x, y) -> float:
        Ay = self.model.generate_vector(PARAMETER)
        Ay.array[:] = 0.0
        if self.gauss_newton_approx:
            self.GNHessian(y, Ay)
        else:
            self.TrueHessian(y, Ay)
        return float(x.petsc_vec.dot(Ay.petsc_vec))

    # ---- Hessian assemblies -------------------------------------------
    def GNHessian(self, x, y) -> None:
        self._zero(y)
        if self.which[0]:
            self.model.applyC(x, self.rhs_fwd)
            self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
            self.model.applyWuu(self.uhat, self.rhs_adj)
            self.model.solveAdjIncremental(self.phat, self.rhs_adj)
            self.model.applyCt(self.phat, y)
        if not self.misfit_only:
            if self.which[1]:
                self.model.applyR(x, self.yhelp)
                self._axpy(y, 1.0, self.yhelp)
            if self.which[2]:
                self.model.applyRNS(x, self.yhelp)
                self._axpy(y, 1.0, self.yhelp)

    def TrueHessian(self, x, y) -> None:
        self._zero(y)
        if self.which[0]:
            self.model.applyC(x, self.rhs_fwd)
            self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
            self.model.applyWuu(self.uhat, self.rhs_adj)
            self.model.applyWum(x, self.rhs_adj2)
            self._axpy(self.rhs_adj, -1.0, self.rhs_adj2)
            self.model.solveAdjIncremental(self.phat, self.rhs_adj)
            self.model.applyWmm(x, y)
            self.model.applyCt(self.phat, self.yhelp)
            self._axpy(y, 1.0, self.yhelp)
            self.model.applyWmu(self.uhat, self.yhelp)
            self._axpy(y, -1.0, self.yhelp)
        if not self.misfit_only:
            if self.which[1]:
                self.model.applyR(x, self.yhelp)
                self._axpy(y, 1.0, self.yhelp)
            if self.which[2]:
                self.model.applyRNS(x, self.yhelp)
                self._axpy(y, 1.0, self.yhelp)

    # ---- helpers ------------------------------------------------------
    @staticmethod
    def _zero(v) -> None:
        if hasattr(v, "zero"):
            v.zero()
        else:
            v.array[:] = 0.0
            if hasattr(v, "scatter_forward"):
                v.scatter_forward()

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

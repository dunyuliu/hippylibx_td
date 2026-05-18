# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Total Variation prior (primal-dual formulation) for hippylibX (dolfinx).

Faithful dolfinx port of the legacy hippylib classes authored by
**Xindi Gong** (see ``hippylib/modeling/nonsmoothPrior.py`` in
``xindigong/hippylib:tv-enhanced``).

References:
    [1] Chan, Tony F., Gene H. Golub, and Pep Mulet.
        "A nonlinear primal-dual method for total variation-based
        image restoration."
        SIAM J. Sci. Comput. 20(6) (1999): 1964-1977.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import ufl
import dolfinx as dlx
import dolfinx.fem.petsc
import petsc4py
from petsc4py import PETSc
from mpi4py import MPI

from ..utils.vector2function import vector2Function


def _make_lu(A: PETSc.Mat) -> PETSc.KSP:
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    try:
        pc.setFactorSolverType("mumps")
    except Exception:
        pass
    return ksp


def _make_krylov(A: PETSc.Mat, ksp_type: str = "cg", pc_type: str = "jacobi",
                 rel_tol: float = 1e-12, max_iter: int = 100) -> PETSc.KSP:
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=rel_tol, max_it=max_iter)
    pc = ksp.getPC()
    pc.setType(pc_type)
    ksp.setErrorIfNotConverged(True)
    ksp.setInitialGuessNonzero(False)
    return ksp


def _make_solver(A: PETSc.Mat, solver_type: str,
                 rel_tol: float, max_iter: int,
                 ksp_type: str = "cg", pc_type: str = "jacobi") -> PETSc.KSP:
    if solver_type == "krylov":
        return _make_krylov(A, ksp_type, pc_type, rel_tol, max_iter)
    if solver_type == "lu":
        return _make_lu(A)
    raise ValueError(f"Unknown solver_type {solver_type!r}")


def _vec_like(Vh) -> dlx.la.Vector:
    return dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)


class TVPrior:
    """Primal-dual scalar Total Variation prior.

    Functional:
        ``alpha * ∫ sqrt(|∇m|² + β) dx``

    Args:
        Vhm: parameter space.
        Vhw: slack-variable space (typically a vector-valued P0/P1).
        Vhwnorm: scalar space used to store ``|w|``.
        alpha: TV weight.
        beta: smoothing parameter (β > 0 so the integrand is smooth at ∇m=0).
        peps: mass-matrix preconditioner scaling.
        rel_tol, max_iter: Krylov solver tolerances.
        solver_type: ``"krylov"`` (default) or ``"lu"``.
    """

    def __init__(
        self,
        Vhm,
        Vhw,
        Vhwnorm,
        alpha: float,
        beta: float,
        peps: float = 1e-3,
        rel_tol: float = 1e-12,
        max_iter: int = 100,
        solver_type: str = "krylov",
    ):
        msh = Vhm.mesh
        self.alpha = dlx.fem.Constant(msh, dlx.default_scalar_type(float(alpha)))
        self.beta = dlx.fem.Constant(msh, dlx.default_scalar_type(float(beta)))

        self.Vhm = Vhm
        self.Vhw = Vhw
        self.Vhwnorm = Vhwnorm

        self.m_lin: dlx.fem.Function | None = None
        self.w_lin: dlx.fem.Function | None = None
        self.gauss_newton_approx = False

        self.peps = float(peps)
        self.rel_tol = float(rel_tol)
        self.max_iter = int(max_iter)
        self.solver_type = solver_type

        (self.m_trial, self.m_test, self.M, self.Msolver) = self._setupM(Vhm)
        (self.w_trial, self.w_test, self.Mw, self.Mwsolver) = self._setupM(Vhw)
        (self.wnorm_trial, self.wnorm_test,
         self.Mwnorm, self.Mwnormsolver) = self._setupM(Vhwnorm)

        # ncomp == 0 mirrors legacy convention: scalar parameter space
        self.ncomp = Vhm.num_sub_spaces
        self.ndim = Vhm.mesh.topology.dim

    def __del__(self):
        for attr in ("M", "Mw", "Mwnorm", "Msolver", "Mwsolver", "Mwnormsolver"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass

    # ---- internals ----------------------------------------------------
    def _setupM(self, Vh):
        trial = ufl.TrialFunction(Vh)
        test = ufl.TestFunction(Vh)
        varfM = ufl.inner(trial, test) * ufl.dx
        M = dolfinx.fem.petsc.assemble_matrix(dlx.fem.form(varfM), bcs=[])
        M.assemble()
        Msolver = _make_solver(
            M, self.solver_type, self.rel_tol, self.max_iter,
            ksp_type="cg", pc_type="jacobi",
        )
        return trial, test, M, Msolver

    def _fTV(self, m):
        return ufl.sqrt(ufl.inner(ufl.grad(m), ufl.grad(m)) + self.beta)

    # ---- factories ----------------------------------------------------
    def init_vector(self, dim: int = 0) -> dlx.la.Vector:
        """Return a vector compatible with the parameter mass matrix."""
        return _vec_like(self.Vhm)

    def generate_slack(self) -> dlx.la.Vector:
        return _vec_like(self.Vhw)

    def mpi_comm(self):
        return self.Vhm.mesh.comm

    # ---- linearization ------------------------------------------------
    def setLinearizationPoint(self, m, w, gauss_newton_approx: bool) -> None:
        self.m_lin = vector2Function(m, self.Vhm)
        self.w_lin = vector2Function(w, self.Vhw)
        self.gauss_newton_approx = bool(gauss_newton_approx)

    # ---- cost / grad --------------------------------------------------
    def cost(self, m) -> float:
        mfun = vector2Function(m, self.Vhm)
        local = dlx.fem.assemble_scalar(
            dlx.fem.form(self.alpha * self._fTV(mfun) * ufl.dx)
        )
        return float(self.Vhm.mesh.comm.allreduce(local, op=MPI.SUM))

    def grad(self, m, out) -> None:
        out.array[:] = 0.0
        mfun = vector2Function(m, self.Vhm)
        TVm = self._fTV(mfun)
        grad_tv = (
            self.alpha * (dlx.fem.Constant(self.Vhm.mesh, dlx.default_scalar_type(1.0)) / TVm)
            * ufl.inner(ufl.grad(mfun), ufl.grad(self.m_test)) * ufl.dx
        )
        dolfinx.fem.petsc.assemble_vector(out.petsc_vec, dlx.fem.form(grad_tv))
        out.petsc_vec.ghostUpdate(
            PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
        )

    # ---- primal Hessian coefficient ----------------------------------
    def primal_hess_coeff(self, m, w, TVm):
        """Equations (5.1)/(5.2) from Chan-Golub-Mulet, symmetrized."""
        one_half = dlx.fem.Constant(self.Vhm.mesh, dlx.default_scalar_type(0.5))
        one = dlx.fem.Constant(self.Vhm.mesh, dlx.default_scalar_type(1.0))
        if self.ncomp == 0:
            return (one / TVm) * (
                ufl.Identity(self.ndim)
                - one_half * ufl.outer(w, ufl.grad(m) / TVm)
                - one_half * ufl.outer(ufl.grad(m) / TVm, w)
            )
        i, j, k, l = ufl.indices(4)
        eye_ncomp = ufl.Identity(self.ncomp)
        eye_ndim = ufl.Identity(self.ndim)
        eye = ufl.as_tensor(eye_ncomp[i, k] * eye_ndim[j, l], (i, j, k, l))
        return (one / TVm) * (
            eye
            - one_half * ufl.outer(w, ufl.grad(m) / TVm)
            - one_half * ufl.outer(ufl.grad(m) / TVm, w)
        )

    def hess_action(self, m, w, m_dir):
        TVm = self._fTV(m)
        A = self.primal_hess_coeff(m, w, TVm)
        if self.ncomp == 0:
            return self.alpha * ufl.inner(ufl.dot(A, ufl.grad(m_dir)),
                                          ufl.grad(self.m_test)) * ufl.dx
        i, j, k, l = ufl.indices(4)
        return (
            self.alpha
            * ufl.grad(m_dir)[i, j] * A[i, j, k, l] * ufl.grad(self.m_test)[k, l]
            * ufl.dx
        )

    def applyR(self, dm, out) -> None:
        out.array[:] = 0.0
        m_dir = vector2Function(dm, self.Vhm)
        form = self.hess_action(self.m_lin, self.w_lin, m_dir)
        dolfinx.fem.petsc.assemble_vector(out.petsc_vec, dlx.fem.form(form))
        out.petsc_vec.ghostUpdate(
            PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
        )

    def compute_w_hat(self, m, w, m_hat, w_hat) -> None:
        m_f = vector2Function(m, self.Vhm)
        m_hat_f = vector2Function(m_hat, self.Vhm)
        w_f = vector2Function(w, self.Vhw)
        TVm = self._fTV(m_f)
        A = self.primal_hess_coeff(m_f, w_f, TVm)
        if self.ncomp == 0:
            dw = (A * ufl.grad(m_hat_f) - w_f + ufl.grad(m_f) / TVm)
        else:
            i, j, k, l = ufl.indices(4)
            dw = (
                ufl.as_tensor(A[i, j, k, l] * ufl.grad(m_hat_f)[k, l], (i, j))
                - w_f + ufl.grad(m_f) / TVm
            )
        rhs = _vec_like(self.Vhw)
        dolfinx.fem.petsc.assemble_vector(
            rhs.petsc_vec, dlx.fem.form(ufl.inner(self.w_test, dw) * ufl.dx)
        )
        rhs.petsc_vec.ghostUpdate(
            PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
        )
        self.Mwsolver.solve(rhs.petsc_vec, w_hat.petsc_vec)
        w_hat.scatter_forward()

    def wnorm(self, w) -> dlx.la.Vector:
        w_f = vector2Function(w, self.Vhw)
        nw = ufl.inner(w_f, w_f)
        rhs = _vec_like(self.Vhwnorm)
        dolfinx.fem.petsc.assemble_vector(
            rhs.petsc_vec, dlx.fem.form(ufl.inner(self.wnorm_test, nw) * ufl.dx)
        )
        rhs.petsc_vec.ghostUpdate(
            PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
        )
        out = _vec_like(self.Vhwnorm)
        self.Mwnormsolver.solve(rhs.petsc_vec, out.petsc_vec)
        out.scatter_forward()
        return out

    def compute_w(self, m) -> dlx.la.Vector:
        m_f = vector2Function(m, self.Vhm)
        TVm = self._fTV(m_f)
        w_form = ufl.grad(m_f) / TVm
        rhs = _vec_like(self.Vhw)
        dolfinx.fem.petsc.assemble_vector(
            rhs.petsc_vec, dlx.fem.form(ufl.inner(self.w_test, w_form) * ufl.dx)
        )
        rhs.petsc_vec.ghostUpdate(
            PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
        )
        out = _vec_like(self.Vhw)
        self.Mwsolver.solve(rhs.petsc_vec, out.petsc_vec)
        out.scatter_forward()
        return out

    # ---- preconditioner -----------------------------------------------
    def Psolver(self) -> PETSc.KSP:
        varfHTV = self.hess_action(self.m_lin, self.w_lin, self.m_trial)
        varfM = ufl.inner(self.m_trial, self.m_test) * ufl.dx
        varfP = varfHTV + self.peps * varfM
        P = dolfinx.fem.petsc.assemble_matrix(dlx.fem.form(varfP), bcs=[])
        P.assemble()
        if self.solver_type == "krylov":
            return _make_krylov(P, "cg", "hypre", self.rel_tol, self.max_iter)
        return _make_lu(P)


class weightedVTVPrior(TVPrior):
    """Weighted vector Total Variation prior:

        ``∫ sqrt( Σ_i α_i |∇m_i|² + β ) dx``

    Faithful port of legacy ``weightedVTVPrior``. Inherits factories
    and primal/dual helpers from :class:`TVPrior` and overrides the
    integrand-defining methods.
    """

    def __init__(
        self,
        Vhm,
        Vhw,
        Vhwnorm,
        alpha: Sequence[float],
        beta: float,
        peps: float = 1e-3,
        rel_tol: float = 1e-12,
        max_iter: int = 100,
        solver_type: str = "krylov",
    ):
        if len(alpha) <= 1:
            raise ValueError("alpha must be a vector of length > 1")
        if len(alpha) != Vhm.num_sub_spaces:
            raise ValueError("alpha must match the number of parameter components")
        # we intentionally skip TVPrior.__init__ early call to alpha (scalar)
        # and instead build alpha as a diagonal matrix below.
        msh = Vhm.mesh
        self.beta = dlx.fem.Constant(msh, dlx.default_scalar_type(float(beta)))
        self.alpha_vec = np.asarray(alpha, dtype=np.float64)

        self.Vhm = Vhm
        self.Vhw = Vhw
        self.Vhwnorm = Vhwnorm
        self.m_lin = None
        self.w_lin = None
        self.gauss_newton_approx = False

        self.peps = float(peps)
        self.rel_tol = float(rel_tol)
        self.max_iter = int(max_iter)
        self.solver_type = solver_type

        (self.m_trial, self.m_test, self.M, self.Msolver) = self._setupM(Vhm)
        (self.w_trial, self.w_test, self.Mw, self.Mwsolver) = self._setupM(Vhw)
        (self.wnorm_trial, self.wnorm_test,
         self.Mwnorm, self.Mwnormsolver) = self._setupM(Vhwnorm)

        self.ncomp = Vhm.num_sub_spaces
        self.ndim = msh.topology.dim

        # alpha as a diagonal (ncomp, ncomp) ufl matrix
        self.alpha = ufl.as_matrix(np.diag(self.alpha_vec))

    def _fTV(self, m):
        return ufl.sqrt(
            ufl.inner(self.alpha * ufl.grad(m), ufl.grad(m)) + self.beta
        )

    def cost(self, m) -> float:
        mfun = vector2Function(m, self.Vhm)
        local = dlx.fem.assemble_scalar(dlx.fem.form(self._fTV(mfun) * ufl.dx))
        return float(self.Vhm.mesh.comm.allreduce(local, op=MPI.SUM))

    def grad(self, m, out) -> None:
        out.array[:] = 0.0
        mfun = vector2Function(m, self.Vhm)
        wvtv = self._fTV(mfun)
        grad_form = (
            ufl.inner(self.alpha * ufl.grad(mfun), ufl.grad(self.m_test)) / wvtv
        ) * ufl.dx
        dolfinx.fem.petsc.assemble_vector(out.petsc_vec, dlx.fem.form(grad_form))
        out.petsc_vec.ghostUpdate(
            PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
        )

    def primal_hess_coeff(self, m, w, wvtv):
        one_half = dlx.fem.Constant(self.Vhm.mesh, dlx.default_scalar_type(0.5))
        one = dlx.fem.Constant(self.Vhm.mesh, dlx.default_scalar_type(1.0))
        i, j, k, l = ufl.indices(4)
        eye_ndim = ufl.Identity(self.ndim)
        weighted_eye = ufl.as_tensor(self.alpha[i, k] * eye_ndim[j, l], (i, j, k, l))
        v = ufl.grad(m) / wvtv
        return (one / wvtv) * (
            weighted_eye
            - one_half * ufl.outer(self.alpha * w, self.alpha * v)
            - one_half * ufl.outer(self.alpha * v, self.alpha * w)
        )

    def hess_action(self, m, w, m_dir):
        wvtv = self._fTV(m)
        A = self.primal_hess_coeff(m, w, wvtv)
        i, j, k, l = ufl.indices(4)
        return (
            ufl.grad(m_dir)[i, j] * A[i, j, k, l] * ufl.grad(self.m_test)[k, l]
            * ufl.dx
        )

    def compute_w_hat(self, m, w, m_hat, w_hat) -> None:
        m_f = vector2Function(m, self.Vhm)
        m_hat_f = vector2Function(m_hat, self.Vhm)
        w_f = vector2Function(w, self.Vhw)
        wvtv = self._fTV(m_f)
        A = self.primal_hess_coeff(m_f, w_f, wvtv)
        i, j, k, l = ufl.indices(4)
        dw = (
            ufl.as_tensor(A[i, j, k, l] * ufl.grad(m_hat_f)[k, l], (i, j))
            - w_f + (ufl.grad(m_f) / wvtv)
        )
        rhs = _vec_like(self.Vhw)
        dolfinx.fem.petsc.assemble_vector(
            rhs.petsc_vec, dlx.fem.form(ufl.inner(self.w_test, dw) * ufl.dx)
        )
        rhs.petsc_vec.ghostUpdate(
            PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
        )
        self.Mwsolver.solve(rhs.petsc_vec, w_hat.petsc_vec)
        w_hat.scatter_forward()

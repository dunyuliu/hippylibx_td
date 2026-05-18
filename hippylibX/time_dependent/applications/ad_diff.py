# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Advection-Diffusion with initial-condition inversion.

Faithful port of legacy hippylib's `TimeDependentAD` (applications/ad_diff/
model_ad_diff.py). Mirrors the *Model* interface (not the PDEProblem interface)
because legacy `TimeDependentAD` is itself a Model — it owns the prior + misfit
and has no `m` in the varf (the parameter is the initial condition).

Variational form: backward-Euler discretization with GLS streamline stabilization
for advection. Stationary divergence-free wind field is supplied at construction.

PARAMETER and STATE function spaces must be identical, since u^0 = m.
"""

from __future__ import annotations

import numpy as np
import ufl
import dolfinx as dlx
import dolfinx.fem.petsc
from petsc4py import PETSc

from ...modeling.variables import STATE, PARAMETER, ADJOINT
from ..timeDependentVector import TimeDependentVector


class AdvectionDiffusionICModel:
    """Advection-Diffusion model with IC inversion (linear, time-dependent).

    Forward:
        ``u^{n+1} - u^n + Δt * (-div(κ ∇u^{n+1}) + v·∇u^{n+1}) = 0``,  with
        ``u^0 = m``.

    Misfit: provided externally (e.g. :class:`SpaceTimePointwiseStateObservation`).
    Prior: provided externally.

    The class implements the `hippylibX.Model` interface so that the upstream
    :class:`ReducedSpaceNewtonCG` and :class:`ReducedHessian` work unchanged.
    """

    def __init__(self, Vh, prior, misfit, simulation_times, wind_velocity, kappa: float = 1e-3, gls_stab: bool = True):
        # Vh must satisfy Vh[STATE] == Vh[PARAMETER] (so m is a valid IC).
        self.Vh = Vh
        self.prior = prior
        self.misfit = misfit
        self.simulation_times = list(simulation_times)
        assert len(self.simulation_times) >= 2
        self.dt = float(self.simulation_times[1] - self.simulation_times[0])
        self.wind = wind_velocity
        self.kappa = kappa

        msh = Vh[STATE].mesh
        self.mesh = msh
        u = ufl.TrialFunction(Vh[STATE])
        v = ufl.TestFunction(Vh[STATE])

        kappa_const = dlx.fem.Constant(msh, dlx.default_scalar_type(float(kappa)))
        dt_const = dlx.fem.Constant(msh, dlx.default_scalar_type(self.dt))
        zero_const = dlx.fem.Constant(msh, dlx.default_scalar_type(0.0))

        # GLS stabilization parameter tau (Hughes-Tezduyar style).
        # NOTE: `h/vnorm` is undefined at wind-stagnation points where
        # `|wind| = 0`. A small floor `eps_v` is added to the norm to avoid
        # a divide-by-zero on such cells without changing the answer where
        # the wind is non-degenerate.
        h = ufl.CellDiameter(msh)
        eps_v = dlx.fem.Constant(msh, dlx.default_scalar_type(1e-30))
        vnorm = ufl.sqrt(ufl.inner(self.wind, self.wind) + eps_v)
        if gls_stab:
            tau = ufl.min_value(
                h * h / (dlx.fem.Constant(msh, dlx.default_scalar_type(2.0)) * kappa_const),
                h / vnorm,
            )
        else:
            tau = zero_const

        # Residuals (used for SUPG/GLS test-function modifications)
        r_trial = u + dt_const * (-ufl.div(kappa_const * ufl.grad(u)) + ufl.inner(self.wind, ufl.grad(u)))
        r_test = v + dt_const * (-ufl.div(kappa_const * ufl.grad(v)) + ufl.inner(self.wind, ufl.grad(v)))

        # Standard matrices
        M_form = dlx.fem.form(ufl.inner(u, v) * ufl.dx)
        Mstab_form = dlx.fem.form(ufl.inner(u, v + tau * r_test) * ufl.dx)
        Mtstab_form = dlx.fem.form(ufl.inner(u + tau * r_trial, v) * ufl.dx)
        N_form = dlx.fem.form(
            (ufl.inner(kappa_const * ufl.grad(u), ufl.grad(v))
             + ufl.inner(self.wind, ufl.grad(u)) * v) * ufl.dx
        )
        Nt_form = dlx.fem.form(
            (ufl.inner(kappa_const * ufl.grad(v), ufl.grad(u))
             + ufl.inner(self.wind, ufl.grad(v)) * u) * ufl.dx
        )
        stab_form = dlx.fem.form(tau * ufl.inner(r_trial, r_test) * ufl.dx)

        self.M = dolfinx.fem.petsc.assemble_matrix(M_form, bcs=[]); self.M.assemble()
        self.M_stab = dolfinx.fem.petsc.assemble_matrix(Mstab_form, bcs=[]); self.M_stab.assemble()
        self.Mt_stab = dolfinx.fem.petsc.assemble_matrix(Mtstab_form, bcs=[]); self.Mt_stab.assemble()
        N = dolfinx.fem.petsc.assemble_matrix(N_form, bcs=[]); N.assemble()
        Nt = dolfinx.fem.petsc.assemble_matrix(Nt_form, bcs=[]); Nt.assemble()
        stab = dolfinx.fem.petsc.assemble_matrix(stab_form, bcs=[]); stab.assemble()

        # L = M + dt*N + stab,  Lt = M + dt*Nt + stab
        self.L = self.M.duplicate(copy=True)
        self.L.axpy(self.dt, N)
        self.L.axpy(1.0, stab)

        self.Lt = self.M.duplicate(copy=True)
        self.Lt.axpy(self.dt, Nt)
        self.Lt.axpy(1.0, stab)

        N.destroy()
        Nt.destroy()
        stab.destroy()

        # KSPs (LU via MUMPS, matching `TimeDependentPDEVariationalProblem`)
        self.solver = self._make_lu(self.L)
        self.solvert = self._make_lu(self.Lt)

        self.gauss_newton_approx = False
        self.n_fwd_solve = 0
        self.n_adj_solve = 0
        self.n_inc_solve = 0

    def __del__(self):
        for attr in (
            "M", "M_stab", "Mt_stab", "L", "Lt",
            "solver", "solvert",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass

    @staticmethod
    def _make_lu(A: PETSc.Mat) -> PETSc.KSP:
        ksp = PETSc.KSP().create(A.getComm())
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType("mumps")
        except Exception as exc:
            import warnings
            warnings.warn(
                f"MUMPS not available for LU factorization "
                f"({type(exc).__name__}: {exc}); falling back to PETSc default.",
                RuntimeWarning,
                stacklevel=2,
            )
        ksp.setFromOptions()
        return ksp

    # ---- factories ----------------------------------------------------
    def generate_vector(self, component="ALL"):
        if component == "ALL":
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.Vh[STATE])
            m = self.prior.generate_parameter(0)
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.Vh[ADJOINT])
            return [u, m, p]
        elif component == STATE:
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.Vh[STATE])
            return u
        elif component == PARAMETER:
            return self.prior.generate_parameter(0)
        elif component == ADJOINT:
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.Vh[ADJOINT])
            return p
        else:
            raise ValueError(component)

    def init_parameter(self, m):
        return self.prior.generate_parameter(0)

    # ---- cost ---------------------------------------------------------
    def cost(self, x) -> list:
        misfit_cost = self.misfit.cost(x)
        reg_cost = self.prior.cost(x[PARAMETER])
        return [misfit_cost + reg_cost, reg_cost, misfit_cost]

    # ---- forward ------------------------------------------------------
    def solveFwd(self, out, x) -> None:
        self.n_fwd_solve += 1
        out.zero()
        # IC lives in x[PARAMETER]; legacy AD does NOT store it into out at t=0
        # (out[t=0] stays zero — matching legacy semantics).
        u0_vec = x[PARAMETER]
        rhs = self.M.createVecRight()
        uold = u0_vec.petsc_vec.copy()
        u = self.M.createVecRight()
        for t in self.simulation_times[1:]:
            self.M_stab.mult(uold, rhs)
            self.solver.solve(rhs, u)
            tgt = out.view(t)
            tgt.petsc_vec.array[:] = u.array[:]
            tgt.scatter_forward()
            uold.array[:] = u.array[:]
        u.destroy()
        rhs.destroy()
        uold.destroy()

    # ---- adjoint ------------------------------------------------------
    def solveAdj(self, out, x) -> None:
        self.n_adj_solve += 1
        # adjoint RHS = -misfit.grad(STATE, x, .)
        grad_state = TimeDependentVector(self.simulation_times)
        grad_state.initialize(self.Vh[STATE])
        self.misfit.grad(STATE, x, grad_state)

        out.zero()
        pold = self.M.createVecRight()
        p = self.M.createVecRight()
        rhs = self.M.createVecRight()

        for t in reversed(self.simulation_times):
            self.Mt_stab.mult(pold, rhs)
            gs_t = grad_state.view(t)
            rhs.axpy(-1.0, gs_t.petsc_vec)
            self.solvert.solve(rhs, p)
            tgt = out.view(t)
            tgt.petsc_vec.array[:] = p.array[:]
            tgt.scatter_forward()
            pold.array[:] = p.array[:]

        p.destroy()
        pold.destroy()
        rhs.destroy()

    # ---- gradient -----------------------------------------------------
    def evalGradientParameter(self, x, mg, misfit_only: bool = False) -> float:
        if misfit_only:
            mg.array[:] = 0.0
        else:
            self.prior.grad(x[PARAMETER], mg)
        p_at_1 = x[ADJOINT].view(self.simulation_times[1]).petsc_vec
        tmp = self.M.createVecRight()
        self.Mt_stab.mult(p_at_1, tmp)
        mg.petsc_vec.axpy(-1.0, tmp)
        mg.scatter_forward()
        # M^{-1}-norm squared, matching legacy `TimeDependentAD` (which returns
        # `g.inner(mg)` without the sqrt). NewtonCG only uses this as the
        # relative-convergence metric, and consistency between the two stacks
        # matters more than the absolute scaling.
        g = self.prior.generate_parameter(0)
        self.prior.Msolver.solve(mg.petsc_vec, g.petsc_vec)
        grad_norm = float(mg.petsc_vec.dot(g.petsc_vec))
        tmp.destroy()
        return grad_norm

    # ---- Hessian setup -----------------------------------------------
    def setPointForHessianEvaluations(self, x, gauss_newton_approx: bool = False) -> None:
        # Linear problem — only need to record the gauss_newton_approx flag
        # and delegate to misfit/prior so they can prepare.
        self.gauss_newton_approx = gauss_newton_approx
        self.misfit.setLinearizationPoint(x, gauss_newton_approx)
        self.prior.setLinearizationPoint(x[PARAMETER], gauss_newton_approx)

    # ---- incremental solves -----------------------------------------
    def solveFwdIncremental(self, sol, rhs) -> None:
        self.n_inc_solve += 1
        sol.zero()
        uold = self.M.createVecRight()
        u = self.M.createVecRight()
        Muold = self.M.createVecRight()
        my_rhs = self.M.createVecRight()
        for t in self.simulation_times[1:]:
            self.M_stab.mult(uold, Muold)
            r_t = rhs.view(t)
            my_rhs.array[:] = r_t.array[:]
            my_rhs.axpy(1.0, Muold)
            self.solver.solve(my_rhs, u)
            tgt = sol.view(t)
            tgt.petsc_vec.array[:] = u.array[:]
            tgt.scatter_forward()
            uold.array[:] = u.array[:]
        u.destroy(); uold.destroy(); Muold.destroy(); my_rhs.destroy()

    def solveAdjIncremental(self, sol, rhs) -> None:
        self.n_inc_solve += 1
        sol.zero()
        pold = self.M.createVecRight()
        p = self.M.createVecRight()
        Mpold = self.M.createVecRight()
        my_rhs = self.M.createVecRight()
        for t in reversed(self.simulation_times):
            self.Mt_stab.mult(pold, Mpold)
            r_t = rhs.view(t)
            my_rhs.array[:] = r_t.array[:]
            Mpold.axpy(1.0, my_rhs)
            self.solvert.solve(Mpold, p)
            tgt = sol.view(t)
            tgt.petsc_vec.array[:] = p.array[:]
            tgt.scatter_forward()
            pold.array[:] = p.array[:]
        p.destroy(); pold.destroy(); Mpold.destroy(); my_rhs.destroy()

    # ---- KKT blocks ---------------------------------------------------
    def applyC(self, dm, out) -> None:
        """out = -M_stab * dm at t=simulation_times[1], zero elsewhere."""
        out.zero()
        tmp = self.M.createVecRight()
        self.M_stab.mult(dm.petsc_vec, tmp)
        tmp.scale(-1.0)
        t1 = self.simulation_times[1]
        tgt = out.view(t1)
        tgt.petsc_vec.array[:] = tmp.array[:]
        tgt.scatter_forward()
        tmp.destroy()

    def applyCt(self, dp, out) -> None:
        """out = -Mt_stab * dp[simulation_times[1]]."""
        t1 = self.simulation_times[1]
        dp0 = dp.view(t1).petsc_vec
        tmp = self.M.createVecRight()
        self.Mt_stab.mult(dp0, tmp)
        out.petsc_vec.array[:] = -tmp.array[:]
        out.scatter_forward()
        tmp.destroy()

    def applyWuu(self, du, out) -> None:
        out.zero()
        self.misfit.apply_ij(STATE, STATE, du, out)

    def applyWum(self, dm, out) -> None:
        out.zero()

    def applyWmu(self, du, out) -> None:
        out.array[:] = 0.0

    def applyWmm(self, dm, out) -> None:
        out.array[:] = 0.0

    def applyR(self, dm, out) -> None:
        self.prior.R.mult(dm.petsc_vec, out.petsc_vec)

    def Rsolver(self):
        return self.prior.Rsolver

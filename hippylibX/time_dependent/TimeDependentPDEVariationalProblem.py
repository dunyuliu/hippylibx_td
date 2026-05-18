"""Time-dependent PDE variational problem for hippylibX (dolfinx).

Faithful port of hippylib's `TimeDependentPDEVariationalProblem`.
The varf handler must implement ``__call__(u, u_old, m, p, t) -> ufl.Form``
and expose attribute ``dt`` (timestep).
"""

from __future__ import annotations

import numpy as np
import ufl
import dolfinx as dlx
import dolfinx.fem.petsc
import petsc4py
from petsc4py import PETSc

from ..modeling.variables import STATE, PARAMETER, ADJOINT

from .timeDependentVector import TimeDependentVector


def _zero_vec(v: dlx.la.Vector) -> None:
    v.array[:] = 0.0
    v.scatter_forward()


class TimeDependentPDEVariationalProblem:
    def __init__(
        self,
        Vh,
        varf_handler,
        bc,
        bc0,
        u0,
        t_init: float,
        t_final: float,
        is_fwd_linear: bool = False,
    ):
        self.Vh = Vh
        self.varf = varf_handler

        self.fwd_bc = list(bc) if isinstance(bc, (list, tuple)) else [bc]
        self.adj_bc = list(bc0) if isinstance(bc0, (list, tuple)) else [bc0]

        self.mesh = self.Vh[STATE].mesh
        self.init_cond = u0  # a dolfinx.fem.Function
        self.t_init = t_init
        self.t_final = t_final
        self.dt = varf_handler.dt
        self.times = np.arange(
            self.t_init, self.t_final + 0.5 * self.dt, self.dt
        )

        self.linearize_x = None
        self.gauss_newton_approx = False

        self.solverA = None
        self.solverAadj = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear

        self.petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        self.n_calls = {
            "forward": 0,
            "adjoint": 0,
            "incremental_forward": 0,
            "incremental_adjoint": 0,
        }

    # ---- factories -----------------------------------------------------
    def generate_state(self) -> TimeDependentVector:
        u = TimeDependentVector(self.times)
        u.initialize(self.Vh[STATE])
        return u

    def generate_parameter(self) -> dlx.la.Vector:
        return dlx.la.vector(
            self.Vh[PARAMETER].dofmap.index_map,
            self.Vh[PARAMETER].dofmap.index_map_bs,
        )

    def generate_static_state(self) -> dlx.la.Vector:
        return dlx.la.vector(
            self.Vh[STATE].dofmap.index_map,
            self.Vh[STATE].dofmap.index_map_bs,
        )

    def generate_static_adjoint(self) -> dlx.la.Vector:
        return dlx.la.vector(
            self.Vh[ADJOINT].dofmap.index_map,
            self.Vh[ADJOINT].dofmap.index_map_bs,
        )

    # ---- helpers -------------------------------------------------------
    def _createLUSolver(self) -> petsc4py.PETSc.KSP:
        ksp = petsc4py.PETSc.KSP().create(self.mesh.comm)
        prefix = f"hippylibx_td_solve_{id(ksp)}"
        ksp.setOptionsPrefix(prefix)
        opts = petsc4py.PETSc.Options()
        opts.prefixPush(prefix)
        for k, v in self.petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        ksp.setFromOptions()
        return ksp

    @staticmethod
    def _copy_function_from_vec(fun: dlx.fem.Function, vec: dlx.la.Vector) -> None:
        fun.x.array[:] = vec.array[:]
        fun.x.scatter_forward()

    # ---- forward solve -------------------------------------------------
    def solveFwd(self, out: TimeDependentVector, x: list) -> None:
        out.zero()
        self.n_calls["forward"] += 1
        if self.solverA is None:
            self.solverA = self._createLUSolver()

        u_old = dlx.fem.Function(self.Vh[STATE])
        u_old.x.array[:] = self.init_cond.x.array[:]
        u_old.x.scatter_forward()
        # store initial condition into out at t_init
        out.store(u_old.x, self.times[0])

        m = dlx.fem.Function(self.Vh[PARAMETER])
        self._copy_function_from_vec(m, x[PARAMETER])

        if self.is_fwd_linear:
            du = ufl.TrialFunction(self.Vh[STATE])
            dp = ufl.TestFunction(self.Vh[ADJOINT])
            u_vec = self.generate_static_state()

            for t in self.times[1:]:
                res_form = self.varf(du, u_old, m, dp, t)
                A_form = dlx.fem.form(ufl.lhs(res_form))
                b_form = dlx.fem.form(ufl.rhs(res_form))

                A = dolfinx.fem.petsc.assemble_matrix(A_form, bcs=self.fwd_bc)
                A.assemble()
                b = dolfinx.fem.petsc.assemble_vector(b_form)
                dolfinx.fem.petsc.apply_lifting(b, [A_form], [self.fwd_bc])
                b.ghostUpdate(
                    PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
                )
                dolfinx.fem.petsc.set_bc(b, self.fwd_bc)

                self.solverA.setOperators(A)
                self.solverA.solve(b, u_vec.petsc_vec)
                u_vec.scatter_forward()

                out.store(u_vec, t)
                u_old.x.array[:] = u_vec.array[:]
                u_old.x.scatter_forward()

                A.destroy()
                b.destroy()
        else:
            # Nonlinear Newton inline (one step at a time)
            u = dlx.fem.Function(self.Vh[STATE])
            u.x.array[:] = u_old.x.array[:]
            dp = ufl.TestFunction(self.Vh[ADJOINT])
            du = ufl.TrialFunction(self.Vh[STATE])

            for t in self.times[1:]:
                res_form = self.varf(u, u_old, m, dp, t)
                jac_form = ufl.derivative(res_form, u, du)
                R = dlx.fem.form(res_form)
                J = dlx.fem.form(jac_form)

                # Simple Newton iteration
                max_iter = 25
                tol = 1e-10
                for k_it in range(max_iter):
                    A = dolfinx.fem.petsc.assemble_matrix(J, bcs=self.fwd_bc)
                    A.assemble()
                    b = dolfinx.fem.petsc.assemble_vector(R)
                    dolfinx.fem.petsc.apply_lifting(b, [J], [self.fwd_bc], x0=[u.x.petsc_vec], alpha=-1.0)
                    b.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
                    # set residual to zero where u is BC-prescribed
                    dolfinx.fem.petsc.set_bc(b, self.fwd_bc, u.x.petsc_vec, -1.0)
                    nrm = b.norm()
                    if nrm < tol:
                        A.destroy()
                        b.destroy()
                        break
                    self.solverA.setOperators(A)
                    du_vec = A.createVecLeft()
                    self.solverA.solve(b, du_vec)
                    u.x.petsc_vec.axpy(-1.0, du_vec)
                    u.x.scatter_forward()
                    A.destroy()
                    b.destroy()
                    du_vec.destroy()
                out.store(u.x, t)
                u_old.x.array[:] = u.x.array[:]
                u_old.x.scatter_forward()

    # ---- adjoint solve -------------------------------------------------
    def solveAdj(
        self,
        out: TimeDependentVector,
        x: list,
        adj_rhs: TimeDependentVector,
    ) -> None:
        """Solve the adjoint problem backwards in time.

        adj_rhs is the (negative misfit gradient w.r.t. state) at each time.
        """
        out.zero()
        self.n_calls["adjoint"] += 1
        if self.solverAadj is None:
            self.solverAadj = self._createLUSolver()

        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        p_old = dlx.fem.Function(self.Vh[ADJOINT])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        self._copy_function_from_vec(m, x[PARAMETER])

        du = ufl.TestFunction(self.Vh[STATE])
        dp = ufl.TrialFunction(self.Vh[ADJOINT])

        p_vec = self.generate_static_adjoint()

        for t in reversed(self.times[1:]):
            x[STATE].retrieve(u.x, t)

            form = self.varf(u, u_old, m, p, t)
            adj_form = ufl.derivative(ufl.derivative(form, u, du), p, dp)
            b_form_ufl = -ufl.derivative(ufl.derivative(form, u_old, du), p, p_old)

            A_form = dlx.fem.form(adj_form)
            B_form = dlx.fem.form(b_form_ufl)

            Aadj = dolfinx.fem.petsc.assemble_matrix(A_form, bcs=self.adj_bc)
            Aadj.assemble()
            b = dolfinx.fem.petsc.assemble_vector(B_form)
            dolfinx.fem.petsc.apply_lifting(b, [A_form], [self.adj_bc])
            b.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b, self.adj_bc)

            # add the misfit-driven adjoint rhs (already lifted)
            rhs_t = adj_rhs.view(t)
            b.axpy(1.0, rhs_t.petsc_vec)

            self.solverAadj.setOperators(Aadj)
            self.solverAadj.solve(b, p_vec.petsc_vec)
            p_vec.scatter_forward()
            out.store(p_vec, t)

            p_old.x.array[:] = p_vec.array[:]
            p_old.x.scatter_forward()

            Aadj.destroy()
            b.destroy()

    # ---- gradient w.r.t. parameter ------------------------------------
    def evalGradientParameter(self, x: list, out: dlx.la.Vector) -> None:
        out.array[:] = 0.0

        dm = ufl.TestFunction(self.Vh[PARAMETER])
        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        self._copy_function_from_vec(m, x[PARAMETER])

        x[STATE].retrieve(u_old.x, self.times[0])

        out_t = self.generate_parameter()

        for t in self.times[1:]:
            x[STATE].retrieve(u.x, t)
            x[ADJOINT].retrieve(p.x, t)

            form = self.varf(u, u_old, m, p, t)
            grad_form = dlx.fem.form(ufl.derivative(form, m, dm))

            out_t.array[:] = 0.0
            dolfinx.fem.petsc.assemble_vector(out_t.petsc_vec, grad_form)
            out_t.petsc_vec.ghostUpdate(
                PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
            )
            out.array[:] += out_t.array[:]

            u_old.x.array[:] = u.x.array[:]
            u_old.x.scatter_forward()

        out.scatter_forward()

    # ---- linearization point ------------------------------------------
    def setLinearizationPoint(self, x: list, gauss_newton_approx: bool = False) -> None:
        self.linearize_x = x
        self.gauss_newton_approx = gauss_newton_approx
        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()

    # ---- incremental forward ------------------------------------------
    def _solveIncrementalFwd(
        self, out: TimeDependentVector, rhs: TimeDependentVector
    ) -> None:
        out.zero()
        self.n_calls["incremental_forward"] += 1

        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        uhat_old = dlx.fem.Function(self.Vh[STATE])
        self._copy_function_from_vec(m, self.linearize_x[PARAMETER])

        dp = ufl.TestFunction(self.Vh[ADJOINT])
        du = ufl.TrialFunction(self.Vh[STATE])

        uhat_vec = self.generate_static_state()

        self.linearize_x[STATE].retrieve(u_old.x, self.times[0])

        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.x, t)
            form = self.varf(u, u_old, m, dp, t)
            Ainc_form_ufl = ufl.derivative(form, u, du)
            binc_form_ufl = -ufl.derivative(form, u_old, uhat_old)

            A_form = dlx.fem.form(Ainc_form_ufl)
            B_form = dlx.fem.form(binc_form_ufl)

            Ainc = dolfinx.fem.petsc.assemble_matrix(A_form, bcs=self.adj_bc)
            Ainc.assemble()
            b = dolfinx.fem.petsc.assemble_vector(B_form)
            dolfinx.fem.petsc.apply_lifting(b, [A_form], [self.adj_bc])
            b.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b, self.adj_bc)

            # incremental rhs from caller
            rhs_t = rhs.view(t)
            b.axpy(1.0, rhs_t.petsc_vec)

            self.solver_fwd_inc.setOperators(Ainc)
            self.solver_fwd_inc.solve(b, uhat_vec.petsc_vec)
            uhat_vec.scatter_forward()

            uhat_old.x.array[:] = uhat_vec.array[:]
            uhat_old.x.scatter_forward()

            # advance the linearization u_old to time t
            self.linearize_x[STATE].retrieve(u_old.x, t)

            out.store(uhat_vec, t)

            Ainc.destroy()
            b.destroy()

    # ---- incremental adjoint ------------------------------------------
    def _solveIncrementalAdj(
        self, out: TimeDependentVector, rhs: TimeDependentVector
    ) -> None:
        out.zero()
        self.n_calls["incremental_adjoint"] += 1

        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        phat_old = dlx.fem.Function(self.Vh[ADJOINT])
        self._copy_function_from_vec(m, self.linearize_x[PARAMETER])

        dp = ufl.TrialFunction(self.Vh[ADJOINT])
        du = ufl.TestFunction(self.Vh[STATE])
        du_old = ufl.TestFunction(self.Vh[STATE])

        phat_vec = self.generate_static_adjoint()

        times_rev = list(reversed(self.times[1:]))
        for idx, t in enumerate(times_rev):
            self.linearize_x[STATE].retrieve(u.x, t)
            # u_old at time t corresponds to the previous frame in physical time
            t_prev = self.times[0] if t == self.times[1] else self.times[
                list(self.times).index(t) - 1
            ]
            self.linearize_x[STATE].retrieve(u_old.x, t_prev)
            self.linearize_x[ADJOINT].retrieve(p.x, t)

            form = self.varf(u, u_old, m, p, t)
            A_adj_form_ufl = ufl.derivative(ufl.derivative(form, u, du), p, dp)
            b_adj_form_ufl = -ufl.derivative(
                ufl.derivative(form, u_old, du_old), p, phat_old
            )

            A_form = dlx.fem.form(A_adj_form_ufl)
            B_form = dlx.fem.form(b_adj_form_ufl)

            Aadj = dolfinx.fem.petsc.assemble_matrix(A_form, bcs=self.adj_bc)
            Aadj.assemble()
            b = dolfinx.fem.petsc.assemble_vector(B_form)
            dolfinx.fem.petsc.apply_lifting(b, [A_form], [self.adj_bc])
            b.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b, self.adj_bc)

            rhs_t = rhs.view(t)
            b.axpy(1.0, rhs_t.petsc_vec)

            self.solver_adj_inc.setOperators(Aadj)
            self.solver_adj_inc.solve(b, phat_vec.petsc_vec)
            phat_vec.scatter_forward()

            phat_old.x.array[:] = phat_vec.array[:]
            phat_old.x.scatter_forward()

            out.store(phat_vec, t)

            Aadj.destroy()
            b.destroy()

    def solveIncremental(
        self, out: TimeDependentVector, rhs: TimeDependentVector, is_adj: bool
    ) -> None:
        if is_adj:
            self._solveIncrementalAdj(out, rhs)
        else:
            self._solveIncrementalFwd(out, rhs)

    # ---- second-derivative blocks (KKT) -------------------------------
    def _bc_zero_rows_on_petsc(self, vec: petsc4py.PETSc.Vec) -> None:
        """Zero adjoint-BC dofs in a PETSc vector."""
        dolfinx.fem.petsc.set_bc(vec, self.adj_bc, alpha=0.0)

    def applyC(self, dm: dlx.la.Vector, out: TimeDependentVector) -> None:
        """out_t = (d^2 F / dp dm)[dm]  for each t."""
        out.zero()
        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        dm_fun = dlx.fem.Function(self.Vh[PARAMETER])

        self._copy_function_from_vec(m, self.linearize_x[PARAMETER])
        dm_fun.x.array[:] = dm.array[:]
        dm_fun.x.scatter_forward()

        dp = ufl.TestFunction(self.Vh[ADJOINT])

        self.linearize_x[STATE].retrieve(u_old.x, self.times[0])

        out_t = self.generate_static_adjoint()
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.x, t)
            self.linearize_x[ADJOINT].retrieve(p.x, t)
            form = self.varf(u, u_old, m, p, t)
            cvarf = ufl.derivative(ufl.derivative(form, p, dp), m, dm_fun)

            out_t.array[:] = 0.0
            dolfinx.fem.petsc.assemble_vector(out_t.petsc_vec, dlx.fem.form(cvarf))
            out_t.petsc_vec.ghostUpdate(
                PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
            )
            self._bc_zero_rows_on_petsc(out_t.petsc_vec)
            out.store(out_t, t)

            self.linearize_x[STATE].retrieve(u_old.x, t)

    def applyCt(self, dp_tdv: TimeDependentVector, out: dlx.la.Vector) -> None:
        out.array[:] = 0.0

        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        dp_fun = dlx.fem.Function(self.Vh[ADJOINT])

        self._copy_function_from_vec(m, self.linearize_x[PARAMETER])

        dm = ufl.TestFunction(self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.x, self.times[0])

        out_t = self.generate_parameter()
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.x, t)
            self.linearize_x[ADJOINT].retrieve(p.x, t)
            dp_tdv.retrieve(dp_fun.x, t)

            form = self.varf(u, u_old, m, p, t)
            cvarf_adj = ufl.derivative(ufl.derivative(form, p, dp_fun), m, dm)
            out_t.array[:] = 0.0
            dolfinx.fem.petsc.assemble_vector(out_t.petsc_vec, dlx.fem.form(cvarf_adj))
            out_t.petsc_vec.ghostUpdate(
                PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
            )
            out.array[:] += out_t.array[:]

            self.linearize_x[STATE].retrieve(u_old.x, t)

        out.scatter_forward()

    def applyWuu(self, du: TimeDependentVector, out: TimeDependentVector) -> None:
        out.zero()
        if self.gauss_newton_approx:
            return

        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        du_fun = dlx.fem.Function(self.Vh[STATE])
        du_old = dlx.fem.Function(self.Vh[STATE])
        self._copy_function_from_vec(m, self.linearize_x[PARAMETER])

        du_test = ufl.TestFunction(self.Vh[STATE])
        du_old_test = ufl.TestFunction(self.Vh[STATE])

        self.linearize_x[STATE].retrieve(u_old.x, self.times[0])
        du.retrieve(du_old.x, self.times[0])

        out_t = self.generate_static_state()
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.x, t)
            self.linearize_x[ADJOINT].retrieve(p.x, t)
            du.retrieve(du_fun.x, t)

            form = self.varf(u, u_old, m, p, t)
            varf = ufl.derivative(
                ufl.derivative(form, u, du_fun), u, du_test
            ) + ufl.derivative(
                ufl.derivative(form, u_old, du_old), u_old, du_old_test
            )

            out_t.array[:] = 0.0
            dolfinx.fem.petsc.assemble_vector(out_t.petsc_vec, dlx.fem.form(varf))
            out_t.petsc_vec.ghostUpdate(
                PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
            )
            self._bc_zero_rows_on_petsc(out_t.petsc_vec)

            self.linearize_x[STATE].retrieve(u_old.x, t)
            du.retrieve(du_old.x, t)

            out.store(out_t, t)

    def applyWum(self, dm: dlx.la.Vector, out: TimeDependentVector) -> None:
        out.zero()
        if self.gauss_newton_approx:
            return

        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        dm_fun = dlx.fem.Function(self.Vh[PARAMETER])

        self._copy_function_from_vec(m, self.linearize_x[PARAMETER])
        dm_fun.x.array[:] = dm.array[:]
        dm_fun.x.scatter_forward()

        du_test = ufl.TestFunction(self.Vh[STATE])
        du_old_test = ufl.TestFunction(self.Vh[STATE])

        self.linearize_x[STATE].retrieve(u_old.x, self.times[0])

        out_t = self.generate_static_state()
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.x, t)
            self.linearize_x[ADJOINT].retrieve(p.x, t)

            form = self.varf(u, u_old, m, p, t)
            varf = ufl.derivative(
                ufl.derivative(form, m, dm_fun), u, du_test
            ) + ufl.derivative(
                ufl.derivative(form, m, dm_fun), u_old, du_old_test
            )

            out_t.array[:] = 0.0
            dolfinx.fem.petsc.assemble_vector(out_t.petsc_vec, dlx.fem.form(varf))
            out_t.petsc_vec.ghostUpdate(
                PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
            )
            self._bc_zero_rows_on_petsc(out_t.petsc_vec)

            self.linearize_x[STATE].retrieve(u_old.x, t)
            out.store(out_t, t)

    def applyWmu(self, du: TimeDependentVector, out: dlx.la.Vector) -> None:
        out.array[:] = 0.0
        if self.gauss_newton_approx:
            return

        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        du_fun = dlx.fem.Function(self.Vh[STATE])
        du_old = dlx.fem.Function(self.Vh[STATE])
        self._copy_function_from_vec(m, self.linearize_x[PARAMETER])

        dm_test = ufl.TestFunction(self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.x, self.times[0])
        du.retrieve(du_old.x, self.times[0])

        out_t = self.generate_parameter()
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.x, t)
            self.linearize_x[ADJOINT].retrieve(p.x, t)
            du.retrieve(du_fun.x, t)

            form = self.varf(u, u_old, m, p, t)
            varf = ufl.derivative(
                ufl.derivative(form, u, du_fun), m, dm_test
            ) + ufl.derivative(
                ufl.derivative(form, u_old, du_old), m, dm_test
            )

            out_t.array[:] = 0.0
            dolfinx.fem.petsc.assemble_vector(out_t.petsc_vec, dlx.fem.form(varf))
            out_t.petsc_vec.ghostUpdate(
                PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
            )
            out.array[:] += out_t.array[:]

            self.linearize_x[STATE].retrieve(u_old.x, t)
            du.retrieve(du_old.x, t)

        out.scatter_forward()

    def applyWmm(self, dm: dlx.la.Vector, out: dlx.la.Vector) -> None:
        out.array[:] = 0.0
        if self.gauss_newton_approx:
            return

        u = dlx.fem.Function(self.Vh[STATE])
        u_old = dlx.fem.Function(self.Vh[STATE])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        m = dlx.fem.Function(self.Vh[PARAMETER])
        dm_fun = dlx.fem.Function(self.Vh[PARAMETER])

        self._copy_function_from_vec(m, self.linearize_x[PARAMETER])
        dm_fun.x.array[:] = dm.array[:]
        dm_fun.x.scatter_forward()

        dm_test = ufl.TestFunction(self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.x, self.times[0])

        out_t = self.generate_parameter()
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.x, t)
            self.linearize_x[ADJOINT].retrieve(p.x, t)

            form = self.varf(u, u_old, m, p, t)
            varf = ufl.derivative(ufl.derivative(form, m, dm_fun), m, dm_test)
            out_t.array[:] = 0.0
            dolfinx.fem.petsc.assemble_vector(out_t.petsc_vec, dlx.fem.form(varf))
            out_t.petsc_vec.ghostUpdate(
                PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE
            )
            out.array[:] += out_t.array[:]

            self.linearize_x[STATE].retrieve(u_old.x, t)

        out.scatter_forward()

    def apply_ij(self, i: int, j: int, dir, out) -> None:
        KKT = {
            (STATE, STATE): self.applyWuu,
            (STATE, PARAMETER): self.applyWum,
            (PARAMETER, STATE): self.applyWmu,
            (PARAMETER, PARAMETER): self.applyWmm,
            (ADJOINT, PARAMETER): self.applyC,
            (PARAMETER, ADJOINT): self.applyCt,
        }
        KKT[(i, j)](dir, out)

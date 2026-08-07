# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import petsc4py.PETSc
from .variables import STATE, PARAMETER, ADJOINT
import dolfinx as dlx
import petsc4py


# decorator for functions in classes that are not used -> may not be needed in the final
# version of X
def unused_function(func):
    return None


class ReducedHessian:
    """
    This class implements matrix free application of the reduced Hessian operator.
    The constructor takes the following parameters:

    - :code:`model`:               the object which contains the description of the problem.
    - :code:`misfit_only`:         a boolean flag that describes whenever the full Hessian or only the misfit component of the Hessian is used.

    Type :code:`help(modelTemplate)` for more information on which methods model should implement.
    """

    def __init__(self, model, misfit_only=False):
        """
        Construct the reduced Hessian Operator
        """
        self.model = model
        self.gauss_newton_approx = self.model.gauss_newton_approx
        self.misfit_only = misfit_only
        self.ncalls = 0

        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(ADJOINT)
        self.rhs_adj2 = model.generate_vector(ADJOINT)
        self.uhat = model.generate_vector(STATE)
        self.phat = model.generate_vector(ADJOINT)
        self.yhelp = model.generate_vector(PARAMETER)

        self.petsc_wrapper = petsc4py.PETSc.Mat().createPython(
            self.model.prior.M.getSizes(), comm=self.model.prior.Vh.mesh.comm
        )
        self.petsc_wrapper.setPythonContext(self)
        self.petsc_wrapper.setUp()

    def destroy(self) -> None:
        """Free the work vectors and the Python Mat wrapper held by this Hessian.

        B5: `ReducedSpaceNewtonCG` builds a fresh ReducedHessian on EVERY Newton
        iteration (NewtonCG.py:277). For a time-dependent problem `rhs_fwd`,
        `rhs_adj`, `rhs_adj2`, `uhat` and `phat` are each a TimeDependentVector
        holding `nsteps` PETSc Vecs, so without this each iteration leaked
        ~5*nsteps Vecs -- measured at 500 Vecs and ~0.75 GB per iteration in 2D
        with nsteps=100, which made a long 3D run OOM.

        Call only once the CG solve using this operator has finished; the caller
        owns the object and nothing else aliases these vectors.
        """
        for name in ("rhs_fwd", "rhs_adj", "rhs_adj2", "uhat", "phat", "yhelp"):
            v = getattr(self, name, None)
            if v is None:
                continue
            d = getattr(v, "destroy", None)          # TimeDependentVector
            if callable(d):
                try:
                    d()
                except Exception:
                    pass
            else:                                     # plain la.Vector
                try:
                    v.petsc_vec.destroy()
                except Exception:
                    pass
            setattr(self, name, None)
        w = getattr(self, "petsc_wrapper", None)
        if w is not None:
            try:
                w.destroy()
            except Exception:
                pass
            self.petsc_wrapper = None

    def __del__(self):
        # `destroy()` may already have run and set this to None (it is called
        # explicitly at the end of each Newton iteration); __del__ still fires
        # later at GC time, so it must tolerate an already-freed wrapper.
        w = getattr(self, "petsc_wrapper", None)
        if w is not None:
            try:
                w.destroy()
            except Exception:
                pass

    @property
    def mat(self) -> petsc4py.PETSc.Mat:
        return self.petsc_wrapper

    def mult(self, mat, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        """
        Apply the reduced Hessian (or the Gauss-Newton approximation) to the vector :code:`x`. Return the result in :code:`y`.
        """

        x_dlx = self.model.generate_vector(PARAMETER)
        y_dlx = self.model.generate_vector(PARAMETER)

        x_dlx.petsc_vec.axpy(1.0, x)

        if self.gauss_newton_approx:
            self.GNHessian(x_dlx, y_dlx)
        else:
            self.TrueHessian(x_dlx, y_dlx)

        y.axpby(1.0, 0.0, y_dlx.petsc_vec)  # y = 1. y_dlx + 0.*y

        self.ncalls += 1

    def GNHessian(self, x: dlx.la.Vector, y: dlx.la.Vector) -> None:
        """
        Apply the Gauss-Newton approximation of the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyCt(self.phat, y)

        if not self.misfit_only:
            self.model.applyR(x, self.yhelp)
            y.array[:] += self.yhelp.array

    def TrueHessian(self, x: dlx.la.Vector, y: dlx.la.Vector) -> None:
        """
        Apply the the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWum(x, self.rhs_adj2)
        self.rhs_adj.array[:] = self.rhs_adj.array + (-1.0) * self.rhs_adj2.array
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyWmm(x, y)
        self.model.applyCt(self.phat, self.yhelp)
        y.array[:] += self.yhelp.array
        self.model.applyWmu(self.uhat, self.yhelp)
        y.array[:] -= self.yhelp.array
        if not self.misfit_only:
            self.model.applyR(x, self.yhelp)
            y.array[:] += self.yhelp.array

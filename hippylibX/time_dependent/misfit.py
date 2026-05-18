# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Time-dependent misfit components.

- :class:`ContinuousStateObservation` — single-time L²(X) misfit
  (mass-matrix based, mirrors the legacy hippylib class).
- :class:`SpaceTimePointwiseStateObservation` — pointwise sensors at a list
  of observation times.
- :class:`MisfitTD` — time-summed wrapper over per-time misfits.
"""

from __future__ import annotations

import numpy as np
import ufl
import dolfinx as dlx
import dolfinx.fem.petsc

from ..modeling.variables import STATE, PARAMETER
from ..modeling.pointwiseInterpolationMatrix import pointwiseInterpolationMatrix
from .timeDependentVector import TimeDependentVector


class ContinuousStateObservation:
    """L^2(X) misfit at a single (frozen) time:
        1/(2*sigma^2) * ||u - d||_{L^2(X)}^2
    where X is the integration domain induced by ``dX``. ``bcs`` are the
    homogeneous Dirichlet BCs whose dofs are zeroed in the mass matrix W.
    """

    def __init__(self, Vh, dX, bcs, data=None, noise_variance: float | None = None):
        self.Vh = Vh
        u = ufl.TrialFunction(Vh)
        v = ufl.TestFunction(Vh)
        W_form = dlx.fem.form(ufl.inner(u, v) * dX)
        self.W = dolfinx.fem.petsc.assemble_matrix(W_form, bcs=[])
        self.W.assemble()

        if bcs is None:
            bcs = []
        if not isinstance(bcs, (list, tuple)):
            bcs = [bcs]
        self.bcs = list(bcs)

        if self.bcs:
            # Zero rows of W for BC dofs.
            for bc in self.bcs:
                dof_indices = bc._cpp_object.dof_indices()[0]
                self.W.zeroRows(dof_indices, diag=0.0)
            self.W.assemble()
            # Zero corresponding columns (transpose, zero rows, transpose back).
            self.W.transpose()
            for bc in self.bcs:
                dof_indices = bc._cpp_object.dof_indices()[0]
                self.W.zeroRows(dof_indices, diag=0.0)
            self.W.assemble()
            self.W.transpose()

        if data is None:
            self.d = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
        else:
            self.d = data

        self.noise_variance = noise_variance

    def __del__(self):
        if getattr(self, "W", None) is not None:
            try:
                self.W.destroy()
            except Exception:
                pass

    def _check_nv(self):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        if self.noise_variance == 0:
            raise ZeroDivisionError(
                "Noise Variance must not be 0.0 (set to 1.0 for deterministic problems)"
            )

    def cost(self, x: list) -> float:
        self._check_nv()
        r = self.d.petsc_vec.copy()
        r.axpy(-1.0, x[STATE].petsc_vec)
        Wr = self.W.createVecLeft()
        self.W.mult(r, Wr)
        c = r.dot(Wr) / (2.0 * self.noise_variance)
        r.destroy()
        Wr.destroy()
        return float(c)

    def grad(self, i: int, x: list, out: dlx.la.Vector) -> None:
        self._check_nv()
        if i == STATE:
            r = x[STATE].petsc_vec.copy()
            r.axpy(-1.0, self.d.petsc_vec)
            self.W.mult(r, out.petsc_vec)
            out.petsc_vec.scale(1.0 / self.noise_variance)
            r.destroy()
        elif i == PARAMETER:
            out.array[:] = 0.0
        else:
            raise IndexError(i)

    def setLinearizationPoint(self, x, gauss_newton_approx=False) -> None:
        return  # quadratic, nothing to do

    def apply_ij(self, i: int, j: int, dir, out) -> None:
        self._check_nv()
        if i == STATE and j == STATE:
            self.W.mult(dir.petsc_vec, out.petsc_vec)
            out.petsc_vec.scale(1.0 / self.noise_variance)
        else:
            out.array[:] = 0.0


class SpaceTimePointwiseStateObservation:
    """Space-time pointwise misfit.

    Observes the state at a fixed list of sensor coordinates at each of the
    ``observation_times`` and compares against time-keyed data ``d`` with
    Gaussian noise of variance ``noise_variance``.

    Misfit:
        ``1/(2*sigma^2) * sum_t || B u(t) - d(t) ||^2``

    where ``B`` is the pointwise interpolation matrix mapping a state in
    ``Vh`` to its values at the sensor coordinates.

    Mirrors hippylib's `SpaceTimePointwiseStateObservation`.
    """

    def __init__(self, Vh, observation_times, targets, d=None, noise_variance=None):
        self.Vh = Vh
        self.observation_times = list(observation_times)
        # targets: (ntargets, ndim) array of sensor coordinates
        self.targets = np.asarray(targets, dtype=np.float64)
        self.B = pointwiseInterpolationMatrix(Vh, self.targets)

        if d is None:
            # build a TDV-like storage compatible with B (row-side)
            self._d_petsc = [self.B.createVecLeft() for _ in self.observation_times]
            self.d = _PointwiseTDStorage(self._d_petsc, self.observation_times)
        else:
            self.d = d

        self.noise_variance = noise_variance

        # scratch
        self._Bu = self.B.createVecLeft()

    def __del__(self):
        for attr in ("B", "_Bu"):
            v = getattr(self, attr, None)
            if v is not None:
                try:
                    v.destroy()
                except Exception:
                    pass
        for v in getattr(self, "_d_petsc", []) or []:
            try:
                v.destroy()
            except Exception:
                pass

    def _check_nv(self):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        if self.noise_variance == 0:
            raise ZeroDivisionError(
                "Noise Variance must not be 0.0 (set 1.0 for deterministic problems)"
            )

    def cost(self, x: list) -> float:
        self._check_nv()
        c = 0.0
        for t in self.observation_times:
            u_t = x[STATE].view(t)
            self.B.mult(u_t.petsc_vec, self._Bu)
            self._Bu.axpy(-1.0, self.d.view(t))
            c += float(self._Bu.dot(self._Bu))
        return c / (2.0 * self.noise_variance)

    def grad(self, i: int, x: list, out) -> None:
        self._check_nv()
        if i == STATE:
            out.zero()
            for t in self.observation_times:
                u_t = x[STATE].view(t)
                self.B.mult(u_t.petsc_vec, self._Bu)
                self._Bu.axpy(-1.0, self.d.view(t))
                self._Bu.scale(1.0 / self.noise_variance)
                self.B.multTranspose(self._Bu, out.view(t).petsc_vec)
                out.view(t).scatter_forward()
        elif i == PARAMETER:
            out.array[:] = 0.0
        else:
            raise IndexError(i)

    def setLinearizationPoint(self, x, gauss_newton_approx=False) -> None:
        return

    def apply_ij(self, i: int, j: int, direction, out) -> None:
        self._check_nv()
        if i == STATE and j == STATE:
            out.zero()
            for t in self.observation_times:
                self.B.mult(direction.view(t).petsc_vec, self._Bu)
                self._Bu.scale(1.0 / self.noise_variance)
                self.B.multTranspose(self._Bu, out.view(t).petsc_vec)
                out.view(t).scatter_forward()
        else:
            if hasattr(out, "zero"):
                out.zero()
            else:
                out.array[:] = 0.0


class _PointwiseTDStorage:
    """Minimal TDV-like wrapper around per-time PETSc.Vec snapshots living
    in B's row space (sensor space, not the FE state space). Only supports
    ``view(t)``."""

    def __init__(self, vecs, times, tol=1e-10):
        self._vecs = list(vecs)
        self._times = list(times)
        self.tol = tol

    def _index(self, t):
        for i, ti in enumerate(self._times):
            if abs(ti - t) < self.tol:
                return i
        raise KeyError(f"Time {t} not in {self._times}")

    def view(self, t):
        return self._vecs[self._index(t)]

    def set(self, t, vec):
        self._vecs[self._index(t)].axpby(1.0, 0.0, vec)


class MisfitTD:
    """Time-summed misfit: ``cost(x) = sum_t misfit_t.cost(u(t), m)``.

    State is expected to be a :class:`TimeDependentVector`; parameter is a
    static :class:`dolfinx.la.Vector`.
    """

    def __init__(self, misfits, sim_times):
        self.misfits = misfits
        self.sim_times = list(sim_times)

    def cost(self, x: list) -> float:
        c = 0.0
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            c += misfit.cost([x[STATE].view(t), x[PARAMETER], None])
        return c

    def grad(self, i: int, x: list, out) -> None:
        if i == STATE:
            out.zero()
            for itime, misfit in enumerate(self.misfits):
                t = self.sim_times[itime]
                misfit.grad(i, [x[STATE].view(t), x[PARAMETER], None], out.view(t))
        elif i == PARAMETER:
            out.array[:] = 0.0
            out_t = dlx.la.vector(x[PARAMETER].index_map, x[PARAMETER].block_size)
            for itime, misfit in enumerate(self.misfits):
                t = self.sim_times[itime]
                out_t.array[:] = 0.0
                misfit.grad(i, [x[STATE].view(t), x[PARAMETER], None], out_t)
                out.array[:] += out_t.array[:]
            out.scatter_forward()
        else:
            raise IndexError(i)

    def setLinearizationPoint(self, x: list, gauss_newton_approx=False) -> None:
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.setLinearizationPoint(
                [x[STATE].view(t), x[PARAMETER], None], gauss_newton_approx
            )

    def _apply_STATE_STATE(self, direction, out) -> None:
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.apply_ij(STATE, STATE, direction.view(t), out.view(t))

    def _apply_STATE_PARAMETER(self, direction, out) -> None:
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.apply_ij(STATE, PARAMETER, direction, out.view(t))

    def _apply_PARAMETER_STATE(self, direction, out) -> None:
        out.array[:] = 0.0
        out_t = dlx.la.vector(out.index_map, out.block_size)
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            out_t.array[:] = 0.0
            misfit.apply_ij(PARAMETER, STATE, direction.view(t), out_t)
            out.array[:] += out_t.array[:]
        out.scatter_forward()

    def _apply_PARAMETER_PARAMETER(self, direction, out) -> None:
        out.array[:] = 0.0
        out_t = dlx.la.vector(out.index_map, out.block_size)
        for misfit in self.misfits:
            out_t.array[:] = 0.0
            misfit.apply_ij(PARAMETER, PARAMETER, direction, out_t)
            out.array[:] += out_t.array[:]
        out.scatter_forward()

    def apply_ij(self, i: int, j: int, direction, out) -> None:
        if i == STATE and j == STATE:
            out.zero()
            self._apply_STATE_STATE(direction, out)
        elif i == STATE and j == PARAMETER:
            out.zero()
            self._apply_STATE_PARAMETER(direction, out)
        elif i == PARAMETER and j == STATE:
            self._apply_PARAMETER_STATE(direction, out)
        elif i == PARAMETER and j == PARAMETER:
            self._apply_PARAMETER_PARAMETER(direction, out)
        else:
            raise IndexError((i, j))

# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Time-dependent vector for hippylibX (dolfinx).

Faithful port of hippylib's :class:`TimeDependentVector`. Stores one
:class:`dolfinx.la.Vector` snapshot per time frame, keyed by absolute time.

Architectural note — the ``.array[:]`` proxy
--------------------------------------------
:class:`TimeDependentVector` exposes a :class:`_TDVArrayProxy` via the
``.array`` property. The proxy implements ``__setitem__`` for the full-
slice form ``a[:] = …`` so that the upstream :class:`hippylibX.Model`,
:class:`hippylibX.ReducedHessian`, and :func:`hippylibX.modelVerify` can
operate on time-dependent state/adjoint vectors **without modification**.

In particular, the upstream code paths
``out.array[:] *= -1.0``, ``out.array[:] += tmp.array``, and
``out.array[:] = src.array`` all dispatch correctly per snapshot when
``out`` is a :class:`TimeDependentVector`. This is what allowed the
TD port to drop the early `ModelTD` / `ReducedHessianTD` / `modelVerifyTD`
shims that an earlier version of this code carried.

Reads via the proxy return a *copy* (concatenation), so arithmetic such as
``a.array + 0.5 * b.array`` yields a plain :class:`numpy.ndarray`.
Writes via ``__setitem__`` propagate back to the underlying snapshots.
Partial-slice assignment (``a[i:j] = …``) is deliberately unsupported and
raises :class:`NotImplementedError`.
"""

from __future__ import annotations

import numpy as np
import dolfinx as dlx
import petsc4py
from petsc4py import PETSc


# ---------------------------------------------------------------------------
# Communicator-sharing snapshot storage
# ---------------------------------------------------------------------------
# DOLFINx builds a *fresh* neighbour communicator (MPI_Dist_graph_create_adjacent)
# for every `dolfinx.la.Vector` at construction.  A TimeDependentVector holds one
# snapshot per time frame (nt+1 vectors), and an inversion keeps several
# TimeDependentVectors alive at once, so nt × (#TDVs) neighbour comms quickly
# exhaust MPICH's hard 2048-context limit under MPI (np>1) — the run aborts with
# "Too many communicators".  PETSc's Vec.duplicate(), by contrast, *shares* the
# parent's communicator (PetscCommDuplicate ref-counting), so we back every
# snapshot with a ghosted PETSc Vec duplicated from a single per-TDV template.
# This yields exactly ONE neighbour comm per TimeDependentVector regardless of nt.
#
# `_DupVector` mirrors the small slice of the `dolfinx.la.Vector` API that the
# TD code touches: `.array` (full [owned|ghost] numpy view, read+write),
# `.scatter_forward()`, and `.petsc_vec`.  Numerics are bit-identical to a real
# la.Vector (verified: full-array equality after scatter, identical petsc dot).


class _LocalArrayProxy:
    """numpy-like view over a ghosted PETSc Vec's full [owned|ghost] local form.

    Reproduces `dolfinx.la.Vector.array` semantics (length = size_local +
    num_ghosts) so that ``v.array[:] = ...``, ``np.asarray(v.array)``,
    ``v.array.shape[0]`` and in-place ops behave exactly as before.
    """

    __slots__ = ("_pv",)

    def __init__(self, pv: "PETSc.Vec"):
        self._pv = pv

    def __array__(self, dtype=None):
        with self._pv.localForm() as lf:
            a = np.array(lf.array, copy=True)
        return a if dtype is None else a.astype(dtype)

    @property
    def shape(self):
        with self._pv.localForm() as lf:
            return lf.array.shape

    def __len__(self) -> int:
        with self._pv.localForm() as lf:
            return int(lf.array.shape[0])

    def __getitem__(self, key):
        with self._pv.localForm() as lf:
            return np.array(lf.array[key], copy=True)

    def __setitem__(self, key, value) -> None:
        with self._pv.localForm() as lf:
            lf.array[key] = value


class _DupVector:
    """la.Vector-compatible snapshot backed by a comm-sharing ghosted PETSc Vec."""

    __slots__ = ("petsc_vec", "_arr")

    def __init__(self, template_pv: "PETSc.Vec"):
        self.petsc_vec = template_pv.duplicate()   # shares neighbour comm
        self.petsc_vec.set(0.0)
        self._arr = _LocalArrayProxy(self.petsc_vec)

    @property
    def array(self) -> _LocalArrayProxy:
        return self._arr

    def scatter_forward(self) -> None:
        self.petsc_vec.ghostUpdate(
            PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
        )


class _TDVArrayProxy:
    """A proxy that lets ``tdv.array[:] = something`` work transparently.

    ``hippylibX``'s `ReducedSpaceNewtonCG` and `ReducedHessian` perform
    in-place numpy assignment on the STATE/ADJOINT vector's `.array`. For
    time-dependent vectors we forward those slice-assignments to all
    snapshots in the right order.
    """

    def __init__(self, tdv: "TimeDependentVector"):
        self.tdv = tdv

    def __array__(self, dtype=None):
        out = np.concatenate([np.asarray(v.array) for v in self.tdv.data])
        return out if dtype is None else out.astype(dtype)

    def __len__(self) -> int:
        return sum(v.array.shape[0] for v in self.tdv.data)

    def __getitem__(self, key):
        return np.asarray(self)[key]

    def __setitem__(self, key, value) -> None:
        if not (isinstance(key, slice) and key == slice(None, None, None)):
            raise NotImplementedError(
                "Only full-slice assignment array[:] = ... is supported on a TDV"
            )
        if isinstance(value, _TDVArrayProxy):
            if value.tdv.nsteps != self.tdv.nsteps:
                raise ValueError(
                    f"nsteps mismatch on assignment: "
                    f"dst={self.tdv.nsteps} src={value.tdv.nsteps}"
                )
            for i in range(self.tdv.nsteps):
                self.tdv.data[i].array[:] = value.tdv.data[i].array[:]
                self.tdv.data[i].scatter_forward()
            return
        arr = np.asarray(value).ravel()
        offs = 0
        for v in self.tdv.data:
            n = v.array.shape[0]
            v.array[:] = arr[offs : offs + n]
            v.scatter_forward()
            offs += n

    # arithmetic returns concatenated ndarrays so that
    # ``x[STATE].array + alpha * h.array`` works.
    def __add__(self, other):
        return np.asarray(self) + (np.asarray(other) if hasattr(other, "__array__") or hasattr(other, "__iter__") else other)

    def __radd__(self, other):
        return (np.asarray(other) if hasattr(other, "__array__") or hasattr(other, "__iter__") else other) + np.asarray(self)

    def __sub__(self, other):
        return np.asarray(self) - (np.asarray(other) if hasattr(other, "__array__") or hasattr(other, "__iter__") else other)

    def __rsub__(self, other):
        return (np.asarray(other) if hasattr(other, "__array__") or hasattr(other, "__iter__") else other) - np.asarray(self)

    def __mul__(self, other):
        return np.asarray(self) * other

    def __rmul__(self, other):
        return other * np.asarray(self)

    def __imul__(self, other):
        for v in self.tdv.data:
            v.array[:] *= other
            v.scatter_forward()
        return self


class TimeDependentVector:
    """A container for time-dependent vectors.

    Snapshots are stored/retrieved by specifying the time of the snapshot.
    The list of valid time frames is fixed at construction.
    """

    # B5 instrumentation: count TDV constructions and record WHERE from, so the
    # leak site is measured rather than inferred. Two prior guesses (the
    # factorization cache, then solveAdj's rhs) were both wrong.
    _ve_count = 0
    _ve_sites = {}

    def __init__(self, times, tol: float = 1e-10):
        import os as _os
        if _os.environ.get("VE_TDV_TRACE"):
            import traceback as _tb
            TimeDependentVector._ve_count += 1
            # Walk out past this module and the generate_vector/generate_state
            # wrappers to name the CALLER -- the wrapper body is not the leak site.
            st = _tb.extract_stack()[:-1]
            k = None
            for fr in reversed(st):
                bn = _os.path.basename(fr.filename)
                if bn == "timeDependentVector.py":
                    continue
                if fr.name in ("generate_state", "generate_vector", "generate_parameter"):
                    continue
                k = f"{bn}:{fr.lineno} {fr.name}"
                break
            k = k or "unknown"
            TimeDependentVector._ve_sites[k] = TimeDependentVector._ve_sites.get(k, 0) + 1
        self.times = list(times)
        self.nsteps = len(self.times)
        self.tol = tol
        self.Vh = None
        self._template = None      # one la.Vector per TDV; holds the shared comm
        self.data: list[_DupVector] = []

    # ---- factories -----------------------------------------------------
    def initialize(self, Vh) -> None:
        """Allocate snapshots compatible with the function space ``Vh``.

        All snapshots share a single neighbour communicator (see module note):
        one template la.Vector is created, and every frame is a comm-sharing
        PETSc duplicate of it.
        """
        self.Vh = Vh
        self._template = dlx.la.vector(
            Vh.dofmap.index_map, Vh.dofmap.index_map_bs
        )
        tpv = self._template.petsc_vec
        self.data = [_DupVector(tpv) for _ in range(self.nsteps)]

    def destroy(self) -> None:
        """Free the PETSc Vecs backing this TDV's snapshots.

        PETSc objects are not deterministically collected by Python's GC, so a
        TDV that goes out of scope leaks ``nsteps`` Vecs. Measured with
        ``-log_view`` (2D, 6 Newton iterations): Vector 13453 created / 6762
        destroyed, ~1115 leaked per iteration -- about 11 TDVs of 100 snapshots
        each -- costing ~0.75 GB per Newton iteration.

        ONLY call this on a TDV you own. Freeing one that another object still
        aliases is worse than the leak, so this is deliberately NOT wired into
        ``__del__``: callers opt in at sites where the TDV is provably local.

        Idempotent, and safe on a TDV that was never ``initialize``d.
        """
        for v in self.data:
            try:
                v.petsc_vec.destroy()
            except Exception:
                pass  # already destroyed / never built -- never fatal
        self.data = []
        if self._template is not None:
            try:
                self._template.petsc_vec.destroy()
            except Exception:
                pass
            self._template = None

    def copy(self) -> "TimeDependentVector":
        res = TimeDependentVector(self.times, tol=self.tol)
        res.Vh = self.Vh
        res._template = dlx.la.vector(
            self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs
        )
        tpv = res._template.petsc_vec
        res.data = []
        for v in self.data:
            new_v = _DupVector(tpv)
            new_v.array[:] = v.array[:]
            new_v.scatter_forward()
            res.data.append(new_v)
        return res

    # ---- snapshot index helpers ---------------------------------------
    def _index(self, t: float) -> int:
        i = 0
        while i < self.nsteps - 1 and 2 * t > self.times[i] + self.times[i + 1]:
            i += 1
        if abs(t - self.times[i]) >= self.tol:
            raise KeyError(
                f"Time {t!r} not in time frames "
                f"(closest {self.times[i]!r}, tol={self.tol})"
            )
        return i

    # ---- snapshot ops --------------------------------------------------
    def store(self, u, t: float) -> None:
        """Copy snapshot ``u`` into the frame at time ``t``.

        ``u`` may be a ``dolfinx.la.Vector`` or a ``petsc4py.PETSc.Vec``.
        """
        i = self._index(t)
        target = self.data[i]
        if hasattr(u, "array"):
            target.array[:] = u.array[:]
        else:  # PETSc.Vec
            target.petsc_vec.array[:] = u.array[:]
        target.scatter_forward()

    def retrieve(self, u, t: float) -> None:
        """Copy snapshot at time ``t`` into ``u``."""
        i = self._index(t)
        src = self.data[i]
        if hasattr(u, "array"):
            u.array[:] = src.array[:]
            if hasattr(u, "scatter_forward"):
                u.scatter_forward()
        else:  # PETSc.Vec
            u.array[:] = src.array[:]

    def view(self, t: float) -> dlx.la.Vector:
        """Return the underlying ``dolfinx.la.Vector`` at time ``t``."""
        return self.data[self._index(t)]

    # ---- linear algebra -----------------------------------------------
    def zero(self) -> None:
        for d in self.data:
            d.array[:] = 0.0
            d.scatter_forward()

    def scale(self, a: float) -> None:
        for d in self.data:
            d.array[:] *= a
            d.scatter_forward()

    def __imul__(self, a: float) -> "TimeDependentVector":
        self.scale(a)
        return self

    def axpy(self, a: float, other: "TimeDependentVector") -> None:
        if other.nsteps != self.nsteps:
            raise ValueError(
                f"nsteps mismatch: self={self.nsteps} other={other.nsteps}"
            )
        for i in range(self.nsteps):
            self.data[i].array[:] += a * other.data[i].array[:]
            self.data[i].scatter_forward()

    def inner(self, other: "TimeDependentVector") -> float:
        if other.nsteps != self.nsteps:
            raise ValueError(
                f"nsteps mismatch: self={self.nsteps} other={other.nsteps}"
            )
        s = 0.0
        for i in range(self.nsteps):
            s += float(self.data[i].petsc_vec.dot(other.data[i].petsc_vec))
        return s

    def norm(self, time_norm: str, space_norm: str) -> float:
        """Space-time norm. Currently only ``time_norm == 'linf'`` is supported."""
        if time_norm != "linf":
            raise NotImplementedError(
                f"time_norm={time_norm!r}: only 'linf' is supported"
            )

        if space_norm == "linf":
            ptype = petsc4py.PETSc.NormType.NORM_INFINITY
        elif space_norm == "l2":
            ptype = petsc4py.PETSc.NormType.NORM_2
        elif space_norm == "l1":
            ptype = petsc4py.PETSc.NormType.NORM_1
        else:
            raise ValueError(f"Unknown space_norm: {space_norm}")
        s = 0.0
        for d in self.data:
            v = d.petsc_vec.norm(ptype)
            if v > s:
                s = v
        return s

    @property
    def array(self) -> _TDVArrayProxy:
        return _TDVArrayProxy(self)

    def scatter_forward(self) -> None:
        for d in self.data:
            d.scatter_forward()

    def get_local(self) -> np.ndarray:
        return np.concatenate([np.asarray(v.array) for v in self.data])

    def set_local(self, v: np.ndarray) -> None:
        vv = np.reshape(v, (self.nsteps, -1))
        for i in range(self.nsteps):
            self.data[i].array[:] = vv[i]
            self.data[i].scatter_forward()

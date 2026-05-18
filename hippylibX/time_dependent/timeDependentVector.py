"""Time-dependent vector for hippylibX (dolfinx).

Faithful port of hippylib's `TimeDependentVector`. Stores one
`dolfinx.la.Vector` snapshot per time frame, keyed by absolute time.
"""

from __future__ import annotations

import numpy as np
import dolfinx as dlx


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
            assert value.tdv.nsteps == self.tdv.nsteps
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

    def __init__(self, times, tol: float = 1e-10):
        self.times = list(times)
        self.nsteps = len(self.times)
        self.tol = tol
        self.Vh = None
        self.data: list[dlx.la.Vector] = []

    # ---- factories -----------------------------------------------------
    def initialize(self, Vh) -> None:
        """Allocate snapshots compatible with the function space ``Vh``."""
        self.Vh = Vh
        self.data = [
            dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
            for _ in range(self.nsteps)
        ]

    def copy(self) -> "TimeDependentVector":
        res = TimeDependentVector(self.times, tol=self.tol)
        res.Vh = self.Vh
        res.data = []
        for v in self.data:
            new_v = dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)
            new_v.array[:] = v.array[:]
            new_v.scatter_forward()
            res.data.append(new_v)
        return res

    # ---- snapshot index helpers ---------------------------------------
    def _index(self, t: float) -> int:
        i = 0
        while i < self.nsteps - 1 and 2 * t > self.times[i] + self.times[i + 1]:
            i += 1
        assert abs(t - self.times[i]) < self.tol, (
            f"Time {t} not in time frames (closest {self.times[i]})"
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
        assert other.nsteps == self.nsteps
        for i in range(self.nsteps):
            self.data[i].array[:] += a * other.data[i].array[:]
            self.data[i].scatter_forward()

    def inner(self, other: "TimeDependentVector") -> float:
        assert other.nsteps == self.nsteps
        s = 0.0
        for i in range(self.nsteps):
            s += float(self.data[i].petsc_vec.dot(other.data[i].petsc_vec))
        return s

    def norm(self, time_norm: str, space_norm: str) -> float:
        """Space-time norm. Currently only ``time_norm == 'linf'`` is supported."""
        assert time_norm == "linf"
        import petsc4py

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

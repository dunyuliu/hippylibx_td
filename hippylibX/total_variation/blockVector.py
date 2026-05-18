# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Block vector for primal-dual optimization with TV priors.

Faithful dolfinx port of the legacy hippylib class authored by
**Xindi Gong** (see ``hippylib/modeling/blockVector.py`` in
``xindigong/hippylib:tv-enhanced``).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import dolfinx as dlx


class BlockVector:
    """A container that stores multiple vectors as one block.

    Used to hold the primal-dual pair ``(m, w)`` (and possibly extra
    state/adjoint snapshots) during TV optimization. Each element of
    :attr:`data` is either a :class:`dolfinx.la.Vector` or another
    :class:`BlockVector` (hierarchical).

    Mirrors the legacy ``hippylib.BlockVector`` API: ``nv``,
    :py:meth:`axpy`, :py:meth:`zero`, ``__imul__``, :py:meth:`copy`,
    :py:meth:`randn_perturb`, plus the ``from*`` class methods.
    """

    def __init__(self, data: list):
        self.data = list(data)

    # ---- introspection -------------------------------------------------
    @property
    def nv(self) -> int:
        return len(self.data)

    @property
    def isHierarchical(self) -> bool:
        return self.nv > 0 and hasattr(self.data[0], "nv")

    # ---- factories -----------------------------------------------------
    @classmethod
    def fromOther(cls, other: "BlockVector") -> "BlockVector":
        return cls([_copy_vec(d) for d in other.data])

    @classmethod
    def fromVector(cls, v, Nv: int) -> "BlockVector":
        return cls([_copy_vec(v) for _ in range(Nv)])

    @classmethod
    def fromFunctionSpace(cls, Vh, Nv: int) -> "BlockVector":
        return cls([_make_vec(Vh) for _ in range(Nv)])

    @classmethod
    def fromFunctionSpaces(cls, Vhs: Iterable) -> "BlockVector":
        return cls([_make_vec(Vh) for Vh in Vhs])

    # ---- BLAS-ish ------------------------------------------------------
    def randn_perturb(self, std_dev: float, rng: np.random.Generator | None = None) -> None:
        """Add ``N(0, std_dev^2 I)`` to each snapshot."""
        if rng is None:
            rng = np.random.default_rng()
        for d in self.data:
            if hasattr(d, "randn_perturb"):
                d.randn_perturb(std_dev, rng=rng)
            else:
                d.array[:] += rng.normal(0.0, std_dev, size=d.array.shape)
                if hasattr(d, "scatter_forward"):
                    d.scatter_forward()

    def axpy(self, a: float, other: "BlockVector") -> None:
        """``self += a * other`` snapshot per snapshot."""
        if self.nv != other.nv:
            raise ValueError(
                f"nv mismatch: self={self.nv} other={other.nv}"
            )
        for i in range(self.nv):
            if hasattr(self.data[i], "axpy") and hasattr(other.data[i], "axpy"):
                self.data[i].axpy(a, other.data[i])
            elif hasattr(self.data[i], "petsc_vec"):
                self.data[i].petsc_vec.axpy(a, other.data[i].petsc_vec)
                self.data[i].scatter_forward()
            else:
                self.data[i].array[:] += a * other.data[i].array[:]
                if hasattr(self.data[i], "scatter_forward"):
                    self.data[i].scatter_forward()

    def zero(self) -> None:
        for d in self.data:
            if hasattr(d, "zero"):
                d.zero()
            else:
                d.array[:] = 0.0
                if hasattr(d, "scatter_forward"):
                    d.scatter_forward()

    def __imul__(self, alpha: float) -> "BlockVector":
        for d in self.data:
            if hasattr(d, "__imul__"):
                d *= alpha
            else:
                d.array[:] *= alpha
                if hasattr(d, "scatter_forward"):
                    d.scatter_forward()
        return self

    def copy(self) -> "BlockVector":
        return BlockVector.fromOther(self)


# ---- helpers ----------------------------------------------------------
def _make_vec(Vh) -> dlx.la.Vector:
    return dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)


def _copy_vec(v):
    if hasattr(v, "copy"):
        return v.copy()
    # dlx.la.Vector — manual copy
    if hasattr(v, "array"):
        out = dlx.la.vector(v.index_map, v.block_size)
        out.array[:] = v.array[:]
        out.scatter_forward()
        return out
    raise TypeError(f"don't know how to copy {type(v).__name__}")

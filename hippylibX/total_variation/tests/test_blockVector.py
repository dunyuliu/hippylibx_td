# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Unit tests for `BlockVector` (TV subpackage)."""

from __future__ import annotations

import numpy as np
import pytest
import dolfinx as dlx
from mpi4py import MPI

from hippylibX.total_variation import BlockVector


@pytest.fixture(scope="module")
def Vh():
    msh = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    return dlx.fem.functionspace(msh, ("Lagrange", 1))


def test_fromFunctionSpace_dimensions(Vh):
    bv = BlockVector.fromFunctionSpace(Vh, 3)
    assert bv.nv == 3
    assert not bv.isHierarchical
    for v in bv.data:
        assert v.array.shape[0] >= Vh.dofmap.index_map.size_local


def test_fromFunctionSpaces_mixed(Vh):
    bv = BlockVector.fromFunctionSpaces([Vh, Vh, Vh])
    assert bv.nv == 3


def test_fromOther_is_deep_copy(Vh):
    a = BlockVector.fromFunctionSpace(Vh, 2)
    for k, v in enumerate(a.data):
        v.array[:] = float(k + 1)
        v.scatter_forward()
    b = BlockVector.fromOther(a)
    # mutating b shouldn't touch a
    b.data[0].array[:] = -99.0
    b.data[0].scatter_forward()
    assert np.allclose(a.data[0].array, 1.0)
    assert np.allclose(b.data[0].array, -99.0)


def test_zero(Vh):
    a = BlockVector.fromFunctionSpace(Vh, 2)
    for v in a.data:
        v.array[:] = 7.0
        v.scatter_forward()
    a.zero()
    for v in a.data:
        assert np.allclose(v.array, 0.0)


def test_imul_and_axpy(Vh):
    a = BlockVector.fromFunctionSpace(Vh, 2)
    b = BlockVector.fromFunctionSpace(Vh, 2)
    for v in a.data:
        v.array[:] = 1.0
        v.scatter_forward()
    for v in b.data:
        v.array[:] = 2.0
        v.scatter_forward()

    a *= 3.0          # a = 3
    for v in a.data:
        assert np.allclose(v.array, 3.0)

    a.axpy(0.5, b)    # a = 3 + 0.5*2 = 4
    for v in a.data:
        assert np.allclose(v.array, 4.0)


def test_copy_is_independent(Vh):
    a = BlockVector.fromFunctionSpace(Vh, 2)
    for v in a.data:
        v.array[:] = 5.0
        v.scatter_forward()
    b = a.copy()
    b *= 0.0
    for v in a.data:
        assert np.allclose(v.array, 5.0)


def test_axpy_size_mismatch_raises(Vh):
    a = BlockVector.fromFunctionSpace(Vh, 2)
    b = BlockVector.fromFunctionSpace(Vh, 3)
    with pytest.raises(ValueError):
        a.axpy(1.0, b)


def test_randn_perturb_seeded(Vh):
    a = BlockVector.fromFunctionSpace(Vh, 2)
    a.randn_perturb(1.0, rng=np.random.default_rng(0))
    # nonzero somewhere
    nonzero = any(np.any(v.array != 0.0) for v in a.data)
    assert nonzero

    # determinism
    b = BlockVector.fromFunctionSpace(Vh, 2)
    b.randn_perturb(1.0, rng=np.random.default_rng(0))
    for va, vb in zip(a.data, b.data):
        assert np.allclose(va.array, vb.array)


def test_hierarchical(Vh):
    inner = BlockVector.fromFunctionSpace(Vh, 2)
    outer = BlockVector([inner, BlockVector.fromOther(inner)])
    assert outer.nv == 2
    assert outer.isHierarchical

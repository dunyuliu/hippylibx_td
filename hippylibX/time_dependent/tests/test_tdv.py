# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Unit tests for `TimeDependentVector`. Run only inside the `fenicsx` env."""

from __future__ import annotations

import numpy as np
import pytest
import dolfinx as dlx
from mpi4py import MPI

from hippylibX.time_dependent.timeDependentVector import TimeDependentVector


@pytest.fixture(scope="module")
def Vh():
    msh = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    return dlx.fem.functionspace(msh, ("Lagrange", 1))


@pytest.fixture
def times():
    return np.linspace(0.0, 1.0, 5)


def _new_tdv(Vh, times, fill: float = 0.0):
    tdv = TimeDependentVector(times)
    tdv.initialize(Vh)
    for v in tdv.data:
        v.array[:] = fill
        v.scatter_forward()
    return tdv


def test_initialize_and_dimensions(Vh, times):
    tdv = TimeDependentVector(times)
    tdv.initialize(Vh)
    assert tdv.nsteps == len(times)
    n_local = Vh.dofmap.index_map.size_local * Vh.dofmap.index_map_bs
    for v in tdv.data:
        # local size match (ghost rows ≥ 0 — only check non-strict)
        assert v.array.shape[0] >= n_local


def test_store_retrieve_view(Vh, times):
    tdv = _new_tdv(Vh, times, fill=0.0)
    u = dlx.fem.Function(Vh)
    u.x.array[:] = 7.5
    u.x.scatter_forward()

    tdv.store(u.x, times[2])
    v = dlx.fem.Function(Vh)
    tdv.retrieve(v.x, times[2])
    assert np.allclose(v.x.array, 7.5)

    view_t2 = tdv.view(times[2])
    assert np.allclose(view_t2.array, 7.5)
    # other times still zero
    assert np.allclose(tdv.view(times[0]).array, 0.0)
    assert np.allclose(tdv.view(times[4]).array, 0.0)


def test_zero(Vh, times):
    tdv = _new_tdv(Vh, times, fill=3.14)
    tdv.zero()
    for v in tdv.data:
        assert np.allclose(v.array, 0.0)


def test_scale_and_imul(Vh, times):
    tdv = _new_tdv(Vh, times, fill=2.0)
    tdv.scale(3.0)
    for v in tdv.data:
        assert np.allclose(v.array, 6.0)
    tdv *= 0.5
    for v in tdv.data:
        assert np.allclose(v.array, 3.0)


def test_axpy(Vh, times):
    x = _new_tdv(Vh, times, fill=1.0)
    y = _new_tdv(Vh, times, fill=2.0)
    x.axpy(0.5, y)  # x <- x + 0.5*y
    for v in x.data:
        assert np.allclose(v.array, 2.0)


def test_inner(Vh, times):
    x = _new_tdv(Vh, times, fill=1.0)
    y = _new_tdv(Vh, times, fill=2.0)
    s = x.inner(y)
    expected = 2.0 * x.nsteps * sum(v.array.shape[0] for v in x.data) / x.nsteps
    # 1*2 * total_dof_count
    total = sum(v.array.shape[0] for v in x.data)
    assert s == pytest.approx(2.0 * total)


def test_norm(Vh, times):
    x = _new_tdv(Vh, times, fill=0.0)
    # set one frame to nonzero
    x.data[2].array[:] = 4.0
    x.data[2].scatter_forward()
    nrm = x.norm("linf", "linf")
    assert nrm == pytest.approx(4.0)


def test_copy_independence(Vh, times):
    x = _new_tdv(Vh, times, fill=5.0)
    y = x.copy()
    y.data[0].array[:] = 99.0
    y.data[0].scatter_forward()
    # x must not be affected
    assert np.allclose(x.data[0].array, 5.0)


def test_array_proxy_full_slice_assign_from_numpy(Vh, times):
    x = _new_tdv(Vh, times, fill=0.0)
    n_per = x.data[0].array.shape[0]
    flat = np.arange(x.nsteps * n_per, dtype=float)
    x.array[:] = flat
    for i, v in enumerate(x.data):
        assert np.allclose(v.array, flat[i * n_per : (i + 1) * n_per])


def test_array_proxy_full_slice_assign_from_proxy(Vh, times):
    a = _new_tdv(Vh, times, fill=1.0)
    b = _new_tdv(Vh, times, fill=2.0)
    a.array[:] = b.array  # proxy-to-proxy copy
    for v in a.data:
        assert np.allclose(v.array, 2.0)


def test_array_proxy_arithmetic(Vh, times):
    a = _new_tdv(Vh, times, fill=1.0)
    b = _new_tdv(Vh, times, fill=2.0)
    out = a.array + 0.5 * b.array  # returns concatenated ndarray
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, 2.0)


def test_index_out_of_frame_raises(Vh, times):
    tdv = _new_tdv(Vh, times)
    u = dlx.fem.Function(Vh)
    with pytest.raises(KeyError):
        tdv.store(u.x, 0.42)  # not in frames

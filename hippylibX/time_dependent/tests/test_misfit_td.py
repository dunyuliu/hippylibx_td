"""Unit tests for `ContinuousStateObservation` and `MisfitTD` (fenicsx env)."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
import dolfinx as dlx
from mpi4py import MPI

import hippylibX as hpx
from hippylibX.time_dependent.misfit import (
    ContinuousStateObservation,
    MisfitTD,
)
from hippylibX.time_dependent.timeDependentVector import TimeDependentVector


@pytest.fixture(scope="module")
def Vh():
    msh = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    return dlx.fem.functionspace(msh, ("Lagrange", 1))


def _make_obs(Vh, sigma=1.0):
    obs = ContinuousStateObservation(Vh, ufl.dx, bcs=[])
    obs.noise_variance = sigma * sigma
    return obs


def test_cost_zero_when_state_equals_data(Vh):
    obs = _make_obs(Vh)
    u = dlx.fem.Function(Vh)
    u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u.x.scatter_forward()
    obs.d.array[:] = u.x.array[:]
    obs.d.scatter_forward()
    c = obs.cost([u.x, None, None])
    assert c == pytest.approx(0.0, abs=1e-14)


def test_grad_zero_when_state_equals_data(Vh):
    obs = _make_obs(Vh)
    u = dlx.fem.Function(Vh)
    u.interpolate(lambda x: x[0])
    u.x.scatter_forward()
    obs.d.array[:] = u.x.array[:]
    obs.d.scatter_forward()
    out = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
    obs.grad(hpx.STATE, [u.x, None, None], out)
    assert np.allclose(out.array, 0.0, atol=1e-12)


def test_grad_matches_finite_difference(Vh):
    obs = _make_obs(Vh, sigma=0.5)
    u = dlx.fem.Function(Vh)
    u.interpolate(lambda x: 0.5 * x[0] + 0.3 * x[1])
    u.x.scatter_forward()
    obs.d.array[:] = 0.0
    obs.d.scatter_forward()

    g = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
    obs.grad(hpx.STATE, [u.x, None, None], g)

    # FD: cost(u + eps h) ≈ cost(u) + eps <g, h>
    rng = np.random.default_rng(0)
    h = dlx.fem.Function(Vh)
    h.x.array[:] = rng.normal(0.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    c0 = obs.cost([u.x, None, None])
    eps = 1e-6
    u_plus = dlx.fem.Function(Vh)
    u_plus.x.array[:] = u.x.array[:] + eps * h.x.array[:]
    u_plus.x.scatter_forward()
    c1 = obs.cost([u_plus.x, None, None])

    fd = (c1 - c0) / eps
    analytic = float(np.dot(g.array, h.x.array))
    rel = abs(fd - analytic) / max(abs(analytic), 1e-30)
    assert rel < 1e-4, f"grad FD: {fd} vs analytic {analytic} (rel {rel:.2e})"


def test_apply_ij_state_state_is_W_div_sigma2(Vh):
    obs = _make_obs(Vh, sigma=0.5)
    direction = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
    direction.array[:] = 1.0
    direction.scatter_forward()
    out = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
    obs.apply_ij(hpx.STATE, hpx.STATE, direction, out)
    # Compare to W*dir / sigma^2 manually
    expected = obs.W.createVecLeft()
    obs.W.mult(direction.petsc_vec, expected)
    expected.scale(1.0 / obs.noise_variance)
    assert np.allclose(out.array, expected.array)


def test_apply_ij_cross_is_zero(Vh):
    obs = _make_obs(Vh)
    direction = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
    out = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
    obs.apply_ij(hpx.STATE, hpx.PARAMETER, direction, out)
    assert np.allclose(out.array, 0.0)


# --- MisfitTD -------------------------------------------------------------


def test_misfittd_cost_is_sum(Vh):
    times = np.linspace(0.0, 1.0, 4)
    obs_list = [_make_obs(Vh) for _ in times]

    # Build a TDV with state, and per-time observation set to a different constant
    tdv = TimeDependentVector(times)
    tdv.initialize(Vh)
    expected_total = 0.0
    for k, t in enumerate(times):
        tdv.data[k].array[:] = float(k)  # state = k
        tdv.data[k].scatter_forward()
        obs_list[k].d.array[:] = 0.0
        obs_list[k].d.scatter_forward()
        # local cost = 0.5/sigma^2 * (k)^2 * (mass-integral of 1)
        # mass-integral of 1 = total area = 1.0 for unit square — but the W
        # matrix here is the inner-product matrix on P1, not the integral of
        # the function. So just sum the observed costs directly.
        expected_total += obs_list[k].cost([tdv.data[k], None, None])

    misfit = MisfitTD(obs_list, times)
    total = misfit.cost([tdv, None, None])
    assert total == pytest.approx(expected_total, rel=1e-12)


def test_misfittd_grad_state_per_frame(Vh):
    times = np.linspace(0.0, 1.0, 3)
    obs_list = [_make_obs(Vh) for _ in times]

    tdv_state = TimeDependentVector(times)
    tdv_state.initialize(Vh)
    for k, t in enumerate(times):
        tdv_state.data[k].array[:] = float(k + 1)
        tdv_state.data[k].scatter_forward()
        obs_list[k].d.array[:] = 0.0
        obs_list[k].d.scatter_forward()

    out_tdv = TimeDependentVector(times)
    out_tdv.initialize(Vh)

    misfit = MisfitTD(obs_list, times)
    misfit.grad(hpx.STATE, [tdv_state, None, None], out_tdv)

    # Each frame's grad must match the per-frame ObsContinuous grad
    for k, t in enumerate(times):
        ref = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
        obs_list[k].grad(hpx.STATE, [tdv_state.data[k], None, None], ref)
        assert np.allclose(out_tdv.view(t).array, ref.array)

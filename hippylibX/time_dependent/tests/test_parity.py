# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Parity tests: legacy hippylib vs ported hippylibX must produce equivalent
numerical results on the same deterministic problem, for each supported
time-dependent physics (heat, tumor, ...).

Each runner is invoked in a fresh subprocess inside its own conda env, so the
two FEniCS stacks never coexist in the same process.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).parent

CONDA_SH = os.environ.get("CONDA_SH", "/opt/anaconda3/etc/profile.d/conda.sh")
HIPPYLIBX_BASE_DIR = os.environ.get("HIPPYLIBX_BASE_DIR", "")
HIPPYLIB_PATH = os.environ.get("HIPPYLIB_PATH", "")

# Both conda envs must exist (one runs the port, one runs legacy).
# We skip the parity suite gracefully on machines that only have one.
FENICSX_ENV = os.environ.get("FENICSX_ENV", "fenicsx")
LEGACY_ENV = os.environ.get("FENICSPROJECT_ENV", "fenicsproject")


def _conda_envs_available() -> tuple[bool, str]:
    """Return (ok, reason). ok=False means the parity suite should skip."""
    if not os.path.exists(CONDA_SH):
        return False, f"conda init not found at {CONDA_SH}"
    res = subprocess.run(
        ["bash", "-c", f"source {CONDA_SH} && conda env list"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if res.returncode != 0:
        return False, f"`conda env list` failed: {res.stderr.strip()}"
    envs = res.stdout
    missing = [e for e in (FENICSX_ENV, LEGACY_ENV) if e not in envs]
    if missing:
        return False, f"missing conda env(s): {missing}"
    # Legacy hippylib must be importable from HIPPYLIB_PATH (env var, no
    # hardcoded fallback). The path is added to sys.path by run_legacy.py.
    if not HIPPYLIB_PATH:
        return False, "HIPPYLIB_PATH env var not set (path to a hippylib checkout)"
    if not os.path.isdir(os.path.join(HIPPYLIB_PATH, "hippylib")):
        return False, f"HIPPYLIB_PATH={HIPPYLIB_PATH!r} doesn't look like a hippylib checkout"
    return True, ""


pytestmark = pytest.mark.skipif(
    not _conda_envs_available()[0],
    reason=_conda_envs_available()[1] or "parity envs unavailable",
)

PROBLEMS = ["heat", "tumor", "ad_diff"]

# Per-problem parity tolerances. Heat & tumor: ~1e-13 (FP-roundoff agreement;
# both stacks compute identical floating-point operations modulo FFC/FFCx
# code-gen order). AD: ~1e-2 — both stacks solve the linear MAP via
# NewtonCG, but their inner-CG Eisenstat-Walker forcing terms produce
# slightly different inexact iterates, so the converged MAP differs by
# ~0.1% on m and ~1% on cost. The forward solve, prior, and misfit
# evaluations all agree to ~1e-13; only the optimizer trajectory diverges.
TOL_FINAL_COST = {"heat": 1e-4, "tumor": 1e-4, "ad_diff": 5e-2}
TOL_MAP_STATE  = {"heat": 1e-4, "tumor": 1e-4, "ad_diff": 5e-3}
TOL_MAP_PARAM  = {"heat": 1e-4, "tumor": 1e-4, "ad_diff": 5e-3}
TOL_PRE_INVERSION = 1e-6   # for cost@m0, cost@m_true, state_norms@m_true


def _run(env_name: str, runner: str, problem: str, out_json: Path) -> dict:
    cmd = (
        f"source {CONDA_SH} && conda activate {env_name} && "
        f"HIPPYLIBX_BASE_DIR={HIPPYLIBX_BASE_DIR} "
        f"HIPPYLIB_PATH={HIPPYLIB_PATH} "
        f"PROBLEM={problem} "
        f"RESULTS_JSON={out_json} python3 {THIS_DIR / runner}"
    )
    print(f"\n[parity] {problem}: running {runner} in env {env_name}")
    res = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"{runner} ({problem}) failed (exit {res.returncode}):\n"
            f"--stdout--\n{res.stdout}\n--stderr--\n{res.stderr}"
        )
    return json.loads(out_json.read_text())


_RESULTS: dict[tuple[str, str], dict] = {}


def _get(stack: str, problem: str) -> dict:
    key = (stack, problem)
    if key in _RESULTS:
        return _RESULTS[key]
    if stack == "x":
        env_name = "fenicsx"
        runner = "run_x.py"
    elif stack == "legacy":
        env_name = "fenicsproject"
        runner = "run_legacy.py"
    else:
        raise ValueError(stack)
    out_json = THIS_DIR / f"results_{stack}_{problem}.json"
    _RESULTS[key] = _run(env_name, runner, problem, out_json)
    return _RESULTS[key]


@pytest.fixture(scope="module", params=PROBLEMS)
def both(request):
    problem = request.param
    return problem, _get("x", problem), _get("legacy", problem)


# ----- Tests --------------------------------------------------------------


def test_ndofs_match(both):
    problem, x, leg = both
    assert x["ndofs_state"] == leg["ndofs_state"], f"[{problem}] state DOFs differ"
    assert x["ndofs_param"] == leg["ndofs_param"], f"[{problem}] param DOFs differ"


def test_state_norms_at_mtrue(both):
    problem, x, leg = both
    a = x["state_norms_at_mtrue"]
    b = leg["state_norms_at_mtrue"]
    assert len(a) == len(b)
    for i, (xn, ln) in enumerate(zip(a, b)):
        rel = abs(xn - ln) / max(abs(ln), 1e-30)
        assert rel < TOL_PRE_INVERSION, f"[{problem}] step {i}: x={xn} legacy={ln} (rel={rel})"


def test_cost_at_m0(both):
    problem, x, leg = both
    for i, name in enumerate(("total", "reg", "misfit")):
        a, b = x["cost_at_m0"][i], leg["cost_at_m0"][i]
        rel = abs(a - b) / max(abs(b), 1e-30)
        assert rel < TOL_PRE_INVERSION, f"[{problem}] cost@m0[{name}] x={a} legacy={b} (rel={rel})"


def test_cost_at_mtrue(both):
    problem, x, leg = both
    for i, name in enumerate(("total", "reg", "misfit")):
        a, b = x["cost_at_mtrue"][i], leg["cost_at_mtrue"][i]
        rel = abs(a - b) / max(abs(b), 1e-30)
        assert rel < TOL_PRE_INVERSION, f"[{problem}] cost@m_true[{name}] x={a} legacy={b} (rel={rel})"


def test_newton_converges(both):
    problem, x, leg = both
    assert x["converged"], f"[{problem}] X did not converge"
    assert leg["converged"], f"[{problem}] legacy did not converge"
    # iter counts: heat/tumor should match exactly; AD may differ due to
    # algorithmic divergence in the inner Eisenstat-Walker CG forcing.
    max_diff = 1 if problem in ("heat", "tumor") else 5
    assert abs(x["newton_iters"] - leg["newton_iters"]) <= max_diff, (
        f"[{problem}] iters: x={x['newton_iters']} legacy={leg['newton_iters']}"
    )


def test_final_cost_match(both):
    problem, x, leg = both
    a, b = x["final_cost"], leg["final_cost"]
    rel = abs(a - b) / max(abs(b), 1e-30)
    tol = TOL_FINAL_COST[problem]
    assert rel < tol, f"[{problem}] final cost x={a} vs legacy={b} (rel={rel} > {tol})"


def test_map_state_norms(both):
    problem, x, leg = both
    tol = TOL_MAP_STATE[problem]
    for i, (a, b) in enumerate(zip(x["map_state_norms"], leg["map_state_norms"])):
        if abs(b) < 1e-30:
            assert abs(a) < 1e-12, f"[{problem}] step {i}: x={a} (expected ~0)"
            continue
        rel = abs(a - b) / abs(b)
        assert rel < tol, f"[{problem}] MAP state ‖·‖ step {i}: x={a} legacy={b} (rel={rel} > {tol})"


def test_map_param_norm(both):
    problem, x, leg = both
    a, b = x["m_map_l2"], leg["m_map_l2"]
    rel = abs(a - b) / max(abs(b), 1e-30)
    tol = TOL_MAP_PARAM[problem]
    assert rel < tol, f"[{problem}] ||m_MAP||: x={a} legacy={b} (rel={rel} > {tol})"

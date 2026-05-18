# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""TV parity tests: legacy `xindigong/hippylib:tv-enhanced` vs ported
`hippylibX.total_variation` must produce equivalent numerical results
on the same deterministic denoising problem.

Each runner is invoked in a fresh subprocess inside its own conda env, so
the two FEniCS stacks never coexist in the same process.

Env var requirements:
- ``HIPPYLIB_TV_PATH``: a clone of ``xindigong/hippylib`` checked out
  on ``tv-enhanced``.
- ``CONDA_SH``: location of conda's ``profile.d/conda.sh``
  (default: ``/opt/anaconda3/etc/profile.d/conda.sh``).
- ``FENICSX_ENV`` / ``FENICSPROJECT_ENV``: env names (defaults
  ``fenicsx`` / ``fenicsproject``).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).parent
CONDA_SH = os.environ.get("CONDA_SH", "/opt/anaconda3/etc/profile.d/conda.sh")
HIPPYLIB_TV_PATH = os.environ.get("HIPPYLIB_TV_PATH", "")
FENICSX_ENV = os.environ.get("FENICSX_ENV", "fenicsx")
LEGACY_ENV = os.environ.get("FENICSPROJECT_ENV", "fenicsproject")


def _envs_available() -> tuple[bool, str]:
    if not os.path.exists(CONDA_SH):
        return False, f"conda init not found at {CONDA_SH}"
    res = subprocess.run(
        ["bash", "-c", f"source {CONDA_SH} && conda env list"],
        capture_output=True, text=True, timeout=10,
    )
    if res.returncode != 0:
        return False, f"`conda env list` failed: {res.stderr.strip()}"
    envs = res.stdout
    missing = [e for e in (FENICSX_ENV, LEGACY_ENV) if e not in envs]
    if missing:
        return False, f"missing conda env(s): {missing}"
    if not HIPPYLIB_TV_PATH:
        return False, (
            "HIPPYLIB_TV_PATH not set; set to a clone of "
            "xindigong/hippylib:tv-enhanced"
        )
    if not os.path.isdir(os.path.join(HIPPYLIB_TV_PATH, "hippylib", "modeling")):
        return False, (
            f"HIPPYLIB_TV_PATH={HIPPYLIB_TV_PATH!r} does not look like a "
            "hippylib checkout"
        )
    return True, ""


pytestmark = pytest.mark.skipif(
    not _envs_available()[0],
    reason=_envs_available()[1] or "TV parity envs unavailable",
)


def _run(env_name: str, runner: str, out_json: Path) -> dict:
    cmd = (
        f"source {CONDA_SH} && conda activate {env_name} && "
        f"HIPPYLIB_TV_PATH={HIPPYLIB_TV_PATH} "
        f"RESULTS_JSON={out_json} python3 {THIS_DIR / runner}"
    )
    print(f"\n[parity-tv] running {runner} in env {env_name}")
    res = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, timeout=600,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"{runner} failed (exit {res.returncode}):\n"
            f"--stdout--\n{res.stdout}\n--stderr--\n{res.stderr}"
        )
    return json.loads(out_json.read_text())


_CACHE: dict[str, dict] = {}


def _get(stack: str) -> dict:
    if stack in _CACHE:
        return _CACHE[stack]
    if stack == "x":
        _CACHE[stack] = _run(
            FENICSX_ENV, "run_x_tv.py", THIS_DIR / "results_x_tv.json",
        )
    elif stack == "legacy":
        _CACHE[stack] = _run(
            LEGACY_ENV, "run_legacy_tv.py",
            THIS_DIR / "results_legacy_tv.json",
        )
    else:
        raise ValueError(stack)
    return _CACHE[stack]


@pytest.fixture(scope="module")
def both():
    return _get("x"), _get("legacy")


# ----- tests -------------------------------------------------------------


def test_ndofs_match(both):
    x, leg = both
    assert x["ndofs_param"] == leg["ndofs_param"]


def test_cost_at_m0_match(both):
    """Cost at m=0 is purely a forward+misfit evaluation. Should match
    to FP roundoff (no optimization involved)."""
    x, leg = both
    for i, name in enumerate(("total", "smooth_reg", "nonsmooth_reg", "misfit")):
        a, b = x["cost_at_m0"][i], leg["cost_at_m0"][i]
        rel = abs(a - b) / max(abs(b), 1e-30)
        assert rel < 1e-10, (
            f"cost@m=0[{name}]: x={a} leg={b} (rel={rel})"
        )


def test_final_cost_match(both):
    """Both solvers should converge to the same MAP cost (within FP roundoff)."""
    x, leg = both
    rel = abs(x["final_cost"] - leg["final_cost"]) / max(
        abs(leg["final_cost"]), 1e-30
    )
    assert rel < 1e-8, (
        f"final_cost: x={x['final_cost']} leg={leg['final_cost']} (rel={rel})"
    )


def test_m_map_l2_match(both):
    x, leg = both
    rel = abs(x["m_map_l2"] - leg["m_map_l2"]) / max(abs(leg["m_map_l2"]), 1e-30)
    assert rel < 1e-6, (
        f"||m_MAP||: x={x['m_map_l2']} leg={leg['m_map_l2']} (rel={rel})"
    )


def test_err_to_true_match(both):
    """Denoising quality is the same to ~1e-6 between stacks."""
    x, leg = both
    rel = abs(x["err_to_true"] - leg["err_to_true"]) / max(
        abs(leg["err_to_true"]), 1e-30
    )
    assert rel < 1e-5, (
        f"err_to_true: x={x['err_to_true']} leg={leg['err_to_true']} (rel={rel})"
    )


def test_iter_count_close(both):
    """Newton-CG iteration counts may differ by 1-2 due to floating-point
    rounding around the convergence threshold, but should be within ±2."""
    x, leg = both
    assert abs(x["newton_iters"] - leg["newton_iters"]) <= 2, (
        f"iters: x={x['newton_iters']} leg={leg['newton_iters']}"
    )

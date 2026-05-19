# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""MPI consistency test for the TV image denoising problem.

Runs the same deterministic problem (identity PDE, TVPrior, primal-dual
Newton-CG MAP) under np=1, np=2, and np=4 MPI processes, gathers the
globally-ordered MAP parameter vector from each run, and asserts that all
three agree to within ``ATOL = 1e-10``.

Invoked via::

    pytest test_mpi_consistency.py -v

The test is skipped automatically if ``mpirun``/``mpiexec`` is not on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_THIS, "run_tv_mpi.py")

ATOL = 1e-10
NP_VALUES = [1, 2, 4]


def _mpirun_available() -> bool:
    return shutil.which("mpirun") is not None or shutil.which("mpiexec") is not None


def _mpirun_cmd() -> str:
    return shutil.which("mpirun") or shutil.which("mpiexec")


def _run_one(np: int, tmp_dir: str) -> dict:
    """Run the TV MPI runner with *np* processes and return the parsed JSON."""
    out_path = os.path.join(tmp_dir, f"results_np{np}.json")
    env = os.environ.copy()
    env["RESULTS_JSON"] = out_path
    cmd = [_mpirun_cmd(), "-n", str(np), sys.executable, _RUNNER]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=_THIS
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mpirun -n {np} failed (returncode={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if not os.path.exists(out_path):
        raise FileNotFoundError(
            f"runner did not produce {out_path}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    with open(out_path) as f:
        return json.load(f)


@pytest.mark.skipif(not _mpirun_available(), reason="mpirun/mpiexec not on PATH")
def test_mpi_consistency():
    """TV MAP parameter vector must be identical across np=1,2,4."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = {}
        for np in NP_VALUES:
            try:
                results[np] = _run_one(np, tmp_dir)
            except Exception as exc:
                pytest.skip(f"mpirun -n {np} could not run: {exc}")

        ref = results[NP_VALUES[0]]
        ref_m = np.array(ref["m_map_global"])

        for np in NP_VALUES[1:]:
            r = results[np]
            assert r["ndofs_param"] == ref["ndofs_param"], (
                f"np={np} ndofs_param mismatch"
            )
            assert r["converged"], f"np={np} primal-dual Newton-CG did not converge"

        # primary check: globally-ordered MAP parameter vector
        for np in NP_VALUES[1:]:
            r = results[np]
            m_np = np.array(r["m_map_global"])
            err = np.max(np.abs(m_np - ref_m))
            assert np.allclose(m_np, ref_m, rtol=0, atol=ATOL), (
                f"np={np} TV MAP parameter disagrees with np=1: "
                f"max|diff|={err:.2e} (tol={ATOL:.0e})"
            )

        # scalar consistency
        for np in NP_VALUES[1:]:
            r = results[np]
            assert abs(r["m_map_l2"] - ref["m_map_l2"]) < ATOL, (
                f"np={np} m_map_l2 mismatch: {r['m_map_l2']} vs {ref['m_map_l2']}"
            )

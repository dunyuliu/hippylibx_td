# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Shared problem definition for TV parity tests.

Both the legacy and the ported runner import these constants, plus the
analytic two-disk "true image" function, so the two stacks denoise the
same image with the same parameters.

No random noise is used (``NOISE_STD = 0``); ``d = m_true`` exactly.
DOF ordering differs between dolfin and dolfinx, so any random noise
applied at the DOF level would diverge between stacks.
"""

# --- problem ---
NX = NY = 12          # small for fast parity sweeps
ALPHA = 1e-2          # TV weight
BETA = 1e-3           # TV smoothing
PEPS_FACTOR = 0.5     # peps = PEPS_FACTOR * ALPHA

# --- noise (deterministic) ---
NOISE_STD = 0.0       # zero noise → d = m_true exactly
NOISE_VARIANCE = 1.0  # used in the misfit (matches legacy convention)

# --- solver ---
REL_TOL = 1e-6
ABS_TOL = 1e-12
MAX_ITER = 15
CG_MAX_ITER = 30
GN_ITER = 5
PRINT_LEVEL = -1

# --- analytic test image: single disk at (0.5, 0.5), radius 0.2 ---
DISK_CX = 0.5
DISK_CY = 0.5
DISK_R = 0.2


def two_disk_image_numpy(x):
    """Single-disk indicator. Used by run_x_tv.py via dolfinx
    `interpolate(callable)`. ``x`` is a (gdim, N) numpy array."""
    import numpy as np
    return np.where(
        (x[0] - DISK_CX) ** 2 + (x[1] - DISK_CY) ** 2 < DISK_R ** 2, 1.0, 0.0,
    )


# --- legacy dolfin Expression string equivalent ---
# Use C++ ternary; can't use std::pow inside Expression on macOS clang.
LEGACY_M_TRUE_EXPR = (
    f"((x[0]-{DISK_CX})*(x[0]-{DISK_CX}) + "
    f"(x[1]-{DISK_CY})*(x[1]-{DISK_CY}) < {DISK_R}*{DISK_R}) ? 1.0 : 0.0"
)

# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Model-style classes for specific time-dependent inverse problems.

Each module here defines a self-contained :class:`hippylibX.Model`-like
class (owning its own prior + misfit) tailored to one application:

- :mod:`ad_diff` — advection-diffusion with initial-condition inversion.

The generic :class:`hippylibX.time_dependent.TimeDependentPDEVariational
Problem` is preferred when the parameter enters the PDE residual as a
coefficient (e.g. heat, tumor). Application-specific Models live here
when their structure (IC inversion, custom misfit, stabilization)
doesn't fit the generic template.
"""

from .ad_diff import AdvectionDiffusionICModel  # noqa

# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Variable index constants (vendored).

Mirrors :mod:`hippylibX.modeling.variables` so the
:mod:`hippylibX.total_variation` subpackage has zero imports from
upstream `hippylibX/`. SLACK is the extra slot for non-smooth
(primal-dual TV) variables.
"""

STATE = 0
PARAMETER = 1
ADJOINT = 2
SLACK = 3
NVAR = 4

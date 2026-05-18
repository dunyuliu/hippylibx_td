# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Time-dependent extension for hippylibX.

Faithful port of the legacy hippylib time-dependent classes
(:class:`TimeDependentVector`, :class:`TimeDependentPDEVariationalProblem`,
:class:`MisfitTD`, :class:`ContinuousStateObservation`,
:class:`SpaceTimePointwiseStateObservation`) onto dolfinx, plus the
standalone :class:`AdvectionDiffusionICModel` for initial-condition
inversion problems.

Usage::

    import hippylibX as hpx
    from hippylibX.time_dependent import (
        TimeDependentVector,
        TimeDependentPDEVariationalProblem,
        MisfitTD,
        ContinuousStateObservation,
        SpaceTimePointwiseStateObservation,
    )

The upstream :class:`hippylibX.Model`, :class:`hippylibX.ReducedHessian`,
and :func:`hippylibX.modelVerify` work directly on
:class:`TimeDependentVector` state/adjoint variables via the in-place
``.array[:]`` proxy on TDV — no model-level shims required.
"""

from .timeDependentVector import TimeDependentVector  # noqa
from .TimeDependentPDEVariationalProblem import (  # noqa
    TimeDependentPDEVariationalProblem,
)
from .misfit import (  # noqa
    MisfitTD,
    ContinuousStateObservation,
    SpaceTimePointwiseStateObservation,
)
from .applications.ad_diff import AdvectionDiffusionICModel  # noqa

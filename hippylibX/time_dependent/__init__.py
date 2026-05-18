# --------------------------------------------------------------------------bc-
# Time-dependent extension for hippylibX.
#
# Faithful port of the legacy hippylib time-dependent classes
# (TimeDependentVector, TimeDependentPDEVariationalProblem, MisfitTD,
# ContinuousStateObservation) onto dolfinx.
#
# Usage:
#     import hippylibX as hpx
#     from hippylibX.time_dependent import (
#         TimeDependentVector,
#         TimeDependentPDEVariationalProblem,
#         MisfitTD,
#         ContinuousStateObservation,
#     )
#
# The upstream `hpx.Model`, `hpx.ReducedHessian`, and `hpx.modelVerify`
# work directly on `TimeDependentVector` state/adjoint variables via the
# in-place `.array[:]` proxy on TDV — no model-level shims required.
# --------------------------------------------------------------------------ec-

from .timeDependentVector import TimeDependentVector  # noqa
from .TimeDependentPDEVariationalProblem import (  # noqa
    TimeDependentPDEVariationalProblem,
)
from .misfit import (  # noqa
    MisfitTD,
    ContinuousStateObservation,
    SpaceTimePointwiseStateObservation,
)
from .ad_diff_problem import AdvectionDiffusionICModel  # noqa

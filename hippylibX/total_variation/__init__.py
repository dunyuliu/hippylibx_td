# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""Total Variation prior + primal-dual optimization for hippylibX (dolfinx).

This subpackage is a faithful port of the Total Variation (TV) work
originally developed by **Xindi Gong** in the legacy `hippylib` fork at

    https://github.com/xindigong/hippylib/tree/tv-enhanced

The legacy implementation lives across:
    hippylib/modeling/nonsmoothPrior.py     -> :class:`TVPrior`
    hippylib/modeling/nonsmoothModel.py     -> :class:`ModelNS`
    hippylib/modeling/blockVector.py        -> :class:`BlockVector`
    hippylib/modeling/multiPDEProblem.py    -> :class:`MultiPDEProblem`
    hippylib/algorithms/PDNewtonCG.py       -> :class:`PDNewtonCG`
    hippylib/modeling/reducedHessian.py     -> :class:`NSReducedHessian`
    hippylib/modeling/variables.py          -> adds :data:`SLACK`

Mathematical references:
    [1] Chan, Tony F., Gene H. Golub, and Pep Mulet. "A nonlinear
        primal-dual method for total variation-based image restoration."
        SIAM J. Sci. Comput. 20.6 (1999): 1964-1977.

All credit for the algorithmic design, derivations, and reference
implementation goes to Xindi Gong. Any defects in the dolfinx port are
the maintainers' responsibility.
"""

# variable indices (vendored from ._variables so this subpackage has zero
# imports from upstream `hippylibX.modeling`).
from ._variables import STATE, PARAMETER, ADJOINT, SLACK, NVAR  # noqa

NVAR_NS = NVAR  # kept as a friendly alias

from .blockVector import BlockVector  # noqa
from .multiPDEProblem import MultiPDEProblem  # noqa
from .nonsmoothPrior import TVPrior, weightedVTVPrior  # noqa
from .nonsmoothModel import ModelNS  # noqa
from .nsReducedHessian import NSReducedHessian  # noqa
from .pdNewtonCG import (  # noqa
    ReducedSpacePDNewtonCG,
    ReducedSpacePDNewtonCG_ParameterList,
    LS_ParameterList,
)

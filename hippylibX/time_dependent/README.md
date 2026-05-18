# `hippylibX.time_dependent`

Time-dependent extension for hippylibX. Faithful port of the legacy
`hippylib` time-dependent classes onto dolfinx.

## Contents

- `timeDependentVector.py` — `TimeDependentVector`: snapshot container
  indexed by time, backed by `dolfinx.la.Vector` per frame. Exposes an
  `.array[:]` numpy-proxy so upstream `hippylibX.Model` /
  `hippylibX.ReducedHessian` / `hippylibX.modelVerify` operate on it
  unchanged (no model-level shims needed).
- `TimeDependentPDEVariationalProblem.py` —
  `TimeDependentPDEVariationalProblem`: stepper that drives a
  `varf(u, u_old, m, p, t)` UFL form through `times`, with
  `solveFwd` / `solveAdj` / incremental fwd+adj solvers / per-step
  parameter gradient / second-derivative blocks (`applyC/Ct/Wuu/Wum/Wmu/Wmm`).
- `misfit.py` — `ContinuousStateObservation` (single-time L²(X) mass-matrix
  misfit, mirrors legacy hippylib) and `MisfitTD` (time-summed wrapper).

## Usage

```python
import hippylibX as hpx
from hippylibX import time_dependent as td

# build the time-dependent forward problem from a UFL form factory
pde = td.TimeDependentPDEVariationalProblem(
    Vh, varf_handler, bc=[bc], bc0=[bc0], u0=u0,
    t_init=0.0, t_final=1.0, is_fwd_linear=True,
)

# time-summed misfit
misfits = [td.ContinuousStateObservation(Vh[hpx.STATE], ufl.dx, bcs=[bc0])
           for _ in pde.times]
# ... populate misfits[k].d and misfits[k].noise_variance ...
misfit = td.MisfitTD(misfits, pde.times)

# upstream Model + Newton-CG, no shims
model = hpx.Model(pde, prior, misfit)
hpx.modelVerify(model, m0)
x = hpx.ReducedSpaceNewtonCG(model, params).solve(x)
```

See `examples/heat.py` for a full worked tutorial.

## Tests

```bash
conda activate fenicsx
pytest -q hippylibX/time_dependent/tests/
```

`test_parity.py` additionally invokes legacy `hippylib` + dolfin in a
sibling conda env (`fenicsproject`) to check numerical equivalence on a
deterministic problem.

## Credits

Initial port of the legacy `hippylib` time-dependent classes to dolfinx
was carried out with assistance from Anthropic's Claude (Opus 4.7) acting
as a pair-programming agent. The mathematical specification, design
decisions, integration into this fork, and verification are owned by the
maintainers of this repository.

## Design notes

The `.array[:]` proxy on `TimeDependentVector` is what lets upstream
`Model` / `ReducedHessian` / `modelVerify` operate without modification:
expressions like `out.array[:] *= -1.0` or
`out.array[:] += tmp.array` evaluate through the proxy and dispatch
per-snapshot. As a result this subpackage adds new files but touches
**no existing hippylibX file** other than one re-export line in
`hippylibX/__init__.py`.

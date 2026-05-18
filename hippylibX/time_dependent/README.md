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
- `misfit.py` — three misfit classes:
  - `ContinuousStateObservation` — single-time L²(X) mass-matrix misfit
  - `SpaceTimePointwiseStateObservation` — pointwise sensor grid at a
    list of observation times
  - `MisfitTD` — time-summed wrapper around per-time misfits
- `ad_diff_problem.py` — `AdvectionDiffusionICModel`: standalone
  Model-style class for linear advection-diffusion with **initial-
  condition inversion** (mirrors legacy `TimeDependentAD`). Uses
  SUPG/GLS streamline stabilization. Inverts the IC, not a coefficient.

## Coverage of legacy hippylib time-dependent applications

| legacy application | covered by | parity-tested |
|---|---|---|
| `applications/time_dependent/model_heat.py` | `TimeDependentPDEVariationalProblem` + `MisfitTD` | ✓ |
| `applications/time_dependent/model_tumor.py` (nonlinear) | same, with `is_fwd_linear=False` | ✓ |
| `applications/ad_diff/model_ad_diff.py` (IC inversion + pointwise sensors) | `AdvectionDiffusionICModel` + `SpaceTimePointwiseStateObservation` | ✓ |

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

## Install (this subpackage)

Two conda envs are needed if you want to run the full test suite (unit
+ parity vs legacy hippylib). Only the first is needed for normal use.

### `fenicsx` — required (port runtime + unit tests)

```bash
conda create -n fenicsx -c conda-forge \
    fenics-dolfinx=0.10 mpich petsc=*=*complex* python=3.12 \
    matplotlib numpy pytest
conda activate fenicsx
# clone the hippylibX fork (this repo)
cd <path>/hippylibx_td
# the subpackage is importable as long as the repo root is on PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

### `fenicsproject` — optional (for parity tests only)

```bash
conda create -n fenicsproject -c conda-forge fenics=2019.1 python=3.13 \
    matplotlib numpy pytest
conda activate fenicsproject
# clone legacy hippylib
git clone https://github.com/hippylib/hippylib.git $HOME/hippylib
export HIPPYLIB_PATH=$HOME/hippylib
```

Then, from `fenicsx`:

```bash
conda activate fenicsx
export HIPPYLIB_PATH=$HOME/hippylib    # legacy checkout
pytest -q hippylibX/time_dependent/tests/
```

If `HIPPYLIB_PATH` is unset or the `fenicsproject` env is missing, the
parity tests skip cleanly; unit tests still run.

## Tests

```bash
conda activate fenicsx
pytest -q hippylibX/time_dependent/tests/
```

`test_parity.py` additionally invokes legacy `hippylib` + dolfin in a
sibling conda env (`fenicsproject`) to check numerical equivalence on a
deterministic problem. The parity tests auto-skip if either env or the
`HIPPYLIB_PATH` env var is unavailable.

### Continuous integration

A minimal GitHub Actions snippet for the unit subset is provided at
`ci/unit_tests.yml` in this directory. It is **not** wired into the
repo-root `.github/workflows/` — copy or include it from there when you
want CI on this subpackage.

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

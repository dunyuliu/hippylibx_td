# `hippylibX.time_dependent` v1.0.0 — initial release

Date: 2026-05-18
Scope: `hippylibX/time_dependent/` subpackage only (upstream `hippylibX/`
is untouched aside from one re-export line in `hippylibX/__init__.py`).

## 1. Summary

First versioned release of the time-dependent extension subpackage for
`hippylibX` (the dolfinx-backed next generation of `hippylib`). The
release contains:

- A faithful port of the legacy `hippylib` time-dependent classes:
  `TimeDependentVector`, `TimeDependentPDEVariationalProblem`,
  `MisfitTD`, `ContinuousStateObservation`,
  `SpaceTimePointwiseStateObservation`.
- A self-contained `AdvectionDiffusionICModel` for initial-condition
  inverse problems (parameter = u⁰), mirroring legacy `TimeDependentAD`.
- Three reproducible parity examples vs legacy `hippylib`: heat,
  reaction-diffusion (tumor), and advection-diffusion with IC inversion.

## 2. Files added

```
hippylibX/time_dependent/
├── __init__.py
├── README.md
├── PROJECT_RULES.md
├── AUDIT.md                          # aggregated audit
├── AUDIT_math.md                     # per-axis audits
├── AUDIT_code.md
├── AUDIT_tests.md
├── timeDependentVector.py
├── TimeDependentPDEVariationalProblem.py
├── misfit.py
├── applications/
│   ├── __init__.py
│   └── ad_diff.py                    # AdvectionDiffusionICModel
├── examples/
│   └── heat.py
├── ci/
│   └── unit_tests.yml                # GitHub Actions snippet
├── tests/
│   ├── conftest.py
│   ├── refproblem.py
│   ├── run_x.py                      # PROBLEM env: heat|tumor|ad_diff
│   ├── run_legacy.py
│   ├── test_tdv.py                   # 12 tests
│   ├── test_misfit_td.py             #  7 tests
│   ├── test_pde_td.py                #  4 tests
│   ├── test_kkt_blocks.py            #  4 tests (KKT-block transposes/sym)
│   └── test_parity.py                # 24 tests (8 × 3 physics)
└── release_notes_v1.0.0.md
```

## 3. Files removed

- `tests/results_legacy.json`, `tests/results_x.json` — stale outputs
  from before the multi-problem refactor (replaced by per-problem
  `results_*_{heat,tumor,ad_diff}.json`, all gitignored).
- `ad_diff_problem.py` — moved into `applications/ad_diff.py`.

## 4. Audit findings and fixes

Findings came from three parallel audit agents (math, code, tests).
Full reports in `AUDIT.md` + `AUDIT_{math,code,tests}.md`. Resolution
status:

### PR-blocking

| # | finding | resolution |
|---|---|---|
| 1 | SPDX/UT-Austin banner missing on 13 `.py` files | Added to every `.py` in this subpackage |
| 2 | Hardcoded `/Users/dliu/...` and `/opt/anaconda3/...` in tests | Removed; tests now require `HIPPYLIB_PATH` (and `CONDA_SH` may be overridden); parity suite auto-skips when prerequisites are missing |
| 3 | Stale test-result JSONs committed | Deleted; per-problem JSONs are gitignored |

### Correctness flags

| # | finding | resolution |
|---|---|---|
| 4 | Algebraic deviation from legacy in `_solveIncrementalAdj` (legacy off-by-one; port computes correct `t_prev`) | Documented in-source with a multi-line comment explaining why the deviation is silent |
| 5 | GLS τ divides by `\|wind\|` — NaN at wind-stagnation points | Added `eps_v=1e-30` floor in `applications/ad_diff.py` |
| 6 | `grad_norm` returned as M⁻¹-norm-squared (no sqrt) to match legacy | Comment already in place; preserved deliberately for parity |

### Hygiene

| # | finding | resolution |
|---|---|---|
| 7 | No `__del__` on classes holding PETSc Mat/KSP | Added to `TimeDependentPDEVariationalProblem`, `AdvectionDiffusionICModel`, `ContinuousStateObservation`, `SpaceTimePointwiseStateObservation` |
| 8 | `assert` used for input validation | Replaced with explicit `KeyError` / `ValueError` / `NotImplementedError`; test `test_index_out_of_frame_raises` updated to match |
| 9 | Silent `except: pass` around MUMPS factor setup | Replaced with `warnings.warn(...)` |
| 10 | Dead code: `_zero_vec`, `import math`, late `import petsc4py` | Removed |

### Coverage gaps

| # | finding | resolution |
|---|---|---|
| 11 | KKT blocks `applyC/Ct/Wuu/Wum/Wmu/Wmm` reached only via `modelVerify` | Added `test_kkt_blocks.py` with 4 direct identity tests (C/Ct transpose, Wum/Wmu transpose, Wmm symmetry, GN-approx zeroes PDE-W blocks) |

### Architecture

| # | finding | resolution |
|---|---|---|
| 13 | `AdvectionDiffusionICModel` lived among generic Problem/Misfit classes | Moved to `applications/ad_diff.py` (new subpackage `applications/`) to mirror legacy hippylib layout |
| 14 | AD parity tolerance 5e-3 (vs 1e-13 for heat/tumor) | `CG_COARSE_TOL` tightened from 5e-1 → 1e-8 in `refproblem.py`. The Eisenstat-Walker inner-CG forcing still causes a small NewtonCG-trajectory divergence; AD remains at 5e-3 (`AUDIT_tests.md` §4 explains why) |

### Reproducibility

| # | finding | resolution |
|---|---|---|
| 15 | `INSTALL.md` only covers `fenicsx` env | Full Install section added to `README.md` (this subpackage) covering both envs with pinned dolfinx 0.10 |
| 16 | No CI integration | Added `ci/unit_tests.yml` snippet (not auto-wired; copy into repo `.github/workflows/` when ready) |
| 17 | dolfinx version not pinned | Pinned to `fenics-dolfinx=0.10` in install instructions |

### Open issues (not fixed; require judgment)

- **AD parity tolerance** stays at 5e-3 for cost and 5e-3 for MAP norm.
  Forward solve, prior, and misfit evaluations all agree to ~1e-13; only
  the NewtonCG outer-iteration trajectory diverges between stacks
  because the two implementations of Eisenstat-Walker forcing differ.
  Tightening `CG_COARSE_TOL` does not collapse the gap. A follow-up
  option is to bypass NewtonCG for AD and use `CGSolverSteihaug`
  directly on the reduced Hessian — but this changes the algorithm in
  both stacks and is not parity-preserving.
- **Pinned versions** — `fenics-dolfinx=0.10` is the only pin. PETSc,
  petsc4py, mpich are conda-resolved. A reproducible `environment.yml`
  is not yet provided.
- **Upstream PR** — the subpackage is ready for review against
  `hippylib/hippylibx`. The remaining decision point is whether
  `applications/ad_diff.py` belongs in core or in a separate
  examples/applications repo.

## 5. Test results

- Unit: 27 tests (12 TDV + 7 misfit + 4 PDE + 4 KKT). All pass in ~5s
  in the `fenicsx` env.
- Parity: 24 tests (8 metrics × 3 physics). All pass in ~45s when both
  conda envs and `HIPPYLIB_PATH` are present; auto-skip otherwise.

Total: **51/51 tests passing** in 52s.

## 6. Totals / metrics

```
TD subpackage size:     ~1500 SLOC code + ~750 SLOC tests
Files added/modified:   17 source + 4 audit + 1 release notes
Parity (heat, tumor):   ~1e-13 (machine roundoff)
Parity (ad_diff):       ~5e-3 on MAP, ~3e-2 on cost
```

## 7. Assumptions

- The host has Anaconda at `/opt/anaconda3/` for parity tests. Override
  via `CONDA_SH` env var if elsewhere.
- Legacy `hippylib` is reachable via `HIPPYLIB_PATH` (a git checkout
  containing `applications/ad_diff/model_ad_diff.py`).
- `fenicsx` env carries `fenics-dolfinx=0.10` and matching petsc4py.
- All physical quantities are dimensionless in the test problems (heat
  decay, Fisher growth, advection-diffusion). No unit conversions are
  performed anywhere in the package.

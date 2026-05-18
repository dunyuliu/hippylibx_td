# Project rules — `hippylibX.time_dependent`

Rules the release workflow audits against. Keep this file short.

## Scope

This subpackage lives at `hippylibX/time_dependent/` inside the
`hippylibx_td` fork. **Anything outside this directory is upstream and
must not be modified.** Releases described in `release_notes_v*.md`
here apply only to this subpackage.

## File layout

- `__init__.py` — public re-exports only; no logic.
- `timeDependentVector.py` — `TimeDependentVector` + `_TDVArrayProxy`.
- `TimeDependentPDEVariationalProblem.py` — generic TD PDE problem.
- `misfit.py` — `ContinuousStateObservation`, `MisfitTD`,
  `SpaceTimePointwiseStateObservation`, `_PointwiseTDStorage`.
- `applications/` — Model-style classes for specific problems
  (e.g. `ad_diff.py: AdvectionDiffusionICModel`).
- `tests/` — pytest suite (`test_*.py`), parity runners
  (`run_x.py`, `run_legacy.py`, `refproblem.py`), `conftest.py`.
- `examples/` — runnable tutorials.
- `ci/` — workflow snippets (not auto-wired).
- `docs/` — archived release notes.
- `release_notes_v*.md` — latest release notes at the subpackage root;
  older ones live in `docs/`.

## Source-file rules

- Every `.py` file (except `tests/results_*.json` artifacts) carries
  the upstream `# ---bc-` / `SPDX-License-Identifier: GPL-2.0-only`
  banner. Module docstrings live below the banner.
- No `/Users/dliu/`, `/opt/anaconda3/`, or other maintainer-specific
  paths anywhere in code that ships. Tests may rely on env vars
  (`HIPPYLIB_PATH`, `HIPPYLIBX_BASE_DIR`, `CONDA_SH`,
  `FENICSPROJECT_ENV`) and must skip cleanly when those are unset or
  point at missing locations.
- PETSc / dolfinx objects allocated in `__init__` must be destroyed in
  `__del__`. Mirror the pattern in `hippylibX/modeling/PDEProblem.py`.
- Use explicit `raise KeyError/ValueError/NotImplementedError` for
  input validation. Never use `assert` for validation — `python -O`
  strips them.

## Test rules

- The unit suite (`test_tdv.py`, `test_misfit_td.py`, `test_pde_td.py`,
  `test_kkt_blocks.py`) must run in the `fenicsx` env alone, with no
  legacy dependency.
- The parity suite (`test_parity.py`) requires `fenicsx` +
  `fenicsproject` envs and `HIPPYLIB_PATH` pointed at a legacy hippylib
  checkout. It must auto-skip cleanly when prerequisites are missing.
- Per-problem parity tolerances are declared in `test_parity.py`. Heat
  and tumor target FP-roundoff (~1e-13). AD-diff is looser (~5e-2 on
  cost, ~5e-3 on MAP norm) because the NewtonCG inner-CG
  Eisenstat-Walker forcing diverges slightly between stacks.
- Test result JSONs (`tests/results_*.json`) are not committed.
- `refproblem.py` keeps the deterministic test problem definitions;
  random sampling and unseeded `parRandom` are forbidden in tests.

## Reproducibility rules

- `README.md` must list the exact conda commands for `fenicsx` (and
  optionally `fenicsproject`) and pin dolfinx to a major.minor.
- Examples must not hardcode absolute paths.

## Release rules

- Patch bumps: source polish, test additions, doc updates.
- Minor bumps: new public class or method, new application under
  `applications/`, breaking-change to a private helper.
- Major bumps: any breaking change to the public API listed in
  `__init__.py`.
- Release notes describe the final post-audit filesystem state, not
  the raw git diff. Findings that need human judgment are listed under
  open issues.

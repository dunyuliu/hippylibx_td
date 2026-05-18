# Code Audit: `hippylibX/time_dependent/`

## Executive summary (5 lines)

1. **Top win:** the `.array[:]` proxy on `TimeDependentVector` lets the upstream `Model` / `ReducedHessian` / `modelVerify` machinery work *unchanged* — the subpackage adds files without patching existing hippylibX code, which is exactly what an upstreamable extension should do.
2. **Top win:** parity tests (heat / tumor / ad_diff) drive a real numerical comparison against legacy hippylib in a sibling conda env; this is more rigorous than typical PR test coverage.
3. **Top risk (blocking PR):** every file in `time_dependent/` is missing the upstream SPDX/copyright banner (`# SPDX-License-Identifier: GPL-2.0-only` + UT‑Austin block). Upstream `hippylibX/modeling/__init__.py:1-8` shows the canonical banner; none of the new files have it.
4. **Top risk (blocking PR):** `tests/test_parity.py:21-26` and `tests/run_legacy.py:20,171` hardcode `/Users/dliu/...` as fallback defaults; this will fail on every other developer's machine even though the env-var override exists.
5. **Top risk (architectural):** `AdvectionDiffusionICModel` is a *Model* (owns prior + misfit), unlike its file neighbours which are *Problems* / *Misfits*; co-locating it under the same `__init__.py` re-exports as `TimeDependentPDEVariationalProblem` is the strongest naming inconsistency in the package and will confuse reviewers.

---

## 1. Public API surface (`__init__.py`)

`time_dependent/__init__.py` exports four symbols:

```
TimeDependentVector,
TimeDependentPDEVariationalProblem,
MisfitTD, ContinuousStateObservation, SpaceTimePointwiseStateObservation,
AdvectionDiffusionICModel
```

Issues:

- **No SPDX/copyright banner.** Compare with `hippylibX/modeling/__init__.py:1-8`. The banner is `# --------------------------------------------------------------------------bc-` (NOT `bc-` as in the current `__init__.py:1` — that one uses 74 dashes; the new one uses 74 too but the rest of the new files have *no* banner at all, see §4). The new `__init__.py:1-20` uses the banner *style* but replaces the copyright line with a freeform usage docstring. For an upstream PR, replace with the standard copyright/SPDX line + a short module docstring *below* it (the `"""..."""` form, not `#`).
- **`AdvectionDiffusionICModel` should not be re-exported here.** It is a different abstraction (Model, owns prior+misfit) from the other three (Problem / Misfit / TDV container). See §2.
- Private helpers are correctly underscore-prefixed (`_TDVArrayProxy`, `_PointwiseTDStorage`, `_zero_vec`) and *not* re-exported. Good.

## 2. Module structure & boundaries

- `TimeDependentPDEVariationalProblem`, `MisfitTD`, `ContinuousStateObservation`, `SpaceTimePointwiseStateObservation` all depend on `TimeDependentVector` only via `view(t)` / `store` / `retrieve` / `zero` / the `.array[:]` proxy. Coupling is appropriately narrow.
- `_PointwiseTDStorage` (`misfit.py:194-214`) is duplicating a *subset* of `TimeDependentVector`'s API. Two paths forward, both <1 day:
  1. Drop it and reuse `TimeDependentVector` for the row-space (B-image) storage too — but `TDV.initialize(Vh)` is keyed on a FunctionSpace, not a PETSc row map, so this requires a tiny refactor of `TDV.initialize` to optionally accept `(index_map, bs)`.
  2. Keep `_PointwiseTDStorage` but move it out of `misfit.py` to its own private file `_sensor_storage.py` so `misfit.py` stays focused on misfits.
- **`AdvectionDiffusionICModel` does not belong next to generic classes.** Legacy hippylib places this under `applications/ad_diff/`. The straightforward port:
  - Create `time_dependent/applications/__init__.py` and `time_dependent/applications/ad_diff.py` containing `AdvectionDiffusionICModel`.
  - Drop it from `time_dependent/__init__.py`; users would do `from hippylibX.time_dependent.applications.ad_diff import AdvectionDiffusionICModel`. This mirrors legacy layout.

## 3. Style consistency with upstream

| convention | upstream `modeling/` | `time_dependent/` |
|---|---|---|
| `--bc-` / `--ec-` banner with SPDX + UT‑Austin copyright | yes (all files) | **only** `__init__.py` has the banner shape, and even there the copyright is replaced by a usage docstring. All other 4 source files: **none** |
| module-level docstring | not used (banner only) | yes (`"""..."""` at top of each file) |
| `import hippylibX as hpx` inside the package | `modeling/misfit.py:10` does this | not used here (correctly avoids cyclic import) |
| type hints | inconsistent in upstream | richer here (the new files annotate most signatures — *better* than upstream, no issue) |
| docstring style | sparse, RST-ish | NumPy/Markdown blend; OK |

**Action (≤1 day):** prepend the standard banner to every file under `time_dependent/` (including tests/ and examples/heat.py). Keep the existing module docstring directly under the banner.

## 4. Dead code / unused imports

- `TimeDependentPDEVariationalProblem.py:22-24` — `_zero_vec(v)` is defined and used nowhere. Grep confirms. **Delete.**
- `ad_diff_problem.py:16` — `import math` is unused (only `np.exp`, `dlx.fem.Constant`, `ufl.*` used). **Delete.**
- `misfit.py:10` — `import numpy as np` is used only at `misfit.py:126` (`np.asarray(targets, dtype=np.float64)`); legitimate but borderline. Keep.
- `misfit.py:14` — `import dolfinx.fem.petsc` is unused in this file (no `dolfinx.fem.petsc.*` reference after removing imports that go through `dlx.fem.petsc` ... actually `dolfinx.fem.petsc` *is* unused: grep for `dolfinx.fem.petsc` in `misfit.py` returns only `assemble_matrix` at line 32 which uses `dolfinx.fem.petsc.assemble_matrix`. Wait — line 32 *does* reference it. Keep.)  Verified: keep.
- `_PointwiseTDStorage` (`misfit.py:194-214`) — referenced exactly once at `misfit.py:132`. Used. Keep (or refactor, §2).
- `petsc4py` import in `timeDependentVector.py:185` is a function-local import inside `norm()`. Move it to module top-level for consistency with the other modules.

## 5. Hardcoded paths

| location | issue |
|---|---|
| `tests/test_parity.py:21-26` | `HIPPYLIBX_BASE_DIR` falls back to `/Users/dliu/scratch/visco_inversion/src/hippylibx`; `HIPPYLIB_PATH` falls back to `/Users/dliu/scratch/visco_inversion/src/hippylib`. |
| `tests/run_legacy.py:20,171` | Same maintainer-specific paths embedded as fallbacks. |

**Proposed portable defaults (<1 hr fix):**

```python
HIPPYLIBX_BASE_DIR = os.environ.get(
    "HIPPYLIBX_BASE_DIR",
    os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..")),  # repo root
)
HIPPYLIB_PATH = os.environ.get("HIPPYLIB_PATH")  # no default
```

The legacy parity tests should `pytest.skip(...)` cleanly when `HIPPYLIB_PATH` is unset or the `fenicsproject` conda env is missing, instead of hard-failing. `tests/test_parity.py:43-63` already throws `RuntimeError` on subprocess failure — wrap that into a skip when the cause is a missing env/path.

`examples/heat.py:27-30` and `tests/conftest.py:10-14` use relative repo-root resolution. Good.

## 6. Test artifacts in tree

Files present that should be regeneratable, not committed:

```
tests/results_legacy.json            (older, no problem suffix — STALE)
tests/results_x.json                 (older, no problem suffix — STALE)
tests/results_legacy_heat.json
tests/results_legacy_tumor.json
tests/results_legacy_ad_diff.json
tests/results_x_heat.json
tests/results_x_tumor.json
tests/results_x_ad_diff.json
```

`git check-ignore` confirms **none of them are ignored** (only `__pycache__/` is in `.gitignore`).

**Fixes (<1 hr):**

1. Add to `.gitignore` at repo root:
   ```
   hippylibX/time_dependent/tests/results_*.json
   ```
2. Delete the stale `results_legacy.json` and `results_x.json` (no problem suffix). These pre-date the multi-problem refactor where the runners switched to `results_{stack}_{problem}.json` (see `tests/test_parity.py:81`); the suffix-less files are dead artifacts.

## 7. `.array[:]` proxy on `TimeDependentVector` — contract

Implementation: `timeDependentVector.py:13-78` (`_TDVArrayProxy`), exposed as `TimeDependentVector.array` property at line 202-204.

**Contract as currently implemented:**

- `tdv.array[:] = ndarray` — full-slice assignment with a 1-D array of length `sum_i len(tdv.data[i].array)`. Writes are sliced contiguously per snapshot in time order; each snapshot then `scatter_forward()`s. Partial-slice assignment raises `NotImplementedError` (`timeDependentVector.py:36-39`).
- `tdv.array[:] = other_tdv.array` — proxy-to-proxy copy, snapshot-by-snapshot. Asserts equal `nsteps`.
- `tdv.array + scalar`, `tdv.array * other.array`, etc. — returns a *concatenated* `np.ndarray`; the result is no longer connected to the TDV. This is what makes `out.array[:] += tmp.array` work in upstream Newton-CG inner loops.
- `np.asarray(tdv.array)` returns the concatenation (`__array__` at line 25-27).

**Surprise factor for new readers:** medium-high. A property called `.array` that returns a *proxy* (not a real ndarray) and that mutates state on `[:] = ...` is unusual; it's only obvious if you read `_TDVArrayProxy` first. The `.array` property name is chosen specifically because upstream code reads/writes `.array[:]` on `dolfinx.la.Vector`, so a TDV must mimic that — this is *load-bearing duck typing*. Renaming to `.flat` would break the duck-type and require model-level shims, which is the very thing the design avoids.

**Recommendations (≤1 day each):**

1. **Document the contract** explicitly in the `TimeDependentVector` class docstring (currently `timeDependentVector.py:82-86` is one line; expand to ~15 lines describing what `.array` does, including the "duck-types `dolfinx.la.Vector.array`" rationale).
2. Tighten `__setitem__` to assert that `value` has the expected total length when it's an ndarray — currently `arr[offs:offs+n]` silently produces empty slices on mismatch.
3. Consider adding `__iadd__` / `__isub__` to `_TDVArrayProxy`. Currently only `__imul__` (`timeDependentVector.py:74-78`) is defined; for parity with `dolfinx.la.Vector.array` semantics in upstream Newton-CG steps, `out.array[:] += ...` works because `+` returns an ndarray and then `[:] =` is called, but `out.array += ...` (no `[:]`) would silently break since it would rebind `out.array` (a property!) — though in practice that path isn't exercised. Document the exclusion.

## 8. Naming

`time_dependent/__init__.py:23-31` re-exports:

- `TimeDependentPDEVariationalProblem` — a *PDEProblem* (`solveFwd`, `solveAdj`, `applyC/Ct/Wuu/...`, no prior/misfit)
- `AdvectionDiffusionICModel` — a *Model* (owns `prior`, `misfit`; has `cost(x)`, `setPointForHessianEvaluations`, no `applyC` block API)

Co-exporting both as peers under a package called `time_dependent` is confusing: the Model-vs-Problem distinction is exactly what upstream's `modeling/` separates (`PDEVariationalProblem` vs `Model`). Suggested renames + moves (≤1 day total):

- Keep `TimeDependentPDEVariationalProblem` as-is.
- Rename `AdvectionDiffusionICModel` → keep the name but move to `time_dependent/applications/ad_diff.py` (see §2). The class name "Model" is fine; the file location is the issue. After the move, the only re-export at the package level should be the generic Problem/TDV/Misfit symbols.

## 9. Error handling

- `_TDVArrayProxy.__setitem__` (`timeDependentVector.py:36-39`) — raises `NotImplementedError` cleanly for non-full slices. Good.
- `TimeDependentVector._index` (`timeDependentVector.py:116-123`) — `assert` instead of explicit exception; if a user runs with `python -O` the assert becomes a silent no-op and a wrong snapshot is returned. **Replace with explicit `raise ValueError`.**
- `ContinuousStateObservation.apply_ij` (`misfit.py:97-103`) — silently zeros `out` for any (i,j) other than (STATE,STATE). Mirrors legacy hippylib, but **not documented**. Add one line to the docstring: "cross blocks (STATE,PARAM), (PARAM,STATE), (PARAM,PARAM) are zero".
- `SpaceTimePointwiseStateObservation.apply_ij` (`misfit.py:178-191`) — same silent zero-out for non-(STATE,STATE). Same documentation gap.
- `MisfitTD.apply_ij` (`misfit.py:289-301`) — explicitly raises `IndexError` for the (ADJOINT, *) blocks. Good, but inconsistent with the per-time misfits above.
- `AdvectionDiffusionICModel.applyWum / applyWmu / applyWmm` (`ad_diff_problem.py:304-311`) — silently zero out. Legitimate (model is linear in m), but should carry a one-line comment explaining why.
- `AdvectionDiffusionICModel._make_lu` (`ad_diff_problem.py:123-126`) — bare `except Exception: pass` around `pc.setFactorSolverType("mumps")`. This will mask serious errors; narrow the except to `PETSc.Error` or log a warning when MUMPS isn't available.
- `examples/heat.py:233-242` — wraps XDMF write in `try/except Exception as e: print(...)`. Acceptable in a tutorial but the bare `Exception` is too broad.

## 10. `__del__` / resource cleanup

Compare with `hippylibX/modeling/PDEProblem.py:64-80` which has a `__del__` that destroys `Wuu/Wmu/Wum/Wmm`.

- `TimeDependentPDEVariationalProblem` (this package) — owns four KSP objects (`solverA`, `solverAadj`, `solver_fwd_inc`, `solver_adj_inc`) created in `_createLUSolver` (`TimeDependentPDEVariationalProblem.py:102-112`). **No `__del__`.** PETSc Mats `A`, `Aadj`, `Ainc` are destroyed inside the time loops (good), but the KSPs leak until interpreter shutdown.
- `AdvectionDiffusionICModel` — owns `self.M`, `self.M_stab`, `self.Mt_stab`, `self.L`, `self.Lt`, `self.solver`, `self.solvert` (all assembled once in `__init__`, `ad_diff_problem.py:87-109`). **No `__del__`.** These persist for the lifetime of the model; in a long-running inversion driver that creates / discards multiple model instances, this leaks PETSc memory.
- `ContinuousStateObservation.W` (`misfit.py:32`) — also no `__del__`.
- `SpaceTimePointwiseStateObservation` — owns `self.B`, `self._Bu`, and `self._d_petsc` list. No `__del__`.

**Fix (<1 day):** add a `__del__` to each of the four classes following the `modeling/PDEProblem.py:64-80` pattern. Guard each destroy with an `is not None` / try-except since attributes may not be set if `__init__` raised partway through.

## 11. Misc

- `ad_diff_problem.py:46-48` — `assert len(self.simulation_times) >= 2` — should be `ValueError`.
- `ad_diff_problem.py:87-92` — chained `.assemble()` calls on one line via `;` (`self.M = ...; self.M.assemble()`). Style-inconsistent with the rest of the package which keeps one statement per line. ≤30 min reflow.
- `tests/run_x.py:212` — `# noqa: SLF001` suppression for `misfit._Bu` private access. The cleaner fix is to give `SpaceTimePointwiseStateObservation` a public helper `_set_data_from_state(self, t, u_vec)` that does the `B*u` step internally, then drop the noqa.
- `tests/conftest.py` has no SPDX banner. Same fix as §3.
- `TimeDependentPDEVariationalProblem.solveFwd` nonlinear branch (`TimeDependentPDEVariationalProblem.py:164-203`) inlines a hand-rolled Newton loop rather than delegating to a `dolfinx.nls`-style nonlinear solver. Acceptable since it mirrors legacy hippylib exactly, but worth a one-line comment so reviewers don't suggest replacing it with `NonlinearProblem`.

---

## Concrete change list (each <1 day)

1. Add SPDX/UT‑Austin banner to every file in `time_dependent/` (incl. tests + examples). — 1 hr
2. Delete `_zero_vec` (`TimeDependentPDEVariationalProblem.py:22-24`) and `import math` (`ad_diff_problem.py:16`). — 5 min
3. Move `AdvectionDiffusionICModel` to `time_dependent/applications/ad_diff.py`; remove from package `__init__.py`. — 1 hr
4. Replace `assert` in `TimeDependentVector._index` and `ad_diff_problem.__init__` with `ValueError`. — 15 min
5. Add `__del__` cleanup to `TimeDependentPDEVariationalProblem`, `AdvectionDiffusionICModel`, `ContinuousStateObservation`, `SpaceTimePointwiseStateObservation`. — 2 hr
6. Replace hardcoded `/Users/dliu/...` fallbacks in `tests/test_parity.py:21-26` and `tests/run_legacy.py:20,171` with repo-relative defaults + `pytest.skip` when legacy env unavailable. — 1 hr
7. Add `hippylibX/time_dependent/tests/results_*.json` to `.gitignore`; `git rm` the eight committed result files (including the two stale unsuffixed ones). — 15 min
8. Document the `.array` proxy contract in `TimeDependentVector` docstring; document `apply_ij` cross-block zeroing in both pointwise/continuous observation classes. — 1 hr
9. Move `import petsc4py` in `timeDependentVector.py:185` to module top-level. — 1 min
10. Narrow `except Exception` in `ad_diff_problem.py:125`. — 5 min

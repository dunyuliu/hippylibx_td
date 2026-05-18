# `hippylibX.time_dependent` — Audit

Date: 2026-05-18
Target: `/Users/dliu/scratch/hippylibx_td/hippylibX/time_dependent/`
Reviewed by: three parallel audit agents (math/numerics, code/API, tests/reproducibility).

Full sub-reports: `AUDIT_math.md`, `AUDIT_code.md`, `AUDIT_tests.md`.

---

## Executive summary

**Top 3 wins**

1. The mathematical port is faithful: every KKT block (`applyC/Ct/Wuu/Wum/Wmu/Wmm`), the adjoint backward time-loop, and the IC-inversion gradient derive correctly from the Lagrangian (`AUDIT_math.md` §1–§3). Heat & tumor parity vs legacy `hippylib` is at floating-point roundoff (~1e-13).
2. The `.array[:]` proxy on `TimeDependentVector` (`timeDependentVector.py:_TDVArrayProxy`) is the architectural keystone — it lets upstream `Model` / `ReducedHessian` / `modelVerify` operate on TDVs **without modification**. The expected operations (`a[:] = b`, `a[:] *= c`, `a[:] += b.array`) all dispatch correctly per-snapshot.
3. All three legacy TD applications (heat, tumor, AD-diff) reproduce in the port with passing parity tests (47/47 = 23 unit + 24 parity), and the suite runs end-to-end in ~30 s.

**Top 3 risks (PR-blocking before upstream)**

1. **Copyright/license banners** — every source file (except `__init__.py`) is missing the `# ----bc-` / SPDX banner present on every upstream `hippylibX/modeling/*.py` (`AUDIT_code.md` §1.B).
2. **Hardcoded paths** — `tests/test_parity.py:21-26` and `tests/run_legacy.py:20,171` fall back to maintainer-specific paths (`/Users/dliu/scratch/visco_inversion/...`, `/opt/anaconda3/...`) without graceful skip when those locations are missing (`AUDIT_code.md` §3, `AUDIT_tests.md` §3).
3. **Test result JSONs committed** — `tests/results_*.json` are not in `.gitignore` scope (the repo `.gitignore:169` matches but they were committed previously); two stale pre-refactor files (`results_legacy.json`, `results_x.json`) sit in the tree.

**One latent-deviation finding (not a bug, worth noting)**

The port silently *fixes* an off-by-one bug in legacy `_solveIncrementalAdj` (legacy `times[it-1]` wraps to `times[-1]` on the first reverse step; port uses an explicit `t_prev` lookup at `TimeDependentPDEVariationalProblem.py:386-388`). Either revert (strict line-by-line) or add a comment explaining why we diverge. See `AUDIT_math.md` §2.

---

## 1. Goal & implementation

**Goal**: provide a faithful port of legacy `hippylib`'s time-dependent inverse-problem machinery to `dolfinx`, packaged as the isolated subpackage `hippylibX.time_dependent`, with one re-export line into `hippylibX/__init__.py`.

**End-to-end flow** (heat tutorial; `examples/heat.py`):

```
nx,ny=64,64 ; nt=20 ; T=1 ; γ=0.1, δ=0.5
→ build [Vh2,Vh1,Vh2] (heat.py:61-63)
→ HeatEquationVarf θ=1 (heat.py:96-115)
→ TimeDependentPDEVariationalProblem (heat.py:120)
→ BiLaplacianPrior(γ,δ, robin_bc=True) (heat.py:133)
→ pde.solveFwd at m_true sampled from prior (heat.py:148)
→ ContinuousStateObservation per t (heat.py:155-162)
→ MisfitTD wrapping per-time obs (heat.py:163)
→ hpx.Model(pde, prior, misfit) — upstream Model, no shim (heat.py:169)
→ hpx.modelVerify (FD gradient + Hessian symmetry) (heat.py:178)
→ hpx.ReducedSpaceNewtonCG.solve → MAP (heat.py:219-220)
```

The legacy heat tutorial maps 1-to-1 onto this flow.

**README claims vs implementation**: all four classes claimed in `README.md` are implemented (`TimeDependentVector`, `TimeDependentPDEVariationalProblem`, `ContinuousStateObservation` + `MisfitTD`, `SpaceTimePointwiseStateObservation`, `AdvectionDiffusionICModel`). No dead claims.

## 2. Inventory & stale items

| path | role | status |
|---|---|---|
| `__init__.py` | public re-exports | ✓ keep |
| `timeDependentVector.py` | snapshot container + numpy-proxy | ✓ keep |
| `TimeDependentPDEVariationalProblem.py` | generic TD PDE problem | ✓ keep |
| `misfit.py` | `ContinuousStateObservation`, `MisfitTD`, `SpaceTimePointwiseStateObservation`, `_PointwiseTDStorage` | ✓ keep |
| `ad_diff_problem.py` | `AdvectionDiffusionICModel` (model-like) | **move** — should live under `time_dependent/applications/ad_diff.py` to mirror legacy layout (`AUDIT_code.md` §2) |
| `examples/heat.py` | tutorial | ✓ keep |
| `tests/conftest.py`, `tests/refproblem.py`, `tests/run_x.py`, `tests/run_legacy.py`, `tests/test_*.py` | test suite | ✓ keep |
| `tests/results_legacy.json`, `tests/results_x.json` | **stale** — pre-multi-problem refactor | **delete** (`AUDIT_tests.md` §6) |
| `tests/results_x_{heat,tumor,ad_diff}.json`, `tests/results_legacy_{...}.json` | last parity-test outputs | regenerated on each run — gitignore (`AUDIT_code.md` §3) |
| `AUDIT_math.md`, `AUDIT_code.md`, `AUDIT_tests.md`, this file | audit artifacts | gitignore or commit alongside release notes |

## 3. Reproducibility

- **From a clean checkout**: the unit tests reproduce if `fenicsx` env exists; the parity tests additionally need `fenicsproject` (legacy `dolfin` + `hippylib`). Only `fenicsx` setup is documented in `INSTALL.md`. `README.md:65-71` mentions the second env but with no pin or install spec.
- **Hardcoded paths**: `tests/test_parity.py:21-26` (`CONDA_SH`, `HIPPYLIBX_BASE_DIR`, `HIPPYLIB_PATH`) default to maintainer paths. Env-var override works but missing env → `RuntimeError`, not `pytest.skip`.
- **Sample inputs**: none required (problems are analytic). Good.
- **Randomness**: deterministic. `refproblem.py:29-30` uses zero noise; the only RNG in tests is `default_rng(0)` in `test_misfit_td.py:66`. `examples/heat.py` uses unseeded `parRandom` but is not a test.
- **Pinned versions**: not present. `INSTALL.md` says "dolfinx v0.10" but doesn't pin a specific patch version, hippylibX upstream SHA, or petsc4py.

## 4. Physics & numerics

- **Units**: this is method-test code (heat eq, AD-diff, tumor toy). No units enforced or needed.
- **Sign conventions**: `solveAdj` requires the caller to pre-negate the misfit gradient (legacy and upstream Model both do this in `model.py:136`). Confirmed in code; no inversion bug observed.
- **Magic numbers** that should be exposed:
  - `kappa = 1e-3` in `ad_diff_problem.py:41,202` — diffusivity for AD; only used in the AD problem, but if anyone wants a different regime they have to edit source.
  - GLS τ formula `min(h²/(2κ), h/|v|)` at `ad_diff_problem.py:65` — matches legacy verbatim.
- **Physical bounds**: heat solution can stay negative (no positivity enforcement) — same as legacy. Acceptable.
- **NaN/Inf risk**: `ad_diff_problem.py:65` divides by `|wind|` — undefined at wind-stagnation points. Not triggered with the constant non-zero wind used in tests. Suspected risk if anyone uses a wind that vanishes locally. (`AUDIT_math.md` §4)
- **Conservation**: heat eq with zero source and homogeneous Dirichlet BC should give monotone decrease of `‖u‖²_M` (energy dissipation). Currently checked qualitatively in `test_pde_td.py::test_forward_with_nontrivial_ic_decays_in_time` via L²-norm decrease. A stricter `M`-norm energy test would catch sign errors in the diffusion term. (`AUDIT_math.md` §6)
- **Adjoint algebra**: every KKT block in `TimeDependentPDEVariationalProblem.py` matches legacy line-for-line. Spot-checked `applyC`, `applyCt`, `applyWuu`, `applyWum`, `applyWmu`, `applyWmm`, `evalGradientParameter`, `solveAdj`. See `AUDIT_math.md` §1–§3.
- **AD-diff IC gradient**: `evalGradientParameter` returns `R(m·) − Mt_stab · p^1` where `p^1` is the adjoint at `simulation_times[1]`. Derivation in `AUDIT_math.md` §3 confirms this is the correct Lagrangian gradient when IC = parameter.
- **`grad_norm = mg · Msolver(mg)` (no sqrt)** in `ad_diff_problem.py:225-228` — deliberate match to legacy `TimeDependentAD.evalGradientParameter` to keep NewtonCG convergence quantitatively identical. Flagged as known idiosyncrasy.

## 5. Implementation consistency

- **Legacy port faithfulness**: the math is line-by-line, with **one deliberate deviation** (the `_solveIncrementalAdj` off-by-one fix at `TimeDependentPDEVariationalProblem.py:386-388`). Either revert for strict parity or add an inline comment. (`AUDIT_math.md` §2)
- **Two preserved legacy approximations**: (a) `solveAdj` differentiates `form(t_n)` w.r.t. `u_old` instead of `form(t_{n+1})`, (b) `solveAdj` never updates `u_old`. Both correct for all three apps' BDF1 mass-matrix coupling. (`AUDIT_math.md` §2)
- **Public API surface vs upstream conventions**: re-exports in `__init__.py` are clean. `AdvectionDiffusionICModel` is a model-like class living next to problem/misfit classes — type mismatch with the rest of the package. Should move into `applications/` subdir (`AUDIT_code.md` §2).
- **Naming consistency**: PDE & TDV use camelCase classes + camelCase files (`TimeDependentPDEVariationalProblem.py`). Upstream uses both (`PDEProblem.py`, `pointwiseInterpolationMatrix.py`). Inconsistent, both within upstream and our addition — not a regression.
- **`grad_norm` square vs sqrt**: `AdvectionDiffusionICModel.evalGradientParameter` returns the M⁻¹-norm-squared (no sqrt) to match legacy `TimeDependentAD`. Upstream `hpx.Model.evalGradientParameter` returns sqrt. Documented but worth a code comment. (`AUDIT_math.md` §3, `AUDIT_tests.md` §4)
- **Resource cleanup**: `TimeDependentPDEVariationalProblem` (4 KSPs), `AdvectionDiffusionICModel` (2 KSPs, 5 Mats), `ContinuousStateObservation` (1 Mat), `SpaceTimePointwiseStateObservation` (1 Mat + N Vecs) — none implement `__del__`. Upstream `PDEVariationalProblem.__del__` at `modeling/PDEProblem.py:64-80` is the template. (`AUDIT_code.md` §5)

## 6. Logging & error handling

- **Error pathways**:
  - `TimeDependentVector._index` uses `assert` to validate times — disappears under `python -O`. Replace with explicit `KeyError`/`ValueError`. (`AUDIT_code.md` §4.A)
  - `ContinuousStateObservation._check_nv` / `SpaceTimePointwiseStateObservation._check_nv` raise loudly on `noise_variance == 0` or `None`. Good.
  - `apply_ij` for non-`(STATE, STATE)` cross-blocks in pointwise misfit silently zeros `out` — semantically correct for the misfit role, but undocumented. Add a one-line docstring.
- **Silent fall-throughs**:
  - `ad_diff_problem.py:125` has `try: pc.setFactorSolverType("mumps") except Exception: pass`. If MUMPS isn't available, we silently fall back to whatever LU PETSc picks. Should at least log a warning.
- **No application-level logging**: numerical methods rely on Newton-CG's own printer (`print_level=-1` in parity tests). Acceptable.

## 7. Performance & scaling

- Test suite total runtime ~30 s; ~90% spent in subprocess parity (`tumor` parity dominates at ~15 s due to nonlinear Newton in both stacks). (`AUDIT_tests.md` §7)
- The TD machinery itself is per-timestep assembly inside `solveFwd/Adj/Incremental`. Each step assembles A and b via `dolfinx.fem.petsc`; that's the same pattern as upstream `hippylibX.PDEVariationalProblem` and legacy `TimeDependentPDEVariationalProblem`. No regression.
- Memory: `TimeDependentVector` holds `nsteps × ndofs` doubles per state/adjoint. For the AD problem at NX=NY=16, that's negligible. Will scale linearly in `nt`.

## 8. Top-N priorities (actionable, <1 day each)

1. **Add SPDX/UT-Austin banners** to all six source files. Copy-paste from any `hippylibX/modeling/*.py`. (PR-blocking. ~15 min.)
2. **Delete stale `tests/results_legacy.json` and `tests/results_x.json`**; ensure the per-problem `results_*_*.json` files are correctly gitignored. (~5 min.)
3. **Add `__del__` cleanup** to `TimeDependentPDEVariationalProblem`, `AdvectionDiffusionICModel`, both observation misfits — mirror `PDEVariationalProblem.__del__`. (~30 min.)
4. **Replace asserts with explicit errors** in `TimeDependentVector._index`. (~10 min.)
5. **Document the `.array[:]` proxy** in the `TimeDependentVector` docstring — this is the architectural keystone of the port and a future-reader will be surprised by it. (~15 min.)
6. **Document the `grad_norm` no-sqrt idiosyncrasy** in `AdvectionDiffusionICModel.evalGradientParameter` with an inline comment that points at legacy line numbers. (~10 min.)
7. **Add `pytest.skip` for missing parity env** in `tests/test_parity.py` so the unit suite still runs cleanly when `fenicsproject` isn't installed. (~30 min.)
8. **Pin dolfinx version** in `INSTALL.md` and add the `fenicsproject` env spec (or a one-paragraph "to run parity tests, also install legacy FEniCS via ..."). (~30 min.)
9. **Move `AdvectionDiffusionICModel` to `time_dependent/applications/ad_diff.py`** to match legacy hippylib's layout (applications subpackage). Drop from top-level re-exports if you keep one source-of-truth. (~30 min.)
10. **Add direct unit tests for the Hessian KKT blocks** (compare `applyC.mult` against an FD approximation; same for `applyCt`, `applyWuu` on a tiny problem). Currently all are reached only via `modelVerify`. (~2 hours.)
11. **(Optional, tighten AD parity)** — set `CG_COARSE_TOL` from 5e-1 → 1e-8 in `refproblem.py:37` to force both inner CGs into the asymptotic regime; may collapse the 5e-3 vs 1e-13 gap. Run-and-see. (~15 min.)

---

*Companion sub-reports*: `AUDIT_math.md` (gradient/Hessian algebra, line-by-line port verification), `AUDIT_code.md` (API surface, dead code, resource hygiene), `AUDIT_tests.md` (coverage table, reproducibility, time budget).

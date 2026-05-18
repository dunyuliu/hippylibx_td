# AUDIT — `hippylibX/time_dependent/` test suite & reproducibility

## Executive summary
- 47 tests pass in ~33 s wall-clock; the suite is dominated entirely by the three subprocess parity fixtures (~30 s of the 33 s), and the in-process unit suite is fast (<3 s).
- Direct unit coverage of `TimeDependentPDEVariationalProblem` is **indirect**: `solveFwd` (linear branch) and the gradient/Hessian KKT machinery are exercised via `hpx.modelVerify` in `test_pde_td.py`, but no test invokes `evalGradientParameter`, `solveAdj`, `solveIncremental`, `applyC/Ct/Wuu/Wum/Wmu/Wmm` by name — they are reached only transitively.
- The nonlinear forward branch (`is_fwd_linear=False`) and `AdvectionDiffusionICModel` have **zero direct unit tests** — only parity tests cover them (`refproblem.py` tumor + ad_diff).
- Parity infra is brittle for CI: it hard-codes `/opt/anaconda3/etc/profile.d/conda.sh` and `/Users/dliu/scratch/visco_inversion/...` paths (env-overridable but with no `pytest.skip` fallback when envs are absent).
- Reproducibility: `refproblem.py` is fully deterministic (zero noise, analytic `m_true`); the only RNG in the test tree is a seeded `np.random.default_rng(0)` in `test_misfit_td.py:66`. `examples/heat.py` uses `parRandom`/`prior.sample` without a seed — not a test concern but the example is non-deterministic across runs.

## 1. Coverage by class/method

Citations are `file:line`. "Indirect via modelVerify" means the symbol is reached only through `hpx.modelVerify` or `hpx.ReducedSpaceNewtonCG`, never asserted on directly.

### `TimeDependentPDEVariationalProblem.py`

| Method | Defined at | Direct unit test | Indirect | Parity |
|---|---|---|---|---|
| `generate_state/parameter/static_state/static_adjoint` | `TimeDependentPDEVariationalProblem.py:78-99` | test_pde_td.py:91,106,125 (state only) | yes | yes |
| `solveFwd` (linear) | `:135-163` | test_pde_td.py:92,107,126 | yes | heat |
| `solveFwd` (nonlinear Newton inner loop) | `:164-203` | **none** | no | tumor (run_x.py:135 `is_fwd_linear=False`) |
| `solveAdj` | `:206-263` | **none** | via modelVerify (test_pde_td.py:144) | heat, tumor |
| `evalGradientParameter` | `:266-297` | **none** | via modelVerify | heat, tumor |
| `setLinearizationPoint` | `:300-305` | **none** | via modelVerify | heat, tumor |
| `_solveIncrementalFwd` | `:308-360` | **none** | via modelVerify Hessian | heat, tumor |
| `_solveIncrementalAdj` | `:363-421` | **none** | via modelVerify Hessian | heat, tumor |
| `solveIncremental` (dispatch) | `:423-429` | **none** | via modelVerify | yes |
| `applyC` | `:436-468` | **none** | via modelVerify (Hessian KKT) | yes |
| `applyCt` | `:470-502` | **none** | via modelVerify | yes |
| `applyWuu` | `:504-546` | **none** | via modelVerify | yes |
| `applyWum` | `:548-588` | **none** | via modelVerify | yes |
| `applyWmu` | `:590-631` | **none** | via modelVerify | yes |
| `applyWmm` | `:633-668` | **none** | via modelVerify | yes |
| `apply_ij` dispatcher | `:670-679` | **none** | via modelVerify | yes |
| `gauss_newton_approx=True` early-return in `applyWuu/Wum/Wmu/Wmm` | `:506,550,592,635` | **none** | not exercised (parity uses `GN_iter=0` per refproblem.py:38) | no |

The `gauss_newton_approx=True` branch is observably **unexercised** in any test in this suite — `refproblem.py:38` sets `GN_iter = 0` precisely to avoid algorithmic divergence with legacy, and `test_pde_td.py` never sets the flag. Suspected: the Gauss–Newton fast path is silently untested.

### `misfit.py`

| Class.method | Defined at | Direct test |
|---|---|---|
| `ContinuousStateObservation.__init__` (W mass matrix, BC zeroing) | `misfit.py:27-60` | test_misfit_td.py:25 (no-BC); BC-zeroing branch (`misfit.py:41-53`) tested only indirectly via `test_pde_td.py:130` |
| `ContinuousStateObservation.cost` | `:70-79` | test_misfit_td.py:31,42,109 |
| `ContinuousStateObservation.grad` (STATE) | `:81-88` | test_misfit_td.py:42,54 |
| `ContinuousStateObservation.grad` (PARAMETER returns zero) | `:89-90` | **none** |
| `ContinuousStateObservation.grad` raises `IndexError` for ADJOINT | `:91-92` | **none** |
| `ContinuousStateObservation.setLinearizationPoint` (no-op) | `:94-95` | **none** |
| `ContinuousStateObservation.apply_ij` STATE,STATE | `:99-101` | test_misfit_td.py:84 |
| `ContinuousStateObservation.apply_ij` cross/other | `:102-103` | test_misfit_td.py:98 |
| `ContinuousStateObservation._check_nv` (None / zero) | `:62-68` | **none** (error paths not tested) |
| `SpaceTimePointwiseStateObservation.__init__` | `:122-139` | **none directly**; built in run_x.py:196 (parity) |
| `SpaceTimePointwiseStateObservation.cost` | `:149-157` | **none direct** |
| `SpaceTimePointwiseStateObservation.grad` | `:159-173` | **none direct** |
| `SpaceTimePointwiseStateObservation.apply_ij` | `:178-191` | **none direct** |
| `_PointwiseTDStorage.set/view/_index` | `:194-214` | **none direct** |
| `MisfitTD.cost` | `:228-233` | test_misfit_td.py:109 |
| `MisfitTD.grad` STATE | `:236-240` | test_misfit_td.py:133 |
| `MisfitTD.grad` PARAMETER | `:241-249` | **none direct** (only via Newton path) |
| `MisfitTD.setLinearizationPoint` | `:253-258` | **none direct** |
| `MisfitTD.apply_ij` (all four blocks + `_apply_*` helpers) | `:289-301` | **none direct** |

### `AdvectionDiffusionICModel` (`ad_diff_problem.py`)
- **Zero direct unit tests.** The class is exercised only by parity test_parity.py:28 (`PROBLEMS = [..., "ad_diff"]`) via `run_x.py:155-215` (`setup_ad_diff`).
- All public methods (`solveFwd:162`, `solveAdj:183`, `evalGradientParameter:210`, `solveFwdIncremental:239`, `solveAdjIncremental:258`, `applyC:278`, `applyCt:290`, `applyWuu/Wum/Wmu/Wmm:300-311`, `applyR:313`, `Rsolver:316`, `setPointForHessianEvaluations:231`, `init_parameter:152`, `generate_vector:131`) are touched only inside the subprocess parity run.
- If `fenicsproject` env is missing, **all of these methods become entirely uncovered**.

### `timeDependentVector.py`
Solid direct coverage in `test_tdv.py` (12 tests covering init, store/retrieve/view, zero, scale/imul, axpy, inner, norm, copy independence, array-proxy assign, array-proxy arithmetic, out-of-frame assertion). No gaps worth flagging.

## 2. Parity-test rigor

Observed (`test_parity.py:37-39`):
```
TOL_FINAL_COST = {"heat": 1e-4, "tumor": 1e-4, "ad_diff": 5e-2}
TOL_MAP_STATE  = {"heat": 1e-4, "tumor": 1e-4, "ad_diff": 5e-3}
TOL_MAP_PARAM  = {"heat": 1e-4, "tumor": 1e-4, "ad_diff": 5e-3}
```
- The 1e-4 vs 5e-3 split is **documented in-place** at `test_parity.py:30-36`: forward, prior, misfit, gradient, Hessian agree to ~1e-13; the inner-CG Eisenstat–Walker forcing term diverges and so the linear MAP iterate diverges by ~0.1% on `m` and ~1% on cost.
- The justification is plausible (observed): for `ad_diff`, both stacks call `ReducedSpaceNewtonCG` (run_x.py:340, presumably analogous in legacy) but inner-CG truncation depends on the dynamically updated relative-grad-norm. AD is a linear inverse problem so the *exact* MAP is unique; the optimizers stop at different inexact iterates.
- Suspected tightening path (not implemented here): drive `ad_diff` to a much tighter `cg_coarse_tolerance` (currently `5e-1`, `refproblem.py:37`) — e.g. `1e-8` — to force both inner CGs into the asymptotic regime; the two should then agree to ~`1e-6`. Trade-off: parity wall-clock would increase. The current `5e-3` tolerance is **adequate but not tight**; documentation in source is honest about it.
- Sloppiness signal: `test_newton_converges` (`test_parity.py:133`) allows iter count to differ by up to 5 for `ad_diff` vs 1 for heat/tumor — this is a permissive bound that masks any optimizer regression smaller than ~5 outer iters.

## 3. Reproducibility from clean checkout

### Env setup
- `INSTALL.md` documents only one env (`fenicsx`, docker image `dolfinx/dolfinx:v0.8.0`). It says **nothing** about the legacy `fenicsproject` conda env required by `test_parity.py`. (`INSTALL.md:1-55`, no mention of legacy hippylib at all.)
- `README.md:65-71` mentions both envs in one line but does **not** document how to create `fenicsproject`. No `environment.yml` / `requirements.txt` for either env exists in this directory.
- Versions: `INSTALL.md:20` pins `FEniCSx 0.8.0`, `petsc4py >= 3.21`, `slepc4py >= 3.21`. No pinning for `fenicsproject` (legacy dolfin) anywhere.

### Hardcoded maintainer paths
- `test_parity.py:20`: `CONDA_SH = os.environ.get("CONDA_SH", "/opt/anaconda3/etc/profile.d/conda.sh")` — macOS-specific default.
- `test_parity.py:22`: `HIPPYLIBX_BASE_DIR` defaults to `/Users/dliu/scratch/visco_inversion/src/hippylibx` — unrelated repo path.
- `test_parity.py:25`: `HIPPYLIB_PATH` defaults to `/Users/dliu/scratch/visco_inversion/src/hippylib` — unrelated repo path.
- All three are env-overridable but the test offers no graceful skip; a missing env yields a `RuntimeError` from `_run` (`test_parity.py:58-62`) presented as test failure, not skip.

### Pytest invocation
- `pytest -q hippylibX/time_dependent/tests/` works in the `fenicsx` env once the parity envs are set up; `conftest.py:11` prepends repo root to `sys.path`, so the test can be run from any cwd.

### `examples/heat.py`
- `examples/heat.py:51-54` writes outputs to `../results/heat_tutorial_x_...` — a relative path computed from cwd, not from `_THIS`. If invoked from anywhere other than `examples/`, the results directory lands in an unexpected location (observed at `heat.py:55` `os.makedirs(..., exist_ok=True)` silently creates it).
- Non-deterministic: `heat.py:137,139,159,175,176` use `hpx.parRandom.normal` / `prior.sample` with no fixed seed — each run produces different `m_true` and `m0`. Not a test issue (the test suite doesn't run the example), but reproducibility of the tutorial figures is lost across runs.

## 4. Deterministic seeding audit

`grep` for `parRandom|prior.sample|np.random|random.` across `time_dependent/` (excluding `__pycache__`):

| Location | Use | Concern |
|---|---|---|
| `tests/test_misfit_td.py:66` | `rng = np.random.default_rng(0)` | seeded, deterministic — OK |
| `examples/heat.py:137,139,159,175,176` | `parRandom.normal`, `prior.sample`, `normal_perturb` | unseeded; **example only**, not a test |
| `refproblem.py:29-30` | `NOISE_STD = 0.0; NOISE_VARIANCE = 1e-6` | deterministic by construction — OK |
| `run_x.py`, `run_legacy.py` | no RNG | OK |

No random sampling sneaks into the parity path. The dolfinx vs dolfin DOF-ordering rationale at `refproblem.py:26-29` is correct: using random noise would make the two stacks observe different `d` arrays even at the same dof.

## 5. Subprocess parity infrastructure

Observed (`test_parity.py:43-63`):
- Each problem spawns a fresh `bash -c "source ... && conda activate ... && ... python3 runner"` with a 600 s timeout.
- Failure mode if `fenicsproject` is absent: `conda activate fenicsproject` exits non-zero inside the heredoc, `subprocess.run` returns non-zero, `_run` raises `RuntimeError` (`test_parity.py:58`), pytest reports 8 of 24 parity tests as **errored** (not skipped) per problem. Net effect: a CI runner without legacy fenics shows 24 failures with confusing tracebacks.
- Same brittleness if `CONDA_SH` doesn't exist (`bash -c "source /opt/anaconda3/..."` exits 1).

Robustness for CI: **insufficient as written**. A graceful skip would require a session-scoped fixture that probes:
1. `os.path.exists(CONDA_SH)` — else `pytest.skip("conda not found")`;
2. `bash -c "source $CONDA_SH && conda env list | grep fenicsproject"` returncode == 0 — else `pytest.skip("fenicsproject env not present")`;
3. `os.path.isdir(HIPPYLIB_PATH)` — else skip.
None of these checks exist in the current `test_parity.py`.

## 6. Test artifact hygiene

Observed:
- `tests/results_x_{heat,tumor,ad_diff}.json` and `tests/results_legacy_{heat,tumor,ad_diff}.json` are written each run (`run_x.py:362`, analogous in `run_legacy.py`).
- `.gitignore:169` covers them: `hippylibX/time_dependent/tests/results_*.json` — **OK**.
- **Stale leftover from before refactor**: `tests/results_legacy.json` (802 B, dated `May 18 16:26`) and `tests/results_x.json` (802 B, dated `May 18 16:25`) exist alongside the per-problem suffix variants. These match the legacy `run_legacy.py`/`run_x.py` naming before the `PROBLEMS = [heat, tumor, ad_diff]` parametrization (`test_parity.py:28`); current code never writes `results_legacy.json` (no suffix). The two files are dead artifacts and should be removed.
- No `tmp_path` / cleanup in `test_parity.py` — JSONs persist between runs. The pattern is intentional (the fixture caches results in `_RESULTS` module-level dict, `test_parity.py:66`) but the on-disk JSONs are not cleaned.

## 7. Time budget

Total: 33.04 s. Slowest 5 (from `--durations=10`):

| Rank | Time | Test |
|---|---|---|
| 1 | 14.58 s | `test_parity.py::test_ndofs_match[tumor]` (setup — first subprocess launch for tumor; legacy + x runner) |
| 2 | 9.67 s  | `test_parity.py::test_ndofs_match[heat]` (setup) |
| 3 | 5.36 s  | `test_parity.py::test_ndofs_match[ad_diff]` (setup) |
| 4 | 1.75 s  | `test_pde_td.py::test_fd_gradient_slope_one` |
| 5 | 1.07 s  | `test_pde_td.py::test_hessian_symmetry_with_prior` |

Dominant cost (~29.6 s, 90% of wall-clock): the three subprocess parity setups, which spawn **2 conda-activated Python processes each** (one fenicsx, one fenicsproject) and run a full Newton-CG inversion. The slowest is `tumor` because of the nonlinear inner Newton at every timestep on both stacks. The in-process unit suite is ~3.4 s. Acceptable.

## 8. CI integration

Observed `.github/workflows/CI_testing.yml`:
- Runs in `dolfinx/dolfinx:stable` container (lines 16-17). **Only one fenics stack** is available — no `fenicsproject` / legacy `hippylib`.
- Invokes individual `test_*.py` files via `cd ./hippylibX/test && mpirun -n N python3 test_X.py` (lines 28-66) — does **not** use `pytest`, and does not touch `hippylibX/time_dependent/tests/` at all.

Slotting the new suite in:
- The pytest unit subset (`test_tdv.py`, `test_misfit_td.py`, `test_pde_td.py`) is directly runnable: add a step `cd ./hippylibX && python3 -m pytest -q time_dependent/tests/test_tdv.py time_dependent/tests/test_misfit_td.py time_dependent/tests/test_pde_td.py`.
- The parity subset (`test_parity.py`) **cannot run** in the upstream CI container — there is no second conda env and no legacy hippylib install. It needs either (a) the graceful skip described in §5, or (b) a separate workflow file with both envs in a non-container runner.

## 9. Concrete findings — file:line list

- `test_parity.py:20,22,25` — hardcoded maintainer-machine paths (env-overridable, no skip fallback).
- `test_parity.py:58-62` — missing env raises `RuntimeError` instead of `pytest.skip`.
- `TimeDependentPDEVariationalProblem.py:164-203` — nonlinear Newton forward path has no direct unit test.
- `TimeDependentPDEVariationalProblem.py:506,550,592,635` — `gauss_newton_approx=True` early-return is unreached by any test (refproblem.py:38 forces `GN_iter=0`).
- `ad_diff_problem.py:27-318` — entire class has no direct unit test (covered only by parity).
- `misfit.py:106-191` — `SpaceTimePointwiseStateObservation` (cost, grad, apply_ij) has no direct unit test.
- `misfit.py:62-68` — `_check_nv` error paths (None / 0) untested.
- `misfit.py:91-92` — `IndexError` for ADJOINT in `grad` untested.
- `tests/results_legacy.json`, `tests/results_x.json` — stale artifacts from before per-problem refactor; no current code writes them.
- `examples/heat.py:51-54` — relative `results_dir` is cwd-sensitive.
- `examples/heat.py:137-176` — unseeded RNG in example (not a test concern, but tutorial outputs aren't reproducible).
- `README.md:65-71` — mentions `fenicsproject` env but provides no setup instructions; `INSTALL.md` doesn't mention the parity env at all.
- `.github/workflows/CI_testing.yml` — does not run any `time_dependent/tests/` file; no path forward for parity without env-skip logic.

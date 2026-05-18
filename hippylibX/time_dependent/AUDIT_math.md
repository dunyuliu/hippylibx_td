# Mathematical/Numerical Audit — `hippylibX/time_dependent/`

## Executive summary (5 lines)

1. The port is a faithful line-for-line reproduction of legacy `TimeDependentPDEVariationalProblem` and `TimeDependentAD`; all Lagrangian-derivative algebra in `evalGradientParameter`/`applyC`/`applyCt`/`applyWuu`/`applyWum`/`applyWmu`/`applyWmm` agrees with the legacy hippylib reference, and the `R(m) - Mt_stab*p^1` identity in the IC-inversion gradient is correct.
2. The port silently **fixes** one legacy off-by-one bug in `_solveIncrementalAdj` (legacy uses `times[it-1]` which wraps to `times[-1]` on the first reverse step). This is a behavioral departure from "no deviation" and should be confirmed acceptable.
3. Two latent algebraic shortcuts inherited from legacy (and therefore preserved) are flagged: the adjoint forward-coupling term differentiates `form(t_n)` rather than `form(t_{n+1})` w.r.t. `u_old`; `solveAdj` never sets a meaningful `u_old` for the linearization point. Both are harmless for the three supplied applications because their u_old-coupling is the linear mass matrix.
4. The `_TDVArrayProxy` correctly handles `array[:] = …`, `array[:] *= …`, `array[:] += …` (the last by routing through numpy and then per-chunk `__setitem__`), so upstream `Model`/`ReducedHessian`/`modelVerify` interact correctly with `TimeDependentVector`.
5. The GLS stabilization formula in `ad_diff_problem.py` matches legacy verbatim, but has a divide-by-zero hazard at points where the wind vanishes; flagged but not actionable for the current test cases (Re=100 Stokes wind is non-vanishing in the interior).

---

## Observed vs suspected — quick table

| # | Location | Observation | Suspected status |
|---|----------|-------------|------------------|
| 1 | `TimeDependentPDEVariationalProblem.py:386-388` | Uses fixed `t_prev` lookup for u_old | **Intentional fix** of legacy bug `TimeDependentPDEVariationalProblem.py:311` (legacy reads `times[it-1]` → `times[-1]` when `it==0`). Deviation from strict line-by-line port. |
| 2 | `TimeDependentPDEVariationalProblem.py:236-238` (solveAdj), `:392-396` (_solveIncrementalAdj) | `dF/du_old` evaluated at `form(t_n)` not `form(t_{n+1})` | **Algebraic shortcut, preserved from legacy.** Correct iff u_old-coupling has no explicit t-dependence (true for BDF1 mass-matrix coupling in heat/tumor/ad_diff). Flag for forms with time-varying u_old coefficients. |
| 3 | `TimeDependentPDEVariationalProblem.py:233-260` (solveAdj) | `u_old` Function is never assigned from `x[STATE]`; remains zero throughout | **Preserved from legacy** (`:188` defines `u_old = Function(...)` then never retrieves into it). Correct iff `∂(∂F/∂u)/∂u_old = 0`, which holds for the linear/affine u_old coupling in the supplied apps. |
| 4 | `TimeDependentPDEVariationalProblem.py:266-297` (evalGradientParameter) | u_old IS rolled forward through the loop (line 294) | **Correct** and matches legacy `:235-238, 243`. |
| 5 | `TimeDependentPDEVariationalProblem.py:266-297`, line 297 `out.scatter_forward()` | Extra scatter at end of grad | Harmless. Legacy relied on DOLFIN auto-ghost. |
| 6 | `TimeDependentPDEVariationalProblem.py:504-547` (applyWuu) | Only diagonal blocks `∂²F/∂u∂u` and `∂²F/∂u_old∂u_old`; the cross terms `(u,u_old)` and `(u_old,u)` are omitted | **Preserved from legacy** `:429-430`. Mathematically an approximation to the full Hessian (it is the exact Hessian if F is separable in `(u,u_old)`, which IS true for typical BDF1 forms `(u-u_old)/dt·v + a(u,v)`). |
| 7 | `TimeDependentPDEVariationalProblem.py:548-588` (applyWum), `:590-631` (applyWmu) | Same omission of cross-terms; faithful to legacy | OK. |
| 8 | `TimeDependentPDEVariationalProblem.py:252` `b.axpy(1.0, rhs_t.petsc_vec)` | Sign convention requires caller to pre-negate misfit grad | **Convention matches legacy** `:218`. Verify upstream `Model` wrapper in `hippylibX/modeling/model.py` actually does this negation; if `Model` reuses the static-PDE convention you may have a sign error. **Action: confirm caller negates.** |
| 9 | `ad_diff_problem.py:215-218` (evalGradientParameter) | `mg = R(m) − Mt_stab · p^1` | **Correct.** Derive: `L = R(m) + ½‖u−d‖² + Σ_{n=1}^N ⟨p^n, L u^n − M_stab u^{n−1}⟩`; with `u^0 = m`, `∂L/∂m = R'(m) − M_stab^T p^1 = R(m) − Mt_stab p^1`. |
| 10 | `ad_diff_problem.py:225-228` | `grad_norm = mg·Msolver(mg)` (no sqrt) | **Known idiosyncrasy.** Quadratic, not norm. Matches legacy `model_ad_diff.py:266-270`. NewtonCG only uses relative ratios, so consistency between stacks matters more than the absolute scaling. Flag for future cleanup, but **do not change** — required for parity. |
| 11 | `ad_diff_problem.py:278-288` (applyC) | `out[t1] = −M_stab · dm`, zero elsewhere | **Correct.** Derived: `∂²L/(∂p^n ∂m) = −M_stab` iff n=1 else 0. Matches legacy `:325-336`. |
| 12 | `ad_diff_problem.py:290-298` (applyCt) | `out = −Mt_stab · dp[t1]` | **Correct.** Adjoint of (11). Matches legacy `:338-344`. |
| 13 | `ad_diff_problem.py:65` | `tau = min(h²/(2κ), h/|v|)` | **Matches legacy** verbatim (`model_ad_diff.py:144-147`). **Hazard:** divide-by-zero where `|v| = 0`. Not triggered by the Stokes wind in tests, but worth a guard (`vnorm + eps`) for general use. |
| 14 | `ad_diff_problem.py:162-179` (solveFwd) | Does not store IC at `out[t=0]` | **Matches legacy** `:207-219`. `out[t=0]` remains zero. Safe because no misfit is evaluated at t=0 (observation_times starts at t_1 > 0). If a user adds an obs at t=0, this will silently mis-evaluate. |
| 15 | `ad_diff_problem.py:195` | `for t in reversed(self.simulation_times)` — includes t=0 | Loop produces p at t=0, which is mathematically unused but harmlessly costs one solve. Matches legacy `:241`. |
| 16 | `timeDependentVector.py:35-52` (`_TDVArrayProxy.__setitem__`) | Full-slice assignment iterates chunks; non-full slice raises | **Correct.** End-to-end trace below. |
| 17 | `timeDependentVector.py:74-78` (`_TDVArrayProxy.__imul__`) | Iterates `v.array *= scalar` per snapshot | **Correct** for `array[:] *= -1`. |
| 18 | `timeDependentVector.py:155-159` (`zero`), `:160-167` (`scale`) | Naïve loops; correct ghost scatter at each step | OK. |
| 19 | `timeDependentVector.py:175-180` (`inner`) | `Σ_t pᵗ·qᵗ` — discrete time integral with unit weights (no `dt`) | **Matches legacy** semantics. This is an `ℓ²(times)` inner product, not an L²(0,T) integral. Documented limitation. |
| 20 | `TimeDependentPDEVariationalProblem.py:50-52` | `np.arange(t_init, t_final + 0.5*dt, dt)` | Same as legacy. Robust to floating-point end-point inclusion. ✓ |
| 21 | `TimeDependentPDEVariationalProblem.py:436-466` (applyC) | Uses `_bc_zero_rows_on_petsc` rather than `bc.apply` | dolfinx idiom; semantically equivalent to legacy. ✓ |
| 22 | `TimeDependentPDEVariationalProblem.py:466` | Stores `out_t` **then** advances `u_old` from `x[STATE]` at time t. The advance happens AFTER the assemble | ✓ Matches legacy ordering `:367-371`. |
| 23 | `TimeDependentPDEVariationalProblem.py:451-468` applyC vs `:478-501` applyCt | `applyC` advances `u_old` AFTER store at end of loop iter; `applyCt` advances `u_old` AT end too (line 500) | Consistent with each other and with legacy. |

---

## Detailed findings

### 1. Adjoint correctness — derivation reproduced

Discrete Lagrangian (per port semantics):

```
L(u, m, p) = J_misfit(u) + R(m) + Σ_{n=1}^N ⟨p^n, F(u^n, u^{n-1}, m, t_n)⟩
```

Stationarity wrt `u^n` (for `1 ≤ n < N`):

```
∂J/∂u^n + (∂F/∂u)^T(t_n) p^n + (∂F/∂u_old)^T(t_{n+1}) p^{n+1} = 0
```

The port (`TimeDependentPDEVariationalProblem.py:236-238`) builds:

```
adj_form = derivative(derivative(form(t_n), u, du), p, dp)            # ✓ correct
b_form   = − derivative(derivative(form(t_n), u_old, du), p, p_old)   # uses form(t_n), should be form(t_{n+1})
```

Because legacy `:209-211` is identical, this is a faithful port. For all three model applications, `∂F/∂u_old` is a constant bilinear form (mass-matrix), independent of t, so `form(t_n)` and `form(t_{n+1})` yield the same operator. **Future hazard:** add a varying-coefficient u_old coupling (e.g. time-varying capacity) and this will produce a wrong adjoint.

At `t = t_N` (final), `p_old = 0` initially — correct boundary condition for the terminal adjoint.

### 2. The legacy off-by-one in incremental adjoint, "fixed" in port

Legacy `TimeDependentPDEVariationalProblem.py:309-311`:

```python
for it, t in enumerate( reversed(self.times[1:]) ):
    self.linearize_x[STATE].retrieve(u.vector(), t)
    self.linearize_x[STATE].retrieve(u_old.vector(), self.times[it-1])
```

When `it = 0`, `self.times[it-1] = self.times[-1]` = final time — pulls the future state instead of the past state. This is a legacy bug.

Port `TimeDependentPDEVariationalProblem.py:382-389`:

```python
times_rev = list(reversed(self.times[1:]))
for idx, t in enumerate(times_rev):
    self.linearize_x[STATE].retrieve(u.x, t)
    t_prev = self.times[0] if t == self.times[1] else \
             self.times[ list(self.times).index(t) - 1 ]
    self.linearize_x[STATE].retrieve(u_old.x, t_prev)
```

This always picks the chronologically earlier frame. **This is a real algebraic divergence from legacy.** Whether to consider it a bug-fix or a violation of the "line-by-line, no deviation" rule is a policy call; raise with maintainers. (The parity tests pass to ~1e-13 for heat/tumor because the heat/tumor forms have constant u_old-coupling, so the wrongly retrieved `u_old` value never actually feeds into the bilinear form — only its trial-function symbol does. The discrepancy is invisible in those tests.)

### 3. `solveAdj` never sets `u_old`

`TimeDependentPDEVariationalProblem.py:222` declares `u_old = dlx.fem.Function(...)`, but lines 233-260 never assign a value from `x[STATE]`. So `u_old.x.array == 0` for the entire backward loop.

Legacy is identical (`:188`). Same caveat as #2: harmless when `∂(∂F/∂u)/∂u_old ≡ 0` (always true in supplied apps). Flag for general users.

### 4. Sign convention on adjoint RHS

`solveAdj` (`TimeDependentPDEVariationalProblem.py:252`) does `b.axpy(+1.0, rhs_t)`. Legacy `:218` is identical. Per the docstring at `:215`, the caller is expected to pass `adj_rhs = −∂J/∂u`. This is consistent with hippylib's `Model.solveAdj` wrapper.

**Action item (cannot verify without running):** trace `hippylibX/modeling/model.py::solveAdj` and confirm it negates `misfit.grad(STATE, …)` before invoking `pde.solveAdj`. If the upstream `Model` wrapper assumes the static-PDE convention (where the misfit gradient is passed un-negated and the PDE negates internally), there is a sign error. The ~1e-13 parity for heat/tumor suggests the convention IS correctly threaded, but please verify.

### 5. AD-diff IC gradient — explicit derivation

For `u^0 = m`, the linear backward-Euler form is:

```
L u^n − M_stab u^{n-1} = 0,   n=1,…,N
```

The Lagrangian:

```
𝓛 = R(m) + ½‖u − d‖²_W + Σ_{n=1}^N ⟨p^n, L u^n − M_stab u^{n-1}⟩
```

With `u^0 ≡ m`:

```
∂𝓛/∂m = R'(m) − M_stab^T p^1 = R(m) − Mt_stab p^1            ✓ matches ad_diff_problem.py:215-218
```

For the C and Cᵀ blocks (KKT Hessian):

```
∂²𝓛/(∂p^n ∂m) = −M_stab · [n==1]
∂²𝓛/(∂m ∂p^n) = −M_stab^T · [n==1]
```

Hence `applyC(dm)` returns `−M_stab dm` at `t = simulation_times[1]` and zero elsewhere (`ad_diff_problem.py:280-288` ✓), and `applyCt(dp)` returns `−Mt_stab · dp[t1]` (`:290-298` ✓).

### 6. GLS τ — numerical conditioning

`ad_diff_problem.py:62-65`:

```python
h = ufl.CellDiameter(msh)
vnorm = ufl.sqrt(ufl.inner(self.wind, self.wind))
tau = ufl.min_value(h*h/(2*kappa_const), h/vnorm)
```

Matches legacy `model_ad_diff.py:142-147` exactly. Both branches are positive when `κ > 0`, `|v| > 0`, `h > 0`. **Failure mode:** if the supplied wind has interior stagnation points (`|v| = 0`), `h/vnorm` produces NaN/Inf. UFL's `min_value` does not screen these out. Not exercised by the test problem (Re=100 lid-driven Stokes wind is nowhere zero in Ω). Recommend `vnorm + ufl.Constant(eps)` if used with general flows.

### 7. `_TDVArrayProxy` semantics — end-to-end trace for `modelVerify`

Trace: `modelVerify` (in `hippylibX/utils/`) typically does:

```python
h_norm = la.norm(h_state.array)            # → np.asarray(_TDVArrayProxy) — calls __array__ → concatenated ndarray ✓
x_pert.array[:] = x.array[:] + alpha * h.array[:]
# RHS:  np.asarray(x.array) + alpha * np.asarray(h.array)   →  ndarray
# LHS:  _TDVArrayProxy.__setitem__(slice(None), ndarray)
# Effect: chunked write per snapshot ✓
```

For `array[:] *= -1`:

```python
proxy = tdv.array                          # _TDVArrayProxy
proxy[:] *= -1
# Python expands to:  proxy[:] = proxy[:].__imul__(-1)
# proxy[:].__getitem__(slice(None)) → np.asarray(proxy)[slice] → ndarray (a copy)
# ndarray.__imul__(-1) → negated ndarray
# proxy.__setitem__(slice(None), ndarray) → chunked write ✓
```

But note: `proxy *= -1` (no `[:]`) goes through `_TDVArrayProxy.__imul__` (`timeDependentVector.py:74-78`), which scales the underlying `tdv.data` in place. Also correct, but a different code path. **Both styles are exercised by modelVerify** and both end at consistent state.

For `array[:] += other.array`:

```python
a = tdv1.array; b = tdv2.array
a[:] += b
# → a[:] = a[:].__iadd__(b)
# a[:] = ndarray + np.asarray(b) → ndarray
# a.__setitem__(slice(None), ndarray) → chunked write ✓
```

The only invariant required is **per-snapshot dof count matches** — guaranteed because the assignment loop in `__setitem__` uses `v.array.shape[0]` chunks of the concatenation.

### 8. Conservation invariant suggestion (heat eq, source=0, hom. Dirichlet u=0 on ∂Ω)

With BDF1, the discrete energy identity gives (with `M` mass matrix, `K` stiffness):

```
½ ‖u^{n+1}‖²_M  +  Δt · ‖∇u^{n+1}‖²_K  =  ½ ‖u^n‖²_M  −  ½ ‖u^{n+1} − u^n‖²_M
```

Hence ‖u^n‖_M is **monotone non-increasing** in n, and strictly decreasing for any nonzero IC. **Concrete test recommendation:** in `tests/`, add an assertion that

```python
energies = [u.view(t).petsc_vec.dot(M @ u.view(t).petsc_vec) for t in times]
assert all(energies[i+1] <= energies[i] + 1e-12 for i in range(len(energies)-1))
```

For tumor (nonlinear reaction term), this does NOT hold and the invariant should be a positivity check (`u^n ≥ 0` cell-wise if the IC is) rather than monotone energy decay.

### 9. Unit consistency

Nothing physically dimensional. The only place `dt` appears as a coefficient is `varf_handler.dt`, propagated consistently through the form. No `dt` enters `TimeDependentVector.inner` (it computes Σ_t, not Σ_t dt) — this is a **legacy convention** and matches; just note that any "L²(0,T)" claim about the inner product is wrong.

### 10. Things I did NOT run

- I did not execute the parity tests or `modelVerify`.
- I did not exercise the BC-lifting path on the Newton solver in `solveFwd` nonlinear branch (`:177-203`) — the `apply_lifting(..., alpha=-1.0)` plus `set_bc(b, ..., u.x.petsc_vec, -1.0)` pattern looks like the correct dolfinx idiom but should be unit-tested against a problem with non-homogeneous Dirichlet BCs (the heat test uses zero BCs and would not exercise it).
- Cross-check of the Newton convergence tolerance (`tol = 1e-10` absolute, `max_iter = 25`) against the legacy `NonlinearVariationalSolver` defaults — the absolute `b.norm()` check is coarser than the relative + absolute mix used in DOLFIN. **Action: confirm tolerance is tight enough for the tumor app's parity test at 1e-13.**

---

## Concrete action items (ordered by impact)

1. **Confirm sign convention** of `adj_rhs` passed to `solveAdj` from the upstream `Model` wrapper. If `Model` does NOT negate `misfit.grad(STATE,…)`, port has a sign error invisible to symmetric quadratic misfits. (File: `hippylibX/modeling/model.py`.)
2. **Decide policy** on the `_solveIncrementalAdj` off-by-one "fix" at `TimeDependentPDEVariationalProblem.py:386-388`. Either revert to bit-faithful legacy behavior and document the bug, or document the intentional fix in `README.md`. Current state silently diverges from the stated "line-by-line" rule.
3. **Add a stagnation guard** on `vnorm` in `ad_diff_problem.py:63` for robustness on general wind fields (`vnorm + eps`).
4. **Add the heat-equation energy-decay test** suggested in §8.
5. **Document the limitations** (cross-block Hessian truncation in `applyWuu`/`applyWum`, missing u_old retrieval in `solveAdj`, time-independent u_old assumption in adjoint b_form) in the module docstrings — currently only README mentions parity tolerances, not algorithmic assumptions.
6. **Verify the Newton tolerance** in `solveFwd` nonlinear path is consistent with legacy convergence for the tumor parity test.

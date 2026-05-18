"""Shared problem definitions for parity tests.

Both the legacy and the ported runner import these constants so the two
stacks solve *exactly* the same problem (same mesh, same true parameter,
same noise, same initial guess, same algorithm params).

Three physics are covered:

- ``heat``: linear diffusion, parameter = log-diffusivity. Backward Euler.
- ``tumor``: nonlinear reaction-diffusion (Fisher-type), parameter =
  log-diffusivity. ``is_fwd_linear=False`` exercises the Newton inner loop.
- ``ad_diff``: linear advection-diffusion with **initial-condition
  inversion**. Uses ``SpaceTimePointwiseStateObservation`` (sensor grid).

Select the active problem in a runner via the ``PROBLEM`` env var.
"""

# Shared discretization
NX, NY = 16, 16
NT = 4
T_INIT, T_FINAL = 0.0, 1.0

# Shared prior settings (BiLaplacianPrior(Vh_param, gamma, delta, robin_bc=True))
GAMMA, DELTA = 0.1, 0.5
ROBIN_BC = True

# Noise (deterministic): zero so that `d == u_true` exactly. The DOF order
# can differ between dolfin and dolfinx, so random noise would diverge.
NOISE_STD = 0.0
NOISE_VARIANCE = 1e-6

# Newton-CG parameters
REL_TOL = 1e-9
ABS_TOL = 1e-15
MAX_ITER = 60
CG_COARSE_TOL = 5e-1
GLOB = "LS"
GN_ITER = 0  # use the full Hessian throughout (no Gauss-Newton warm-up)

# Analytic m_true expression strings (legal in both UFL/dolfinx and
# dolfin.Expression). No `log` (clang JIT issue on legacy macOS).
M_TRUE_EXPR_STR_HEAT = "0.5 + 0.5*sin(pi*x[0])*sin(pi*x[1])"
M_TRUE_EXPR_STR_TUMOR = "0.5 + 0.3*sin(pi*x[0])*sin(pi*x[1])"

# AD-diff: parameter is the initial condition u0. Use a localised bump.
# dolfin Expression uses std::exp; dolfinx uses numpy.exp via the
# python-callable interpolate. The runner builds them separately.
AD_DIFF_M_CX = 0.35       # bump centre x
AD_DIFF_M_CY = 0.7        # bump centre y
AD_DIFF_M_AMP = 1.0
AD_DIFF_M_SIGMA = 0.10    # bump width

# AD-diff: sensor grid 4x4 inside the unit square (16 sensors)
AD_DIFF_N_SENS_X = 4
AD_DIFF_N_SENS_Y = 4

# Tumor reaction coefficient
TUMOR_LAMBDA = 1.0

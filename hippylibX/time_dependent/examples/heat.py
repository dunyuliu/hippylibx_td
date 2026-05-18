#!/usr/bin/env python
# coding: utf-8
"""Time-dependent heat-equation inversion — hippylibX + dolfinx tutorial.

Faithful port of the legacy hippylib heat-equation tutorial. Uses
``hippylibX.time_dependent`` for the time-dependent classes and the
upstream :class:`hippylibX.Model` / :func:`hippylibX.modelVerify` /
:class:`hippylibX.ReducedSpaceNewtonCG` for the model and inversion driver.

Run from the repo root::

    python hippylibX/time_dependent/examples/heat.py
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import ufl
import dolfinx as dlx
import dolfinx.fem.petsc
from mpi4py import MPI

# allow `python examples/heat.py` from the repo root or anywhere else
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import hippylibX as hpx
from hippylibX import time_dependent as td

STATE, PARAMETER, ADJOINT = hpx.STATE, hpx.PARAMETER, hpx.ADJOINT

t0 = time.time()
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


# ----------------------------------------------------------------------
# Mesh, function spaces, BCs
# ----------------------------------------------------------------------
nx, ny = 64, 64
nt = 20
T_INIT, T_FINAL = 0.0, 1.0
GAMMA, DELTA = 0.1, 0.5
REL_NOISE = 0.01

results_dir = (
    f"../results/heat_tutorial_x_nx{nx}_ny{ny}_nt{nt}_T{T_FINAL}_"
    f"gamma{GAMMA}_delta{DELTA}_noise{REL_NOISE}"
)
os.makedirs(results_dir, exist_ok=True)
print(f"Results dir: {results_dir}")

comm = MPI.COMM_WORLD
msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.triangle)

Vh2 = dlx.fem.functionspace(msh, ("Lagrange", 2))
Vh1 = dlx.fem.functionspace(msh, ("Lagrange", 1))
Vh = [Vh2, Vh1, Vh2]

ndofs = [Vh[i].dofmap.index_map.size_global * Vh[i].dofmap.index_map_bs for i in range(3)]
print(f"ndofs: state={ndofs[STATE]}, parameter={ndofs[PARAMETER]}, adjoint={ndofs[ADJOINT]}", flush=True)


def top_bottom(x):
    return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))


fdim = msh.topology.dim - 1
tb_facets = dlx.mesh.locate_entities_boundary(msh, fdim, top_bottom)
tb_dofs_state = dlx.fem.locate_dofs_topological(Vh[STATE], fdim, tb_facets)

uD = dlx.fem.Function(Vh[STATE])
uD.interpolate(lambda x: 0.0 * x[0])
uD.x.scatter_forward()
bc = dlx.fem.dirichletbc(uD, tb_dofs_state)
bc0 = bc  # homogeneous, same


# ----------------------------------------------------------------------
# Initial condition u0 = x(1-x) y(1-y), and source f = 0
# ----------------------------------------------------------------------
u0 = dlx.fem.Function(Vh[STATE])
u0.interpolate(lambda x: x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]))
u0.x.scatter_forward()
f = dlx.fem.Constant(msh, dlx.default_scalar_type(0.0))


# ----------------------------------------------------------------------
# Variational form: theta-scheme
# ----------------------------------------------------------------------
class HeatEquationVarf:
    def __init__(self, dt: float, f, theta: float = 1.0):
        self._dt = float(dt)
        self.dt_inv = dlx.fem.Constant(msh, dlx.default_scalar_type(1.0 / dt))
        self.f = f
        self.theta = dlx.fem.Constant(msh, dlx.default_scalar_type(theta))

    @property
    def dt(self):
        return self._dt

    def __call__(self, u, u_old, m, p, t):
        theta = self.theta
        one = dlx.fem.Constant(msh, dlx.default_scalar_type(1.0))
        u_mid = theta * u + (one - theta) * u_old
        return (
            (u - u_old) * p * self.dt_inv * ufl.dx
            + ufl.exp(m) * ufl.inner(ufl.grad(u_mid), ufl.grad(p)) * ufl.dx
            - self.f * p * ufl.dx
        )


dt = (T_FINAL - T_INIT) / nt
pde_varf = HeatEquationVarf(dt, f, theta=1.0)
pde = td.TimeDependentPDEVariationalProblem(
    Vh, pde_varf, bc=[bc], bc0=[bc0], u0=u0, t_init=T_INIT, t_final=T_FINAL,
    is_fwd_linear=True,
)


# ----------------------------------------------------------------------
# Prior and ground-truth parameter
# ----------------------------------------------------------------------
prior_mean = dlx.fem.Function(Vh[PARAMETER])
prior_mean.x.array[:] = 0.0
prior_mean = prior_mean.x  # use the underlying vector

prior = hpx.BiLaplacianPrior(Vh[PARAMETER], GAMMA, DELTA, mean=prior_mean)

# draw true parameter
noise = prior.generate_parameter("noise")
hpx.parRandom.normal(1.0, noise)
m_true = prior.generate_parameter(0)
prior.sample(noise, m_true)
print(f"Prior: gamma={GAMMA}, delta={DELTA}")


# ----------------------------------------------------------------------
# Synthetic observations
# ----------------------------------------------------------------------
u_true = pde.generate_state()
x_true = [u_true, m_true, None]
pde.solveFwd(u_true, x_true)

max_state = u_true.norm("linf", "linf")
noise_std = REL_NOISE * max_state
print(f"max_state={max_state:.4e}, noise_std={noise_std:.4e}")

misfits = []
for t in pde.times:
    misfit_t = td.ContinuousStateObservation(Vh[STATE], ufl.dx, bcs=[bc0])
    # set d = u_true(t) + noise
    misfit_t.d.array[:] = u_true.view(t).array[:]
    hpx.parRandom.normal_perturb(noise_std, misfit_t.d)
    misfit_t.d.scatter_forward()
    misfit_t.noise_variance = noise_std * noise_std
    misfits.append(misfit_t)
misfit = td.MisfitTD(misfits, pde.times)


# ----------------------------------------------------------------------
# Build the model and run modelVerify
# ----------------------------------------------------------------------
model = hpx.Model(pde, prior, misfit)

print("\n" + "#" * 80)
print(" modelVerify (misfit_only=False)")
print("#" * 80)
m0 = prior.generate_parameter(0)
hpx.parRandom.normal(1.0, noise)
prior.sample(noise, m0)

verify = hpx.modelVerify(
    model, m0, is_quadratic=False, misfit_only=False, verbose=(comm.rank == 0)
)
plt.savefig(f"{results_dir}/modelVerify.png", dpi=150, bbox_inches="tight")
plt.close("all")

print(f"\nFD gradient errors (first/last): {verify['err_grad'][0]:.3e} / {verify['err_grad'][-1]:.3e}")
print(f"FD Hessian errors  (first/last): {verify['err_H'][0]:.3e}    / {verify['err_H'][-1]:.3e}")
print(f"Symmetry error: {verify['sym_Hessian_value']:.3e}")
# dump the FD convergence curves
print("\nFD curve (eps, err_grad, err_H):")
for e, g, h in zip(verify["eps"], verify["err_grad"], verify["err_H"]):
    print(f"  eps={e:.3e}   err_grad={g:.3e}   err_H={h:.3e}")


# ----------------------------------------------------------------------
# Find the MAP via Newton-CG
# ----------------------------------------------------------------------
print("\n" + "#" * 80)
print(" Find the MAP point")
print("#" * 80)

initial_guess_m = prior.generate_parameter(0)
initial_guess_m.array[:] = prior_mean.array[:]

x = [
    model.generate_vector(STATE),
    initial_guess_m,
    model.generate_vector(ADJOINT),
]

parameters = hpx.ReducedSpaceNewtonCG_ParameterList()
parameters["rel_tolerance"] = 1e-6
parameters["abs_tolerance"] = 1e-9
parameters["max_iter"] = 30
parameters["cg_coarse_tolerance"] = 5e-1
parameters["globalization"] = "LS"
parameters["GN_iter"] = 10
if comm.rank != 0:
    parameters["print_level"] = -1

solver = hpx.ReducedSpaceNewtonCG(model, parameters)
x = solver.solve(x)
if solver.converged:
    print(f"\nConverged in {solver.it} iterations.")
else:
    print("\nNOT converged")
print(f"Termination: {solver.termination_reasons[solver.reason]}")
print(f"Final cost:  {solver.final_cost:.6e}")
print(f"Final grad:  {solver.final_grad_norm:.3e}")


# ----------------------------------------------------------------------
# Save MAP and true parameter as images
# ----------------------------------------------------------------------
try:
    m_map_fun = hpx.vector2Function(x[PARAMETER], Vh[PARAMETER], name="m_map")
    m_true_fun = hpx.vector2Function(m_true, Vh[PARAMETER], name="m_true")
    with dlx.io.XDMFFile(msh.comm, f"{results_dir}/parameters.xdmf", "w") as fid:
        fid.write_mesh(msh)
        fid.write_function(m_true_fun, 0.0)
        fid.write_function(m_map_fun, 1.0)
    print(f"Wrote {results_dir}/parameters.xdmf")
except Exception as e:
    print(f"Skipped XDMF write: {e}")

print(f"\nTotal: {time.time() - t0:.1f}s")

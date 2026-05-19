# --------------------------------------------------------------------------bc-
# Copyright (C) 2026 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""MPI consistency runner — solve the heat problem under arbitrary np
and gather the globally-ordered (state, gradient, MAP-param) vectors.

The output is a JSON dump suitable for comparing serial vs parallel.
``$RESULTS_JSON`` controls the dump path.

Run via::

    mpirun -n N python3 run_x_mpi.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import ufl
import dolfinx as dlx
import dolfinx.fem.petsc
from mpi4py import MPI

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import hippylibX as hpx
from hippylibX import time_dependent as td

STATE, PARAMETER, ADJOINT = hpx.STATE, hpx.PARAMETER, hpx.ADJOINT
comm = MPI.COMM_WORLD


def _gather_global(vec: dlx.la.Vector, V) -> np.ndarray | None:
    """Return the globally-ordered scalar array on rank 0; None elsewhere.

    Uses the dofmap's index_map to map local owned dofs → global indices,
    gathers via MPI, and assembles the global vector in canonical
    (global-index) order on rank 0.
    """
    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    nlocal = imap.size_local
    # owned dofs only (skip ghosts)
    local_arr = np.asarray(vec.array)[: nlocal * bs].copy()
    # global indices of the owned dofs
    global_idx = np.concatenate(
        [imap.local_to_global(np.arange(nlocal, dtype=np.int32)) * bs + b
         for b in range(bs)]
    )
    if bs > 1:
        # interleave: dof i, block b => global = imap.l2g(i)*bs + b
        glob = np.empty(nlocal * bs, dtype=np.int64)
        l2g = imap.local_to_global(np.arange(nlocal, dtype=np.int32))
        for b in range(bs):
            glob[b::bs] = l2g * bs + b
        global_idx = glob
    # gather lengths
    counts = comm.gather(local_arr.size, root=0)
    idxs = comm.gather(global_idx.astype(np.int64), root=0)
    vals = comm.gather(local_arr.astype(np.float64), root=0)
    if comm.rank != 0:
        return None
    n_total = imap.size_global * bs
    out = np.full(n_total, np.nan, dtype=np.float64)
    for ids, vs in zip(idxs, vals):
        out[ids] = vs
    if np.isnan(out).any():
        raise RuntimeError("global gather missed some entries")
    return out


def main(out_path: str) -> None:
    nx, ny, nt = 12, 12, 4
    T_INIT, T_FINAL = 0.0, 1.0
    GAMMA, DELTA = 0.1, 0.5

    msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.triangle)
    Vh2 = dlx.fem.functionspace(msh, ("Lagrange", 2))
    Vh1 = dlx.fem.functionspace(msh, ("Lagrange", 1))
    Vh = [Vh2, Vh1, Vh2]

    def top_bottom(x):
        return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

    fdim = msh.topology.dim - 1
    facets = dlx.mesh.locate_entities_boundary(msh, fdim, top_bottom)
    dofs = dlx.fem.locate_dofs_topological(Vh[STATE], fdim, facets)
    uD = dlx.fem.Function(Vh[STATE]); uD.interpolate(lambda x: 0.0 * x[0]); uD.x.scatter_forward()
    bc = dlx.fem.dirichletbc(uD, dofs)

    u0 = dlx.fem.Function(Vh[STATE])
    u0.interpolate(lambda x: x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]))
    u0.x.scatter_forward()

    class HeatVarf:
        def __init__(self, dt):
            self._dt = float(dt)
            self.dt_inv = dlx.fem.Constant(msh, dlx.default_scalar_type(1.0 / dt))

        @property
        def dt(self):
            return self._dt

        def __call__(self, u, u_old, m, p, t):
            return ((u - u_old) * p * self.dt_inv * ufl.dx
                    + ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx)

    pde = td.TimeDependentPDEVariationalProblem(
        Vh, HeatVarf((T_FINAL - T_INIT) / nt),
        bc=[bc], bc0=[bc], u0=u0, t_init=T_INIT, t_final=T_FINAL, is_fwd_linear=True,
    )

    # Deterministic m_true
    m_true_fun = dlx.fem.Function(Vh[PARAMETER])
    m_true_fun.interpolate(lambda x: 0.5 + 0.5 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    m_true_fun.x.scatter_forward()
    m_true = m_true_fun.x

    # Forward solve at m_true
    u_true = pde.generate_state()
    pde.solveFwd(u_true, [u_true, m_true, None])

    # Misfit at zero noise (d = u_true)
    misfits = []
    for t in pde.times:
        obs = td.ContinuousStateObservation(Vh[STATE], ufl.dx, bcs=[bc])
        obs.d.array[:] = u_true.view(t).array[:]
        obs.d.scatter_forward()
        obs.noise_variance = 1e-6
        misfits.append(obs)
    misfit = td.MisfitTD(misfits, pde.times)

    prior_mean = dlx.fem.Function(Vh[PARAMETER]); prior_mean.x.array[:] = 0.0
    prior = hpx.BiLaplacianPrior(
        Vh[PARAMETER], GAMMA, DELTA, mean=prior_mean.x, robin_bc=True,
    )

    model = hpx.Model(pde, prior, misfit)

    # cost@m_true (involves global allreduce inside misfit + prior.cost)
    x_true_eval = [u_true, m_true, model.generate_vector(ADJOINT)]
    model.solveAdj(x_true_eval[ADJOINT], x_true_eval)
    cost_at_mtrue = model.cost(x_true_eval)

    # MAP at m0=0
    m0 = prior.generate_parameter(0); m0.array[:] = 0.0; m0.scatter_forward()
    x = [model.generate_vector(STATE), m0, model.generate_vector(ADJOINT)]
    params = hpx.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-6
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = 20
    params["cg_coarse_tolerance"] = 0.5
    params["globalization"] = "LS"
    params["GN_iter"] = 5
    params["print_level"] = -1
    solver = hpx.ReducedSpaceNewtonCG(model, params)
    x = solver.solve(x)

    # gather globally
    m_map_global = _gather_global(x[PARAMETER], Vh[PARAMETER])
    # also gather forward state norms at each timestep
    state_norms = [float(u_true.view(t).petsc_vec.norm()) for t in pde.times]
    map_state_norms = [float(x[STATE].view(t).petsc_vec.norm()) for t in pde.times]
    m_map_l2 = float(x[PARAMETER].petsc_vec.norm())

    if comm.rank == 0:
        out = {
            "np": comm.size,
            "ndofs_param": int(Vh[PARAMETER].dofmap.index_map.size_global * Vh[PARAMETER].dofmap.index_map_bs),
            "ndofs_state": int(Vh[STATE].dofmap.index_map.size_global * Vh[STATE].dofmap.index_map_bs),
            "cost_at_mtrue": [float(c) for c in cost_at_mtrue],
            "state_norms_at_mtrue": state_norms,
            "newton_iters": int(solver.it),
            "converged": bool(solver.converged),
            "final_cost": float(solver.final_cost),
            "m_map_l2": m_map_l2,
            "map_state_norms": map_state_norms,
            "m_map_global": m_map_global.tolist(),
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {out_path} (np={comm.size})")


if __name__ == "__main__":
    main(os.environ.get("RESULTS_JSON", f"results_x_mpi_np{comm.size}.json"))

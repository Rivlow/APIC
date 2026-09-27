"""Solver : reçoit un dict de paramètres, des matrices (nx × ny) et la table des parois, alloue les champs
Taichi, calcule.

    solver = Solver(params, matrices, wall_table)     # semis + allocation + état initial
    solver.step(20)                       # 20 sous-pas
    solver.stats() ; solver.positions()   # lectures GPU -> CPU à la demande
    solver.render(x0, y0, scale, grid, tint) -> image (res, res, 3) u8

Pendant la simulation rien ne quitte le GPU, sauf ce que l'on demande (stats, positions, image rendue).
"""
from __future__ import annotations

import os
import time

import numpy as np
import taichi as ti

from Solver.physics import grid_step
from ui import kernels as K
from ui import kernels_inc as M
from ui.kernels_solid import (G2P_solid, P2G_solid, clear_eps, scatter_eps, solid_colors,
                              solid_stats, update_damage)

STRUCTURAL = ("nx", "ny", "bound", "ppc", "capacity", "seed", "res", "incompressible")   # le reste s'applique à chaud

_arch: str | None = None


def ensure_taichi() -> str:
    """Initialize Taichi once and return the chosen arch.

    **Outputs**

    - `str` : arch name

    **Note** : APIC_UI_ARCH (vulkan | cuda | cpu) forces the arch; otherwise tries Vulkan, then CUDA, then CPU.
    """
    global _arch
    if _arch is None:
        forced = os.environ.get("APIC_UI_ARCH")
        for arch in ([getattr(ti, forced)] if forced else [ti.vulkan, ti.cuda, ti.cpu]):
            try:
                ti.init(arch=arch)
                break
            except Exception as exc:
                print(f"[ui] arch {arch} indisponible ({exc})")
        _arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    return _arch


def _fluid_points(cells: np.ndarray, ppc: int, rng: np.random.Generator, dx: float) -> np.ndarray:
    """Seed ppc² uniform random points per active cell.

    **Inputs**

    - `cells` : np.ndarray bool (nx, ny)   active cells
    - `ppc` : int                        particles per cell side
    - `rng` : np.random.Generator
    - `dx` : float                      cell size

    **Outputs**

    - np.ndarray f64 (N, 2)   positions in domain units
    """
    idx = np.argwhere(cells)
    base = np.repeat(idx, ppc * ppc, axis=0).astype(np.float64)
    return (base + rng.random(base.shape)) * dx


def _solid_points(cells: np.ndarray, ppc: int, dx: float) -> np.ndarray:
    """Seed a regular ppc × ppc lattice per active cell.

    **Inputs**

    - `cells` : np.ndarray bool (nx, ny)   active cells
    - `ppc` : int                        particles per cell side
    - `dx` : float                      cell size

    **Outputs**

    - np.ndarray f64 (N, 2)   positions in domain units
    """
    idx = np.argwhere(cells)
    o = (np.arange(ppc) + 0.5) / ppc
    ox, oy = np.meshgrid(o, o, indexing="ij")
    offsets = np.column_stack([ox.ravel(), oy.ravel()])
    return (idx[:, None, :] + offsets[None, :, :]).reshape(-1, 2) * dx


class Solver:
    def __init__(self, params: dict, matrices: dict, wall_table):
        """Seed particles, allocate Taichi fields and set the initial state.

        **Inputs**

        - `params` : dict                              simulation parameters
        - `matrices` : dict of np.ndarray (nx, ny)       fluid/solid/obstacle masks, vx0/vy0
        - `wall_table` : WallTable (5 arrays, (4, max(nx, ny)[, 2]))   type, v, depth, pressure, friction
        """
        self.arch = ensure_taichi()
        self.p = dict(params)
        p = self.p
        nx, ny, ppc, bound = p["nx"], p["ny"], p["ppc"], p["bound"]
        nm = max(nx, ny)                              # cellules carrées : dx = 1 / max(nx, ny)
        dx0 = 1.0 / nm
        m = matrices
        # Solver.walls.WallTable (4, max(nx, ny)) : type, v, depth, pressure, friction
        wtype, wvel, wdepth, wpress, wfric = wall_table
        # zone utilisable : le stencil 3x3 sortirait de la grille dans la bande de paroi
        lo = np.array([bound, bound]) * dx0
        hi = np.array([nx - bound, ny - bound]) * dx0

        # ---- semis (numpy, une fois) ; vitesse initiale = celle de la cellule d'origine
        rng = np.random.default_rng(p["seed"])
        xf = _fluid_points(m["fluid"], ppc, rng, dx0)
        xs = _solid_points(m["solid"], ppc, dx0)

        def vel_of(pts):
            c = (pts * nm).astype(int)
            ci, cj = np.clip(c[:, 0], 0, nx - 1), np.clip(c[:, 1], 0, ny - 1)
            return np.column_stack([m["vx0"][ci, cj], m["vy0"][ci, cj]])

        vf, vs = vel_of(xf), vel_of(xs)
        xf, xs = np.clip(xf, lo, hi), np.clip(xs, lo, hi)

        # ---- grille des cellules (bits) ; les conditions aux limites sont sur les parois (wtype, wvel)
        cells = (m["fluid"] * K.FLUID0 | m["solid"] * K.SOLID0 | m["obstacle"] * K.OBSTACLE).astype(np.int32)
        self.has_inlet, self.has_outlet = bool((wtype == K.INLET).any()), bool((wtype == K.OUTLET).any())
        self.incompressible = bool(p.get("incompressible", False))
        # sortie à pression imposée p > 0 : l'eau peut aussi y rentrer (mode incompressible)
        self.has_pressure_outlet = bool((wpress > 0).any()) and self.incompressible
        if not self.incompressible and (wpress != 0).any():     # pression imposée ignorée : sorties libres
            print("[solver] pression imposée en sortie ignorée : elle n'agit que sur la projection "
                  "incompressible (incompressible=True)")
            wpress = np.zeros_like(wpress)

        # ---- capacités
        self.n_fluid_init = len(xf)
        self.has_fluid = self.n_fluid_init > 0 or self.has_inlet
        cap = self.n_fluid_init
        if self.has_inlet or self.has_pressure_outlet:
            cap = max(cap, p["capacity"] or ppc * ppc * nx * ny)
        self.capacity = max(cap, 1)                    # dense(ti.i, 0) est invalide
        self.n_solid = len(xs)
        self.has_solid = self.n_solid > 0
        self._x0f = np.zeros((self.capacity, 2), np.float32)
        self._v0f = np.zeros((self.capacity, 2), np.float32)
        self._x0f[:self.n_fluid_init], self._v0f[:self.n_fluid_init] = xf, vf
        self._x0s = (xs if self.has_solid else np.full((1, 2), 0.5)).astype(np.float32)
        self._v0s = (vs if self.has_solid else np.zeros((1, 2))).astype(np.float32)

        # ---- champs Taichi (un FieldsBuilder, libéré d'un bloc par release())
        fb = ti.FieldsBuilder()
        self.x_f, self.v_f = ti.Vector.field(2, ti.f32), ti.Vector.field(2, ti.f32)
        self.C_f, self.J_f = ti.Matrix.field(2, 2, ti.f32), ti.field(ti.f32)
        self.alive, self.free_stack = ti.field(ti.i32), ti.field(ti.i32)
        self.sc_f = ti.field(ti.f32)                   # quantité colorée par particule (voir fluid_mode)
        fb.dense(ti.i, self.capacity).place(self.x_f, self.v_f, self.C_f, self.J_f, self.alive, self.free_stack,
                                            self.sc_f)
        self.free_top = ti.field(ti.i32)
        fb.dense(ti.i, 1).place(self.free_top)

        self.x_s, self.v_s = ti.Vector.field(2, ti.f32), ti.Vector.field(2, ti.f32)
        self.C_s, self.F_s = ti.Matrix.field(2, 2, ti.f32), ti.Matrix.field(2, 2, ti.f32)
        self.D_s, self.broken_s, self.col_s = ti.field(ti.f32), ti.field(ti.i32), ti.Vector.field(3, ti.f32)
        fb.dense(ti.i, len(self._x0s)).place(self.x_s, self.v_s, self.C_s, self.F_s, self.D_s, self.broken_s, self.col_s)

        self.grid_v, self.grid_m = ti.Vector.field(2, ti.f32), ti.field(ti.f32)
        self.grid_e, self.grid_w = ti.field(ti.f32), ti.field(ti.f32)
        self.cells = ti.field(ti.i32)
        fb.dense(ti.ij, (nx, ny)).place(self.grid_v, self.grid_m, self.grid_e, self.grid_w, self.cells)
        self.wall_type, self.wall_v, self.emit_acc = ti.field(ti.i32), ti.Vector.field(2, ti.f32), ti.field(ti.f32)
        self.wall_d = ti.field(ti.i32)                 # position du mur (cellules depuis le bord) : bande ou face d'obstacle
        self.wall_p = ti.field(ti.f32)                 # pression imposée sur une sortie (0 : sortie libre)
        self.wall_f = ti.field(ti.f32)                 # frottement des murs (0 glissant, 1 adhérent)
        fb.dense(ti.ij, (4, nm)).place(self.wall_type, self.wall_v, self.emit_acc, self.wall_d, self.wall_p,
                                       self.wall_f)
        self.img = ti.Vector.field(3, ti.u8)
        fb.dense(ti.ij, (p["res"], p["res"])).place(self.img)
        if self.incompressible:                        # grille décalée + gradient conjugué
            self.u, self.mu = ti.field(ti.f32), ti.field(ti.f32)
            self.v, self.mv = ti.field(ti.f32), ti.field(ti.f32)
            fb.dense(ti.ij, (nx + 1, ny)).place(self.u, self.mu)
            fb.dense(ti.ij, (nx, ny + 1)).place(self.v, self.mv)
            self.ctype = ti.field(ti.i32)
            self.q, self.r, self.pd, self.Ap, self.rhs = (ti.field(ti.f32) for _ in range(5))
            fb.dense(ti.ij, (nx, ny)).place(self.ctype, self.q, self.r, self.pd, self.Ap, self.rhs)
            self.cg = ti.field(ti.f32)
            fb.dense(ti.i, 3).place(self.cg)
            self.fp = ti.Vector.field(2, ti.f32)      # accélération de pression sur les nœuds du solide
            fb.dense(ti.ij, (nx, ny)).place(self.fp)
            # projection de densité (positions) : rho / rho0 aux centres, potentiel phi, déplacement sur les faces
            self.dens, self.phi = ti.field(ti.f32), ti.field(ti.f32)
            fb.dense(ti.ij, (nx, ny)).place(self.dens, self.phi)
            self.du, self.dv = ti.field(ti.f32), ti.field(ti.f32)
            fb.dense(ti.ij, (nx + 1, ny)).place(self.du)
            fb.dense(ti.ij, (nx, ny + 1)).place(self.dv)
        self._tree = fb.finalize()

        self.cells.from_numpy(cells)
        self.wall_type.from_numpy(np.ascontiguousarray(wtype, np.int32))
        self.wall_v.from_numpy(np.ascontiguousarray(wvel, np.float32))
        self.wall_d.from_numpy(np.ascontiguousarray(wdepth, np.int32))
        self.wall_p.from_numpy(np.ascontiguousarray(wpress, np.float32))
        self.wall_f.from_numpy(np.ascontiguousarray(wfric, np.float32))
        self.t = 0.0
        self.last_step_ms = 0.0
        self.cg_last_iters = 0
        self.fluid_mode = 0                            # indice dans kernels.FLUID_MODES
        self.scalar_max = 0.0                          # échelle de couleur, mise à jour dans stats()
        self._inlet_speed = float(np.hypot(wvel[..., 0], wvel[..., 1])[wtype == K.INLET].max()) if self.has_inlet else 0.0
        self.set_params(p)
        self.reset()

    # ------------------------------------------------------------ paramètres à chaud
    def set_params(self, params: dict) -> None:
        """Apply non-structural parameters and recompute derived constants and dt.

        **Inputs**

        - `params` : dict   parameter overrides (STRUCTURAL keys ignored)

        **Outputs**

        - self.p, dx, masses, Lamé coefficients, dt, dt_solid, tau_D updated
        """
        for k, v in params.items():
            if k not in STRUCTURAL:
                self.p[k] = v
        p = self.p
        self.nx, self.ny = p["nx"], p["ny"]
        self.dx = 1.0 / max(self.nx, self.ny)
        self.inv_dx = float(max(self.nx, self.ny))
        self.p_spacing = self.dx / p["ppc"]
        p_vol = self.p_spacing ** 2
        self.p_mass_f, self.p_mass_s = p_vol * p["fluid_rho"], p_vol * p["solid_rho"]
        self.p_vol = p_vol
        self.mu_s = p["solid_E"] / (2 * (1 + p["solid_nu"]))
        self.la_s = p["solid_E"] * p["solid_nu"] / ((1 + p["solid_nu"]) * (1 - 2 * p["solid_nu"]))
        speeds = [1e-6]
        if self.has_fluid:
            speeds.append((p["fluid_E"] / p["fluid_rho"]) ** 0.5)
        if self.has_solid:
            speeds.append(((self.la_s + 2 * self.mu_s) / p["solid_rho"]) ** 0.5)
        if self.has_inlet:
            speeds.append(self._inlet_speed)
        self.dt = p["cfl"] * self.dx / max(speeds)
        # pas de temps propre du solide (ondes élastiques) : en incompressible, le fluide avance au pas
        # d'advection et le solide est sous-cyclé à dt_solid
        c_s = ((self.la_s + 2 * self.mu_s) / p["solid_rho"]) ** 0.5
        self.dt_solid = p["cfl"] * self.dx / c_s
        self.tau_D = max(p["tau_D"], 2.0 * min(self.dt, self.dt_solid))   # dD <= dt / tau_D doit rester < 1 par pas

    # ------------------------------------------------------------ cycle de vie
    def reset(self) -> None:
        """Reset to the initial state.

        **Outputs**

        - particle fields re-uploaded and re-initialized, t = 0

        **Note** : one CPU -> GPU transfer of the seeded positions, then init kernels.
        """
        if self.incompressible:
            self.q.fill(0.0)
            self.fp.fill(0.0)
        self.emit_acc.fill(0.0)
        self.x_f.from_numpy(self._x0f)
        self.v_f.from_numpy(self._v0f)
        K.init_pool(self.alive, self.x_f, self.C_f, self.J_f, self.n_fluid_init, self.free_stack, self.free_top)
        self.x_s.from_numpy(self._x0s)
        self.v_s.from_numpy(self._v0s)
        K.init_solid_state(self.C_s, self.F_s, self.D_s, self.broken_s)
        self.t = 0.0

    def release(self) -> None:
        """Free all Taichi fields of this solver.

        **Outputs**

        - field tree destroyed, self._tree = None
        """
        if self._tree is not None:
            self._tree.destroy()
            self._tree = None

    # ------------------------------------------------------------ simulation (incompressible)
    def _adapt_dt(self) -> None:
        """Set the advection time step from the CFL on max fluid speed (incompressible).

        **Outputs**

        - self.dt updated

        **Note** : reads max speed GPU -> CPU; no acoustic limit.
        """
        p = self.p
        vmax = max(float(M.max_speed(self.v_f, self.alive)), self._inlet_speed, 0.25)
        dt = p["cfl"] * self.dx / vmax
        if p["gravity"] > 0:
            dt = min(dt, 0.5 * (self.dx / p["gravity"]) ** 0.5)
        self.dt = dt

    def _project(self) -> None:
        """Pressure projection on the MAC grid by conjugate gradient.

        **Outputs**

        - self.q (pressure), self.u, self.v updated

        **Note** : fixed `cg_iters` iterations, fully on GPU with no read-back; warm-started from previous pressure.
        """
        p = self.p
        M.cg_init(self.q, self.r, self.pd, self.rhs, self.u, self.v, self.ctype, self.cg,
                  self.wall_type, self.wall_d, self.wall_p, self.dx, self.dt, 1.0 / p["fluid_rho"],
                  self.nx, self.ny, p["bound"])
        for _ in range(p["cg_iters"]):
            M.cg_apply(self.pd, self.Ap, self.ctype, self.cg, self.nx, self.ny)
            M.cg_update(self.q, self.r, self.pd, self.Ap, self.ctype, self.cg)
        self.cg_last_iters = p["cg_iters"]
        M.mac_project(self.u, self.v, self.q, self.ctype, self.dx, self.nx, self.ny)

    def _density_projection(self) -> None:
        """Density projection: shift fluid particle positions so that rho = rho0.

        **Outputs**

        - self.x_f updated (velocities untouched)

        **Note** : fixed `density_iters` CG iterations, no read-back; shares CG buffers with the pressure solve.
        """
        p = self.p
        kappa = float(p["volume_correction"])
        if kappa <= 0.0:
            return
        nx, ny = self.nx, self.ny
        M.particle_density(self.x_f, self.alive, self.dens, self.inv_dx, int(p["ppc"]))
        M.density_cg_init(self.phi, self.r, self.pd, self.rhs, self.dens, self.ctype, self.cg, kappa, self.dx,
                          nx, ny)
        for _ in range(int(p["density_iters"])):
            M.cg_apply(self.pd, self.Ap, self.ctype, self.cg, nx, ny)
            M.cg_update(self.phi, self.r, self.pd, self.Ap, self.ctype, self.cg)
        M.density_gradient(self.phi, self.du, self.dv, self.ctype, self.dx, nx, ny)
        M.density_shift(self.x_f, self.alive, self.du, self.dv, self.ctype, self.inv_dx, self.dx, p["bound"], nx, ny)

    def _solid_substep(self, dt_s: float, with_pressure: bool) -> None:
        """Advance the solid alone by one MPM step on the collocated grid.

        **Inputs**

        - `dt_s` : float   solid time step
        - `with_pressure` : bool    add the fluid pressure acceleration self.fp

        **Outputs**

        - solid particle state (x_s, v_s, C_s, F_s, D_s, broken_s) updated
        """
        p, dx, inv_dx, nx, ny, bound = self.p, self.dx, self.inv_dx, self.nx, self.ny, self.p["bound"]
        self.grid_m.fill(0.0)
        self.grid_v.fill(0.0)
        P2G_solid(self.grid_m, self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.D_s, self.broken_s,
                  inv_dx, dt_s, dx, self.mu_s, self.la_s, self.p_mass_s, self.p_vol, p["k_res"])
        grid_step(self.grid_m, self.grid_v, self.cells, self.wall_type, self.wall_v, self.wall_d, self.wall_f,
                  dt_s, p["gravity"], 0.0, p["obstacle_friction"], bound, nx, ny)
        if with_pressure:
            M.add_accel(self.grid_v, self.grid_m, self.fp, dt_s)
        G2P_solid(self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.broken_s, inv_dx, dt_s, dx, bound, nx, ny)
        if p["use_damage"]:
            clear_eps(self.grid_e, self.grid_w)
            scatter_eps(self.grid_e, self.grid_w, self.x_s, self.F_s, self.broken_s, inv_dx)
            update_damage(self.grid_e, self.grid_w, self.x_s, self.F_s, self.D_s, self.broken_s,
                          inv_dx, dt_s, p["eps0"], p["epsf"], self.tau_D, int(p["use_rupture"]))

    def _substep_incompressible(self) -> None:
        """Advance one incompressible substep with partitioned fluid-solid coupling.

        **Outputs**

        - fluid and solid state updated

        **Note** : solid sub-cycled at dt_solid imposes its velocity on occupied faces; fluid returns -grad p to it.
        """
        p, dt, dx, inv_dx, nx, ny, bound = self.p, self.dt, self.dx, self.inv_dx, self.nx, self.ny, self.p["bound"]
        if self.has_solid:
            n_in = max(1, int(np.ceil(dt / self.dt_solid)))
            dt_s = dt / n_in
            for _ in range(n_in):
                self._solid_substep(dt_s, True)
            # grille collocalisée du solide (masse + vitesse) pour les faces des cellules MOVING
            self.grid_m.fill(0.0)
            self.grid_v.fill(0.0)
            P2G_solid(self.grid_m, self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.D_s, self.broken_s,
                      inv_dx, dt_s, dx, self.mu_s, self.la_s, self.p_mass_s, self.p_vol, p["k_res"])
            grid_step(self.grid_m, self.grid_v, self.cells, self.wall_type, self.wall_v, self.wall_d, self.wall_f,
                      0.0, 0.0, 0.0, p["obstacle_friction"], bound, nx, ny)
        M.mac_p2g(self.x_f, self.v_f, self.C_f, self.alive, self.u, self.v, self.mu, self.mv, inv_dx, dx)
        M.mac_classify(self.ctype, self.cells, self.wall_type, self.wall_v, self.wall_d, self.x_f, self.alive, self.x_s,
                       int(self.has_solid), inv_dx, nx, ny, bound, int(p["free_surface"]))
        M.mac_bc(self.u, self.v, self.ctype, self.wall_type, self.wall_v, self.wall_d, self.wall_f, self.grid_v,
                 self.grid_m, dt, p["gravity"], p["obstacle_friction"], nx, ny, bound, 1)
        self._project()
        M.mac_bc(self.u, self.v, self.ctype, self.wall_type, self.wall_v, self.wall_d, self.wall_f, self.grid_v,
                 self.grid_m, dt, p["gravity"], p["obstacle_friction"], nx, ny, bound, 0)
        if self.has_solid:
            M.pressure_force(self.fp, self.grid_m, self.q, self.ctype, inv_dx,
                             -(p["fluid_rho"] / p["solid_rho"]) / dt, nx, ny)
        M.mac_g2p(self.x_f, self.v_f, self.C_f, self.alive, self.u, self.v, self.mu, self.mv, inv_dx, dx)
        self._advect_and_emit(dt)
        self._density_projection()

    def _advect_and_emit(self, dt: float) -> None:
        """Advect fluid particles, then emit at inlets and pressure outlets.

        **Inputs**

        - `dt` : float   time step

        **Outputs**

        - x_f, v_f, alive, free stack updated
        """
        p, dx, inv_dx, nx, ny, bound = self.p, self.dx, self.inv_dx, self.nx, self.ny, self.p["bound"]
        K.advect_fluid(self.x_f, self.v_f, self.alive, self.wall_type, self.wall_d, inv_dx, dt, bound, dx, nx, ny,
                       self.free_stack, self.free_top)
        if self.has_inlet:
            K.emit_wall(self.x_f, self.v_f, self.C_f, self.J_f, self.alive, self.wall_type, self.wall_v,
                        self.wall_d, self.emit_acc, float(p["ppc"] * p["ppc"]), self.free_stack, self.free_top, dt, dx, bound,
                        nx, ny)
        if self.has_pressure_outlet:
            M.emit_pressure_outlet(self.x_f, self.v_f, self.C_f, self.J_f, self.alive, self.wall_type, self.wall_d,
                                   self.wall_p, self.emit_acc, self.ctype, self.u, self.v, int(p["ppc"]),
                                   self.free_stack, self.free_top, dt, dx, bound, nx, ny)

    # ------------------------------------------------------------ simulation (faiblement compressible)
    def _substep(self) -> None:
        """Advance one weakly compressible MPM substep (fluid + solid on the collocated grid).

        **Outputs**

        - fluid and solid state updated
        """
        p, dt, dx, inv_dx, nx, ny, bound = self.p, self.dt, self.dx, self.inv_dx, self.nx, self.ny, self.p["bound"]
        if self.has_fluid:
            K.P2G_fluid(self.grid_m, self.grid_v, self.x_f, self.v_f, self.C_f, self.J_f, self.alive,
                        inv_dx, dt, dx, p["fluid_E"], self.p_mass_f, self.p_vol)
        else:
            self.grid_m.fill(0.0)
            self.grid_v.fill(0.0)
        if self.has_solid:
            P2G_solid(self.grid_m, self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.D_s, self.broken_s,
                      inv_dx, dt, dx, self.mu_s, self.la_s, self.p_mass_s, self.p_vol, p["k_res"])
        grid_step(self.grid_m, self.grid_v, self.cells, self.wall_type, self.wall_v, self.wall_d, self.wall_f,
                  dt, p["gravity"], 0.0, p["obstacle_friction"], bound, nx, ny)
        if self.has_fluid:
            K.G2P_fluid(self.grid_v, self.x_f, self.v_f, self.C_f, self.J_f, self.alive, inv_dx, dt, dx)
        if self.has_solid:
            G2P_solid(self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.broken_s, inv_dx, dt, dx, bound, nx, ny)
            if p["use_damage"]:
                clear_eps(self.grid_e, self.grid_w)
                scatter_eps(self.grid_e, self.grid_w, self.x_s, self.F_s, self.broken_s, inv_dx)
                update_damage(self.grid_e, self.grid_w, self.x_s, self.F_s, self.D_s, self.broken_s,
                              inv_dx, dt, p["eps0"], p["epsf"], self.tau_D, int(p["use_rupture"]))
        if self.has_fluid:
            self._advect_and_emit(dt)

    def step(self, substeps: int | None = None) -> None:
        """Advance the simulation by several substeps.

        **Inputs**

        - `substeps` : int | None   number of substeps (None: p["substeps"])

        **Outputs**

        - state, self.t and self.last_step_ms updated
        """
        t0 = time.perf_counter()
        n_sub = substeps or self.p["substeps"]
        if self.incompressible:
            self._adapt_dt()
            for _ in range(n_sub):
                self._substep_incompressible()
        else:
            for _ in range(n_sub):
                self._substep()
        ti.sync()
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        self.t += n_sub * self.dt

    # ------------------------------------------------------------ lectures (GPU -> CPU à la demande)
    def stats(self) -> dict:
        """Read simulation diagnostics.

        **Outputs**

        - `dict` : t, dt, n_fluid, capacity, n_solid, n_broken, D_max, ms, scalar_max (+ cg_iters, cg_rr, div_max)

        **Note** : GPU -> CPU reads on demand only; also updates the smoothed self.scalar_max.
        """
        n_broken, d_max = 0, 0.0
        if self.has_solid:
            s = solid_stats(self.D_s, self.broken_s)
            n_broken, d_max = int(s[0]), float(s[1])
        st = {"t": self.t, "dt": self.dt, "n_fluid": int(K.count_alive(self.alive)) if self.has_fluid else 0,
              "capacity": self.capacity, "n_solid": self.n_solid, "n_broken": n_broken, "D_max": d_max,
              "ms": self.last_step_ms}
        if self.incompressible:                        # diagnostics, lus seulement quand stats() est appelé
            st["cg_iters"] = self.cg_last_iters
            st["cg_rr"] = float(M.cg_residual(self.cg))
            st["div_max"] = float(M.divergence_max(self.u, self.v, self.ctype, self.dx))
        if self.has_fluid and self.fluid_mode > 0:    # échelle de couleur lissée (max de la quantité affichée)
            m = float(K.scalar_absmax(self.sc_f, self.alive))
            self.scalar_max = m if self.scalar_max <= 0 else 0.7 * self.scalar_max + 0.3 * m
        st["scalar_max"] = self.scalar_max
        return st

    def positions(self) -> np.ndarray:
        """Read alive fluid particle positions (GPU -> CPU).

        **Outputs**

        - np.ndarray f32 (N, 2)
        """
        return self.x_f.to_numpy()[self.alive.to_numpy() == 1] if self.has_fluid else np.zeros((0, 2), np.float32)

    def velocities(self) -> np.ndarray:
        """Read alive fluid particle velocities (GPU -> CPU).

        **Outputs**

        - np.ndarray f32 (N, 2)
        """
        return self.v_f.to_numpy()[self.alive.to_numpy() == 1] if self.has_fluid else np.zeros((0, 2), np.float32)

    def solid_positions(self) -> np.ndarray:
        """Read solid particle positions (GPU -> CPU).

        **Outputs**

        - np.ndarray f32 (Ns, 2)
        """
        return self.x_s.to_numpy() if self.has_solid else np.zeros((0, 2), np.float32)

    def damage(self) -> np.ndarray:
        """Read solid particle damage D (GPU -> CPU).

        **Outputs**

        - np.ndarray f32 (Ns,)
        """
        return self.D_s.to_numpy() if self.has_solid else np.zeros(0, np.float32)

    def render(self, x0: float = 0.0, y0: float = 0.0, scale: float = 1.0,
               grid: bool = False, tint: bool = False) -> np.ndarray:
        """Render the view to an image on the GPU.

        **Inputs**

        - `x0`, `y0`, `scale` : float   view origin and zoom (view = [x0, x0 + 1/scale]²)
        - `grid`, `tint` : bool    draw mesh / initial-cell tints

        **Outputs**

        - np.ndarray u8 (res, res, 3)
        """
        res = self.p["res"]
        if self.has_solid:
            solid_colors(self.F_s, self.D_s, self.broken_s, self.col_s, int(self.p["color_mode"]), self.p["epsf"])
        r_px = max(0, int(0.5 * self.p_spacing * res * scale + 0.5))
        mode = int(self.fluid_mode)
        if self.has_fluid and mode > 0:
            if self.incompressible:
                K.fluid_scalar_inc(mode, self.x_f, self.v_f, self.alive, self.u, self.v, self.q, self.sc_f,
                                   self.p["fluid_rho"] / self.dt, self.inv_dx, self.nx, self.ny)
            else:
                K.fluid_scalar_wc(mode, self.x_f, self.v_f, self.J_f, self.alive, self.grid_v, self.sc_f,
                                  self.p["fluid_E"], self.inv_dx, self.nx, self.ny)
        K.render(self.img, res, x0, y0, scale, self.cells, self.wall_type, self.wall_d, self.nx, self.ny,
                 self.p["bound"],
                 int(grid), int(tint),
                 self.x_f, self.alive, int(self.has_fluid), 0.35, 0.65, 1.0, r_px,
                 self.sc_f, mode, 1.0 / max(self.scalar_max, 1e-9),
                 self.x_s, self.col_s, int(self.has_solid), r_px)
        return self.img.to_numpy()

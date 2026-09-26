"""Solver : reçoit un dict de paramètres et des matrices (n × n), alloue les champs Taichi, calcule.

    solver = Solver(params, matrices)     # semis + allocation + état initial
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

from Code_tuto.mpm_solid import (G2P_solid, P2G_solid, clear_eps, scatter_eps, solid_colors,
                                 solid_stats, update_damage)
from ui import kernels as K

STRUCTURAL = ("n", "bound", "ppc", "capacity", "seed", "res")   # tout le reste s'applique à chaud

_arch: str | None = None


def ensure_taichi() -> str:
    """ti.init une seule fois : APIC_UI_ARCH (vulkan | cuda | cpu), sinon Vulkan, puis CUDA, puis CPU."""
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


def _fluid_points(cells: np.ndarray, ppc: int, rng: np.random.Generator) -> np.ndarray:
    """ppc² points uniformes par cellule active, en unités domaine."""
    n = cells.shape[0]
    idx = np.argwhere(cells)
    base = np.repeat(idx, ppc * ppc, axis=0).astype(np.float64)
    return (base + rng.random(base.shape)) / n


def _solid_points(cells: np.ndarray, ppc: int) -> np.ndarray:
    """Réseau régulier ppc × ppc par cellule active (comme init_beam de Code_tuto/mpm_solid.py)."""
    n = cells.shape[0]
    idx = np.argwhere(cells)
    o = (np.arange(ppc) + 0.5) / ppc
    ox, oy = np.meshgrid(o, o, indexing="ij")
    offsets = np.column_stack([ox.ravel(), oy.ravel()])
    return (idx[:, None, :] + offsets[None, :, :]).reshape(-1, 2) / n


class Solver:
    def __init__(self, params: dict, matrices: dict):
        self.arch = ensure_taichi()
        self.p = dict(params)
        p = self.p
        n, ppc, bound = p["n"], p["ppc"], p["bound"]
        m = matrices
        lo, hi = bound / n, 1.0 - bound / n           # bande de paroi : le stencil 3x3 sortirait de la grille

        # ---- semis (numpy, une fois) ; vitesse initiale = celle de la cellule d'origine
        rng = np.random.default_rng(p["seed"])
        xf = _fluid_points(m["fluid"], ppc, rng)
        xs = _solid_points(m["solid"], ppc)

        def vel_of(pts):
            c = np.clip((pts * n).astype(int), 0, n - 1)
            return np.column_stack([m["vx0"][c[:, 0], c[:, 1]], m["vy0"][c[:, 0], c[:, 1]]])

        vf, vs = vel_of(xf), vel_of(xs)
        xf, xs = np.clip(xf, lo, hi), np.clip(xs, lo, hi)

        # ---- grille des cellules (bits) ; entrées / sorties hors bande de paroi uniquement
        usable = np.zeros((n, n), bool)
        usable[bound:n - bound, bound:n - bound] = True
        inlet, outlet = m["inlet"] & usable, m["outlet"] & usable
        cells = (m["fluid"] * K.FLUID0 | m["solid"] * K.SOLID0 | m["obstacle"] * K.OBSTACLE
                 | inlet * K.INLET | outlet * K.OUTLET).astype(np.int32)
        self.has_inlet, self.has_outlet = bool(inlet.any()), bool(outlet.any())

        # ---- capacités
        self.n_fluid_init = len(xf)
        self.has_fluid = self.n_fluid_init > 0 or self.has_inlet
        cap = self.n_fluid_init
        if self.has_inlet:
            cap = max(cap, p["capacity"] or ppc * ppc * n * n)
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
        fb.dense(ti.i, self.capacity).place(self.x_f, self.v_f, self.C_f, self.J_f, self.alive, self.free_stack)
        self.free_top = ti.field(ti.i32)
        fb.dense(ti.i, 1).place(self.free_top)

        self.x_s, self.v_s = ti.Vector.field(2, ti.f32), ti.Vector.field(2, ti.f32)
        self.C_s, self.F_s = ti.Matrix.field(2, 2, ti.f32), ti.Matrix.field(2, 2, ti.f32)
        self.D_s, self.broken_s, self.col_s = ti.field(ti.f32), ti.field(ti.i32), ti.Vector.field(3, ti.f32)
        fb.dense(ti.i, len(self._x0s)).place(self.x_s, self.v_s, self.C_s, self.F_s, self.D_s, self.broken_s, self.col_s)

        self.grid_v, self.grid_m = ti.Vector.field(2, ti.f32), ti.field(ti.f32)
        self.grid_e, self.grid_w = ti.field(ti.f32), ti.field(ti.f32)
        self.cells, self.cell_count, self.bc_v = ti.field(ti.i32), ti.field(ti.i32), ti.Vector.field(2, ti.f32)
        fb.dense(ti.ij, (n, n)).place(self.grid_v, self.grid_m, self.grid_e, self.grid_w,
                                      self.cells, self.cell_count, self.bc_v)
        self.img = ti.Vector.field(3, ti.u8)
        fb.dense(ti.ij, (p["res"], p["res"])).place(self.img)
        self._tree = fb.finalize()

        self.cells.from_numpy(cells)
        self.bc_v.from_numpy(np.stack([m["inlet_vx"], m["inlet_vy"]], axis=-1).astype(np.float32))
        self.t = 0.0
        self.last_step_ms = 0.0
        self.set_params(p)
        self.reset()

    # ------------------------------------------------------------ paramètres à chaud
    def set_params(self, params: dict) -> None:
        """Applique les paramètres non structurels (matériaux, gravité, CFL...) et recalcule dt."""
        for k, v in params.items():
            if k not in STRUCTURAL:
                self.p[k] = v
        p = self.p
        self.dx = 1.0 / p["n"]
        self.inv_dx = float(p["n"])
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
            speeds.append(float(np.hypot(self.bc_v.to_numpy()[..., 0], self.bc_v.to_numpy()[..., 1]).max()))
        self.dt = p["cfl"] * self.dx / max(speeds)
        self.tau_D = max(p["tau_D"], 2.0 * self.dt)    # dD <= dt / tau_D doit rester < 1 par pas

    # ------------------------------------------------------------ cycle de vie
    def reset(self) -> None:
        """État initial : un transfert CPU -> GPU des positions semées, puis kernels d'initialisation."""
        self.x_f.from_numpy(self._x0f)
        self.v_f.from_numpy(self._v0f)
        K.init_pool(self.alive, self.x_f, self.C_f, self.J_f, self.n_fluid_init, self.free_stack, self.free_top)
        self.x_s.from_numpy(self._x0s)
        self.v_s.from_numpy(self._v0s)
        K.init_solid_state(self.C_s, self.F_s, self.D_s, self.broken_s)
        self.t = 0.0

    def release(self) -> None:
        if self._tree is not None:
            self._tree.destroy()
            self._tree = None

    # ------------------------------------------------------------ simulation
    def _substep(self) -> None:
        p, dt, dx, inv_dx, n, bound = self.p, self.dt, self.dx, self.inv_dx, self.p["n"], self.p["bound"]
        if self.has_fluid:
            K.P2G_fluid(self.grid_m, self.grid_v, self.x_f, self.v_f, self.C_f, self.J_f, self.alive,
                        inv_dx, dt, dx, p["fluid_E"], self.p_mass_f, self.p_vol)
        else:
            self.grid_m.fill(0.0)
            self.grid_v.fill(0.0)
        if self.has_solid:
            P2G_solid(self.grid_m, self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.D_s, self.broken_s,
                      inv_dx, dt, dx, self.mu_s, self.la_s, self.p_mass_s, self.p_vol, p["k_res"])
        K.grid_update(self.grid_m, self.grid_v, self.cells, self.bc_v, dt, p["gravity"], bound, n)
        if self.has_fluid:
            K.G2P_fluid(self.grid_v, self.x_f, self.v_f, self.C_f, self.J_f, self.alive,
                        self.cells, self.bc_v, inv_dx, dt, dx, n)
        if self.has_solid:
            G2P_solid(self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.broken_s, inv_dx, dt, dx, bound)
            if p["use_damage"]:
                clear_eps(self.grid_e, self.grid_w)
                scatter_eps(self.grid_e, self.grid_w, self.x_s, self.F_s, self.broken_s, inv_dx)
                update_damage(self.grid_e, self.grid_w, self.x_s, self.F_s, self.D_s, self.broken_s,
                              inv_dx, dt, p["eps0"], p["epsf"], self.tau_D, int(p["use_rupture"]))
        if self.has_fluid:
            K.advect_fluid(self.x_f, self.v_f, self.alive, self.cells, self.cell_count, inv_dx, dt, bound, dx,
                           n, self.free_stack, self.free_top)
            if self.has_inlet:
                K.emit(self.x_f, self.v_f, self.C_f, self.J_f, self.alive, self.cells, self.bc_v, self.cell_count,
                       p["ppc"] * p["ppc"], self.free_stack, self.free_top, dx, bound)

    def step(self, substeps: int | None = None) -> None:
        t0 = time.perf_counter()
        for _ in range(substeps or self.p["substeps"]):
            self._substep()
        ti.sync()
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        self.t += (substeps or self.p["substeps"]) * self.dt

    # ------------------------------------------------------------ lectures (GPU -> CPU à la demande)
    def stats(self) -> dict:
        n_broken, d_max = 0, 0.0
        if self.has_solid:
            s = solid_stats(self.D_s, self.broken_s)
            n_broken, d_max = int(s[0]), float(s[1])
        return {"t": self.t, "dt": self.dt, "n_fluid": int(K.count_alive(self.alive)) if self.has_fluid else 0,
                "capacity": self.capacity, "n_solid": self.n_solid, "n_broken": n_broken, "D_max": d_max,
                "ms": self.last_step_ms}

    def positions(self) -> np.ndarray:
        """(N, 2) positions des particules fluides vivantes."""
        return self.x_f.to_numpy()[self.alive.to_numpy() == 1] if self.has_fluid else np.zeros((0, 2), np.float32)

    def velocities(self) -> np.ndarray:
        return self.v_f.to_numpy()[self.alive.to_numpy() == 1] if self.has_fluid else np.zeros((0, 2), np.float32)

    def solid_positions(self) -> np.ndarray:
        return self.x_s.to_numpy() if self.has_solid else np.zeros((0, 2), np.float32)

    def damage(self) -> np.ndarray:
        return self.D_s.to_numpy() if self.has_solid else np.zeros(0, np.float32)

    def render(self, x0: float = 0.0, y0: float = 0.0, scale: float = 1.0,
               grid: bool = False, tint: bool = False) -> np.ndarray:
        """Vue [x0, x0 + 1/scale] × [y0, y0 + 1/scale] rendue sur le GPU, puis copiée : (res, res, 3) u8."""
        res = self.p["res"]
        if self.has_solid:
            solid_colors(self.F_s, self.D_s, self.broken_s, self.col_s, int(self.p["color_mode"]), self.p["epsf"])
        r_px = max(0, int(0.5 * self.p_spacing * res * scale + 0.5))
        K.render(self.img, res, x0, y0, scale, self.cells, self.p["n"], int(grid), int(tint),
                 self.x_f, self.alive, int(self.has_fluid), 0.35, 0.65, 1.0, r_px,
                 self.x_s, self.col_s, int(self.has_solid), r_px)
        return self.img.to_numpy()

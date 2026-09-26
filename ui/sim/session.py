"""SimSession : possède les champs Taichi (via FieldsBuilder), le pas de temps et le dessin GGUI.

Trafic GPU <-> CPU : semis CPU -> GPU au build/reset uniquement ; pendant la simulation rien ne
redescend, sauf le vec2 de solid_stats() (2 floats). Sur Vulkan le dessin est un kernel qui écrit
dans une ti.Texture présentée telle quelle par GGUI (draw_texture) ; ailleurs, repli sur les
primitives GGUI (draw_canvas), qui passent par des tampons numpy dans Taichi 1.7.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import taichi as ti

from APIC.APIC import G2P, P2G, grid_step
from Code_tuto.mpm_solid import (G2P_solid, P2G_solid, clear_eps, scatter_eps, solid_colors,
                                 solid_stats, update_damage)
from ui.model.project import Project
from ui.sim.kernels import (advect_fluid, clear_grid, init_fluid_state, init_solid_state,
                            render_background, render_scene)
from ui.sim.seeding import build_mask, particle_spacing, seed_fluid, seed_solid


@dataclass
class Derived:
    """Scalaires passés aux kernels, recalculés par apply_runtime()."""
    n_grid: int = 250
    bound: int = 3
    dx: float = 1.0 / 250
    inv_dx: float = 250.0
    dt: float = 1e-4
    g: float = 9.81
    p_spacing: float = 0.5 / 250
    # fluide
    E_f: float = 400.0
    p_mass_f: float = 0.0
    p_vol_f: float = 0.0
    fluid_color: tuple = (0.35, 0.65, 1.0)
    # solide
    mu_s: float = 0.0
    la_s: float = 0.0
    p_mass_s: float = 0.0
    p_vol_s: float = 0.0
    eps0: float = 0.05
    epsf: float = 0.20
    tau_D: float = 2e-3
    k_res: float = 1e-3
    use_damage: int = 1
    use_rupture: int = 1
    color_mode: int = 0


_FIELD_NAMES = ("x_f", "v_f", "C_f", "J_f", "x_s", "v_s", "C_s", "F_s", "D_s", "broken_s", "col_s",
                "grid_v", "grid_m", "grid_e", "grid_w", "mask", "bg")


class SimSession:
    def __init__(self) -> None:
        self._tree = None
        self.built = False
        self.has_fluid = False
        self.has_solid = False
        self.n_fluid = 0
        self.n_solid = 0
        self.t = 0.0
        self.last_step_ms = 0.0
        self.d = Derived()
        self._clear_refs()

    # ------------------------------------------------------------ cycle de vie
    def _clear_refs(self) -> None:
        for name in _FIELD_NAMES:
            setattr(self, name, None)
        self._x0f = self._v0f = self._x0s = self._v0s = None

    def release(self) -> None:
        if self._tree is not None:
            self._tree.destroy()
        self._tree = None
        self.built = False
        self._clear_refs()

    def build(self, project: Project) -> None:
        """Alloue les champs à la taille du projet, sème les particules, initialise l'état."""
        self.release()
        errors = project.validate()
        if errors:
            raise ValueError(" ; ".join(errors))

        n_grid = project.domain.n_grid
        rng = np.random.default_rng(project.solver.seed)
        x0f, v0f = seed_fluid(project, rng)
        x0s, v0s = seed_solid(project)
        self.has_fluid, self.has_solid = len(x0f) > 0, len(x0s) > 0
        self.n_fluid, self.n_solid = len(x0f), len(x0s)
        # dense(ti.i, 0) est invalide : une particule factice, jamais utilisée (phases gardées par has_*)
        if not self.has_fluid:
            x0f, v0f = np.full((1, 2), 0.5, np.float32), np.zeros((1, 2), np.float32)
        if not self.has_solid:
            x0s, v0s = np.full((1, 2), 0.5, np.float32), np.zeros((1, 2), np.float32)
        self._x0f, self._v0f, self._x0s, self._v0s = x0f, v0f, x0s, v0s

        fb = ti.FieldsBuilder()
        self.x_f = ti.Vector.field(2, ti.f32)
        self.v_f = ti.Vector.field(2, ti.f32)
        self.C_f = ti.Matrix.field(2, 2, ti.f32)
        self.J_f = ti.field(ti.f32)
        fb.dense(ti.i, len(x0f)).place(self.x_f, self.v_f, self.C_f, self.J_f)

        self.x_s = ti.Vector.field(2, ti.f32)
        self.v_s = ti.Vector.field(2, ti.f32)
        self.C_s = ti.Matrix.field(2, 2, ti.f32)
        self.F_s = ti.Matrix.field(2, 2, ti.f32)
        self.D_s = ti.field(ti.f32)
        self.broken_s = ti.field(ti.i32)
        self.col_s = ti.Vector.field(3, ti.f32)
        fb.dense(ti.i, len(x0s)).place(self.x_s, self.v_s, self.C_s, self.F_s, self.D_s,
                                       self.broken_s, self.col_s)

        self.grid_v = ti.Vector.field(2, ti.f32)
        self.grid_m = ti.field(ti.f32)
        self.grid_e = ti.field(ti.f32)
        self.grid_w = ti.field(ti.f32)
        self.mask = ti.field(ti.f32)
        self.bg = ti.Vector.field(3, ti.f32)
        fb.dense(ti.ij, (n_grid, n_grid)).place(self.grid_v, self.grid_m, self.grid_e, self.grid_w,
                                                self.mask, self.bg)
        self._tree = fb.finalize()

        self.mask.from_numpy(build_mask(project, n_grid))
        render_background(self.bg, self.mask)
        self.apply_runtime(project)
        self.built = True
        self.reset()

    def reset(self) -> None:
        """Remet les particules à leur état initial (un seul transfert CPU -> GPU)."""
        self.x_f.from_numpy(self._x0f)
        self.v_f.from_numpy(self._v0f)
        init_fluid_state(self.C_f, self.J_f)
        self.x_s.from_numpy(self._x0s)
        self.v_s.from_numpy(self._v0s)
        init_solid_state(self.C_s, self.F_s, self.D_s, self.broken_s)
        self.t = 0.0

    def apply_runtime(self, project: Project) -> None:
        """Recalcule les scalaires (dont dt) sans rien réallouer."""
        d = self.d
        d.n_grid = project.domain.n_grid
        d.bound = project.domain.bound
        d.dx = 1.0 / d.n_grid
        d.inv_dx = float(d.n_grid)
        d.g = project.solver.gravity
        d.p_spacing = particle_spacing(project)
        p_vol = d.p_spacing ** 2
        d.use_damage = int(project.solver.use_damage)
        d.use_rupture = int(project.solver.use_rupture)
        d.color_mode = 0 if project.solver.color_mode == "damage" else 1

        speeds = []
        fm = project.fluid_material()
        if fm is not None:
            d.E_f = fm.E
            d.p_vol_f = p_vol
            d.p_mass_f = p_vol * fm.rho
            d.fluid_color = tuple(fm.color)
            speeds.append((fm.E / fm.rho) ** 0.5)
        sm = project.solid_material()
        if sm is not None:
            d.mu_s = sm.E / (2 * (1 + sm.nu))
            d.la_s = sm.E * sm.nu / ((1 + sm.nu) * (1 - 2 * sm.nu))
            d.p_vol_s = p_vol
            d.p_mass_s = p_vol * sm.rho
            d.eps0, d.epsf, d.k_res = sm.eps0, sm.epsf, sm.k_res
            speeds.append(((d.la_s + 2 * d.mu_s) / sm.rho) ** 0.5)
        c_max = max(speeds) if speeds else 1.0
        d.dt = project.solver.cfl * d.dx / c_max
        if sm is not None:
            d.tau_D = max(sm.tau_D, 2.0 * d.dt)      # dD <= dt / tau_D doit rester < 1 par pas

    # ------------------------------------------------------------ simulation
    def _substep(self) -> None:
        d = self.d
        if self.has_fluid:
            P2G(self.grid_m, self.grid_v, self.x_f, self.v_f, self.C_f, self.J_f,
                d.inv_dx, d.dt, d.dx, d.E_f, d.p_mass_f, d.p_vol_f)
        else:
            clear_grid(self.grid_m, self.grid_v)
        if self.has_solid:
            P2G_solid(self.grid_m, self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.D_s, self.broken_s,
                      d.inv_dx, d.dt, d.dx, d.mu_s, d.la_s, d.p_mass_s, d.p_vol_s, d.k_res)
        grid_step(self.grid_m, self.grid_v, self.mask, d.dt, d.g, d.bound, d.n_grid)
        if self.has_fluid:
            G2P(self.grid_m, self.grid_v, self.x_f, self.v_f, self.C_f, self.J_f,
                d.inv_dx, d.dt, d.dx, d.E_f, d.p_mass_f, d.p_vol_f)
        if self.has_solid:
            G2P_solid(self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.broken_s,
                      d.inv_dx, d.dt, d.dx, d.bound)
            if d.use_damage:
                clear_eps(self.grid_e, self.grid_w)
                scatter_eps(self.grid_e, self.grid_w, self.x_s, self.F_s, self.broken_s, d.inv_dx)
                update_damage(self.grid_e, self.grid_w, self.x_s, self.F_s, self.D_s, self.broken_s,
                              d.inv_dx, d.dt, d.eps0, d.epsf, d.tau_D, d.use_rupture)
        if self.has_fluid:
            advect_fluid(self.x_f, self.v_f, d.dt, d.bound, d.dx)

    def step(self, n: int) -> None:
        t0 = time.perf_counter()
        for _ in range(n):
            self._substep()
        ti.sync()
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        self.t += n * self.d.dt

    # ------------------------------------------------------------ affichage / stats
    def draw_texture(self, tex, res: int, seg_a, seg_b, seg_c, n_seg: int) -> None:
        """Rendu zéro copie (Vulkan) : un kernel dessine tout dans la texture présentée par GGUI."""
        d = self.d
        if self.has_solid:
            solid_colors(self.F_s, self.D_s, self.broken_s, self.col_s, d.color_mode, d.epsf)
        r_px = max(0, int(0.5 * d.p_spacing * res + 0.5))
        render_scene(tex, res, self.bg, d.n_grid,
                     self.x_f, int(self.has_fluid), *d.fluid_color, r_px,
                     self.x_s, self.col_s, int(self.has_solid), r_px,
                     seg_a, seg_b, seg_c, n_seg)

    def draw_canvas(self, canvas) -> None:
        """Repli (CUDA/CPU) : primitives GGUI, qui transitent par des tampons numpy dans Taichi 1.7."""
        d = self.d
        canvas.set_image(self.bg)
        if self.has_fluid:
            canvas.circles(self.x_f, radius=0.5 * d.p_spacing, color=d.fluid_color)
        if self.has_solid:
            solid_colors(self.F_s, self.D_s, self.broken_s, self.col_s, d.color_mode, d.epsf)
            canvas.circles(self.x_s, radius=0.6 * d.p_spacing, per_vertex_color=self.col_s)

    def stats(self) -> dict:
        n_broken, d_max = 0, 0.0
        if self.has_solid:
            s = solid_stats(self.D_s, self.broken_s)
            n_broken, d_max = int(s[0]), float(s[1])
        return {"t": self.t, "dt": self.d.dt, "n_fluid": self.n_fluid, "n_solid": self.n_solid,
                "n_broken": n_broken, "D_max": d_max, "last_step_ms": self.last_step_ms}

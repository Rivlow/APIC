"""Solver : reçoit un dict de paramètres, des matrices (nx × ny ou nx × ny × nz) et la table des parois, alloue
les champs Taichi, calcule. 2D ou 3D (`dim`), même code.

    solver = Solver(params, matrices, wall_table)     # semis + allocation + état initial
    solver.step(20)                       # 20 sous-pas
    solver.stats() ; solver.positions()   # lectures GPU -> CPU à la demande
    solver.render(x0, y0, k=..., size=(w, h)) -> image (h, w, 3) u8 d'une vue 2D quelconque (défaut : boîte entière)
    solver.render3d(camera, size=(w, h))  -> image (h, w, 3) u8 de la vue 3D (voir ui/render3d.py)

Pendant la simulation rien ne quitte le GPU, sauf ce que l'on demande (stats, positions, image rendue).
"""
from __future__ import annotations

import os
import time

import numpy as np
import taichi as ti

from Solver import bc_expr
from Solver.physics import grid_step
from Solver.walls import frame
from ui import kernels as K
from ui import kernels_inc as M
from ui.kernels_solid import (G2P_solid, P2G_solid, clear_eps, scatter_eps, solid_colors,
                              solid_stats, update_damage)
from ui.mgpcg import MGPCG

STRUCTURAL = ("dim", "nx", "ny", "nz", "Lx", "Ly", "Lz", "bound", "ppc", "capacity", "seed", "res",
              "incompressible")   # le reste s'applique à chaud

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
    """Seed ppc^dim uniform random points per active cell.

    **Inputs**

    - `cells` : np.ndarray bool (n)   active cells
    - `ppc` : int                    particles per cell side
    - `rng` : np.random.Generator
    - `dx` : float                   cell size

    **Outputs**

    - np.ndarray f64 (N, dim)   positions in domain units
    """
    idx = np.argwhere(cells)
    base = np.repeat(idx, ppc ** cells.ndim, axis=0).astype(np.float64)
    return (base + rng.random(base.shape)) * dx


def _solid_points(cells: np.ndarray, ppc: int, dx: float) -> np.ndarray:
    """Seed a regular ppc^dim lattice per active cell.

    **Inputs**

    - `cells` : np.ndarray bool (n)   active cells
    - `ppc` : int                    particles per cell side
    - `dx` : float                   cell size

    **Outputs**

    - np.ndarray f64 (N, dim)   positions in domain units
    """
    dim = cells.ndim
    idx = np.argwhere(cells)
    o = (np.arange(ppc) + 0.5) / ppc
    offsets = np.stack([g.ravel() for g in np.meshgrid(*([o] * dim), indexing="ij")], axis=1)
    return (idx[:, None, :] + offsets[None, :, :]).reshape(-1, dim) * dx


def _signed_distance(obstacle: np.ndarray, dx: float) -> np.ndarray:
    """Signed distance to the obstacles at cell centers (surface = cell faces).

    **Inputs**

    - `obstacle` : np.ndarray bool (n)
    - `dx` : float cell size (m)

    **Outputs**

    - np.ndarray f32 (n) : > 0 outside, < 0 inside (m)
    """
    from scipy.ndimage import distance_transform_edt
    out = distance_transform_edt(~obstacle) * dx - 0.5 * dx      # cellules libres : distance au centre obstacle le plus proche
    inside = distance_transform_edt(obstacle) * dx - 0.5 * dx
    return np.where(obstacle, -inside, out).astype(np.float32)


def _boundary_volume(solid: np.ndarray) -> np.ndarray:
    """Boundary volume map : density contributed by wall / obstacle cells, as if filled with fluid at rest.

    **Inputs**

    - `solid` : np.ndarray bool (n) boundary cells

    **Outputs**

    - np.ndarray f32 (n) : in [0, 1], = 1 inside a thick boundary

    **Note** : quadratic B-spline integrated over one cell = [1/6, 2/3, 1/6] per axis (separable convolution).
    """
    from scipy.ndimage import correlate1d
    b = solid.astype(np.float64)
    for ax in range(b.ndim):
        b = correlate1d(b, [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], axis=ax, mode="constant")
    return b.astype(np.float32)


def _wall_band(shape: tuple, bound: int, wtype: np.ndarray, open_type: int) -> np.ndarray:
    """Wall band cells (bound layers along every face), open where the face is an outlet.

    **Inputs**

    - `shape` : tuple cells per axis ; `bound` : int
    - `wtype` : np.ndarray int32 (2 dim, na, nb) wall types ; `open_type` : int type left open (OUTLET)

    **Outputs**

    - np.ndarray bool (shape)
    """
    dim = len(shape)
    band = np.zeros(shape, bool)
    for s in range(2 * dim):
        a, plus, ts = frame(s, dim)
        sl = [slice(None)] * dim
        sl[a] = slice(shape[a] - bound, None) if plus else slice(0, bound)
        lens = [shape[t] for t in ts]
        closed = wtype[s, :lens[0], :(lens[1] if dim == 3 else 1)] != open_type
        closed = np.expand_dims(closed.reshape(lens), a)
        band[tuple(sl)] |= closed
    return band


class Solver:
    def __init__(self, params: dict, matrices: dict, wall_table, consts: dict | None = None, rotor: dict | None = None):
        """Seed particles, allocate Taichi fields and set the initial state.

        **Inputs**

        - `params` : dict   simulation parameters (dim, nx, ny[, nz], Lx, …)
        - `matrices` : dict of np.ndarray (n)   fluid/solid/obstacle masks, vx0/vy0[/vz0]
        - `wall_table` : Solver.walls.WallTable (2 dim, na, nb)   type, v, depth, pressure, friction, expr
        - `consts` : dict | None   constants of the boundary expressions
        - `rotor` : dict | None   rotating obstacle (SimulationRunner.rotor : mask at angle 0, center, axis, omega),
          incompressible only
        """
        self.arch = ensure_taichi()
        self.p = dict(params)
        p = self.p
        self.dim = dim = int(p.get("dim", 2))
        nx, ny, ppc, bound = p["nx"], p["ny"], p["ppc"], p["bound"]
        self.shape = shape = (nx, ny) if dim == 2 else (nx, ny, int(p["nz"]))
        dx0 = p["Lx"] / nx                            # cellules carrées, en mètres
        m = matrices
        wt = wall_table
        wtype, wvel, wdepth, wpress, wfric, wexpr = wt.type, wt.v, wt.depth, wt.pressure, wt.friction, wt.expr
        # zone utilisable : le stencil 3^dim sortirait de la grille dans la bande de paroi
        lo = np.full(dim, bound) * dx0
        hi = (np.array(shape) - bound) * dx0

        # ---- semis (numpy, une fois) ; vitesse initiale = celle de la cellule d'origine. Jamais dans la bande de
        # paroi : ses particules seraient ramenées sur la première couche intérieure (surdensité, 2,5x dans une
        # tranche 3D de 10 cellules dont 6 de bande)
        inner = np.zeros(shape, bool)
        inner[tuple(slice(bound, n - bound) for n in shape)] = True
        rng = np.random.default_rng(p["seed"])
        xf = _fluid_points(m["fluid"] & inner, ppc, rng, dx0)
        xs = _solid_points(m["solid"] & inner, ppc, dx0)
        vkeys = ("vx0", "vy0", "vz0")[:dim]

        def vel_of(pts):
            c = np.clip((pts / dx0).astype(int), 0, np.array(shape) - 1)
            ci = tuple(c[:, a] for a in range(dim))
            return np.column_stack([m[k][ci] for k in vkeys]) if len(pts) else np.zeros((0, dim))

        vf, vs = vel_of(xf), vel_of(xs)
        xf, xs = np.clip(xf, lo, hi), np.clip(xs, lo, hi)

        # ---- grille des cellules (bits) ; les conditions aux limites sont sur les parois (wtype, wvel)
        cells = (m["fluid"] * K.FLUID0 | m["solid"] * K.SOLID0 | m["obstacle"] * K.OBSTACLE).astype(np.int32)
        self.has_inlet, self.has_outlet = bool((wtype == K.INLET).any()), bool((wtype == K.OUTLET).any())
        self.incompressible = bool(p.get("incompressible", False))
        # sortie à pression imposée p > 0 : l'eau peut aussi y rentrer (mode incompressible)
        self.has_pressure_outlet = (bool((wpress > 0).any()) or bool((wexpr[..., dim] >= 0).any())) \
            and self.incompressible
        if not self.incompressible and (wpress != 0).any():     # pression imposée ignorée : sorties libres
            print("[solver] pression imposée en sortie ignorée : elle n'agit que sur la projection "
                  "incompressible (incompressible=True)")
            wpress = np.zeros_like(wpress)

        # ---- capacités
        self.n_fluid_init = len(xf)
        self.has_fluid = self.n_fluid_init > 0 or self.has_inlet
        cap = self.n_fluid_init
        if self.has_inlet or self.has_pressure_outlet:
            cap = max(cap, p["capacity"] or ppc ** dim * int(np.prod(shape)))
        self.capacity = max(cap, 1)                    # dense(ti.i, 0) est invalide
        self.n_solid = len(xs)
        self.has_solid = self.n_solid > 0
        self._x0f = np.zeros((self.capacity, dim), np.float32)
        self._v0f = np.zeros((self.capacity, dim), np.float32)
        self._x0f[:self.n_fluid_init], self._v0f[:self.n_fluid_init] = xf, vf
        self._x0s = (xs if self.has_solid else np.full((1, dim), 0.5)).astype(np.float32)
        self._v0s = (vs if self.has_solid else np.zeros((1, dim))).astype(np.float32)

        # ---- champs Taichi (un FieldsBuilder, libéré d'un bloc par release())
        ax = ti.ij if dim == 2 else ti.ijk
        fb = ti.FieldsBuilder()
        self.x_f, self.v_f = ti.Vector.field(dim, ti.f32), ti.Vector.field(dim, ti.f32)
        self.C_f, self.J_f = ti.Matrix.field(dim, dim, ti.f32), ti.field(ti.f32)
        self.alive, self.free_stack = ti.field(ti.i32), ti.field(ti.i32)
        self.sc_f = ti.field(ti.f32)                   # quantité colorée par particule (voir fluid_mode)
        fb.dense(ti.i, self.capacity).place(self.x_f, self.v_f, self.C_f, self.J_f, self.alive, self.free_stack,
                                            self.sc_f)
        self.free_top = ti.field(ti.i32)
        fb.dense(ti.i, 1).place(self.free_top)

        self.x_s, self.v_s = ti.Vector.field(dim, ti.f32), ti.Vector.field(dim, ti.f32)
        self.C_s, self.F_s = ti.Matrix.field(dim, dim, ti.f32), ti.Matrix.field(dim, dim, ti.f32)
        self.D_s, self.broken_s, self.col_s = ti.field(ti.f32), ti.field(ti.i32), ti.Vector.field(3, ti.f32)
        fb.dense(ti.i, len(self._x0s)).place(self.x_s, self.v_s, self.C_s, self.F_s, self.D_s, self.broken_s, self.col_s)

        self.grid_v, self.grid_m = ti.Vector.field(dim, ti.f32), ti.field(ti.f32)
        self.grid_e, self.grid_w = ti.field(ti.f32), ti.field(ti.f32)
        self.cells = ti.field(ti.i32)
        self.sdf = ti.field(ti.f32)                    # distance signée aux obstacles (m, < 0 dedans)
        fb.dense(ax, shape).place(self.grid_v, self.grid_m, self.grid_e, self.grid_w, self.cells, self.sdf)
        self.wall_type, self.wall_v, self.emit_acc = ti.field(ti.i32), ti.Vector.field(dim, ti.f32), ti.field(ti.f32)
        self.wall_d = ti.field(ti.i32)                 # position du mur (cellules depuis le bord) : bande ou face d'obstacle
        self.wall_p = ti.field(ti.f32)                 # pression imposée sur une sortie (0 : sortie libre)
        self.wall_f = ti.field(ti.f32)                 # frottement des murs (0 glissant, 1 adhérent)
        self.wall_e = ti.Vector.field(dim + 1, ti.i32)  # expression dépendant de t pour (v…, p), -1 = aucune
        fb.dense(ti.ijk, wtype.shape).place(self.wall_type, self.wall_v, self.emit_acc, self.wall_d, self.wall_p,
                                            self.wall_f, self.wall_e)
        self.img = ti.Vector.field(3, ti.u8)
        # image aux proportions du domaine (plan x-y) : res pixels sur le grand côté
        if nx >= ny:
            self.res_x, self.res_y = int(p["res"]), max(1, int(round(p["res"] * ny / nx)))
        else:
            self.res_x, self.res_y = max(1, int(round(p["res"] * nx / ny))), int(p["res"])
        fb.dense(ti.ij, (int(p["res"]), int(p["res"]))).place(self.img)   # tampon maximal : on n'en rend qu'une partie
        self.view3d = None
        if dim == 3:                                   # rendu 3D : profondeur + table des primitives (ui/render3d.py)
            from ui.render3d import View3D
            self.view3d = View3D(fb, int(p["res"]))
        self.mac = self.mg = None
        if self.incompressible:                        # grille décalée + gradient conjugué multigrille
            self.mac = M.MAC(fb, shape)
            self.mg = MGPCG(fb, self.mac.ctype, self.mac.r, self.mac.pd, self.mac.Ap, self.mac.cg, shape)
        self._tree = fb.finalize()

        self.cells.from_numpy(cells)
        # obstacles : distance signée (une fois), garde-fou contre la pénétration à l'advection
        self.use_sdf = int(bool(m["obstacle"].any()))
        # obstacle tournant (roue) : distance signée à l'angle 0, une fois ; sa position suit theta = omega t
        self.rotor = rotor if (rotor is not None and self.incompressible and rotor["mask"].any()) else None
        if rotor is not None and self.rotor is None:
            print("[solver] rotor ignoré : il faut le mode incompressible et une primitive non vide")
        self.rot_axis = ti.Vector([0.0, 1.0, 0.0])
        self.rot_omega = 0.0
        if self.rotor is not None:
            self.mac.rsdf.from_numpy(_signed_distance(self.rotor["mask"], dx0))
            self.mac.rc[None] = list(map(float, self.rotor["center"]))
            self.rot_axis = ti.Vector(list(map(float, self.rotor["axis"])))
            self.rot_omega = float(self.rotor["omega"])
        self.sdf.from_numpy(_signed_distance(m["obstacle"], dx0) if self.use_sdf
                            else np.full(shape, 1e3, np.float32))
        if self.incompressible:                        # volume map des parois (statique) pour la densité ;
            band = _wall_band(shape, bound, wtype, K.OUTLET)   # sortie (libre ou sous pression) : pas de paroi
            self.mac.bvol.from_numpy(_boundary_volume(m["obstacle"] | band))
        self.wall_type.from_numpy(np.ascontiguousarray(wtype, np.int32))
        self.wall_v.from_numpy(np.ascontiguousarray(wvel, np.float32))
        self.wall_d.from_numpy(np.ascontiguousarray(wdepth, np.int32))
        self.wall_p.from_numpy(np.ascontiguousarray(wpress, np.float32))
        self.wall_f.from_numpy(np.ascontiguousarray(wfric, np.float32))
        self.wall_e.from_numpy(np.ascontiguousarray(wexpr, np.int32))
        # conditions limites dépendant du temps : un kernel généré réécrit wall_v / wall_p à chaque sous-pas
        self._bc_eval = bc_expr.build_kernel(list(wt.sources), consts, dim) if wt.sources else None
        self.t = 0.0
        self.last_step_ms = 0.0
        self.cg_last_iters = 0
        self.fluid_mode = K.MODE_DENSITY               # identifiant de mode (kernels.fluid_modes)
        self.scalar_max = 0.0                          # échelle de couleur, mise à jour dans stats()
        speeds = np.linalg.norm(wvel, axis=-1)[wtype == K.INLET]
        self._inlet_speed = float(speeds.max()) if speeds.size else 0.0
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
        self.nz = self.shape[2] if self.dim == 3 else 1
        self.dx = p["Lx"] / self.nx                    # cellules carrées, en mètres
        self.inv_dx = 1.0 / self.dx
        self.p_spacing = self.dx / p["ppc"]
        p_vol = self.p_spacing ** self.dim
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
            self.mac.q.fill(0.0)
            self.mac.fp.fill(0.0)
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

    def _viscosity(self) -> None:
        """Implicit viscous diffusion (I - dt div(nu grad)) u* = u on the MAC faces, by conjugate gradient.

        **Outputs**

        - mac.nu_c (cell viscosity), mac.uf updated (no-op if fluid_nu <= 0 and smagorinsky <= 0)

        **Note** : nu = fluid_nu + Smagorinsky (smagorinsky dx)² |S| from the pre-diffusion velocities; fixed
        `visc_iters` iterations per component, no read-back; face values set by the BCs (walls, inlets, friction
        ghosts) are Dirichlet, air faces Neumann.
        """
        p = self.p
        nu0, cs = max(p["fluid_nu"], 0.0), max(p["smagorinsky"], 0.0)
        if (nu0 <= 0 and cs <= 0) or p["visc_iters"] <= 0:
            return
        k = self.dt / self.dx ** 2
        self.mac.visc_nu(nu0, (cs * self.dx) ** 2, self.inv_dx)
        for a in range(self.dim):
            self.mac.visc_classify(a)
            self.mac.visc_cg_init(a, k)
            for _ in range(int(p["visc_iters"])):
                self.mac.visc_apply(a, k)
                self.mac.visc_update(a)

    def _cg_solve(self, x, iters: int) -> None:
        """Run `iters` CG iterations on the cell Laplacian after cg_init / density_cg_init.

        **Inputs**

        - `x` : f32 field (n) unknown (q or phi), updated in place
        - `iters` : int fixed iteration count

        **Outputs**

        - x, r, pd, Ap, cg updated

        **Note** : `multigrid` : V-cycle preconditioner (2 L + 1 kernels per iteration); else plain CG, 2 kernels.
        """
        if iters <= 0:
            return
        if self.p["multigrid"]:
            self.mg.init()
            for _ in range(iters):
                self.mg.iterate(x)
        else:
            for _ in range(iters):
                self.mac.cg_apply()
                self.mac.cg_update(x)

    def _project(self) -> None:
        """Pressure projection on the MAC grid by conjugate gradient.

        **Outputs**

        - mac.q (pressure), mac.uf updated

        **Note** : fixed `cg_iters` iterations, fully on GPU with no read-back; warm-started from previous pressure;
        multigrid-preconditioned if `multigrid`.
        """
        p = self.p
        self.mac.cg_init(self.wall_type, self.wall_d, self.wall_p, self.dx, self.dt, 1.0 / p["fluid_rho"],
                         p["bound"])
        self._cg_solve(self.mac.q, int(p["cg_iters"]))
        self.cg_last_iters = p["cg_iters"]
        self.mac.project(self.dx)

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
        self.mac.particle_density(self.x_f, self.alive, self.inv_dx, int(p["ppc"]))
        self.mac.density_cg_init(kappa, self.dx)
        self._cg_solve(self.mac.phi, int(p["density_iters"]))
        self.mac.density_gradient(self.dx)
        self.mac.density_shift(self.x_f, self.alive, self.inv_dx, self.dx, p["bound"])

    def _solid_substep(self, dt_s: float, with_pressure: bool) -> None:
        """Advance the solid alone by one MPM step on the collocated grid.

        **Inputs**

        - `dt_s` : float   solid time step
        - `with_pressure` : bool    add the fluid pressure acceleration mac.fp

        **Outputs**

        - solid particle state (x_s, v_s, C_s, F_s, D_s, broken_s) updated
        """
        p, dx, inv_dx, bound = self.p, self.dx, self.inv_dx, self.p["bound"]
        self.grid_m.fill(0.0)
        self.grid_v.fill(0.0)
        P2G_solid(self.grid_m, self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.D_s, self.broken_s,
                  inv_dx, dt_s, dx, self.mu_s, self.la_s, self.p_mass_s, self.p_vol, p["k_res"])
        grid_step(self.grid_m, self.grid_v, self.cells, self.wall_type, self.wall_v, self.wall_d, self.wall_f,
                  dt_s, p["gravity"], 0.0, p["obstacle_friction"], bound, self.nx, self.ny)
        if with_pressure:
            self.mac.add_accel(self.grid_v, self.grid_m, dt_s)
        G2P_solid(self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.broken_s, inv_dx, dt_s, dx, bound)
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
        p, dt, dx, inv_dx, bound = self.p, self.dt, self.dx, self.inv_dx, self.p["bound"]
        mac = self.mac
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
                      0.0, 0.0, 0.0, p["obstacle_friction"], bound, self.nx, self.ny)
        mac.p2g(self.x_f, self.v_f, self.C_f, self.alive, inv_dx, dx)
        mac.cell_bc(self.wall_type, self.wall_v, self.wall_d, self.wall_f, bound, p["obstacle_friction"])
        mac.classify(self.cells, self.wall_type, self.wall_v, self.wall_d, self.x_f, self.alive, self.x_s,
                     int(self.has_solid), inv_dx, bound, int(p["free_surface"]))
        if self.rotor is not None:                     # roue à l'angle omega t : cellules ROTOR, vitesse omega × r
            mac.rotor_classify(self.rot_axis, self.rot_omega * self.t, inv_dx)
        mac.bc(self.wall_type, self.wall_v, self.wall_d, self.wall_f, self.grid_v, self.grid_m, dt, p["gravity"],
               p["obstacle_friction"], bound, 1, dx, self.rot_axis, self.rot_omega)
        self._viscosity()
        self._project()
        mac.bc(self.wall_type, self.wall_v, self.wall_d, self.wall_f, self.grid_v, self.grid_m, dt, p["gravity"],
               p["obstacle_friction"], bound, 0, dx, self.rot_axis, self.rot_omega)
        if self.has_solid:
            mac.pressure_force(self.grid_m, inv_dx, -(p["fluid_rho"] / p["solid_rho"]) / dt)
        mac.g2p(self.x_f, self.v_f, self.C_f, self.alive, inv_dx, dx)
        self._advect_and_emit(dt)
        self._density_projection()

    def _advect_and_emit(self, dt: float) -> None:
        """Advect fluid particles, then emit at inlets and pressure outlets.

        **Inputs**

        - `dt` : float   time step

        **Outputs**

        - x_f, v_f, alive, free stack updated
        """
        p, dx, inv_dx, bound = self.p, self.dx, self.inv_dx, self.p["bound"]
        ppc_face = float(p["ppc"] ** self.dim)       # flux : ppc^dim particules par cellule et par (v_n dt / dx)
        K.advect_fluid(self.x_f, self.v_f, self.alive, self.wall_type, self.wall_v, self.wall_d, self.sdf, self.cells,
                       self.use_sdf, inv_dx, dt, bound, dx, self.free_stack, self.free_top)
        if self.rotor is not None:                     # roue à sa position en fin de pas
            K.rotor_push(self.x_f, self.v_f, self.alive, self.mac.rsdf, self.mac.rc, self.rot_axis,
                         self.rot_omega * (self.t + dt), self.rot_omega, inv_dx, dx)
        if self.has_inlet:
            K.emit_wall(self.x_f, self.v_f, self.C_f, self.J_f, self.alive, self.wall_type, self.wall_v,
                        self.wall_d, self.emit_acc, ppc_face, self.free_stack, self.free_top, dt, dx, bound,
                        self.cells)
        if self.has_pressure_outlet:
            self.mac.emit_pressure_outlet(self.x_f, self.v_f, self.C_f, self.J_f, self.alive, self.wall_type,
                                          self.wall_d, self.wall_p, self.emit_acc, ppc_face, self.free_stack,
                                          self.free_top, dt, dx, bound)

    # ------------------------------------------------------------ simulation (faiblement compressible)
    def _substep(self) -> None:
        """Advance one weakly compressible MPM substep (fluid + solid on the collocated grid).

        **Outputs**

        - fluid and solid state updated
        """
        p, dt, dx, inv_dx, bound = self.p, self.dt, self.dx, self.inv_dx, self.p["bound"]
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
                  dt, p["gravity"], 0.0, p["obstacle_friction"], bound, self.nx, self.ny)
        if self.has_fluid:
            K.G2P_fluid(self.grid_v, self.x_f, self.v_f, self.C_f, self.J_f, self.alive, inv_dx, dt, dx)
        if self.has_solid:
            G2P_solid(self.grid_v, self.x_s, self.v_s, self.C_s, self.F_s, self.broken_s, inv_dx, dt, dx, bound)
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
            if self._bc_eval is not None:              # valeurs imposées fonction de t, évaluées sur le GPU
                self._bc_eval(self.wall_e, self.wall_v, self.wall_p, self.wall_d, self.t, self.dx, self.nx, self.ny,
                              self.nz)
            if self.incompressible:
                self._substep_incompressible()
            else:
                self._substep()
            self.t += self.dt
        ti.sync()
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0

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
            st["cg_rr"] = float(M.cg_residual(self.mac.cg))
            st["div_max"] = float(self.mac.divergence_max(self.dx))
        if self.rotor is not None:                     # couple de pression de l'eau sur la roue, autour de son axe
            self.mac.rotor_torque(self.p["fluid_rho"] / self.dt, self.dx)
            tq = self.mac.torque[None]
            ax = self.rotor["axis"]
            t_ax = float(tq[0] * ax[0] + tq[1] * ax[1] + tq[2] * ax[2]) if self.dim == 3 else float(tq[2])
            st["rotor_rpm"] = self.rot_omega * 60.0 / (2.0 * np.pi)
            st["rotor_angle"] = float(np.degrees(self.rot_omega * self.t) % 360.0)
            st["rotor_torque"] = t_ax
            st["rotor_power"] = t_ax * self.rot_omega
        if self.has_fluid and self.fluid_mode > 0:    # échelle de couleur lissée (max de la quantité affichée)
            if self.fluid_mode == K.MODE_DENSITY:      # 3 rms : quelques particules de bord (sortie) hors échelle
                m = 3.0 * float(K.scalar_rms(self.sc_f, self.alive))
            else:
                m = float(K.scalar_absmax(self.sc_f, self.alive))
            self.scalar_max = m if self.scalar_max <= 0 else 0.7 * self.scalar_max + 0.3 * m
        st["scalar_max"] = self.scalar_max
        return st

    def positions(self) -> np.ndarray:
        """Read alive fluid particle positions (GPU -> CPU).

        **Outputs**

        - np.ndarray f32 (N, dim)
        """
        if not self.has_fluid:
            return np.zeros((0, self.dim), np.float32)
        return self.x_f.to_numpy()[self.alive.to_numpy() == 1]

    def velocities(self) -> np.ndarray:
        """Read alive fluid particle velocities (GPU -> CPU).

        **Outputs**

        - np.ndarray f32 (N, dim)
        """
        if not self.has_fluid:
            return np.zeros((0, self.dim), np.float32)
        return self.v_f.to_numpy()[self.alive.to_numpy() == 1]

    def solid_positions(self) -> np.ndarray:
        """Read solid particle positions (GPU -> CPU).

        **Outputs**

        - np.ndarray f32 (Ns, dim)
        """
        return self.x_s.to_numpy() if self.has_solid else np.zeros((0, self.dim), np.float32)

    def damage(self) -> np.ndarray:
        """Read solid particle damage D (GPU -> CPU).

        **Outputs**

        - np.ndarray f32 (Ns,)
        """
        return self.D_s.to_numpy() if self.has_solid else np.zeros(0, np.float32)

    # ------------------------------------------------------------ rendu
    def _colors(self) -> tuple:
        """Update per-particle display scalars and solid colors on the GPU.

        **Outputs**

        - (mode int, signed int, inv_smax float) for the render kernels
        """
        if self.has_solid:
            solid_colors(self.F_s, self.D_s, self.broken_s, self.col_s, int(self.p["color_mode"]), self.p["epsf"])
        mode = int(self.fluid_mode)
        if self.has_fluid and mode > 0:
            if self.incompressible:
                if mode == K.MODE_DENSITY:             # à jour même sans correction de densité
                    self.mac.particle_density(self.x_f, self.alive, self.inv_dx, int(self.p["ppc"]))
                self.mac.fluid_scalar(mode, self.x_f, self.v_f, self.alive, self.sc_f,
                                      self.p["fluid_rho"] / self.dt, self.p["fluid_rho"], self.inv_dx)
            else:
                K.fluid_scalar_wc(mode, self.x_f, self.v_f, self.J_f, self.alive, self.grid_v, self.sc_f,
                                  self.p["fluid_E"], self.p["fluid_rho"], self.inv_dx)
        return mode, int(K.mode_signed(mode, self.dim)), 1.0 / max(self.scalar_max, 1e-9)

    def render(self, x0: float = 0.0, y0: float = 0.0, scale: float = 1.0,
               grid: bool = False, tint: bool = False, k: float | None = None, size=None) -> np.ndarray:
        """Render the 2D view to an image on the GPU.

        **Inputs**

        - `x0`, `y0` : float   view bottom-left corner (m)
        - `scale` : float      zoom, used when k is None (view = [x0, x0 + Lx/scale] × [y0, y0 + Ly/scale])
        - `grid`, `tint` : bool    draw mesh / initial-cell tints
        - `k` : float | None   size of one image pixel (m) ; None : whole box at its aspect ratio
        - `size` : tuple[int, int] | None   image (width, height) px, each <= res (with k)

        **Outputs**

        - np.ndarray u8 (height, width, 3)

        **Note** : 3D solvers render with render3d.
        """
        if self.dim == 3:
            raise ValueError("solveur 3D : utiliser render3d(camera, size)")
        if k is None:                                  # boîte entière, aux proportions de la boîte
            res_x, res_y = self.res_x, self.res_y
            k = self.nx * self.dx / (scale * res_x)
        else:                                          # vue quelconque (UI : proportions de la fenêtre)
            res = int(self.p["res"])
            res_x, res_y = (max(1, min(int(v), res)) for v in size)
        r_px = max(0, int(0.5 * self.p_spacing / k + 0.5))
        mode, signed, inv_smax = self._colors()
        K.render(self.img, res_x, res_y, x0, y0, k, self.cells, self.wall_type, self.wall_d, self.inv_dx,
                 self.p["bound"], int(grid), int(tint),
                 self.x_f, self.alive, int(self.has_fluid), 0.35, 0.65, 1.0, r_px,
                 self.sc_f, mode, signed, inv_smax,
                 self.x_s, self.col_s, int(self.has_solid), r_px)
        return self.img.to_numpy()[:res_y, :res_x]

    def render3d(self, camera, size=None, clip=None, prims=None, obs_alpha: float = 1.0) -> np.ndarray:
        """Render the 3D view (spheres with depth, obstacles, walls) to an image on the GPU.

        **Inputs**

        - `camera` : ui.render3d.Camera
        - `size` : tuple[int, int] | None   image (width, height) px, each <= res ; None : res × res
        - `clip` : tuple (axis, position m, keep_below bool) | None   clipping plane for the particles
        - `prims` : unused (primitive overlays are drawn by the UI)
        - `obs_alpha` : float opacity of the fixed obstacles (< 1 : translucent casing, the fluid shows through)

        **Outputs**

        - np.ndarray u8 (height, width, 3)

        **Note** : one GPU -> CPU copy of the image only.
        """
        if self.dim != 3:
            raise ValueError("solveur 2D : utiliser render(x0, y0, k, size)")
        res = int(self.p["res"])
        w, h = (res, res) if size is None else (max(1, min(int(v), res)) for v in size)
        mode, signed, inv_smax = self._colors()
        self.view3d.draw(self, camera, w, h, mode, signed, inv_smax, clip, prims, obs_alpha)
        return self.img.to_numpy()[:h, :w]

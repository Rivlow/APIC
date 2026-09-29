"""Kernels Taichi du solveur : fluide APIC avec réservoir de particules, parois, couleurs, rendu 2D.

Tous les arguments sont explicites (aucune globale). Génériques en dimension (2D ou 3D) : la dimension est lue
à la compilation sur les champs (`x.n` pour un champ de vecteurs, `len(field.shape)` pour une grille). Les kernels
solides (MLS-MPM, endommagement) sont dans ui/kernels_solid.py, la grille décalée incompressible dans
ui/kernels_inc.py (classe MAC), le rendu 3D dans ui/render3d.py.

Grille `cells` (i32, forme n = (nx, ny[, nz]), cellules carrées, indexée [i, j(, k)] = (x, y(, z))) : bits
FLUID0 = 1 (eau initiale), SOLID0 = 2, OBSTACLE = 4.  Les conditions aux limites (parois, `wall_type` /
`wall_v`, mise à jour de la grille) sont dans Solver/boundary.py et Solver/physics.py ; ici, seulement ce qui
dépend du réservoir de particules : une sortie (OUTLET, ou entrée à vitesse sortante) détruit les particules qui
franchissent le mur, une entrée (INLET) en émet par flux.

Réservoir fluide : capacité fixe, drapeau `alive`, particules mortes garées en (-1, …) et ignorées partout ; pile
de slots libres (`free_stack`, `free_top[0]`) sur le GPU : une sortie y pousse les particules détruites, une
entrée y prend les slots des particules émises. Rien ne redescend au CPU.
"""
import taichi as ti

from Solver.boundary import (FRAMES, INLET, OBSTACLE, OUTLET, WALL,  # noqa: F401
                             band_cell, exit_at)

FLUID0, SOLID0 = 1, 2


@ti.func
def cell_of(xp, inv_dx: float, n: ti.template()):
    """Cell containing a position, clamped to the grid.

    **Inputs**

    - `xp` : vec dim f32 position
    - `inv_dx` : float
    - `n` : static grid shape

    **Outputs**

    - ivec dim cell index
    """
    c = (xp * inv_dx).cast(int)
    return ti.math.clamp(c, 0, ti.Vector(n) - 1)


@ti.func
def weights(fx):
    """Quadratic B-spline weights.

    **Inputs**

    - `fx` : vec dim f32 offset from the base node (in cells)

    **Outputs**

    - list of 3 vec dim weights
    """
    return [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]


@ti.func
def wprod(w, o: ti.template()):
    """Tensor-product weight of stencil offset o.

    **Inputs**

    - `w` : list of 3 vec dim weights (see weights)
    - `o` : static ivec dim offset in {0, 1, 2}^dim

    **Outputs**

    - float
    """
    r = 1.0
    for c in ti.static(range(o.n)):
        r *= w[o[c]][c]
    return r


def stencil3(dim: int):
    """Static 3^dim stencil offsets (use inside ti.static).

    **Inputs**

    - `dim` : int

    **Outputs**

    - iterable of ivec dim
    """
    return ti.grouped(ti.ndrange(*([3] * dim)))


# ---------------------------------------------------------------- initialisation
@ti.kernel
def init_pool(alive: ti.template(), x: ti.template(), C: ti.template(), J: ti.template(), n_init: int,
              free_stack: ti.template(), free_top: ti.template()):
    """Init the fluid particle pool: first n_init alive, others free.

    **Inputs**

    - `alive` : i32 field (cap,) 1 = live particle
    - `x` : vec dim f32 field (cap,) positions
    - `C` : mat dim f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `n_init` : int
    - `free_stack` : i32 field (cap,) free slot stack
    - `free_top` : i32 field (1,) stack size

    **Outputs**

    - all fields written in place

    **Note** : dead particles parked at (-1, …).
    """
    dim = ti.static(x.n)
    for p in alive:
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)
        J[p] = 1.0
        if p < n_init:
            alive[p] = 1
        else:
            alive[p] = 0
            x[p] = ti.Vector.zero(ti.f32, dim) - 1.0
            free_stack[p - n_init] = p
    free_top[0] = alive.shape[0] - n_init


@ti.kernel
def init_solid_state(C: ti.template(), F: ti.template(), D: ti.template(), broken: ti.template()):
    """Reset solid particles (C = 0, F = I, D = 0, intact).

    **Inputs**

    - `C`, `F` : mat dim f32 fields (Ns,)
    - `D` : f32 field (Ns,) damage
    - `broken` : i32 field (Ns,)

    **Outputs**

    - all fields written in place
    """
    dim = ti.static(C.n)
    for p in D:
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)
        F[p] = ti.Matrix.identity(ti.f32, dim)
        D[p] = 0.0
        broken[p] = 0


@ti.kernel
def count_alive(alive: ti.template()) -> ti.i32:
    """Count live fluid particles.

    **Inputs**

    - `alive` : i32 field (cap,) 1 = live particle

    **Outputs**

    - i32 count
    """
    n = 0
    for p in alive:
        n += alive[p]
    return n


# ---------------------------------------------------------------- 1. particules -> grille (fluide faiblement compressible)
@ti.kernel
def P2G_fluid(grid_m: ti.template(), grid_v: ti.template(),
              x: ti.template(), v: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
              inv_dx: float, dt: float, dx: float, E: float, p_mass: float, p_vol: float):
    """Particles to grid (weakly compressible APIC): mass, momentum, pressure.

    **Inputs**

    - `grid_m` : f32 field (n) node masses
    - `grid_v` : vec dim f32 field (n) node momenta
    - `x`, `v` : vec dim f32 field (cap,) positions / velocities
    - `C` : mat dim f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `inv_dx`, `dt`, `dx`, `E`, `p_mass`, `p_vol` : float

    **Outputs**

    - grid_m, grid_v written in place

    **Note** : clears the grid first.
    """
    dim = ti.static(x.n)
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, dim)
        grid_m[I] = 0.0
    for p in x:
        if alive[p] == 1:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base
            w = weights(fx)
            stress = -dt * E * p_vol * (J[p] - 1) * 4.0 * inv_dx * inv_dx
            for o in ti.static(stencil3(dim)):
                d_pos = (o - fx) * dx
                weight = wprod(w, o)
                grid_v[base + o] += weight * (p_mass * (v[p] + C[p] @ d_pos) + stress * d_pos)
                grid_m[base + o] += weight * p_mass


# ---------------------------------------------------------------- 2. grille : Solver/physics.py (grid_step)


# ---------------------------------------------------------------- 3. grille -> particules (fluide faiblement compressible)
@ti.kernel
def G2P_fluid(grid_v: ti.template(),
              x: ti.template(), v: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
              inv_dx: float, dt: float, dx: float):
    """Grid to particles (weakly compressible APIC).

    **Inputs**

    - `grid_v` : vec dim f32 field (n) node velocities
    - `x`, `v` : vec dim f32 field (cap,) positions / velocities
    - `C` : mat dim f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `inv_dx`, `dt`, `dx` : float

    **Outputs**

    - v, C, J written in place
    """
    dim = ti.static(x.n)
    for p in x:
        if alive[p] == 1:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base
            w = weights(fx)
            v_new = ti.Vector.zero(ti.f32, dim)
            C_new = ti.Matrix.zero(ti.f32, dim, dim)
            for o in ti.static(stencil3(dim)):
                d_pos = (o - fx) * dx
                weight = wprod(w, o)
                v_new += weight * grid_v[base + o]
                C_new += weight * grid_v[base + o].outer_product(d_pos) * (4 * inv_dx * inv_dx)
            v[p] = v_new
            C[p] = C_new
            J[p] *= 1.0 + dt * C_new.trace()


# ---------------------------------------------------------------- 4. advection et sorties
@ti.kernel
def advect_fluid(x: ti.template(), v: ti.template(), alive: ti.template(), wall_type: ti.template(),
                 wall_v: ti.template(), wall_d: ti.template(), sdf: ti.template(), cells: ti.template(), use_sdf: int,
                 inv_dx: float, dt: float, bound: int, dx: float,
                 free_stack: ti.template(), free_top: ti.template()):
    """Advect fluid particles; kill those crossing an outlet (pressure or outward velocity); push those entering an
    obstacle back to its surface.

    **Inputs**

    - `x`, `v` : vec dim f32 field (cap,) positions / velocities
    - `alive` : i32 field (cap,) 1 = live particle
    - `wall_type` : i32 field (2 dim, na, nb) wall BC type
    - `wall_v` : vec dim f32 field (2 dim, na, nb) imposed velocity (inlet pointing outwards = velocity outlet)
    - `wall_d` : i32 field (2 dim, na, nb) wall offset from the edge (cells)
    - `sdf` : f32 field (n) signed distance to the obstacles at cell centers (m, < 0 inside)
    - `cells` : i32 field (n) bit flags (OBSTACLE)
    - `use_sdf` : int 1 = scene has obstacles
    - `inv_dx`, `dt` : float
    - `bound` : int
    - `dx` : float
    - `free_stack` : i32 field (cap,) free slot stack
    - `free_top` : i32 field (1,) stack size

    **Outputs**

    - x, v, alive, free_stack, free_top written in place

    **Note** : killed slots pushed to the stack; survivors clamped to [bound dx, (n - bound) dx]. Obstacles are kept
    out mainly by the density projection (boundary volume map) ; this is the safety net for particles that still
    cross the surface in one Euler step (up to cfl * dx) : moved back along the normal, inward velocity removed.
    """
    n = ti.static(cells.shape)
    dim = ti.static(len(n))
    lo = ti.Vector.zero(ti.f32, dim) + bound * dx
    hi = (ti.Vector(n) - bound).cast(ti.f32) * dx
    margin = 0.1 * dx
    for p in x:
        if alive[p] == 1:
            xn = x[p] + dt * v[p]
            c = (xn * inv_dx).cast(int)
            c = ti.math.clamp(c, 0, ti.Vector(n) - 1)
            out = False
            for s in ti.static(range(2 * dim)):
                a, plus, ts = ti.static(FRAMES[dim][s])
                ka = c[ts[0]]
                kb = 0
                if ti.static(dim == 3):
                    kb = c[ts[1]]
                d = wall_d[s, ka, kb] * dx
                crossed = xn[a] > n[a] * dx - d if ti.static(plus) else xn[a] < d
                if crossed and exit_at(wall_type, wall_v, s, ka, kb, n, bound):
                    out = True
            if out:
                alive[p] = 0
                x[p] = ti.Vector.zero(ti.f32, dim) - 1.0
                idx = ti.atomic_add(free_top[0], 1)
                free_stack[idx] = p
            else:
                xn = ti.math.clamp(xn, lo, hi)
                if use_sdf == 1:
                    for _ in ti.static(range(3)):                # distance multilinéaire approchée dans les coins
                        phi, grad = sdf_at(sdf, xn, inv_dx)      # d'un escalier : on reprojette si besoin
                        gn = grad.norm()
                        cc = cell_of(xn, inv_dx, n)
                        in_obs = (cells[cc] & OBSTACLE) != 0     # coin saillant : la distance interpolée
                        if (phi < margin or in_obs) and gn > 1e-6:   # peut y être positive
                            nrm = grad / gn                     # normale sortante de l'obstacle
                            step = margin - phi if phi < margin else 0.5 * dx
                            xn = ti.math.clamp(xn + step * nrm, lo, hi)
                            vn = v[p].dot(nrm)
                            if vn < 0.0:                         # imperméable : plus de vitesse vers l'obstacle
                                v[p] -= vn * nrm
                x[p] = xn


@ti.func
def sdf_at(sdf: ti.template(), xp, inv_dx: float):
    """Multilinear signed distance and its gradient at a point (samples at cell centers).

    **Inputs**

    - `sdf` : f32 field (n) signed distance (m, < 0 inside obstacles)
    - `xp` : vec dim f32 position (m)
    - `inv_dx` : float

    **Outputs**

    - `phi` : f32 distance (m) ; `grad` : vec dim f32 gradient (points out of the obstacle)
    """
    n = ti.static(sdf.shape)
    dim = ti.static(len(n))
    g = xp * inv_dx - 0.5
    i0 = ti.math.clamp(ti.floor(g).cast(int), 0, ti.Vector(n) - 2)
    f = ti.math.clamp(g - i0, 0.0, 1.0)
    phi = 0.0
    grad = ti.Vector.zero(ti.f32, dim)
    for o in ti.static(ti.grouped(ti.ndrange(*([2] * dim)))):
        s = sdf[i0 + o]
        wt = 1.0
        for c in ti.static(range(dim)):
            wt *= f[c] if o[c] == 1 else 1.0 - f[c]
        phi += wt * s
        for c in ti.static(range(dim)):               # d(poids)/d(f_c)
            dw = 1.0 if o[c] == 1 else -1.0
            for e in ti.static(range(dim)):
                if ti.static(e != c):
                    dw *= f[e] if o[e] == 1 else 1.0 - f[e]
            grad[c] += dw * s
    return phi, grad * inv_dx


# ---------------------------------------------------------------- 5. émission par flux sur les parois d'entrée
@ti.kernel
def emit_wall(x: ti.template(), v: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
              wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(), emit_acc: ti.template(),
              ppc_face: float, free_stack: ti.template(), free_top: ti.template(), dt: float, dx: float, bound: int,
              cells: ti.template()):
    """Emit particles from inlet walls by flux (ppc^dim v_n dt / dx per wall cell).

    **Inputs**

    - `x`, `v` : vec dim f32 field (cap,) positions / velocities
    - `C` : mat dim f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `wall_type` : i32 field (2 dim, na, nb) wall BC type
    - `wall_v` : vec dim f32 field (2 dim, na, nb) inlet velocity
    - `wall_d` : i32 field (2 dim, na, nb) wall offset from the edge (cells)
    - `emit_acc` : f32 field (2 dim, na, nb) fractional particle carry
    - `ppc_face` : float ppc^dim (particles per wall cell per unit of v_n dt / dx)
    - `free_stack` : i32 field (cap,) free slot stack
    - `free_top` : i32 field (1,) stack size
    - `dt`, `dx` : float ; `bound` : int
    - `cells` : i32 field (n) (grid shape only)

    **Outputs**

    - x, v, C, J, alive, emit_acc, free_stack, free_top written in place

    **Note** : stops silently when the pool is full.
    """
    n = ti.static(cells.shape)
    dim = ti.static(len(n))
    lo_d = ti.Vector.zero(ti.f32, dim) + bound * dx
    hi_d = (ti.Vector(n) - bound).cast(ti.f32) * dx
    for side, ka, kb in wall_type:
        for s in ti.static(range(2 * dim)):
            a, plus, ts = ti.static(FRAMES[dim][s])
            ok = side == s and wall_type[side, ka, kb] == INLET and bound <= ka < n[ts[0]] - bound
            if ti.static(dim == 3):
                ok = ok and bound <= kb < n[ts[1]] - bound
            if ok:
                vel = wall_v[side, ka, kb]
                vn = -vel[a] if ti.static(plus) else vel[a]
                if vn > 0:
                    emit_acc[side, ka, kb] += ppc_face * vn * dt / dx
                    count = int(emit_acc[side, ka, kb])
                    emit_acc[side, ka, kb] -= count
                    d = wall_d[side, ka, kb] * dx
                    for _ in range(count):
                        idx = ti.atomic_sub(free_top[0], 1) - 1
                        if idx >= 0:
                            p = free_stack[idx]
                            depth = ti.random() * vn * dt
                            xp = ti.Vector.zero(ti.f32, dim)
                            xp[a] = n[a] * dx - d - depth if ti.static(plus) else d + depth
                            xp[ts[0]] = (ka + ti.random()) * dx
                            if ti.static(dim == 3):
                                xp[ts[1]] = (kb + ti.random()) * dx
                            x[p] = ti.math.clamp(xp, lo_d, hi_d)
                            v[p] = vel
                            C[p] = ti.Matrix.zero(ti.f32, dim, dim)
                            J[p] = 1.0
                            alive[p] = 1
                        else:
                            ti.atomic_add(free_top[0], 1)     # plus de slot libre : on rend ce qu'on a pris


# ---------------------------------------------------------------- 6. quantité colorée du fluide
# Modes (identifiants fixes) : 0 uniforme, 1 |v|, 2 vx, 3 vy, 8 vz (3D), 4 pression, 5 vorticité (2D : signée ;
# 3D : |ω|), 6 masse volumique (écart rho - rho0), 7 viscosité nu (fluid_nu + Smagorinsky, incompressible).
MODE_UNIFORM, MODE_SPEED, MODE_VX, MODE_VY, MODE_P, MODE_VORT, MODE_DENSITY, MODE_NU, MODE_VZ = range(9)
_LABELS = {MODE_UNIFORM: "uniforme", MODE_SPEED: "vitesse |v|", MODE_VX: "vx", MODE_VY: "vy", MODE_VZ: "vz",
           MODE_P: "pression", MODE_VORT: "vorticité", MODE_DENSITY: "masse volumique ρ − ρ0",
           MODE_NU: "viscosité ν (m²/s)"}


def fluid_modes(dim: int = 2) -> list:
    """Color modes available in a dimension, in menu order.

    **Inputs**

    - `dim` : int 2 or 3

    **Outputs**

    - list of (mode id, label)
    """
    ids = [0, 1, 2, 3, 4, 5, 6, 7] if dim == 2 else [0, 1, 2, 3, 8, 4, 5, 6, 7]
    return [(m, _LABELS[m] if not (m == MODE_VORT and dim == 3) else "vorticité |ω|") for m in ids]


FLUID_MODES = [lbl for _, lbl in fluid_modes(2)]      # libellés 2D (index = identifiant)


def mode_signed(mode: int, dim: int = 2) -> bool:
    """Whether a color mode is signed (diverging colormap, scale ±max) or positive (jet, 0..max).

    **Inputs**

    - `mode` : int ; `dim` : int

    **Outputs**

    - bool
    """
    return mode in (MODE_VX, MODE_VY, MODE_VZ, MODE_P, MODE_DENSITY) or (mode == MODE_VORT and dim == 2)


@ti.func
def cmap_jet(t):
    """Jet colormap (blue, cyan, yellow, red).

    **Inputs**

    - `t` : float in [0, 1]

    **Outputs**

    - vec3 RGB
    """
    r = ti.math.clamp(1.5 - ti.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = ti.math.clamp(1.5 - ti.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = ti.math.clamp(1.5 - ti.abs(4.0 * t - 1.0), 0.0, 1.0)
    return ti.Vector([r, g, b])


@ti.func
def cmap_signed(t):
    """Diverging colormap (blue, white, red), 0.5 = zero.

    **Inputs**

    - `t` : float in [0, 1]

    **Outputs**

    - vec3 RGB
    """
    blue = ti.Vector([0.15, 0.35, 1.0])
    white = ti.Vector([0.95, 0.95, 0.95])
    red = ti.Vector([1.0, 0.2, 0.1])
    c = blue * (1.0 - 2.0 * t) + white * (2.0 * t)
    if t > 0.5:
        c = white * (2.0 - 2.0 * t) + red * (2.0 * t - 1.0)
    return c


@ti.func
def particle_color(s: float, fluid_mode: int, signed: int, inv_smax: float, base):
    """Display color of a particle.

    **Inputs**

    - `s` : float scalar ; `fluid_mode` : int (0 uniform) ; `signed` : int 1 = diverging colormap
    - `inv_smax` : float 1 / scale ; `base` : vec3 uniform color

    **Outputs**

    - vec3 RGB
    """
    cp = base
    if fluid_mode != 0:
        if signed == 1:
            cp = cmap_signed(ti.math.clamp(0.5 + 0.5 * s * inv_smax, 0.0, 1.0))
        else:
            cp = cmap_jet(ti.math.clamp(s * inv_smax, 0.0, 1.0))
    return cp


@ti.kernel
def fluid_scalar_wc(mode: int, x: ti.template(), v: ti.template(), J: ti.template(), alive: ti.template(),
                    grid_v: ti.template(), sc: ti.template(), E: float, rho0: float, inv_dx: float):
    """Per-particle display scalar, weakly compressible mode.

    **Inputs**

    - `mode` : int (mode id, see fluid_modes)
    - `x`, `v` : vec dim f32 field (cap,) positions / velocities
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `grid_v` : vec dim f32 field (n) node velocities
    - `sc` : f32 field (cap,)
    - `E`, `rho0`, `inv_dx` : float

    **Outputs**

    - sc written in place
    """
    n = ti.static(grid_v.shape)
    dim = ti.static(len(n))
    for p in x:
        if alive[p] == 1:
            s = 0.0
            if mode == MODE_SPEED:
                s = v[p].norm()
            elif mode == MODE_VX:
                s = v[p][0]
            elif mode == MODE_VY:
                s = v[p][1]
            elif mode == MODE_VZ:
                if ti.static(dim == 3):
                    s = v[p][dim - 1]
            elif mode == MODE_P:
                s = E * (1.0 - J[p])
            elif mode == MODE_VORT:
                c = ti.math.clamp(cell_of(x[p], inv_dx, n), 1, ti.Vector(n) - 2)
                g = ti.Matrix.zero(ti.f32, dim, dim)                 # g[a, b] = d v_b / d x_a
                for a in ti.static(range(dim)):
                    e = ti.Vector.unit(dim, a, ti.i32)
                    dv = (grid_v[c + e] - grid_v[c - e]) * 0.5 * inv_dx
                    for b in ti.static(range(dim)):
                        g[a, b] = dv[b]
                if ti.static(dim == 2):
                    s = g[0, 1] - g[1, 0]
                else:
                    s = ti.Vector([g[1, 2] - g[2, 1], g[2, 0] - g[0, 2], g[0, 1] - g[1, 0]]).norm()
            elif mode == MODE_DENSITY:
                s = rho0 * (1.0 / ti.max(J[p], 1e-6) - 1.0)
            sc[p] = s


@ti.kernel
def scalar_absmax(sc: ti.template(), alive: ti.template()) -> ti.f32:
    """Max |sc| over live particles.

    **Inputs**

    - `sc` : f32 field (cap,)
    - `alive` : i32 field (cap,) 1 = live particle

    **Outputs**

    - f32 max
    """
    m = 0.0
    for p in sc:
        if alive[p] == 1:
            ti.atomic_max(m, ti.abs(sc[p]))
    return m


@ti.kernel
def scalar_rms(sc: ti.template(), alive: ti.template()) -> ti.f32:
    """Root mean square of sc over live particles (robust color scale: ignores a few outliers).

    **Inputs**

    - `sc` : f32 field (cap,)
    - `alive` : i32 field (cap,) 1 = live particle

    **Outputs**

    - f32 rms
    """
    s, n = 0.0, 0
    for p in sc:
        if alive[p] == 1:
            s += sc[p] * sc[p]
            n += 1
    return ti.sqrt(s / ti.max(n, 1))


# ---------------------------------------------------------------- 7. rendu 2D
@ti.kernel
def render(img: ti.template(), res_x: int, res_y: int, x0: float, y0: float, k: float,
           cells: ti.template(), wall_type: ti.template(), wall_d: ti.template(), inv_dx: float,
           bound: int,
           grid_on: int, tint_on: int,
           x_f: ti.template(), alive: ti.template(), has_fluid: int, fr: float, fg: float, fb: float, r_f: int,
           sc: ti.template(), fluid_mode: int, signed: int, inv_smax: float,
           x_s: ti.template(), col_s: ti.template(), has_solid: int, r_s: int):
    """Render domain, walls, obstacles, grid and particles to an image (2D).

    **Inputs**

    - `img` : u8 vec3 field (res, res), only [0:res_y, 0:res_x] is written
    - `res_x`, `res_y` : int image size (px)
    - `x0`, `y0` : float view bottom-left corner (m) ; `k` : float view size of one pixel (m)
    - `cells` : i32 field (nx, ny) bit flags (FLUID0, SOLID0, OBSTACLE)
    - `wall_type` : i32 field (4, nm, 1) wall BC type
    - `wall_d` : i32 field (4, nm, 1) wall offset from the edge (cells)
    - `inv_dx` : float (1 / m) ; `bound`, `grid_on`, `tint_on` : int
    - `x_f` : vec2 f32 field (cap,) fluid positions
    - `alive` : i32 field (cap,) 1 = live particle
    - `has_fluid` : int
    - fr, fg, fb float fluid color
    - `r_f` : int fluid dot radius (px)
    - `sc` : f32 field (cap,) display scalar
    - `fluid_mode` : int (0 uniform) ; `signed` : int 1 = diverging colormap
    - `inv_smax` : float
    - `x_s` : vec2 f32 field (Ns,) solid positions
    - `col_s` : vec3 f32 field (Ns,) solid colors
    - `has_solid`, `r_s` : int

    **Outputs**

    - img written in place

    **Note** : indexed [row, col], row 0 at the top (QImage RGB888).
    """
    n = ti.static(cells.shape)
    nx, ny = n[0], n[1]
    cell_px = 1.0 / (inv_dx * k)                      # taille d'une cellule en pixels
    Lx, Ly = nx / inv_dx, ny / inv_dx
    for row, col in ti.ndrange(res_y, res_x):         # partie utile du tampon (res_y, res_x) <= (res, res)
        xd = x0 + (col + 0.5) * k
        yd = y0 + (res_y - row - 0.5) * k
        c = ti.Vector([0.02, 0.02, 0.08])
        if 0.0 <= xd < Lx and 0.0 <= yd < Ly:
            ci = ti.math.clamp(int(xd * inv_dx), 0, nx - 1)
            cj = ti.math.clamp(int(yd * inv_dx), 0, ny - 1)
            f = cells[ci, cj]
            side, ka, kb = band_cell(ti.Vector([ci, cj]), n, bound, wall_d)
            t = WALL
            if side >= 0:
                t = wall_type[side, ka, kb]
            bc_col = ti.Vector([0.16, 0.16, 0.19])
            if t == INLET:
                bc_col = ti.Vector([0.15, 0.50, 0.22])
            elif t == OUTLET:
                bc_col = ti.Vector([0.55, 0.14, 0.14])
            in_band = ci < bound or cj < bound or ci >= nx - bound or cj >= ny - bound
            if in_band:
                c = bc_col
            if f & OBSTACLE:
                c = ti.Vector([0.35, 0.35, 0.35])
                if not in_band and side >= 0:                 # face d'obstacle qui porte la condition du mur
                    c = bc_col
            elif not in_band and tint_on == 1 and (f & FLUID0):
                c = ti.Vector([0.06, 0.10, 0.22])
            elif not in_band and tint_on == 1 and (f & SOLID0):
                c = ti.Vector([0.22, 0.20, 0.08])
            if grid_on == 1 and cell_px >= 6.0:
                gx = (xd * inv_dx - ci) * cell_px
                gy = (yd * inv_dx - cj) * cell_px
                if gx < 1.0 or gy < 1.0:
                    c = c * 0.6 + ti.Vector([0.25, 0.25, 0.30])
        else:
            c = ti.Vector([0.10, 0.10, 0.12])
        img[row, col] = ti.cast(c * 255, ti.u8)

    for p in x_f:
        if has_fluid == 1 and alive[p] == 1:
            col = int((x_f[p].x - x0) / k)
            row = res_y - 1 - int((x_f[p].y - y0) / k)
            if -r_f <= col < res_x + r_f and -r_f <= row < res_y + r_f:  # hors champ : rien à dessiner
                cp = particle_color(sc[p], fluid_mode, signed, inv_smax, ti.Vector([fr, fg, fb]))
                for a, b in ti.ndrange((-r_f, r_f + 1), (-r_f, r_f + 1)):
                    rr, cc = row + a, col + b
                    if 0 <= rr < res_y and 0 <= cc < res_x:
                        img[rr, cc] = ti.cast(cp * 255, ti.u8)

    for p in x_s:
        if has_solid == 1:
            col = int((x_s[p].x - x0) / k)
            row = res_y - 1 - int((x_s[p].y - y0) / k)
            if -r_s <= col < res_x + r_s and -r_s <= row < res_y + r_s:
                cs = col_s[p]
                for a, b in ti.ndrange((-r_s, r_s + 1), (-r_s, r_s + 1)):
                    rr, cc = row + a, col + b
                    if 0 <= rr < res_y and 0 <= cc < res_x:
                        img[rr, cc] = ti.cast(cs * 255, ti.u8)

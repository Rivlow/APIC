"""Kernels Taichi du solveur : fluide APIC avec réservoir de particules, grille, parois, rendu.

Tous les arguments sont explicites (aucune globale). Les kernels solides (MLS-MPM, endommagement)
sont dans ui/kernels_solid.py (ancien Solver/MpM/mpm_solid.py, inchangé).

Grille `cells` (i32, nx × ny, cellules carrées dx = 1 / max(nx, ny), indexée [i, j] = (x, y)) : bits  FLUID0 = 1 (eau initiale), SOLID0 = 2,
OBSTACLE = 4.  Les conditions aux limites (parois, `wall_type` / `wall_v`, mise à jour de la grille) sont
dans Solver/boundary.py et Solver/physics.py ; ici, seulement ce qui dépend du réservoir de particules :
une sortie (OUTLET) détruit les particules qui franchissent le mur, une entrée (INLET) en émet par flux.

Réservoir fluide : capacité fixe, drapeau `alive`, particules mortes garées en (-1, -1) et ignorées
partout ; pile de slots libres (`free_stack`, `free_top[0]`) sur le GPU : une sortie y pousse les
particules détruites, une entrée y prend les slots des particules émises. Rien ne redescend au CPU.
"""
import taichi as ti

from Solver.boundary import (BOTTOM, INLET, LEFT, OBSTACLE, OUTLET, RIGHT, TOP, WALL,  # noqa: F401
                             band_bc, band_cell, outlet_at)

FLUID0, SOLID0 = 1, 2


@ti.func
def cell_of(xp, inv_dx: float, nx: int, ny: int):
    """Cell containing a position, clamped to the grid.

    **Inputs**

    - `xp` : vec2 f32 position
    - `inv_dx` : float
    - `nx`, `ny` : int

    **Outputs**

    - ivec2 cell index (i, j)
    """
    c = (xp * inv_dx).cast(int)
    return ti.Vector([ti.math.clamp(c.x, 0, nx - 1), ti.math.clamp(c.y, 0, ny - 1)])


@ti.func
def weights(fx):
    """Quadratic B-spline weights.

    **Inputs**

    - `fx` : vec2 f32 offset from the base node (in cells)

    **Outputs**

    - list of 3 vec2 weights
    """
    return [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]


# ---------------------------------------------------------------- initialisation
@ti.kernel
def init_pool(alive: ti.template(), x: ti.template(), C: ti.template(), J: ti.template(), n_init: int,
              free_stack: ti.template(), free_top: ti.template()):
    """Init the fluid particle pool: first n_init alive, others free.

    **Inputs**

    - `alive` : i32 field (cap,) 1 = live particle
    - `x` : vec2 f32 field (cap,) positions
    - `C` : mat2 f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `n_init` : int
    - `free_stack` : i32 field (cap,) free slot stack
    - `free_top` : i32 field (1,) stack size

    **Outputs**

    - all fields written in place

    **Note** : dead particles parked at (-1, -1).
    """
    for p in alive:
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        J[p] = 1.0
        if p < n_init:
            alive[p] = 1
        else:
            alive[p] = 0
            x[p] = [-1.0, -1.0]
            free_stack[p - n_init] = p
    free_top[0] = alive.shape[0] - n_init


@ti.kernel
def init_solid_state(C: ti.template(), F: ti.template(), D: ti.template(), broken: ti.template()):
    """Reset solid particles (C = 0, F = I, D = 0, intact).

    **Inputs**

    - `C`, `F` : mat2 f32 fields (Ns,)
    - `D` : f32 field (Ns,) damage
    - `broken` : i32 field (Ns,)

    **Outputs**

    - all fields written in place
    """
    for p in D:
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        F[p] = ti.Matrix.identity(ti.f32, 2)
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


# ---------------------------------------------------------------- 1. particules -> grille (fluide)
@ti.kernel
def P2G_fluid(grid_m: ti.template(), grid_v: ti.template(),
              x: ti.template(), v: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
              inv_dx: float, dt: float, dx: float, E: float, p_mass: float, p_vol: float):
    """Particles to grid (weakly compressible APIC): mass, momentum, pressure.

    **Inputs**

    - `grid_m` : f32 field (nx, ny) node masses
    - `grid_v` : vec2 f32 field (nx, ny) node momenta
    - `x`, `v` : vec2 f32 field (cap,) positions / velocities
    - `C` : mat2 f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `inv_dx`, `dt`, `dx`, `E`, `p_mass`, `p_vol` : float

    **Outputs**

    - grid_m, grid_v written in place

    **Note** : clears the grid first.
    """
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0
    for p in x:
        if alive[p] == 1:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base
            w = weights(fx)
            stress = -dt * E * p_vol * (J[p] - 1) * 4.0 * inv_dx * inv_dx
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                d_pos = (offset - fx) * dx
                weight = w[i].x * w[j].y
                grid_v[base + offset] += weight * (p_mass * (v[p] + C[p] @ d_pos) + stress * d_pos)
                grid_m[base + offset] += weight * p_mass


# ---------------------------------------------------------------- 2. grille : Solver/physics.py (grid_step)


# ---------------------------------------------------------------- 3. grille -> particules (fluide)
@ti.kernel
def G2P_fluid(grid_v: ti.template(),
              x: ti.template(), v: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
              inv_dx: float, dt: float, dx: float):
    """Grid to particles (weakly compressible APIC).

    **Inputs**

    - `grid_v` : vec2 f32 field (nx, ny) node velocities
    - `x`, `v` : vec2 f32 field (cap,) positions / velocities
    - `C` : mat2 f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `inv_dx`, `dt`, `dx` : float

    **Outputs**

    - v, C, J written in place
    """
    for p in x:
        if alive[p] == 1:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base
            w = weights(fx)
            v_new = ti.Vector.zero(ti.f32, 2)
            C_new = ti.Matrix.zero(ti.f32, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                d_pos = (offset - fx) * dx
                weight = w[i].x * w[j].y
                v_new += weight * grid_v[base + offset]
                C_new += weight * grid_v[base + offset].outer_product(d_pos) * (4 * inv_dx * inv_dx)
            v[p] = v_new
            C[p] = C_new
            J[p] *= 1.0 + dt * C_new.trace()


# ---------------------------------------------------------------- 4. advection et sorties
@ti.kernel
def advect_fluid(x: ti.template(), v: ti.template(), alive: ti.template(), wall_type: ti.template(),
                 wall_d: ti.template(), inv_dx: float, dt: float, bound: int, dx: float, nx: int, ny: int,
                 free_stack: ti.template(), free_top: ti.template()):
    """Advect fluid particles; kill those crossing an outlet.

    **Inputs**

    - `x`, `v` : vec2 f32 field (cap,) positions / velocities
    - `alive` : i32 field (cap,) 1 = live particle
    - `wall_type` : i32 field (4, nm) wall BC type
    - `wall_d` : i32 field (4, nm) wall offset from the edge (cells)
    - `inv_dx`, `dt` : float
    - `bound` : int
    - `dx` : float
    - `nx`, `ny` : int
    - `free_stack` : i32 field (cap,) free slot stack
    - `free_top` : i32 field (1,) stack size

    **Outputs**

    - x, alive, free_stack, free_top written in place

    **Note** : killed slots pushed to the stack; survivors clamped to [bound dx, (n - bound) dx].
    """
    lo = ti.Vector([bound * dx, bound * dx])
    hi = ti.Vector([(nx - bound) * dx, (ny - bound) * dx])
    Lx, Ly = nx * dx, ny * dx
    for p in x:
        if alive[p] == 1:
            xn = x[p] + dt * v[p]
            kj = ti.math.clamp(int(xn.y * inv_dx), 0, ny - 1)
            ki = ti.math.clamp(int(xn.x * inv_dx), 0, nx - 1)
            out = ((xn.x < wall_d[LEFT, kj] * dx and outlet_at(wall_type, LEFT, kj, nx, ny, bound))
                   or (xn.x > Lx - wall_d[RIGHT, kj] * dx and outlet_at(wall_type, RIGHT, kj, nx, ny, bound))
                   or (xn.y < wall_d[BOTTOM, ki] * dx and outlet_at(wall_type, BOTTOM, ki, nx, ny, bound))
                   or (xn.y > Ly - wall_d[TOP, ki] * dx and outlet_at(wall_type, TOP, ki, nx, ny, bound)))
            if out:
                alive[p] = 0
                x[p] = [-1.0, -1.0]
                idx = ti.atomic_add(free_top[0], 1)
                free_stack[idx] = p
            else:
                x[p] = ti.math.clamp(xn, lo, hi)


# ---------------------------------------------------------------- 5. émission par flux sur les parois d'entrée
@ti.kernel
def emit_wall(x: ti.template(), v: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
              wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(), emit_acc: ti.template(),
              ppc2: float, free_stack: ti.template(), free_top: ti.template(), dt: float, dx: float, bound: int,
              nx: int, ny: int):
    """Emit particles from inlet walls by flux (ppc^2 v_n dt / dx per wall cell).

    **Inputs**

    - `x`, `v` : vec2 f32 field (cap,) positions / velocities
    - `C` : mat2 f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `wall_type` : i32 field (4, nm) wall BC type
    - `wall_v` : vec2 f32 field (4, nm) inlet velocity
    - `wall_d` : i32 field (4, nm) wall offset from the edge (cells)
    - `emit_acc` : f32 field (4, nm) fractional particle carry
    - `ppc2` : float
    - `free_stack` : i32 field (cap,) free slot stack
    - `free_top` : i32 field (1,) stack size
    - `dt`, `dx` : float
    - `bound`, `nx`, `ny` : int

    **Outputs**

    - x, v, C, J, alive, emit_acc, free_stack, free_top written in place

    **Note** : stops silently when the pool is full.
    """
    lo_d = ti.Vector([bound * dx, bound * dx])
    hi_d = ti.Vector([(nx - bound) * dx, (ny - bound) * dx])
    for side, k in wall_type:
        ln = ny if side <= RIGHT else nx
        if wall_type[side, k] == INLET and bound <= k < ln - bound:
            lo = wall_d[side, k] * dx
            hi = (nx if side <= RIGHT else ny) * dx - wall_d[side, k] * dx
            vel = wall_v[side, k]
            vn = 0.0
            if side == LEFT:
                vn = vel.x
            elif side == RIGHT:
                vn = -vel.x
            elif side == BOTTOM:
                vn = vel.y
            else:
                vn = -vel.y
            if vn > 0:
                emit_acc[side, k] += ppc2 * vn * dt / dx
                count = int(emit_acc[side, k])
                emit_acc[side, k] -= count
                for _ in range(count):
                    idx = ti.atomic_sub(free_top[0], 1) - 1
                    if idx >= 0:
                        p = free_stack[idx]
                        along = (k + ti.random()) * dx
                        depth = ti.random() * vn * dt
                        xp = ti.Vector([0.0, 0.0])
                        if side == LEFT:
                            xp = ti.Vector([lo + depth, along])
                        elif side == RIGHT:
                            xp = ti.Vector([hi - depth, along])
                        elif side == BOTTOM:
                            xp = ti.Vector([along, lo + depth])
                        else:
                            xp = ti.Vector([along, hi - depth])
                        x[p] = ti.math.clamp(xp, lo_d, hi_d)
                        v[p] = vel
                        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
                        J[p] = 1.0
                        alive[p] = 1
                    else:
                        ti.atomic_add(free_top[0], 1)     # plus de slot libre : on rend ce qu'on a pris


# ---------------------------------------------------------------- 6. quantité colorée du fluide
# modes : 0 uniforme, 1 |v|, 2 vx, 3 vy, 4 pression, 5 vorticité (1 : échelle 0..max ; 2-5 : ±max)
FLUID_MODES = ["uniforme", "vitesse |v|", "vx", "vy", "pression", "vorticité"]


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


@ti.kernel
def fluid_scalar_wc(mode: int, x: ti.template(), v: ti.template(), J: ti.template(), alive: ti.template(),
                    grid_v: ti.template(), sc: ti.template(), E: float, inv_dx: float, nx: int, ny: int):
    """Per-particle display scalar, weakly compressible mode.

    **Inputs**

    - `mode` : int (index in FLUID_MODES)
    - `x`, `v` : vec2 f32 field (cap,) positions / velocities
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `grid_v` : vec2 f32 field (nx, ny) node velocities
    - `sc` : f32 field (cap,)
    - `E`, `inv_dx` : float
    - `nx`, `ny` : int

    **Outputs**

    - sc written in place
    """
    for p in x:
        if alive[p] == 1:
            s = 0.0
            if mode == 1:
                s = v[p].norm()
            elif mode == 2:
                s = v[p].x
            elif mode == 3:
                s = v[p].y
            elif mode == 4:
                s = E * (1.0 - J[p])
            elif mode == 5:
                c = cell_of(x[p], inv_dx, nx, ny)
                i, j = ti.math.clamp(c.x, 1, nx - 2), ti.math.clamp(c.y, 1, ny - 2)
                s = (grid_v[i + 1, j].y - grid_v[i - 1, j].y - grid_v[i, j + 1].x + grid_v[i, j - 1].x) * 0.5 * inv_dx
            sc[p] = s


@ti.kernel
def fluid_scalar_inc(mode: int, x: ti.template(), v: ti.template(), alive: ti.template(),
                     u: ti.template(), vv: ti.template(), q: ti.template(), sc: ti.template(),
                     rho_over_dt: float, inv_dx: float, nx: int, ny: int):
    """Per-particle display scalar, incompressible mode.

    **Inputs**

    - `mode` : int (index in FLUID_MODES)
    - `x`, `v` : vec2 f32 field (cap,) positions / velocities
    - `alive` : i32 field (cap,) 1 = live particle
    - `u`, `vv` : f32 fields (nx+1, ny), (nx, ny+1) MAC face velocities
    - `q` : f32 field (nx, ny) pressure dt / rho
    - `sc` : f32 field (cap,)
    - `rho_over_dt`, `inv_dx` : float
    - `nx`, `ny` : int

    **Outputs**

    - sc written in place
    """
    for p in x:
        if alive[p] == 1:
            s = 0.0
            if mode == 1:
                s = v[p].norm()
            elif mode == 2:
                s = v[p].x
            elif mode == 3:
                s = v[p].y
            else:
                c = cell_of(x[p], inv_dx, nx, ny)
                i, j = ti.math.clamp(c.x, 1, nx - 2), ti.math.clamp(c.y, 1, ny - 2)
                if mode == 4:
                    s = q[i, j] * rho_over_dt
                elif mode == 5:
                    dvdx = ((vv[i + 1, j] + vv[i + 1, j + 1]) - (vv[i - 1, j] + vv[i - 1, j + 1])) * 0.25 * inv_dx
                    dudy = ((u[i, j + 1] + u[i + 1, j + 1]) - (u[i, j - 1] + u[i + 1, j - 1])) * 0.25 * inv_dx
                    s = dvdx - dudy
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


# ---------------------------------------------------------------- 7. rendu
@ti.kernel
def render(img: ti.template(), res: int, x0: float, y0: float, scale: float,
           cells: ti.template(), wall_type: ti.template(), wall_d: ti.template(), nx: int, ny: int, bound: int,
           grid_on: int, tint_on: int,
           x_f: ti.template(), alive: ti.template(), has_fluid: int, fr: float, fg: float, fb: float, r_f: int,
           sc: ti.template(), fluid_mode: int, inv_smax: float,
           x_s: ti.template(), col_s: ti.template(), has_solid: int, r_s: int):
    """Render domain, walls, obstacles, grid and particles to an image.

    **Inputs**

    - `img` : u8 vec3 field (res, res)
    - `res` : int
    - `x0`, `y0`, `scale` : float view origin / zoom
    - `cells` : i32 field (nx, ny) bit flags (FLUID0, SOLID0, OBSTACLE)
    - `wall_type` : i32 field (4, nm) wall BC type
    - `wall_d` : i32 field (4, nm) wall offset from the edge (cells)
    - `nx`, `ny`, `bound`, `grid_on`, `tint_on` : int
    - `x_f` : vec2 f32 field (cap,) fluid positions
    - `alive` : i32 field (cap,) 1 = live particle
    - `has_fluid` : int
    - fr, fg, fb float fluid color
    - `r_f` : int fluid dot radius (px)
    - `sc` : f32 field (cap,) display scalar
    - fluid_mode int
    - `inv_smax` : float
    - `x_s` : vec2 f32 field (Ns,) solid positions
    - `col_s` : vec3 f32 field (Ns,) solid colors
    - `has_solid`, `r_s` : int

    **Outputs**

    - img written in place

    **Note** : indexed [row, col], row 0 at the top (QImage RGB888).
    """
    k = 1.0 / (res * scale)
    nm = ti.max(nx, ny)
    cell_px = res * scale / nm
    Lx, Ly = nx / nm, ny / nm
    for row, col in img:
        xd = x0 + (col + 0.5) * k
        yd = y0 + (res - row - 0.5) * k
        c = ti.Vector([0.02, 0.02, 0.08])
        if 0.0 <= xd < Lx and 0.0 <= yd < Ly:
            ci = ti.math.clamp(int(xd * nm), 0, nx - 1)
            cj = ti.math.clamp(int(yd * nm), 0, ny - 1)
            f = cells[ci, cj]
            side, kk = band_cell(ci, cj, nx, ny, bound, wall_d)
            t = WALL
            if side >= 0:
                t = wall_type[side, kk]
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
                gx = (xd * nm - ci) * cell_px
                gy = (yd * nm - cj) * cell_px
                if gx < 1.0 or gy < 1.0:
                    c = c * 0.6 + ti.Vector([0.25, 0.25, 0.30])
        else:
            c = ti.Vector([0.10, 0.10, 0.12])
        img[row, col] = ti.cast(c * 255, ti.u8)

    for p in x_f:
        if has_fluid == 1 and alive[p] == 1:
            col = int((x_f[p].x - x0) / k)
            row = res - 1 - int((x_f[p].y - y0) / k)
            if -r_f <= col < res + r_f and -r_f <= row < res + r_f:      # hors champ : rien à dessiner
                cp = ti.Vector([fr, fg, fb])
                if fluid_mode == 1:
                    cp = cmap_jet(ti.math.clamp(sc[p] * inv_smax, 0.0, 1.0))
                elif fluid_mode > 1:
                    cp = cmap_signed(ti.math.clamp(0.5 + 0.5 * sc[p] * inv_smax, 0.0, 1.0))
                for a, b in ti.ndrange((-r_f, r_f + 1), (-r_f, r_f + 1)):
                    rr, cc = row + a, col + b
                    if 0 <= rr < res and 0 <= cc < res:
                        img[rr, cc] = ti.cast(cp * 255, ti.u8)

    for p in x_s:
        if has_solid == 1:
            col = int((x_s[p].x - x0) / k)
            row = res - 1 - int((x_s[p].y - y0) / k)
            if -r_s <= col < res + r_s and -r_s <= row < res + r_s:
                cs = col_s[p]
                for a, b in ti.ndrange((-r_s, r_s + 1), (-r_s, r_s + 1)):
                    rr, cc = row + a, col + b
                    if 0 <= rr < res and 0 <= cc < res:
                        img[rr, cc] = ti.cast(cs * 255, ti.u8)

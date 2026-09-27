"""Fluide incompressible : grille décalée (MAC) + projection de pression par gradient conjugué sur GPU.

Grilles (nx × ny cellules carrées, dx = 1 / max(nx, ny)) : u (nx+1, ny) sur les faces verticales, en
(i, j + 0.5) dx ; v (nx, ny+1) sur les faces horizontales, en (i + 0.5, j) dx ; mu, mv = poids de transfert
(0 = face sans particule). Cellules `ctype` (nx, ny) :
AIR = 0 (pression nulle), FLUID = 1 (inconnue), SOLID = 2 (obstacle, entrée, bande de paroi : vitesse
imposée, Neumann). Les cellules de sortie sont de l'air : pression nulle, écoulement libre.

On résout A q = b avec q = dt p / rho, A = laplacien positif à 5 points sur les cellules fluides
(deg = nombre de voisins non solides), b = -dx (u_e - u_w + v_n - v_s), puis u -= grad q.
Frottement aux parois : faces tangentielles dans une cellule solide = valeur fantôme (1 - 2 beta) u_fluide
(beta = 0 glissant, 1 adhérent : vitesse moyenne nulle sur la paroi), voir mac_bc. Correction de densité
sur les positions après l'advection (4 bis).
Le gradient conjugué vit sur le GPU : `cg` = [rr, pAp, rr_new] ; α et β sont calculés dans les kernels.
"""
import taichi as ti

from Solver.boundary import BOTTOM, INLET, LEFT, OBSTACLE, OUTLET, RIGHT, TOP, band_bc, band_cell, in_band, outlet_q

AIR, FLUID, SOLID, MOVING = 0, 1, 2, 3       # MOVING : cellule occupée par le solide MPM (vitesse = celle du solide)


@ti.func
def weights(fx):
    """Quadratic B-spline weights.

    **Inputs**

    - `fx` : vec2 f32 offset from the base node (in cells)

    **Outputs**

    - list of 3 vec2 weights
    """
    return [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]


@ti.func
def stencil(xp, inv_dx: float, ox: float, oy: float):
    """Base node and fraction on a grid staggered by (ox, oy) half cells.

    **Inputs**

    - `xp` : vec2 f32 position
    - `inv_dx`, `ox`, `oy` : float

    **Outputs**

    - (ivec2 base, vec2 fx)
    """
    off = ti.Vector([ox, oy])
    base = (xp * inv_dx - off - 0.5).cast(int)
    fx = xp * inv_dx - off - base
    return base, fx


# ---------------------------------------------------------------- 1. particules -> faces (APIC)
@ti.kernel
def mac_p2g(x: ti.template(), vel: ti.template(), C: ti.template(), alive: ti.template(),
            u: ti.template(), v: ti.template(), mu: ti.template(), mv: ti.template(), inv_dx: float, dx: float):
    """Transfer particles to MAC faces (APIC).

    **Inputs**

    - `x`, `vel` : vec2 f32 field (cap,) positions / velocities
    - `C` : mat2 f32 field (cap,) affine matrices
    - `alive` : i32 field (cap,) 1 = live particle
    - `u`, `v` : f32 fields (nx+1, ny), (nx, ny+1) MAC face velocities
    - `mu`, `mv` : f32 fields, same shapes, transfer weights
    - `inv_dx`, `dx` : float

    **Outputs**

    - u, v, mu, mv written in place
    """
    for I in ti.grouped(u):
        u[I] = 0.0
        mu[I] = 0.0
    for I in ti.grouped(v):
        v[I] = 0.0
        mv[I] = 0.0
    for p in x:
        if alive[p] == 1:
            base, fx = stencil(x[p], inv_dx, 0.0, 0.5)
            w = weights(fx)
            for i, j in ti.static(ti.ndrange(3, 3)):
                node = base + ti.Vector([i, j])
                d = (ti.Vector([node.x, node.y + 0.5]) * dx) - x[p]
                wt = w[i].x * w[j].y
                u[node] += wt * (vel[p].x + C[p][0, 0] * d.x + C[p][0, 1] * d.y)
                mu[node] += wt
            base2, fx2 = stencil(x[p], inv_dx, 0.5, 0.0)
            w2 = weights(fx2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                node = base2 + ti.Vector([i, j])
                d = (ti.Vector([node.x + 0.5, node.y]) * dx) - x[p]
                wt = w2[i].x * w2[j].y
                v[node] += wt * (vel[p].y + C[p][1, 0] * d.x + C[p][1, 1] * d.y)
                mv[node] += wt
    for I in ti.grouped(u):
        if mu[I] > 0:
            u[I] /= mu[I]
    for I in ti.grouped(v):
        if mv[I] > 0:
            v[I] /= mv[I]


# ---------------------------------------------------------------- 2. type des cellules
@ti.kernel
def mac_classify(ctype: ti.template(), cells: ti.template(), wall_type: ti.template(), wall_v: ti.template(),
                 wall_d: ti.template(), x: ti.template(), alive: ti.template(),
                 x_s: ti.template(), has_solid: int, inv_dx: float, nx: int, ny: int, bound: int,
                 free_surface: int):
    """Classify cells (air/fluid/solid/moving).

    **Inputs**

    - `ctype` : i32 field (nx, ny)
    - `cells` : i32 field (nx, ny) bit flags
    - `wall_type` : i32 field (4, nm) wall BC type
    - `wall_v` : vec2 f32 field (4, nm) inlet velocity
    - `wall_d` : i32 field (4, nm) wall offset from the edge (cells)
    - `x` : vec2 f32 field (cap,) fluid positions
    - `alive` : i32 field (cap,) 1 = live particle
    - `x_s` : vec2 f32 field (Ns,) solid positions
    - `has_solid` : int
    - `inv_dx` : float
    - `nx`, `ny`, `bound`, `free_surface` : int

    **Outputs**

    - ctype written in place

    **Note** : outlet wall cells are AIR; free_surface = 0 makes every non-solid cell FLUID.
    """
    for i, j in ctype:
        f = cells[i, j]
        side, _k = band_cell(i, j, nx, ny, bound, wall_d)
        wall = in_band(i, j, nx, ny, bound) or side >= 0
        if wall:
            t, _ = band_bc(wall_type, wall_v, wall_d, i, j, nx, ny, bound)
            ctype[i, j] = AIR if t == OUTLET else SOLID
        elif f & OBSTACLE:
            ctype[i, j] = SOLID
        elif free_surface == 0:
            ctype[i, j] = FLUID
        else:
            ctype[i, j] = AIR
    for p in x_s:
        if has_solid == 1:
            c = (x_s[p] * inv_dx).cast(int)
            ci = ti.math.clamp(c.x, 0, nx - 1)
            cj = ti.math.clamp(c.y, 0, ny - 1)
            if ctype[ci, cj] < SOLID:
                ctype[ci, cj] = MOVING
    for p in x:
        if alive[p] == 1:
            c = (x[p] * inv_dx).cast(int)
            ci = ti.math.clamp(c.x, 0, nx - 1)
            cj = ti.math.clamp(c.y, 0, ny - 1)
            side, _k = band_cell(ci, cj, nx, ny, bound, wall_d)
            wall = in_band(ci, cj, nx, ny, bound) or side >= 0
            if ctype[ci, cj] == AIR and not wall:
                ctype[ci, cj] = FLUID


@ti.func
def solid_beta(wall_f: ti.template(), i: int, j: int, nx: int, ny: int, bound: int, beta_o: float,
               horizontal: int) -> float:
    """Friction of solid cell (i, j): wall friction in the band, else beta_o.

    **Inputs**

    - `wall_f` : f32 field (4, nm) wall friction
    - `i`, `j`, `nx`, `ny`, `bound` : int
    - `beta_o` : float obstacle friction
    - `horizontal` : int 1 = u face (bottom/top walls win in corners)

    **Outputs**

    - float beta
    """
    beta = beta_o
    on_bt = j < bound or j >= ny - bound
    on_lr = i < bound or i >= nx - bound
    if horizontal == 1:
        if on_bt:
            beta = wall_f[BOTTOM, i] if j < bound else wall_f[TOP, i]
        elif on_lr:
            beta = wall_f[LEFT, j] if i < bound else wall_f[RIGHT, j]
    else:
        if on_lr:
            beta = wall_f[LEFT, j] if i < bound else wall_f[RIGHT, j]
        elif on_bt:
            beta = wall_f[BOTTOM, i] if j < bound else wall_f[TOP, i]
    return beta


@ti.kernel
def mac_bc(u: ti.template(), v: ti.template(), ctype: ti.template(), wall_type: ti.template(), wall_v: ti.template(),
           wall_d: ti.template(), wall_f: ti.template(),
           grid_v: ti.template(), grid_m: ti.template(), dt: float, g: float, beta_o: float,
           nx: int, ny: int, bound: int, with_gravity: int):
    """Apply gravity and face BCs (walls, inlets, moving solid, friction ghosts).

    **Inputs**

    - `u`, `v` : f32 fields (nx+1, ny), (nx, ny+1) MAC face velocities
    - `ctype` : i32 field (nx, ny)
    - `wall_type` : i32 field (4, nm) wall BC type
    - `wall_v` : vec2 f32 field (4, nm) inlet velocity
    - `wall_d` : i32 field (4, nm) wall offset from the edge (cells)
    - `wall_f` : f32 field (4, nm) wall friction
    - `grid_v` : vec2 f32 field (nx, ny) solid node velocities
    - `grid_m` : f32 field (nx, ny) solid node masses
    - `dt`, `g`, `beta_o` : float
    - `nx`, `ny`, `bound`, `with_gravity` : int

    **Outputs**

    - u, v written in place
    """
    for I in ti.grouped(v):
        if with_gravity == 1:
            v[I] -= dt * g
    for i, j in u:                                    # face entre les cellules (i-1, j) et (i, j), nœuds (i, j) et (i, j+1)
        il, ir = ti.math.clamp(i - 1, 0, nx - 1), ti.math.clamp(i, 0, nx - 1)
        tl, tr = ctype[il, j], ctype[ir, j]
        if tl >= SOLID or tr >= SOLID:
            val = 0.0
            bl, vl = band_bc(wall_type, wall_v, wall_d, il, j, nx, ny, bound)
            br, vr = band_bc(wall_type, wall_v, wall_d, ir, j, nx, ny, bound)
            if bl == INLET:
                val = vl.x
            elif br == INLET:
                val = vr.x
            elif tl == MOVING or tr == MOVING:
                ii, j1 = ti.math.clamp(i, 0, nx - 1), ti.math.clamp(j + 1, 0, ny - 1)
                m0, m1 = grid_m[ii, j], grid_m[ii, j1]
                if m0 + m1 > 0:
                    val = (grid_v[ii, j].x * m0 + grid_v[ii, j1].x * m1) / (m0 + m1)
            elif tl == SOLID and tr == SOLID and 0 < i < nx:
                # face tangentielle : voisine fluide au-dessus ou au-dessous (ses deux cellules non solides)
                jn = -1
                if j + 1 < ny and ctype[il, j + 1] < SOLID and ctype[ir, j + 1] < SOLID:
                    jn = j + 1
                elif j - 1 >= 0 and ctype[il, j - 1] < SOLID and ctype[ir, j - 1] < SOLID:
                    jn = j - 1
                if jn >= 0:
                    beta = solid_beta(wall_f, ir, j, nx, ny, bound, beta_o, 1)
                    val = (1.0 - 2.0 * beta) * u[i, jn]
            u[i, j] = val
    for i, j in v:                                    # face entre les cellules (i, j-1) et (i, j), nœuds (i, j) et (i+1, j)
        jb, jt = ti.math.clamp(j - 1, 0, ny - 1), ti.math.clamp(j, 0, ny - 1)
        tb, tt = ctype[i, jb], ctype[i, jt]
        if tb >= SOLID or tt >= SOLID:
            val = 0.0
            bb, vb = band_bc(wall_type, wall_v, wall_d, i, jb, nx, ny, bound)
            bt, vt = band_bc(wall_type, wall_v, wall_d, i, jt, nx, ny, bound)
            if bb == INLET:
                val = vb.y
            elif bt == INLET:
                val = vt.y
            elif tb == MOVING or tt == MOVING:
                jj, i1 = ti.math.clamp(j, 0, ny - 1), ti.math.clamp(i + 1, 0, nx - 1)
                m0, m1 = grid_m[i, jj], grid_m[i1, jj]
                if m0 + m1 > 0:
                    val = (grid_v[i, jj].y * m0 + grid_v[i1, jj].y * m1) / (m0 + m1)
            elif tb == SOLID and tt == SOLID and 0 < j < ny:
                ino = -1
                if i + 1 < nx and ctype[i + 1, jb] < SOLID and ctype[i + 1, jt] < SOLID:
                    ino = i + 1
                elif i - 1 >= 0 and ctype[i - 1, jb] < SOLID and ctype[i - 1, jt] < SOLID:
                    ino = i - 1
                if ino >= 0:
                    beta = solid_beta(wall_f, i, jt, nx, ny, bound, beta_o, 0)
                    val = (1.0 - 2.0 * beta) * v[ino, j]
            v[i, j] = val


@ti.kernel
def pressure_force(fp: ti.template(), grid_m: ti.template(), q: ti.template(), ctype: ti.template(),
                   inv_dx: float, coef: float, nx: int, ny: int):
    """Pressure acceleration -grad p / rho_s on solid grid nodes.

    **Inputs**

    - `fp` : vec2 f32 field (nx, ny)
    - `grid_m` : f32 field (nx, ny) solid node masses
    - `q` : f32 field (nx, ny) pressure dt / rho
    - `ctype` : i32 field (nx, ny)
    - `inv_dx`, `coef` : float
    - `nx`, `ny` : int

    **Outputs**

    - fp written in place
    """
    for i, j in fp:
        a = ti.Vector([0.0, 0.0])
        if grid_m[i, j] > 0:
            i0, j0 = ti.math.clamp(i - 1, 0, nx - 1), ti.math.clamp(j - 1, 0, ny - 1)
            i1, j1 = ti.math.clamp(i, 0, nx - 1), ti.math.clamp(j, 0, ny - 1)
            q00 = q[i0, j0] if ctype[i0, j0] == FLUID else 0.0
            q10 = q[i1, j0] if ctype[i1, j0] == FLUID else 0.0
            q01 = q[i0, j1] if ctype[i0, j1] == FLUID else 0.0
            q11 = q[i1, j1] if ctype[i1, j1] == FLUID else 0.0
            a = coef * ti.Vector([(q10 + q11) - (q00 + q01), (q01 + q11) - (q00 + q10)]) * 0.5 * inv_dx
        fp[i, j] = a


@ti.kernel
def add_accel(grid_v: ti.template(), grid_m: ti.template(), fp: ti.template(), dt: float):
    """Add fp dt to grid velocities of massive nodes.

    **Inputs**

    - `grid_v` : vec2 f32 field (nx, ny)
    - `grid_m` : f32 field (nx, ny)
    - `fp` : vec2 f32 field (nx, ny) acceleration
    - `dt` : float

    **Outputs**

    - grid_v written in place
    """
    for i, j in grid_v:
        if grid_m[i, j] > 0:
            grid_v[i, j] += dt * fp[i, j]


# ---------------------------------------------------------------- 4. densité des particules + gradient conjugué
@ti.kernel
def particle_density(x: ti.template(), alive: ti.template(), dens: ti.template(), inv_dx: float, ppc: int):
    """Particle density rho / rho0 at cell centers.

    **Inputs**

    - `x` : vec2 f32 field (cap,) positions
    - `alive` : i32 field (cap,) 1 = live particle
    - `dens` : f32 field (nx, ny)
    - `inv_dx` : float
    - `ppc` : int

    **Outputs**

    - dens written in place (1 = nominal)
    """
    for i, j in dens:
        dens[i, j] = 0.0
    inv = 1.0 / (ppc * ppc)
    for p in x:
        if alive[p] == 1:
            base, fx = stencil(x[p], inv_dx, 0.5, 0.5)                      # centres des cellules
            w = weights(fx)
            for i, j in ti.static(ti.ndrange(3, 3)):
                dens[base + ti.Vector([i, j])] += w[i].x * w[j].y * inv


@ti.func
def apply_A(q, ctype, i, j, nx, ny):
    """Apply the 5-point Laplacian (air: 0, solid: excluded) at cell (i, j).

    **Inputs**

    - `q` : f32 field (nx, ny)
    - `ctype` : i32 field (nx, ny)
    - `i`, `j`, `nx`, `ny` : int

    **Outputs**

    - float (A q)_ij
    """
    deg = 0
    s = 0.0
    for di, dj in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
        ni, nj = i + di, j + dj
        if 0 <= ni < nx and 0 <= nj < ny:
            t = ctype[ni, nj]
            if t < SOLID:                             # solide fixe ou mobile : exclu (Neumann)
                deg += 1
                if t == FLUID:
                    s += q[ni, nj]
    return deg * q[i, j] - s


@ti.kernel
def cg_init(q: ti.template(), r: ti.template(), pd: ti.template(), rhs: ti.template(),
            u: ti.template(), v: ti.template(), ctype: ti.template(), cg: ti.template(),
            wall_type: ti.template(), wall_d: ti.template(), wall_p: ti.template(),
            dx: float, dt: float, inv_rho: float, nx: int, ny: int, bound: int):
    """Solve pressure: CG init (rhs, Dirichlet air cells, r, p, rr).

    **Inputs**

    - `q`, `r`, `pd`, `rhs` : f32 fields (nx, ny)
    - `u`, `v` : f32 fields (nx+1, ny), (nx, ny+1) MAC face velocities
    - `ctype` : i32 field (nx, ny)
    - `cg` : f32 field (3,) [rr, pAp, rr_new]
    - `wall_type` : i32 field (4, nm) wall BC type
    - `wall_d` : i32 field (4, nm) wall offset from the edge (cells)
    - `wall_p` : f32 field (4, nm) outlet pressure
    - `dx`, `dt`, `inv_rho` : float
    - `nx`, `ny`, `bound` : int

    **Outputs**

    - q, r, pd, rhs, cg[0] written in place

    **Note** : warm start from previous q.
    """
    for i, j in ctype:
        if ctype[i, j] != FLUID:
            q[i, j] = 0.0
            if ctype[i, j] == AIR:
                q[i, j] = outlet_q(wall_type, wall_d, wall_p, i, j, nx, ny, bound, dt, inv_rho)
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            b = -dx * (u[i + 1, j] - u[i, j] + v[i, j + 1] - v[i, j])
            for di, dj in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    if ctype[ni, nj] == AIR:
                        b += q[ni, nj]
            rhs[i, j] = b
        else:
            rhs[i, j] = 0.0
    rr = 0.0                                         # réduction locale (pas d'atomiques sur une seule case)
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            r[i, j] = rhs[i, j] - apply_A(q, ctype, i, j, nx, ny)
            pd[i, j] = r[i, j]
            rr += r[i, j] * r[i, j]
        else:
            r[i, j] = 0.0
            pd[i, j] = 0.0
    cg[0] = rr


@ti.kernel
def cg_apply(pd: ti.template(), Ap: ti.template(), ctype: ti.template(), cg: ti.template(), nx: int, ny: int):
    """Solve pressure: CG step A p and pAp.

    **Inputs**

    - `pd`, `Ap` : f32 fields (nx, ny)
    - `ctype` : i32 field (nx, ny)
    - `cg` : f32 field (3,) [rr, pAp, rr_new]
    - `nx`, `ny` : int

    **Outputs**

    - Ap, cg[1] written in place
    """
    pAp = 0.0
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            Ap[i, j] = apply_A(pd, ctype, i, j, nx, ny)
            pAp += pd[i, j] * Ap[i, j]
    cg[1] = pAp


@ti.kernel
def cg_update(q: ti.template(), r: ti.template(), pd: ti.template(), Ap: ti.template(), ctype: ti.template(),
              cg: ti.template()):
    """Solve pressure: CG update of q, r, p.

    **Inputs**

    - `q`, `r`, `pd`, `Ap` : f32 fields (nx, ny)
    - `ctype` : i32 field (nx, ny)
    - `cg` : f32 field (3,) [rr, pAp, rr_new]

    **Outputs**

    - q, r, pd, cg written in place

    **Note** : alpha, beta computed on GPU (no read-back).
    """
    alpha = cg[0] / ti.max(cg[1], 1e-30)
    rr_new = 0.0
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            q[i, j] += alpha * pd[i, j]
            r[i, j] -= alpha * Ap[i, j]
            rr_new += r[i, j] * r[i, j]
    cg[2] = rr_new
    beta = cg[2] / ti.max(cg[0], 1e-30)
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            pd[i, j] = r[i, j] + beta * pd[i, j]
    cg[0] = cg[2]


# ---------------------------------------------------------------- 4 bis. projection de densité (positions)
# Deuxième projection, séparée de celle des vitesses (comme DFSPH : solveur de divergence + solveur de densité
# constante ; Kugelstadt et al. 2019 sur grille). La projection de vitesse rend u à divergence nulle sur la grille,
# mais la vitesse interpolée aux particules ne l'est pas près de la surface libre : les particules se tassent et
# le volume d'eau vu par la grille diminue. On corrige donc les POSITIONS, sans toucher aux vitesses (pas
# d'énergie injectée) : dilatation div(dx_p) = e = rho/rho0 - 1, soit dx_p = -grad phi avec
# A phi = dx² e (même laplacien que la pression, air et sorties : phi = 0, solides exclus).
@ti.kernel
def density_cg_init(phi: ti.template(), r: ti.template(), pd: ti.template(), rhs: ti.template(),
                    dens: ti.template(), ctype: ti.template(), cg: ti.template(), kappa: float, dx: float,
                    nx: int, ny: int):
    """Density projection: CG init with rhs kappa dx^2 (rho/rho0 - 1), phi = 0.

    **Inputs**

    - `phi`, `r`, `pd`, `rhs` : f32 fields (nx, ny)
    - `dens` : f32 field (nx, ny) rho / rho0
    - `ctype` : i32 field (nx, ny)
    - `cg` : f32 field (3,) [rr, pAp, rr_new]
    - `kappa`, `dx` : float
    - `nx`, `ny` : int

    **Outputs**

    - phi, r, pd, rhs, cg[0] written in place

    **Note** : surface cells correct overdensity only.
    """
    for i, j in ctype:
        phi[i, j] = 0.0
        rhs[i, j] = 0.0
        if ctype[i, j] == FLUID:
            # bord : voisin d'air (cellule partiellement remplie) ou de solide (noyau de densité tronqué, pas de
            # particules de l'autre côté) -> densité sous-estimée, on n'y corrige que la surdensité
            surface = False
            for di, dj in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    if ctype[ni, nj] != FLUID:
                        surface = True
            e = dens[i, j] - 1.0
            if surface:
                e = ti.max(e, 0.0)
            rhs[i, j] = kappa * dx * dx * e
    rr = 0.0
    for i, j in ctype:
        r[i, j] = rhs[i, j]
        pd[i, j] = rhs[i, j]
        rr += rhs[i, j] * rhs[i, j]
    cg[0] = rr


@ti.kernel
def density_gradient(phi: ti.template(), du: ti.template(), dv: ti.template(), ctype: ti.template(),
                     dx: float, nx: int, ny: int):
    """Face displacements -grad phi (0 next to solids).

    **Inputs**

    - `phi` : f32 field (nx, ny)
    - `du`, `dv` : f32 fields (nx+1, ny), (nx, ny+1)
    - `ctype` : i32 field (nx, ny)
    - `dx` : float
    - `nx`, `ny` : int

    **Outputs**

    - du, dv written in place
    """
    for i, j in du:
        val = 0.0
        if 0 < i < nx:
            tl, tr = ctype[i - 1, j], ctype[i, j]
            if tl < SOLID and tr < SOLID and (tl == FLUID or tr == FLUID):
                val = -(phi[i, j] - phi[i - 1, j]) / dx
        du[i, j] = val
    for i, j in dv:
        val = 0.0
        if 0 < j < ny:
            tb, tt = ctype[i, j - 1], ctype[i, j]
            if tb < SOLID and tt < SOLID and (tb == FLUID or tt == FLUID):
                val = -(phi[i, j] - phi[i, j - 1]) / dx
        dv[i, j] = val


@ti.kernel
def density_shift(x: ti.template(), alive: ti.template(), du: ti.template(), dv: ti.template(),
                  ctype: ti.template(), inv_dx: float, dx: float, bound: int, nx: int, ny: int):
    """Shift particle positions by the interpolated face displacement.

    **Inputs**

    - `x` : vec2 f32 field (cap,) positions
    - `alive` : i32 field (cap,) 1 = live particle
    - `du`, `dv` : f32 fields (nx+1, ny), (nx, ny+1)
    - `ctype` : i32 field (nx, ny)
    - `inv_dx`, `dx` : float
    - `bound`, `nx`, `ny` : int

    **Outputs**

    - x written in place

    **Note** : shift clamped to 0.5 dx; moves into solid cells rejected; velocities untouched.
    """
    lo = ti.Vector([bound * dx, bound * dx])
    hi = ti.Vector([(nx - bound) * dx, (ny - bound) * dx])
    for p in x:
        if alive[p] == 1:
            base, fx = stencil(x[p], inv_dx, 0.0, 0.5)
            w = weights(fx)
            sx = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):
                sx += w[i].x * w[j].y * du[base + ti.Vector([i, j])]
            base2, fx2 = stencil(x[p], inv_dx, 0.5, 0.0)
            w2 = weights(fx2)
            sy = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):
                sy += w2[i].x * w2[j].y * dv[base2 + ti.Vector([i, j])]
            d = ti.math.clamp(ti.Vector([sx, sy]), -0.5 * dx, 0.5 * dx)
            xn = ti.math.clamp(x[p] + d, lo, hi)
            c = (xn * inv_dx).cast(int)
            if ctype[ti.math.clamp(c.x, 0, nx - 1), ti.math.clamp(c.y, 0, ny - 1)] < SOLID:
                x[p] = xn


# ---------------------------------------------------------------- 5. projection
@ti.kernel
def mac_project(u: ti.template(), v: ti.template(), q: ti.template(), ctype: ti.template(), dx: float,
                nx: int, ny: int):
    """Subtract grad q on faces between non-solid cells (one fluid).

    **Inputs**

    - `u`, `v` : f32 fields (nx+1, ny), (nx, ny+1) MAC face velocities
    - `q` : f32 field (nx, ny) pressure dt / rho
    - `ctype` : i32 field (nx, ny)
    - `dx` : float
    - `nx`, `ny` : int

    **Outputs**

    - u, v written in place
    """
    for i, j in u:
        if 0 < i < nx:
            tl, tr = ctype[i - 1, j], ctype[i, j]
            if tl < SOLID and tr < SOLID and (tl == FLUID or tr == FLUID):
                u[i, j] -= (q[i, j] - q[i - 1, j]) / dx
    for i, j in v:
        if 0 < j < ny:
            tb, tt = ctype[i, j - 1], ctype[i, j]
            if tb < SOLID and tt < SOLID and (tb == FLUID or tt == FLUID):
                v[i, j] -= (q[i, j] - q[i, j - 1]) / dx


# ---------------------------------------------------------------- 6. faces -> particules (APIC)
@ti.kernel
def mac_g2p(x: ti.template(), vel: ti.template(), C: ti.template(), alive: ti.template(),
            u: ti.template(), v: ti.template(), mu: ti.template(), mv: ti.template(), inv_dx: float, dx: float):
    """Transfer face velocities to particles (APIC).

    **Inputs**

    - `x`, `vel` : vec2 f32 field (cap,) positions / velocities
    - `C` : mat2 f32 field (cap,) affine matrices
    - `alive` : i32 field (cap,) 1 = live particle
    - `u`, `v` : f32 fields (nx+1, ny), (nx, ny+1) MAC face velocities
    - `mu`, `mv` : f32 fields, same shapes, transfer weights
    - `inv_dx`, `dx` : float

    **Outputs**

    - vel, C written in place

    **Note** : component kept unchanged when no face has weight > 0.
    """
    k = 4.0 * inv_dx * inv_dx
    for p in x:
        if alive[p] == 1:
            base, fx = stencil(x[p], inv_dx, 0.0, 0.5)
            w = weights(fx)
            su, sw, cx = 0.0, 0.0, ti.Vector([0.0, 0.0])
            for i, j in ti.static(ti.ndrange(3, 3)):
                node = base + ti.Vector([i, j])
                if mu[node] > 0:
                    wt = w[i].x * w[j].y
                    d = (ti.Vector([node.x, node.y + 0.5]) * dx) - x[p]
                    su += wt * u[node]
                    cx += wt * u[node] * d
                    sw += wt
            if sw > 0:
                vel[p].x = su / sw
                C[p][0, 0], C[p][0, 1] = cx.x / sw * k, cx.y / sw * k
            base2, fx2 = stencil(x[p], inv_dx, 0.5, 0.0)
            w2 = weights(fx2)
            sv, sw2, cy = 0.0, 0.0, ti.Vector([0.0, 0.0])
            for i, j in ti.static(ti.ndrange(3, 3)):
                node = base2 + ti.Vector([i, j])
                if mv[node] > 0:
                    wt = w2[i].x * w2[j].y
                    d = (ti.Vector([node.x + 0.5, node.y]) * dx) - x[p]
                    sv += wt * v[node]
                    cy += wt * v[node] * d
                    sw2 += wt
            if sw2 > 0:
                vel[p].y = sv / sw2
                C[p][1, 0], C[p][1, 1] = cy.x / sw2 * k, cy.y / sw2 * k


# ---------------------------------------------------------------- 7. sortie à hauteur imposée = réservoir aval
@ti.func
def _spawn(x: ti.template(), vel: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
           free_stack: ti.template(), free_top: ti.template(), xp, vp):
    """Pop a free slot and spawn one particle.

    **Inputs**

    - `x`, `vel` : vec2 f32 field (cap,) positions / velocities
    - `C` : mat2 f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `free_stack` : i32 field (cap,) free slot stack
    - `free_top` : i32 field (1,) stack size
    - `xp`, `vp` : vec2 f32 position / velocity

    **Outputs**

    - pool fields written in place

    **Note** : no-op when the pool is full.
    """
    idx = ti.atomic_sub(free_top[0], 1) - 1
    if idx >= 0:
        p = free_stack[idx]
        x[p] = xp
        vel[p] = vp
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        J[p] = 1.0
        alive[p] = 1
    else:
        ti.atomic_add(free_top[0], 1)                 # plus de slot libre : on rend ce qu'on a pris


@ti.kernel
def emit_pressure_outlet(x: ti.template(), vel: ti.template(), C: ti.template(), J: ti.template(),
                         alive: ti.template(), wall_type: ti.template(), wall_d: ti.template(), wall_p: ti.template(),
                         acc: ti.template(), ctype: ti.template(), u: ti.template(), v: ti.template(), ppc: int,
                         free_stack: ti.template(), free_top: ti.template(), dt: float, dx: float, bound: int,
                         nx: int, ny: int):
    """Emit particles by flux at pressure outlets (p > 0) with inflow.

    **Inputs**

    - `x`, `vel` : vec2 f32 field (cap,) positions / velocities
    - `C` : mat2 f32 field (cap,) affine matrices
    - `J` : f32 field (cap,) volume ratios
    - `alive` : i32 field (cap,) 1 = live particle
    - `wall_type` : i32 field (4, nm) wall BC type
    - `wall_d` : i32 field (4, nm) wall offset from the edge (cells)
    - `wall_p` : f32 field (4, nm) outlet pressure
    - `acc` : f32 field (4, nm) fractional particle carry
    - `ctype` : i32 field (nx, ny)
    - `u`, `v` : f32 fields (nx+1, ny), (nx, ny+1) MAC face velocities
    - `ppc` : int
    - free_stack i32 field (cap,) free slot stack
    - `free_top` : i32 field (1,) stack size
    - `dt`, `dx` : float
    - `bound`, `nx`, `ny` : int

    **Outputs**

    - pool fields, acc written in place
    """
    ppc2 = float(ppc * ppc)
    for side, k in wall_type:
        ln = ny if side <= RIGHT else nx
        if wall_type[side, k] == OUTLET and wall_p[side, k] > 0.0 and bound <= k < ln - bound:
            d = wall_d[side, k]
            ai, aj, vn = 0, 0, 0.0
            nrm = ti.Vector([0.0, 0.0])                # normale rentrante
            if side == 0:                             # LEFT
                ai, aj, vn, nrm = d, k, u[d, k], ti.Vector([1.0, 0.0])
            elif side == 1:                           # RIGHT
                ai, aj, vn, nrm = nx - d - 1, k, -u[nx - d, k], ti.Vector([-1.0, 0.0])
            elif side == 2:                           # BOTTOM
                ai, aj, vn, nrm = k, d, v[k, d], ti.Vector([0.0, 1.0])
            else:                                     # TOP
                ai, aj, vn, nrm = k, ny - d - 1, -v[k, ny - d], ti.Vector([0.0, -1.0])
            if ctype[ai, aj] == FLUID and vn > 0.0:
                acc[side, k] += ppc2 * vn * dt / dx
                count = int(acc[side, k])
                acc[side, k] -= count
                wall = ti.Vector([(ai + 0.5) * dx, (aj + 0.5) * dx]) - 0.5 * dx * nrm     # point du mur
                tang = ti.Vector([nrm.y, nrm.x])
                for _ in range(count):
                    xp = wall + nrm * (ti.random() * vn * dt) + tang * (ti.random() - 0.5) * dx
                    _spawn(x, vel, C, J, alive, free_stack, free_top, xp, nrm * vn)


@ti.kernel
def cg_residual(cg: ti.template()) -> ti.f32:
    """Current CG residual rr.

    **Inputs**

    - `cg` : f32 field (3,) [rr, pAp, rr_new]

    **Outputs**

    - f32 cg[0]
    """
    return cg[0]


@ti.kernel
def max_speed(vel: ti.template(), alive: ti.template()) -> ti.f32:
    """Max particle speed.

    **Inputs**

    - `vel` : vec2 f32 field (cap,) velocities
    - `alive` : i32 field (cap,) 1 = live particle

    **Outputs**

    - f32 max |vel|
    """
    m = 0.0
    for p in vel:
        if alive[p] == 1:
            ti.atomic_max(m, vel[p].norm())
    return m


@ti.kernel
def divergence_max(u: ti.template(), v: ti.template(), ctype: ti.template(), dx: float) -> ti.f32:
    """Max |div u| over fluid cells (diagnostic).

    **Inputs**

    - `u`, `v` : f32 fields (nx+1, ny), (nx, ny+1) MAC face velocities
    - `ctype` : i32 field (nx, ny)
    - `dx` : float

    **Outputs**

    - f32 max
    """
    m = 0.0
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            ti.atomic_max(m, ti.abs(u[i + 1, j] - u[i, j] + v[i, j + 1] - v[i, j]) / dx)
    return m

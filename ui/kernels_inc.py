"""Fluide incompressible : grille décalée (MAC) + projection de pression par gradient conjugué sur GPU.

Grilles : u (n+1, n) sur les faces verticales, en (i, j + 0.5) dx ; v (n, n+1) sur les faces horizontales,
en (i + 0.5, j) dx ; mu, mv = poids de transfert (0 = face sans particule). Cellules `ctype` (n, n) :
AIR = 0 (pression nulle), FLUID = 1 (inconnue), SOLID = 2 (obstacle, entrée, bande de paroi : vitesse
imposée, Neumann). Les cellules de sortie sont de l'air : pression nulle, écoulement libre.

On résout A q = b avec q = dt p / rho, A = laplacien positif à 5 points sur les cellules fluides
(deg = nombre de voisins non solides), b = -dx (u_e - u_w + v_n - v_s), puis u -= grad q.
Le gradient conjugué vit sur le GPU : `cg` = [rr, pAp, rr_new] ; α et β sont calculés dans les kernels.
"""
import taichi as ti

from Solver.boundary import INLET, OBSTACLE, OUTLET, band_bc, band_cell

AIR, FLUID, SOLID, MOVING = 0, 1, 2, 3       # MOVING : cellule occupée par le solide MPM (vitesse = celle du solide)


@ti.func
def weights(fx):
    return [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]


@ti.func
def stencil(xp, inv_dx: float, ox: float, oy: float):
    """Nœud de base et fraction pour une grille décalée de (ox, oy) demi-cellules."""
    off = ti.Vector([ox, oy])
    base = (xp * inv_dx - off - 0.5).cast(int)
    fx = xp * inv_dx - off - base
    return base, fx


# ---------------------------------------------------------------- 1. particules -> faces (APIC)
@ti.kernel
def mac_p2g(x: ti.template(), vel: ti.template(), C: ti.template(), alive: ti.template(),
            u: ti.template(), v: ti.template(), mu: ti.template(), mv: ti.template(), inv_dx: float, dx: float):
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
                 x_s: ti.template(), has_solid: int, inv_dx: float, n: int, bound: int, free_surface: int):
    """free_surface = 1 : une cellule sans particule est de l'air (p = 0).
    free_surface = 0 : domaine plein, toute cellule non solide est fluide (la pression y est resolue,
    y compris negative).
    Bande de paroi et face d'obstacle qui porte une entrée / sortie : SOLID (paroi glissante ou entrée :
    vitesse imposée sur les faces), AIR pour une sortie (p = 0 exactement sur le mur). Les cellules contenant
    une particule du solide MPM sont MOVING."""
    for i, j in ctype:
        f = cells[i, j]
        side, _k = band_cell(i, j, n, bound, wall_d)
        wall = i < bound or j < bound or i >= n - bound or j >= n - bound or side >= 0
        if wall:
            t, _ = band_bc(wall_type, wall_v, wall_d, i, j, n, bound)
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
            ci = ti.math.clamp(c.x, 0, n - 1)
            cj = ti.math.clamp(c.y, 0, n - 1)
            if ctype[ci, cj] < SOLID:
                ctype[ci, cj] = MOVING
    for p in x:
        if alive[p] == 1:
            c = (x[p] * inv_dx).cast(int)
            ci = ti.math.clamp(c.x, 0, n - 1)
            cj = ti.math.clamp(c.y, 0, n - 1)
            side, _k = band_cell(ci, cj, n, bound, wall_d)
            wall = ci < bound or cj < bound or ci >= n - bound or cj >= n - bound or side >= 0
            if ctype[ci, cj] == AIR and not wall:
                ctype[ci, cj] = FLUID


# ---------------------------------------------------------------- 3. gravité + vitesses imposées
@ti.kernel
def mac_bc(u: ti.template(), v: ti.template(), ctype: ti.template(), wall_type: ti.template(), wall_v: ti.template(),
           wall_d: ti.template(),
           grid_v: ti.template(), grid_m: ti.template(), dt: float, g: float, n: int, bound: int, with_gravity: int):
    """Gravité, puis vitesse imposée sur les faces touchant une cellule solide : 0 (paroi, obstacle),
    (vx, vy) d'une entrée (cellule de bande), ou la vitesse du solide MPM (moyenne des deux nœuds de la
    grille collocalisée qui encadrent la face) pour une cellule MOVING."""
    for I in ti.grouped(v):
        if with_gravity == 1:
            v[I] -= dt * g
    for i, j in u:                                    # face entre les cellules (i-1, j) et (i, j), nœuds (i, j) et (i, j+1)
        il, ir = ti.math.clamp(i - 1, 0, n - 1), ti.math.clamp(i, 0, n - 1)
        tl, tr = ctype[il, j], ctype[ir, j]
        if tl >= SOLID or tr >= SOLID:
            val = 0.0
            bl, vl = band_bc(wall_type, wall_v, wall_d, il, j, n, bound)
            br, vr = band_bc(wall_type, wall_v, wall_d, ir, j, n, bound)
            if bl == INLET:
                val = vl.x
            elif br == INLET:
                val = vr.x
            elif tl == MOVING or tr == MOVING:
                ii, j1 = ti.math.clamp(i, 0, n - 1), ti.math.clamp(j + 1, 0, n - 1)
                m0, m1 = grid_m[ii, j], grid_m[ii, j1]
                if m0 + m1 > 0:
                    val = (grid_v[ii, j].x * m0 + grid_v[ii, j1].x * m1) / (m0 + m1)
            u[i, j] = val
    for i, j in v:                                    # face entre les cellules (i, j-1) et (i, j), nœuds (i, j) et (i+1, j)
        jb, jt = ti.math.clamp(j - 1, 0, n - 1), ti.math.clamp(j, 0, n - 1)
        tb, tt = ctype[i, jb], ctype[i, jt]
        if tb >= SOLID or tt >= SOLID:
            val = 0.0
            bb, vb = band_bc(wall_type, wall_v, wall_d, i, jb, n, bound)
            bt, vt = band_bc(wall_type, wall_v, wall_d, i, jt, n, bound)
            if bb == INLET:
                val = vb.y
            elif bt == INLET:
                val = vt.y
            elif tb == MOVING or tt == MOVING:
                jj, i1 = ti.math.clamp(j, 0, n - 1), ti.math.clamp(i + 1, 0, n - 1)
                m0, m1 = grid_m[i, jj], grid_m[i1, jj]
                if m0 + m1 > 0:
                    val = (grid_v[i, jj].y * m0 + grid_v[i1, jj].y * m1) / (m0 + m1)
            v[i, j] = val


@ti.kernel
def pressure_force(fp: ti.template(), grid_m: ti.template(), q: ti.template(), ctype: ti.template(),
                   inv_dx: float, coef: float, n: int):
    """Accélération −∇p / ρ_s sur les nœuds massiques du solide, à partir de q (cellules fluides seulement,
    0 ailleurs) ; coef = −(ρ_f / ρ_s) / dt_fluide car p = q ρ_f / dt."""
    for i, j in fp:
        a = ti.Vector([0.0, 0.0])
        if grid_m[i, j] > 0:
            i0, j0 = ti.math.clamp(i - 1, 0, n - 1), ti.math.clamp(j - 1, 0, n - 1)
            i1, j1 = ti.math.clamp(i, 0, n - 1), ti.math.clamp(j, 0, n - 1)
            q00 = q[i0, j0] if ctype[i0, j0] == FLUID else 0.0
            q10 = q[i1, j0] if ctype[i1, j0] == FLUID else 0.0
            q01 = q[i0, j1] if ctype[i0, j1] == FLUID else 0.0
            q11 = q[i1, j1] if ctype[i1, j1] == FLUID else 0.0
            a = coef * ti.Vector([(q10 + q11) - (q00 + q01), (q01 + q11) - (q00 + q10)]) * 0.5 * inv_dx
        fp[i, j] = a


@ti.kernel
def add_accel(grid_v: ti.template(), grid_m: ti.template(), fp: ti.template(), dt: float):
    for i, j in grid_v:
        if grid_m[i, j] > 0:
            grid_v[i, j] += dt * fp[i, j]


# ---------------------------------------------------------------- 4. gradient conjugué
@ti.func
def apply_A(q, ctype, i, j, n):
    """(A q)_ij = deg q_ij - somme des q des voisins fluides (air : q = 0 ; solide : exclu)."""
    deg = 0
    s = 0.0
    for di, dj in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n:
            t = ctype[ni, nj]
            if t < SOLID:                             # solide fixe ou mobile : exclu (Neumann)
                deg += 1
                if t == FLUID:
                    s += q[ni, nj]
    return deg * q[i, j] - s


@ti.kernel
def cg_init(q: ti.template(), r: ti.template(), pd: ti.template(), rhs: ti.template(),
            u: ti.template(), v: ti.template(), ctype: ti.template(), cg: ti.template(), dx: float, n: int):
    """Second membre, résidu initial r = b - A q (q = pression précédente comme point de départ)."""
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            rhs[i, j] = -dx * (u[i + 1, j] - u[i, j] + v[i, j + 1] - v[i, j])
        else:
            rhs[i, j] = 0.0
            q[i, j] = 0.0
    rr = 0.0                                          # réduction locale (pas d'atomiques sur une seule case)
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            r[i, j] = rhs[i, j] - apply_A(q, ctype, i, j, n)
            pd[i, j] = r[i, j]
            rr += r[i, j] * r[i, j]
        else:
            r[i, j] = 0.0
            pd[i, j] = 0.0
    cg[0] = rr


@ti.kernel
def cg_apply(pd: ti.template(), Ap: ti.template(), ctype: ti.template(), cg: ti.template(), n: int):
    pAp = 0.0
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            Ap[i, j] = apply_A(pd, ctype, i, j, n)
            pAp += pd[i, j] * Ap[i, j]
    cg[1] = pAp


@ti.kernel
def cg_update(q: ti.template(), r: ti.template(), pd: ti.template(), Ap: ti.template(), ctype: ti.template(),
              cg: ti.template()):
    """q += α p, r -= α A p, puis p = r + β p, avec α = rr / pAp et β = rr_new / rr."""
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


# ---------------------------------------------------------------- 5. projection
@ti.kernel
def mac_project(u: ti.template(), v: ti.template(), q: ti.template(), ctype: ti.template(), dx: float, n: int):
    """u -= grad q sur les faces entre deux cellules non solides dont l'une est fluide (air : q = 0)."""
    for i, j in u:
        if 0 < i < n:
            tl, tr = ctype[i - 1, j], ctype[i, j]
            if tl < SOLID and tr < SOLID and (tl == FLUID or tr == FLUID):
                u[i, j] -= (q[i, j] - q[i - 1, j]) / dx
    for i, j in v:
        if 0 < j < n:
            tb, tt = ctype[i, j - 1], ctype[i, j]
            if tb < SOLID and tt < SOLID and (tb == FLUID or tt == FLUID):
                v[i, j] -= (q[i, j] - q[i, j - 1]) / dx


# ---------------------------------------------------------------- 6. faces -> particules (APIC)
@ti.kernel
def mac_g2p(x: ti.template(), vel: ti.template(), C: ti.template(), alive: ti.template(),
            u: ti.template(), v: ti.template(), mu: ti.template(), mv: ti.template(), inv_dx: float, dx: float):
    """Vitesse et matrice affine depuis les faces valides (poids > 0) ; sans face valide, on garde la vitesse."""
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


@ti.kernel
def cg_residual(cg: ti.template()) -> ti.f32:
    return cg[0]


@ti.kernel
def max_speed(vel: ti.template(), alive: ti.template()) -> ti.f32:
    m = 0.0
    for p in vel:
        if alive[p] == 1:
            ti.atomic_max(m, vel[p].norm())
    return m


@ti.kernel
def divergence_max(u: ti.template(), v: ti.template(), ctype: ti.template(), dx: float) -> ti.f32:
    """Diagnostic : max |div u| sur les cellules fluides (doit tendre vers 0 après projection)."""
    m = 0.0
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            ti.atomic_max(m, ti.abs(u[i + 1, j] - u[i, j] + v[i, j + 1] - v[i, j]) / dx)
    return m

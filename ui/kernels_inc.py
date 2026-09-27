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
                 x_s: ti.template(), has_solid: int, inv_dx: float, nx: int, ny: int, bound: int,
                 free_surface: int):
    """free_surface = 1 : une cellule sans particule est de l'air (p = 0).
    free_surface = 0 : domaine plein, toute cellule non solide est fluide (la pression y est resolue,
    y compris negative).
    Bande de paroi et face d'obstacle qui porte une entrée / sortie : SOLID (paroi glissante ou entrée :
    vitesse imposée sur les faces), AIR pour une sortie (p = 0 exactement sur le mur). Les cellules contenant
    une particule du solide MPM sont MOVING."""
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
    """Frottement de la cellule solide (i, j) : celui du mur si elle est dans la bande de paroi (horizontal = 1 :
    face u, tangentielle aux murs du bas / haut, qui ont la priorité dans un coin ; 0 : face v, murs gauche /
    droite prioritaires), sinon celui des obstacles beta_o."""
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
    """Gravité, puis vitesses sur les faces touchant une cellule solide :
      - face normale (entre une cellule solide et une non solide) : 0 (imperméable), (vx, vy) d'une entrée, ou la
        vitesse du solide MPM (moyenne des deux nœuds de la grille collocalisée qui encadrent la face) ;
      - face tangentielle (entre deux cellules solides fixes, à côté d'une rangée fluide) : valeur fantôme
        (1 - 2 beta) u_fluide, u_fluide = face voisine côté fluide. beta = 0 : glissant (gradient nul),
        beta = 1 : adhérent (moyenne nulle sur la paroi). beta : frottement du mur (wall_f) ou des obstacles."""
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
    """Accélération −∇p / ρ_s sur les nœuds massiques du solide, à partir de q (cellules fluides seulement,
    0 ailleurs) ; coef = −(ρ_f / ρ_s) / dt_fluide car p = q ρ_f / dt."""
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
    for i, j in grid_v:
        if grid_m[i, j] > 0:
            grid_v[i, j] += dt * fp[i, j]


# ---------------------------------------------------------------- 4. densité des particules + gradient conjugué
@ti.kernel
def particle_density(x: ti.template(), alive: ti.template(), dens: ti.template(), inv_dx: float, ppc: int):
    """rho / rho0 aux centres des cellules : somme des poids B-spline quadratiques des particules, divisée par
    ppc² (densité nominale = 1)."""
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
    """(A q)_ij = deg q_ij - somme des q des voisins fluides (air : q = 0 ; solide : exclu)."""
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
    """Second membre, résidu initial r = b - A q (q = pression précédente comme point de départ).
    Cellules d'air : pression de Dirichlet connue, 0 (surface libre) ou pression imposée d'une sortie
    (outlet_q) ; reportée dans le second membre des cellules fluides voisines
    (deg q_i - somme q_j fluides = b + somme q_air), et lue telle quelle par mac_project."""
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
    pAp = 0.0
    for i, j in ctype:
        if ctype[i, j] == FLUID:
            Ap[i, j] = apply_A(pd, ctype, i, j, nx, ny)
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
    """Second membre kappa dx² e : e = rho/rho0 - 1 dans les deux sens à l'intérieur (corriger seulement les
    excès, avec le bruit du semis, gonfle le fluide) ; surdensité seulement dans une cellule de surface (voisine
    de l'air), naturellement sous-dense car partiellement remplie. Point de départ : phi = 0."""
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
    """Déplacement sur les faces : -(phi_j - phi_i) / dx entre deux cellules non solides dont l'une est fluide
    (air : phi = 0) ; 0 ailleurs (aucune poussée vers un mur, un obstacle ou le solide MPM)."""
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
    """x_p += déplacement interpolé depuis les faces (mêmes poids que mac_g2p), borné à 0,5 dx par composante,
    puis écrêté à la bande de paroi. Les vitesses ne changent pas. Un déplacement qui ferait entrer la particule
    dans une cellule solide (obstacle, paroi, solide MPM) est refusé : la correction de densité ne doit jamais
    pousser du fluide dans une structure (le couplage fluide-structure s'emballe)."""
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
    """u -= grad q sur les faces entre deux cellules non solides dont l'une est fluide (air : q = 0)."""
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


# ---------------------------------------------------------------- 7. sortie à hauteur imposée = réservoir aval
@ti.func
def _spawn(x: ti.template(), vel: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
           free_stack: ti.template(), free_top: ti.template(), xp, vp):
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
    """Sortie à pression imposée p > 0 (il y a de l'eau dehors) : si le gradient de pression fait RENTRER
    l'écoulement (vitesse de face projetée v_n > 0, cellule devant le mur fluide), l'eau qui entre est émise au
    flux, comme par une entrée : ppc² v_n dt / dx particules par pas (fraction reportée dans acc), dans la lame
    [mur, mur + v_n dt], à la vitesse normale v_n. Sinon l'eau aspirée laisserait un vide. La sortie de l'eau
    ne demande rien : les particules qui franchissent le mur sont détruites (advect_fluid)."""
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

# Solver/boundary.py -- Application des conditions aux limites sur la grille (Taichi), 2D ou 3D.
#
# Grille de cellules carrées, forme n = (nx, ny) ou (nx, ny, nz) (tuple Python passé en ti.template, lu à la
# compilation). Les parois sont définies par Solver/walls.py (Walls, par le script de lancement ou l'UI) puis
# rastérisées en wall_type (i32), wall_v (vecteur dim), wall_d (i32), wall_p (f32) et wall_f (f32), tables
# (2·dim, na, nb) indexées [côté, ka, kb] (cellule le long des axes tangents ; kb = 0 en 2D), lues ici :
#   - dans la bande de `bound` cellules qui borde le domaine, c'est-à-dire sur le mur lui-même ;
#   - sur la FACE d'un obstacle collé à une entrée / sortie (wall_d > bound) : la dernière couche de l'obstacle
#     le long de la normale du mur (cellule / nœud wall_d - 1 depuis le bord) porte la condition de la zone.
#     Les autres faces de l'obstacle restent des obstacles ordinaires.
# Coins : une cellule dans la bande de deux axes (ou plus) est un mur ordinaire.
# Les obstacles intérieurs vivent dans la grille `cells` (i32) : bit OBSTACLE ; leur frottement est un
# paramètre global beta_o (1 = adhérent : vitesse nulle ; 0 = glissant).
# Les côtés sont parcourus par des boucles ti.static : axe normal et axes tangents sont des constantes de
# compilation (pas d'indexation dynamique de vecteurs).

import taichi as ti

from Solver.walls import (BACK, BOTTOM, FRONT, INLET, LEFT, OBSTACLE, OUTLET, RIGHT, SIDES, TOP,  # noqa: F401
                          WALL, WALL_TYPES, Walls, frame, side_length)

FRAMES = {d: tuple(frame(s, d) for s in range(2 * d)) for d in (2, 3)}   # côté -> (axe normal, plus, tangents)


@ti.func
def tcoords(I, s: ti.template()):
    """Wall-table coordinates of a cell / node on side s.

    **Inputs**

    - `I` : ivec dim cell / node index
    - `s` : static int side

    **Outputs**

    - (ka, kb) int ; kb = 0 in 2D
    """
    ts = ti.static(FRAMES[I.n][s][2])
    kb = 0
    if ti.static(I.n == 3):
        kb = I[ts[1]]
    return I[ts[0]], kb


@ti.func
def tvalid(I, n: ti.template(), s: ti.template(), bound: int) -> bool:
    """Whether the tangent coordinates of I on side s lie outside the corner bands.

    **Inputs**

    - `I` : ivec dim ; `n` : static grid shape ; `s` : static side ; `bound` : int

    **Outputs**

    - bool
    """
    ok = True
    for t in ti.static(FRAMES[len(n)][s][2]):
        ok = ok and bound <= I[t] < n[t] - bound
    return ok


@ti.func
def face_side(I, n: ti.template(), bound: int, wall_d: ti.template()):
    """Locate an obstacle face that carries a wall zone.

    **Inputs**

    - `I` : ivec dim cell / node
    - `n` : static grid shape ; `bound` : int
    - `wall_d` : i32 field (2 dim, na, nb), wall depth in cells from the edge

    **Outputs**

    - (side, ka, kb) int ; (-1, 0, 0) otherwise

    **Note** : same index for cell and node (last obstacle layer before the fluid); sides in order left, right,
    bottom, top, back, front.
    """
    side, ka, kb = -1, 0, 0
    for s in ti.static(range(2 * len(n))):
        a, plus = ti.static(FRAMES[len(n)][s][0], FRAMES[len(n)][s][1])
        if side < 0 and tvalid(I, n, s, bound):
            ca, cb = tcoords(I, s)
            d = wall_d[s, ca, cb]
            if d > bound and I[a] == (n[a] - d if ti.static(plus) else d - 1):
                side, ka, kb = s, ca, cb
    return side, ka, kb


@ti.func
def band_cell(I, n: ti.template(), bound: int, wall_d: ti.template()):
    """Map a wall-band cell or BC obstacle face to its wall-table entry.

    **Inputs**

    - `I` : ivec dim cell
    - `n` : static grid shape ; `bound` : int
    - `wall_d` : i32 field (2 dim, na, nb)

    **Outputs**

    - (side, ka, kb) int ; (-1, 0, 0) if neither band nor BC face, or in a corner
    """
    cnt, side, ka, kb = 0, -1, 0, 0
    for s in ti.static(range(2 * len(n))):
        a, plus = ti.static(FRAMES[len(n)][s][0], FRAMES[len(n)][s][1])
        inb = I[a] >= n[a] - bound if ti.static(plus) else I[a] < bound
        if inb:
            cnt += 1
            side = s
            ka, kb = tcoords(I, s)
    if cnt != 1:                                      # hors bande, ou coin (bande de deux axes) : mur ordinaire
        side = -1
    if cnt == 0:
        side, ka, kb = face_side(I, n, bound, wall_d)
    return side, ka, kb


@ti.func
def in_band(I, n: ti.template(), bound: int) -> bool:
    """Test whether cell I lies in the wall band.

    **Inputs**

    - `I` : ivec dim ; `n` : static grid shape ; `bound` : int

    **Outputs**

    - bool
    """
    inb = False
    for a in ti.static(range(len(n))):
        inb = inb or I[a] < bound or I[a] >= n[a] - bound
    return inb


@ti.func
def band_bc(wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(), I, n: ti.template(),
            bound: int):
    """BC type and imposed velocity of cell I.

    **Inputs**

    - `wall_type`, `wall_d` : i32 field (2 dim, na, nb)
    - `wall_v` : vec dim f32 field (2 dim, na, nb)
    - `I` : ivec dim ; `n` : static grid shape ; `bound` : int

    **Outputs**

    - (t int WALL/INLET/OUTLET, vel vec dim f32) ; (WALL, 0) outside the band or in a corner
    """
    t = WALL
    vel = ti.Vector.zero(ti.f32, ti.static(len(n)))
    side, ka, kb = band_cell(I, n, bound, wall_d)
    if side >= 0:
        t = wall_type[side, ka, kb]
        vel = wall_v[side, ka, kb]
    return t, vel


@ti.func
def outlet_q(wall_type: ti.template(), wall_d: ti.template(), wall_p: ti.template(), I, n: ti.template(),
             bound: int, dt: float, inv_rho: float):
    """Imposed outlet pressure of cell I (wall band or obstacle face), as q = dt p / rho.

    **Inputs**

    - `wall_type`, `wall_d` : i32 field (2 dim, na, nb)
    - `wall_p` : f32 field (2 dim, na, nb)
    - `I` : ivec dim ; `n` : static grid shape ; `bound` : int
    - `dt`, `inv_rho` : float

    **Outputs**

    - q float ; 0 if not an outlet cell
    """
    q = 0.0
    side, ka, kb = band_cell(I, n, bound, wall_d)
    if side >= 0:
        if wall_type[side, ka, kb] == OUTLET:
            q = dt * wall_p[side, ka, kb] * inv_rho
    return q


@ti.func
def exit_at(wall_type: ti.template(), wall_v: ti.template(), s: ti.template(), ka: int, kb: int,
            n: ti.template(), bound: int) -> bool:
    """Test whether particles leave through wall entry (s, ka, kb): outlet, or inlet whose velocity points out of
    the domain (velocity outlet, e.g. u = q / h_aval ; no emission there, emit_wall only emits for inward velocity).

    **Inputs**

    - `wall_type` : i32 field (2 dim, na, nb)
    - `wall_v` : vec dim f32 field (2 dim, na, nb) imposed velocity
    - `s` : static side ; `ka`, `kb` : int ; `n` : static grid shape ; `bound` : int

    **Outputs**

    - bool (corners excluded)
    """
    a, plus, ts = ti.static(FRAMES[len(n)][s])
    ok = bound <= ka < n[ts[0]] - bound
    if ti.static(len(n) == 3):
        ok = ok and bound <= kb < n[ts[1]] - bound
    res = False
    if ok:
        vn = wall_v[s, ka, kb][a]                     # composante entrante (normale intérieure)
        if ti.static(plus):
            vn = -vn
        t = wall_type[s, ka, kb]
        res = t == OUTLET or (t == INLET and vn < 0.0)
    return res


@ti.func
def is_bc_face(I, n: ti.template(), bound: int, wall_d: ti.template()) -> bool:
    """Test whether node I is an obstacle face carrying a wall BC.

    **Inputs**

    - `I` : ivec dim ; `n` : static grid shape ; `bound` : int
    - `wall_d` : i32 field (2 dim, na, nb)

    **Outputs**

    - bool

    **Note** : such nodes must not be zeroed as obstacles.
    """
    side, ka, kb = face_side(I, n, bound, wall_d)
    return side >= 0


@ti.func
def apply_walls(I, v: ti.template(), wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(),
                wall_f: ti.template(), bound: int, n: ti.template()):
    """Apply wall BCs to grid node I (collocated grid).

    **Inputs**

    - `I` : ivec dim node
    - `v` : vec dim f32 velocity (lvalue)
    - `wall_type`, `wall_d` : i32 field (2 dim, na, nb)
    - `wall_v` : vec dim f32 field (2 dim, na, nb)
    - `wall_f` : f32 field (2 dim, na, nb)
    - `bound` : int ; `n` : static grid shape

    **Outputs**

    - v in place (wall: outward component zeroed, tangential x (1 - beta) ; inlet: imposed ; outlet: free)

    **Note** : band nodes are I[a] < bound (low side), I[a] > n[a] - bound (high side; node n - bound is the
    wall), plus obstacle faces.
    """
    fs, _ka, _kb = face_side(I, n, bound, wall_d)
    for s in ti.static(range(2 * len(n))):
        a, plus, ts = ti.static(FRAMES[len(n)][s])
        inb = I[a] > n[a] - bound if ti.static(plus) else I[a] < bound
        if inb or fs == s:
            ka, kb = tcoords(I, s)
            t = wall_type[s, ka, kb] if tvalid(I, n, s, bound) else WALL
            if t == INLET:
                vw = wall_v[s, ka, kb]
                for c in ti.static(range(len(n))):
                    v[c] = vw[c]
            elif t == WALL:
                if ti.static(plus):
                    if v[a] > 0:
                        v[a] = 0.0
                elif v[a] < 0:
                    v[a] = 0.0
                beta = wall_f[s, ka, kb]
                for c in ti.static(ts):
                    v[c] *= 1.0 - beta


@ti.func
def apply_obstacle(I, v: ti.template(), cells: ti.template(), beta: float, n: ti.template()):
    """Apply obstacle BC to node I (collocated grid).

    **Inputs**

    - `I` : ivec dim ; `n` : static grid shape
    - `v` : vec dim f32 velocity (lvalue)
    - `cells` : i32 field (n), OBSTACLE bit
    - `beta` : float friction (>= 1: no-slip)

    **Outputs**

    - v in place: inward component zeroed, tangential x (1 - beta) ; zero if beta >= 1 or no free neighbour

    **Note** : outward normal estimated from the neighbour occupancy gradient.
    """
    dim = ti.static(len(n))
    if beta >= 1.0:
        for c in ti.static(range(dim)):
            v[c] = 0.0
    else:
        nrm = ti.Vector.zero(ti.f32, dim)             # vers les voisins libres
        for a in ti.static(range(dim)):
            e = ti.Vector.unit(dim, a, ti.i32)
            # indices bornés (un ternaire Taichi évalue ses deux branches) ; hors grille = obstacle
            lo = 1.0 if (cells[ti.max(I - e, 0)] & OBSTACLE) or I[a] == 0 else 0.0
            hi = 1.0 if (cells[ti.min(I + e, ti.Vector(n) - 1)] & OBSTACLE) or I[a] == n[a] - 1 else 0.0
            nrm[a] = lo - hi
        ln = nrm.norm()
        if ln < 1e-6:
            for c in ti.static(range(dim)):
                v[c] = 0.0
        else:
            nrm /= ln
            vn = v.dot(nrm)
            vt = v - vn * nrm
            vn = ti.max(vn, 0.0)                      # imperméable : pas de vitesse vers l'obstacle
            w = vn * nrm + (1.0 - beta) * vt
            for c in ti.static(range(dim)):
                v[c] = w[c]

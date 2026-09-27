# Solver/boundary.py -- Application des conditions aux limites sur la grille (Taichi).
#
# Grille nx × ny (cellules carrées dx = 1 / max(nx, ny)). Les parois sont définies par Solver/walls.py (Walls, par
# le script de lancement ou l'UI) puis rastérisées en wall_type (i32), wall_v (vec2), wall_d (i32), wall_p (f32)
# et wall_f (f32), tables (4, max(nx, ny)) indexées [côté, cellule le long du mur], lues ici :
#   - dans la bande de `bound` cellules qui borde le domaine, c'est-à-dire sur le mur lui-même ;
#   - sur la FACE d'un obstacle collé à une entrée / sortie (wall_d > bound) : la dernière couche de l'obstacle
#     le long de la normale du mur (cellule / nœud wall_d - 1 depuis le bord) porte la condition du segment.
#     Les autres faces de l'obstacle restent des obstacles ordinaires.
# Les obstacles intérieurs vivent dans la grille `cells` (i32, nx × ny) : bit OBSTACLE ; leur frottement est un
# paramètre global beta_o (1 = adhérent : vitesse nulle, comme avant ; 0 = glissant).

import taichi as ti

from Solver.walls import (BOTTOM, INLET, LEFT, OBSTACLE, OUTLET, RIGHT, SIDES, TOP,  # noqa: F401
                          WALL, WALL_TYPES, Walls, side_length)


@ti.func
def face_side(i: int, j: int, nx: int, ny: int, bound: int, wall_d: ti.template()):
    """Locate an obstacle face that carries a wall segment.

    **Inputs**

    - `i`, `j` : int cell / node
    - `nx`, `ny`, `bound` : int
    - `wall_d` : i32 field (4, nm), wall depth in cells from the edge

    **Outputs**

    - (side, k) int, side in LEFT/RIGHT/BOTTOM/TOP, k index along the wall ; (-1, 0) otherwise

    **Note** : same index for cell and node (last obstacle layer before the fluid).
    """
    side, k = -1, 0
    if bound <= j < ny - bound:
        dl, dr = wall_d[LEFT, j], wall_d[RIGHT, j]
        if dl > bound and i == dl - 1:
            side, k = LEFT, j
        elif dr > bound and i == nx - dr:
            side, k = RIGHT, j
    if side < 0 and bound <= i < nx - bound:
        db, dt = wall_d[BOTTOM, i], wall_d[TOP, i]
        if db > bound and j == db - 1:
            side, k = BOTTOM, i
        elif dt > bound and j == ny - dt:
            side, k = TOP, i
    return side, k


@ti.func
def band_cell(i: int, j: int, nx: int, ny: int, bound: int, wall_d: ti.template()):
    """Map a wall-band cell or BC obstacle face to its wall-table entry.

    **Inputs**

    - `i`, `j` : int cell
    - `nx`, `ny`, `bound` : int
    - `wall_d` : i32 field (4, nm)

    **Outputs**

    - (side, k) int ; (-1, 0) if neither band nor BC face, or in a corner

    **Note** : corners (k in the transverse band) are always plain walls.
    """
    side, k = -1, 0
    if i < bound:
        side, k = LEFT, j
    elif i >= nx - bound:
        side, k = RIGHT, j
    elif j < bound:
        side, k = BOTTOM, i
    elif j >= ny - bound:
        side, k = TOP, i
    if side >= 0:
        ln = ny if side <= RIGHT else nx
        if k < bound or k >= ln - bound:
            side = -1
    if side < 0:
        side, k = face_side(i, j, nx, ny, bound, wall_d)
    return side, k


@ti.func
def in_band(i: int, j: int, nx: int, ny: int, bound: int) -> bool:
    """Test whether cell (i, j) lies in the wall band.

    **Inputs**

    - `i`, `j`, `nx`, `ny`, `bound` : int

    **Outputs**

    - bool
    """
    return i < bound or j < bound or i >= nx - bound or j >= ny - bound


@ti.func
def band_bc(wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(),
            i: int, j: int, nx: int, ny: int, bound: int):
    """BC type and imposed velocity of cell (i, j).

    **Inputs**

    - `wall_type`, `wall_d` : i32 field (4, nm)
    - `wall_v` : vec2 f32 field (4, nm)
    - `i`, `j`, `nx`, `ny`, `bound` : int

    **Outputs**

    - (t int WALL/INLET/OUTLET, vel vec2 f32) ; (WALL, 0) outside the band or in a corner
    """
    t = WALL
    vel = ti.Vector([0.0, 0.0])
    side, k = band_cell(i, j, nx, ny, bound, wall_d)
    if side >= 0:
        t = wall_type[side, k]
        vel = wall_v[side, k]
    return t, vel


@ti.func
def outlet_q(wall_type: ti.template(), wall_d: ti.template(), wall_p: ti.template(),
             i: int, j: int, nx: int, ny: int, bound: int, dt: float, inv_rho: float):
    """Imposed outlet pressure of cell (i, j) (wall band or obstacle face), as q = dt p / rho.

    **Inputs**

    - `wall_type`, `wall_d` : i32 field (4, nm)
    - `wall_p` : f32 field (4, nm)
    - `i`, `j`, `nx`, `ny`, `bound` : int
    - `dt`, `inv_rho` : float

    **Outputs**

    - q float ; 0 if not an outlet cell
    """
    q = 0.0
    side, k = band_cell(i, j, nx, ny, bound, wall_d)
    if side >= 0:
        if wall_type[side, k] == OUTLET:
            q = dt * wall_p[side, k] * inv_rho
    return q


@ti.func
def outlet_at(wall_type: ti.template(), side: int, k: int, nx: int, ny: int, bound: int) -> bool:
    """Test whether wall entry (side, k) is an outlet (corners excluded).

    **Inputs**

    - `wall_type` : i32 field (4, nm)
    - `side`, `k`, `nx`, `ny`, `bound` : int

    **Outputs**

    - bool
    """
    ln = ny if side <= RIGHT else nx
    return bound <= k < ln - bound and wall_type[side, k] == OUTLET


@ti.func
def is_bc_face(i: int, j: int, nx: int, ny: int, bound: int, wall_d: ti.template()) -> bool:
    """Test whether node (i, j) is an obstacle face carrying a wall BC.

    **Inputs**

    - `i`, `j`, `nx`, `ny`, `bound` : int
    - `wall_d` : i32 field (4, nm)

    **Outputs**

    - bool

    **Note** : such nodes must not be zeroed as obstacles.
    """
    side, k = face_side(i, j, nx, ny, bound, wall_d)
    return side >= 0


@ti.func
def apply_walls(i: int, j: int, v: ti.template(), wall_type: ti.template(), wall_v: ti.template(),
                wall_d: ti.template(), wall_f: ti.template(), bound: int, nx: int, ny: int):
    """Apply wall BCs to grid node (i, j).

    **Inputs**

    - `i`, `j` : int node
    - `v` : vec2 f32 velocity (lvalue)
    - `wall_type`, `wall_d` : i32 field (4, nm)
    - `wall_v` : vec2 f32 field (4, nm)
    - `wall_f` : f32 field (4, nm)
    - `bound`, `nx`, `ny` : int

    **Outputs**

    - v in place (wall: inward component zeroed, tangential x (1 - beta) ; inlet: imposed ; outlet: free)

    **Note** : band nodes are i < bound (left), i > nx - bound (right; node nx - bound is the wall), plus obstacle faces.
    """
    fs, fk = face_side(i, j, nx, ny, bound, wall_d)
    if i < bound or fs == LEFT:
        t = wall_type[LEFT, j] if bound <= j < ny - bound else WALL
        if t == INLET:
            vw = wall_v[LEFT, j]
            v.x, v.y = vw.x, vw.y
        elif t == WALL:
            if v.x < 0:
                v.x = 0.0
            v.y *= 1.0 - wall_f[LEFT, j]
    if i > nx - bound or fs == RIGHT:
        t = wall_type[RIGHT, j] if bound <= j < ny - bound else WALL
        if t == INLET:
            vw = wall_v[RIGHT, j]
            v.x, v.y = vw.x, vw.y
        elif t == WALL:
            if v.x > 0:
                v.x = 0.0
            v.y *= 1.0 - wall_f[RIGHT, j]
    if j < bound or fs == BOTTOM:
        t = wall_type[BOTTOM, i] if bound <= i < nx - bound else WALL
        if t == INLET:
            vw = wall_v[BOTTOM, i]
            v.x, v.y = vw.x, vw.y
        elif t == WALL:
            if v.y < 0:
                v.y = 0.0
            v.x *= 1.0 - wall_f[BOTTOM, i]
    if j > ny - bound or fs == TOP:
        t = wall_type[TOP, i] if bound <= i < nx - bound else WALL
        if t == INLET:
            vw = wall_v[TOP, i]
            v.x, v.y = vw.x, vw.y
        elif t == WALL:
            if v.y > 0:
                v.y = 0.0
            v.x *= 1.0 - wall_f[TOP, i]


@ti.func
def apply_obstacle(i: int, j: int, v: ti.template(), cells: ti.template(), beta: float, nx: int, ny: int):
    """Apply obstacle BC to node (i, j) (collocated grid).

    **Inputs**

    - `i`, `j`, `nx`, `ny` : int
    - `v` : vec2 f32 velocity (lvalue)
    - `cells` : i32 field (nx, ny), OBSTACLE bit
    - `beta` : float friction (>= 1: no-slip)

    **Outputs**

    - v in place: inward component zeroed, tangential x (1 - beta) ; zero if beta >= 1 or no free neighbour

    **Note** : outward normal estimated from the neighbour occupancy gradient.
    """
    if beta >= 1.0:
        v.x, v.y = 0.0, 0.0
    else:
        # indices bornés (un ternaire Taichi évalue ses deux branches) ; hors grille = obstacle
        o_l = 1.0 if (cells[ti.max(i - 1, 0), j] & OBSTACLE) or i == 0 else 0.0
        o_r = 1.0 if (cells[ti.min(i + 1, nx - 1), j] & OBSTACLE) or i == nx - 1 else 0.0
        o_b = 1.0 if (cells[i, ti.max(j - 1, 0)] & OBSTACLE) or j == 0 else 0.0
        o_t = 1.0 if (cells[i, ti.min(j + 1, ny - 1)] & OBSTACLE) or j == ny - 1 else 0.0
        nrm = ti.Vector([o_l - o_r, o_b - o_t])       # vers les voisins libres
        ln = nrm.norm()
        if ln < 1e-6:
            v.x, v.y = 0.0, 0.0
        else:
            nrm /= ln
            vn = v.dot(nrm)
            vt = v - vn * nrm
            vn = ti.max(vn, 0.0)                      # imperméable : pas de vitesse vers l'obstacle
            w = vn * nrm + (1.0 - beta) * vt
            v.x, v.y = w.x, w.y

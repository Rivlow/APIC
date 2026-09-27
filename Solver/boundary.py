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
    """(côté, indice le long du mur) si (i, j) est la face d'un obstacle qui prolonge un segment ; (-1, 0) sinon.
    Même indice pour la cellule et pour le nœud : la dernière couche de l'obstacle avant le fluide."""
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
    """Cellule (i, j) dans la bande de paroi, ou face d'obstacle qui porte un segment -> (côté, indice le long
    du mur) ; (-1, 0) sinon. Les coins (indice dans la bande transversale) sont toujours de la paroi."""
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
    return i < bound or j < bound or i >= nx - bound or j >= ny - bound


@ti.func
def band_bc(wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(),
            i: int, j: int, nx: int, ny: int, bound: int):
    """Type et vitesse imposée de la cellule de bande (i, j) ; WALL et 0 hors bande ou dans un coin."""
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
    """Pression imposée d'une cellule de sortie (bande de paroi ou face d'obstacle), en q = dt p / rho ;
    0 si la cellule n'est pas une sortie."""
    q = 0.0
    side, k = band_cell(i, j, nx, ny, bound, wall_d)
    if side >= 0:
        if wall_type[side, k] == OUTLET:
            q = dt * wall_p[side, k] * inv_rho
    return q


@ti.func
def outlet_at(wall_type: ti.template(), side: int, k: int, nx: int, ny: int, bound: int) -> bool:
    ln = ny if side <= RIGHT else nx
    return bound <= k < ln - bound and wall_type[side, k] == OUTLET


@ti.func
def is_bc_face(i: int, j: int, nx: int, ny: int, bound: int, wall_d: ti.template()) -> bool:
    """Nœud d'obstacle qui porte une condition limite (il ne doit pas être mis à vitesse nulle)."""
    side, k = face_side(i, j, nx, ny, bound, wall_d)
    return side >= 0


@ti.func
def apply_walls(i: int, j: int, v: ti.template(), wall_type: ti.template(), wall_v: ti.template(),
                wall_d: ti.template(), wall_f: ti.template(), bound: int, nx: int, ny: int):
    """Nœuds de bande : i < bound à gauche, i > nx - bound à droite (le nœud nx - bound est le mur), plus les
    nœuds de face d'obstacle (face_side). Mur : composante entrante annulée, composante tangentielle multipliée
    par (1 - beta) (beta = frottement du segment) ; entrée : vitesse imposée ; sortie : nœud libre."""
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
    """Nœud d'obstacle (grille collocalisée). beta >= 1 : adhérent, vitesse nulle. Sinon normale sortante estimée
    par le gradient de l'occupation des voisins : composante entrante annulée, tangentielle multipliée par
    (1 - beta) ; nœud intérieur (aucun voisin libre) : vitesse nulle."""
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

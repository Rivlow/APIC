# Solver/boundary.py -- Application des conditions aux limites sur la grille (Taichi).
#
# Les parois sont définies par Solver/walls.py (Walls, par le script de lancement ou l'UI) puis rastérisées
# en wall_type (i32, 4 x n), wall_v (vec2, 4 x n) et wall_d (i32, 4 x n), lus ici :
#   - dans la bande de `bound` cellules qui borde le domaine, c'est-à-dire sur le mur lui-même ;
#   - sur la FACE d'un obstacle collé à une entrée / sortie (wall_d > bound) : la dernière couche de l'obstacle
#     le long de la normale du mur (cellule / nœud wall_d - 1 depuis le bord) porte la condition du segment.
#     Les autres faces de l'obstacle restent des obstacles ordinaires (vitesse nulle).
# Les obstacles intérieurs vivent dans la grille `cells` (i32, n x n) : bit OBSTACLE -> vitesse nulle.

import taichi as ti

from Solver.walls import (BOTTOM, INLET, LEFT, OBSTACLE, OUTLET, RIGHT, SIDES, TOP, WALL,  # noqa: F401
                          WALL_TYPES, Walls)


@ti.func
def face_side(i: int, j: int, n: int, bound: int, wall_d: ti.template()):
    """(côté, indice le long du mur) si (i, j) est la face d'un obstacle qui prolonge un segment ; (-1, 0) sinon.
    Même indice pour la cellule et pour le nœud : la dernière couche de l'obstacle avant le fluide."""
    side, k = -1, 0
    if bound <= j < n - bound:
        dl, dr = wall_d[LEFT, j], wall_d[RIGHT, j]
        if dl > bound and i == dl - 1:
            side, k = LEFT, j
        elif dr > bound and i == n - dr:
            side, k = RIGHT, j
    if side < 0 and bound <= i < n - bound:
        db, dt = wall_d[BOTTOM, i], wall_d[TOP, i]
        if db > bound and j == db - 1:
            side, k = BOTTOM, i
        elif dt > bound and j == n - dt:
            side, k = TOP, i
    return side, k


@ti.func
def band_cell(i: int, j: int, n: int, bound: int, wall_d: ti.template()):
    """Cellule (i, j) dans la bande de paroi, ou face d'obstacle qui porte un segment -> (côté, indice le long
    du mur) ; (-1, 0) sinon. Les coins (indice dans la bande transversale) sont toujours de la paroi."""
    side, k = -1, 0
    if i < bound:
        side, k = LEFT, j
    elif i >= n - bound:
        side, k = RIGHT, j
    elif j < bound:
        side, k = BOTTOM, i
    elif j >= n - bound:
        side, k = TOP, i
    if side >= 0 and (k < bound or k >= n - bound):
        side = -1
    if side < 0:
        side, k = face_side(i, j, n, bound, wall_d)
    return side, k


@ti.func
def band_bc(wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(),
            i: int, j: int, n: int, bound: int):
    """Type et vitesse imposée de la cellule de bande (i, j) ; WALL et 0 hors bande ou dans un coin."""
    t = WALL
    vel = ti.Vector([0.0, 0.0])
    side, k = band_cell(i, j, n, bound, wall_d)
    if side >= 0:
        t = wall_type[side, k]
        vel = wall_v[side, k]
    return t, vel


@ti.func
def outlet_at(wall_type: ti.template(), side: int, k: int, n: int, bound: int) -> bool:
    return bound <= k < n - bound and wall_type[side, k] == OUTLET


@ti.func
def is_bc_face(i: int, j: int, n: int, bound: int, wall_d: ti.template()) -> bool:
    """Nœud d'obstacle qui porte une condition limite (il ne doit pas être mis à vitesse nulle)."""
    side, k = face_side(i, j, n, bound, wall_d)
    return side >= 0


@ti.func
def apply_walls(i: int, j: int, v: ti.template(), wall_type: ti.template(), wall_v: ti.template(),
                wall_d: ti.template(), bound: int, n: int):
    """Nœuds de bande : i < bound à gauche, i > n - bound à droite (le nœud n - bound est le mur), plus les
    nœuds de face d'obstacle (face_side). Glissant : composante entrante annulée ; entrée : vitesse imposée ;
    sortie : nœud libre."""
    fs, fk = face_side(i, j, n, bound, wall_d)
    if i < bound or fs == LEFT:
        t = wall_type[LEFT, j] if bound <= j < n - bound else WALL
        if t == INLET:
            vw = wall_v[LEFT, j]
            v.x, v.y = vw.x, vw.y
        elif t == WALL and v.x < 0:
            v.x = 0.0
    if i > n - bound or fs == RIGHT:
        t = wall_type[RIGHT, j] if bound <= j < n - bound else WALL
        if t == INLET:
            vw = wall_v[RIGHT, j]
            v.x, v.y = vw.x, vw.y
        elif t == WALL and v.x > 0:
            v.x = 0.0
    if j < bound or fs == BOTTOM:
        t = wall_type[BOTTOM, i] if bound <= i < n - bound else WALL
        if t == INLET:
            vw = wall_v[BOTTOM, i]
            v.x, v.y = vw.x, vw.y
        elif t == WALL and v.y < 0:
            v.y = 0.0
    if j > n - bound or fs == TOP:
        t = wall_type[TOP, i] if bound <= i < n - bound else WALL
        if t == INLET:
            vw = wall_v[TOP, i]
            v.x, v.y = vw.x, vw.y
        elif t == WALL and v.y > 0:
            v.y = 0.0

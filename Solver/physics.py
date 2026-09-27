# Solver/physics.py -- Physique appliquée sur la grille : forces extérieures et conditions aux limites.
#
# grid_step reste UN seul kernel (une passe sur la grille) ; la physique y est composée à partir de ti.func.
# Les conditions aux limites sont définies par le script de lancement ou l'UI (Solver/walls.py : Walls),
# jamais codées en dur ici. Grille nx × ny, cellules carrées dx = 1 / max(nx, ny).

import taichi as ti

from Solver.boundary import OBSTACLE, apply_obstacle, apply_walls, is_bc_face


@ti.func
def apply_external_forces(v: ti.template(), dt: float, g: float, damp: float):
    """Gravité + amortissement léger de la vitesse (damp en 1/s, 0 = aucun)."""
    v *= 1.0 - damp * dt
    v.y -= dt * g


@ti.kernel
def grid_step(grid_m: ti.template(), grid_v: ti.template(), cells: ti.template(),
              wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(), wall_f: ti.template(),
              dt: float, g: float, damp: float, obstacle_friction: float, bound: int, nx: int, ny: int):
    """cells : grille i32 (nx × ny), bit OBSTACLE (frottement obstacle_friction : 1 adhérent, 0 glissant ; sauf
    sur une face qui porte une entrée / sortie). wall_* : table des parois (4 × max(nx, ny)), Solver/boundary.py."""

    for i, j in grid_m:

        if grid_m[i, j] > 0:

            grid_v[i, j] /= grid_m[i, j]      # quantité de mouvement -> vitesse (fin du transfert P2G)
            apply_external_forces(grid_v[i, j], dt, g, damp)
            if cells[i, j] & OBSTACLE and not is_bc_face(i, j, nx, ny, bound, wall_d):
                apply_obstacle(i, j, grid_v[i, j], cells, obstacle_friction, nx, ny)
            apply_walls(i, j, grid_v[i, j], wall_type, wall_v, wall_d, wall_f, bound, nx, ny)


@ti.kernel
def advect(phase: ti.template(), dt: float, bound: int, dx: float, nx: int, ny: int):
    """x <- x + dt v, écrêté à la bande de paroi : une particule à moins de 0.5 dx du bord
    ferait écrire le stencil P2G hors de la grille (domaine [0, nx dx] × [0, ny dx]).
    Une sortie (OUTLET) ne détruit pas les particules ici : il faut une phase à réservoir (voir ui/kernels.py)."""
    lo = ti.Vector([bound * dx, bound * dx])
    hi = ti.Vector([(nx - bound) * dx, (ny - bound) * dx])
    for p in phase.x:
        phase.x[p] += dt * phase.v[p]
        phase.x[p] = ti.math.clamp(phase.x[p], lo, hi)

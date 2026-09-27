# Solver/physics.py -- Physique appliquée sur la grille : forces extérieures et conditions aux limites.
#
# grid_step reste UN seul kernel (une passe sur la grille) ; la physique y est composée à partir de ti.func.
# Les conditions aux limites sont définies par le script de lancement ou l'UI (Solver/boundary.py : Walls),
# jamais codées en dur ici.

import taichi as ti

from Solver.boundary import OBSTACLE, apply_walls, is_bc_face


@ti.func
def apply_external_forces(v: ti.template(), dt: float, g: float, damp: float):
    """Gravité + amortissement léger de la vitesse (damp en 1/s, 0 = aucun)."""
    v *= 1.0 - damp * dt
    v.y -= dt * g


@ti.kernel
def grid_step(grid_m: ti.template(), grid_v: ti.template(), cells: ti.template(),
              wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(),
              dt: float, g: float, damp: float, bound: int, n: int):
    """cells : grille i32 (n x n), bit OBSTACLE -> vitesse nulle (sauf sur une face qui porte une entrée /
    sortie). wall_type, wall_v, wall_d : table des parois (4 x n), voir Solver/boundary.py."""

    for i, j in grid_m:

        if grid_m[i, j] > 0:

            grid_v[i, j] /= grid_m[i, j]      # quantité de mouvement -> vitesse (fin du transfert P2G)
            apply_external_forces(grid_v[i, j], dt, g, damp)
            if cells[i, j] & OBSTACLE and not is_bc_face(i, j, n, bound, wall_d):
                grid_v[i, j] = [0.0, 0.0]
            apply_walls(i, j, grid_v[i, j], wall_type, wall_v, wall_d, bound, n)


@ti.kernel
def advect(phase: ti.template(), dt: float, bound: int, dx: float):
    """x <- x + dt v, écrêté à la bande de paroi : une particule à moins de 0.5 dx du bord
    ferait écrire le stencil P2G hors de la grille (domaine carré [0, 1]^2).
    Une sortie (OUTLET) ne détruit pas les particules ici : il faut une phase à réservoir (voir ui/kernels.py)."""
    for p in phase.x:
        phase.x[p] += dt * phase.v[p]
        phase.x[p] = ti.math.clamp(phase.x[p], bound * dx, 1.0 - bound * dx)

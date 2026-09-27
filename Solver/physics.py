# Solver/physics.py -- Physique appliquée sur la grille : forces extérieures et conditions aux limites.
#
# grid_step reste UN seul kernel (une passe sur la grille) ; la physique y est composée à partir de ti.func.
# Les conditions aux limites sont définies par le script de lancement ou l'UI (Solver/walls.py : Walls),
# jamais codées en dur ici. Grille nx × ny, cellules carrées dx = 1 / max(nx, ny).

import taichi as ti

from Solver.boundary import OBSTACLE, apply_obstacle, apply_walls, is_bc_face


@ti.func
def apply_external_forces(v: ti.template(), dt: float, g: float, damp: float):
    """Apply gravity and linear velocity damping.

    **Inputs**

    - `v` : vec2 f32 grid velocity (lvalue)
    - `dt`, `g`, `damp` : float (damp in 1/s, 0 = none)

    **Outputs**

    - v modified in place
    """
    v *= 1.0 - damp * dt
    v.y -= dt * g


@ti.kernel
def grid_step(grid_m: ti.template(), grid_v: ti.template(), cells: ti.template(),
              wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(), wall_f: ti.template(),
              dt: float, g: float, damp: float, obstacle_friction: float, bound: int, nx: int, ny: int):
    """Grid update: momentum -> velocity, external forces, obstacle and wall BCs.

    **Inputs**

    - `grid_m` : f32 field (nx, ny)
    - `grid_v` : vec2 f32 field (nx, ny), momentum on entry
    - `cells` : i32 field (nx, ny), OBSTACLE bit
    - `wall_type`, `wall_d` : i32 field (4, nm)
    - `wall_v` : vec2 f32 field (4, nm)
    - `wall_f` : f32 field (4, nm)
    - `dt`, `g`, `damp`, `obstacle_friction` : float (friction: 1 no-slip, 0 slip)
    - `bound`, `nx`, `ny` : int

    **Outputs**

    - grid_v = velocity in place (nodes with grid_m > 0)

    **Note** : obstacle faces carrying an inlet / outlet get the wall BC, not the obstacle BC.
    """

    for i, j in grid_m:

        if grid_m[i, j] > 0:

            grid_v[i, j] /= grid_m[i, j]      # quantité de mouvement -> vitesse (fin du transfert P2G)
            apply_external_forces(grid_v[i, j], dt, g, damp)
            if cells[i, j] & OBSTACLE and not is_bc_face(i, j, nx, ny, bound, wall_d):
                apply_obstacle(i, j, grid_v[i, j], cells, obstacle_friction, nx, ny)
            apply_walls(i, j, grid_v[i, j], wall_type, wall_v, wall_d, wall_f, bound, nx, ny)


@ti.kernel
def advect(phase: ti.template(), dt: float, bound: int, dx: float, nx: int, ny: int):
    """Advect particles (x <- x + dt v), clamped to [bound dx, (n - bound) dx].

    **Inputs**

    - `phase` : data_oriented phase: x, v (N,) vec2 f32
    - `dt`, `dx` : float
    - `bound`, `nx`, `ny` : int

    **Outputs**

    - phase.x updated in place

    **Note** : clamp required (within 0.5 dx of the edge P2G writes out of the grid); outlets do not remove particles.
    """
    lo = ti.Vector([bound * dx, bound * dx])
    hi = ti.Vector([(nx - bound) * dx, (ny - bound) * dx])
    for p in phase.x:
        phase.x[p] += dt * phase.v[p]
        phase.x[p] = ti.math.clamp(phase.x[p], lo, hi)

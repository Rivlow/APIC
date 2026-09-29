# Solver/physics.py -- Physique appliquée sur la grille : forces extérieures et conditions aux limites.
#
# grid_step reste UN seul kernel (une passe sur la grille) ; la physique y est composée à partir de ti.func.
# Les conditions aux limites sont définies par le script de lancement ou l'UI (Solver/walls.py : Walls),
# jamais codées en dur ici. Grille 2D ou 3D (forme lue sur grid_m à la compilation), cellules carrées.

import taichi as ti

from Solver.boundary import OBSTACLE, apply_obstacle, apply_walls, is_bc_face


@ti.func
def apply_external_forces(v: ti.template(), dt: float, g: float, damp: float):
    """Apply gravity and linear velocity damping.

    **Inputs**

    - `v` : vec dim f32 grid velocity (lvalue)
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

    - `grid_m` : f32 field (n) node masses
    - `grid_v` : vec dim f32 field (n), momentum on entry
    - `cells` : i32 field (n), OBSTACLE bit
    - `wall_type`, `wall_d` : i32 field (2 dim, na, nb)
    - `wall_v` : vec dim f32 field (2 dim, na, nb)
    - `wall_f` : f32 field (2 dim, na, nb)
    - `dt`, `g`, `damp`, `obstacle_friction` : float (friction: 1 no-slip, 0 slip)
    - `bound` : int ; `nx`, `ny` : int (unused, kept for the 2D scripts ; the shape is read from grid_m)

    **Outputs**

    - grid_v = velocity in place (nodes with grid_m > 0)

    **Note** : obstacle faces carrying an inlet / outlet get the wall BC, not the obstacle BC.
    """
    n = ti.static(grid_m.shape)
    for I in ti.grouped(grid_m):

        if grid_m[I] > 0:

            grid_v[I] /= grid_m[I]            # quantité de mouvement -> vitesse (fin du transfert P2G)
            apply_external_forces(grid_v[I], dt, g, damp)
            if cells[I] & OBSTACLE and not is_bc_face(I, n, bound, wall_d):
                apply_obstacle(I, grid_v[I], cells, obstacle_friction, n)
            apply_walls(I, grid_v[I], wall_type, wall_v, wall_d, wall_f, bound, n)


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

# Solver/APIC.py -- Transferts APIC / MLS-MPM génériques (particules <-> grille), indépendants de la phase.
#
# Ce module ne fait que moyenner / extrapoler entre particules et noeuds de grille.
# La physique est déléguée :
#   - à la phase (objet @ti.data_oriented passé en ti.template(), cf. Solver/Fluid, Solver/Solid) :
#       phase.x, phase.v, phase.C, phase.p_mass, phase.p_vol
#       phase.stress(p)                        -> contrainte de Kirchhoff tau (2x2)
#       phase.update_deformation(p, C_new, dt) -> mise à jour de J ou F à partir de grad v ~ C
#   - à Solver/physics.py (gravité, conditions aux limites) et Solver/Time_integration (advection).
#
# Chaque classe de phase distincte produit sa propre version compilée des kernels (phase est un template).

import taichi as ti


@ti.func
def stencil(xp, inv_dx):
    """Quadratic B-spline weights of the 3x3 stencil.

    **Inputs**

    - `xp` : vec2 f32 particle position
    - `inv_dx` : float

    **Outputs**

    - `base` : vec2 i32 node [0, 0] of the stencil
    - `fx` : vec2 f32 xp inv_dx - base ;
    - `w` : 3x2 f32 weights (row = offset 0..2, column = axis)
    """
    base = (xp * inv_dx - 0.5).cast(int)
    fx = xp * inv_dx - base
    w0 = 0.5 * (1.5 - fx)**2
    w1 = 0.75 - (fx - 1.0)**2
    w2 = 0.5 * (fx - 0.5)**2
    w = ti.Matrix([[w0.x, w0.y], [w1.x, w1.y], [w2.x, w2.y]])
    return base, fx, w


@ti.kernel
def clear_grid(grid_m: ti.template(), grid_v: ti.template()):
    """Reset grid mass and momentum (once per step).

    **Inputs**

    - `grid_m` : f32 field (nx, ny)
    - `grid_v` : vec2 f32 field (nx, ny)

    **Outputs**

    - grid_m, grid_v zeroed in place
    """
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0


@ti.kernel
def P2G(phase: ti.template(), grid_m: ti.template(), grid_v: ti.template(),
        inv_dx: float, dx: float, dt: float):
    """Project particles on grid (mass, momentum, MLS stress).

    **Inputs**

    - `phase` : data_oriented phase: x, v (N,) vec2 f32 ; C (N,) mat2 f32 ; p_mass, p_vol float ; stress(p)
    - `grid_m` : f32 field (nx, ny)
    - `grid_v` : vec2 f32 field (nx, ny)
    - `inv_dx`, `dx`, `dt` : float

    **Outputs**

    - grid_m, grid_v accumulated in place

    **Note** : does not clear the grid (call clear_grid first).
    """

    for p in phase.x:

        base, fx, w = stencil(phase.x[p], inv_dx)
        affine = (-dt * phase.p_vol * 4.0 * inv_dx * inv_dx) * phase.stress(p) + phase.p_mass * phase.C[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            d_pos = (offset - fx) * dx
            weight = w[i, 0] * w[j, 1]
            grid_v[base + offset] += weight * (phase.p_mass * phase.v[p] + affine @ d_pos)
            grid_m[base + offset] += weight * phase.p_mass


@ti.kernel
def G2P(phase: ti.template(), grid_v: ti.template(),
        inv_dx: float, dx: float, dt: float):
    """Gather grid velocity to particles (APIC) and update their deformation.

    **Inputs**

    - `phase` : data_oriented phase: x, v (N,) vec2 f32 ; C (N,) mat2 f32 ; update_deformation(p, C, dt)
    - `grid_v` : vec2 f32 field (nx, ny), velocity (after grid_step)
    - `inv_dx`, `dx`, `dt` : float

    **Outputs**

    - phase.v, phase.C and deformation state (J or F) written in place
    """

    for p in phase.x:

        base, fx, w = stencil(phase.x[p], inv_dx)
        v_new = ti.Vector.zero(ti.f32, 2)
        C_new = ti.Matrix.zero(ti.f32, 2, 2)

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            d_pos = (offset - fx) * dx
            weight = w[i, 0] * w[j, 1]
            g_v = grid_v[base + offset]
            v_new += weight * g_v
            C_new += weight * g_v.outer_product(d_pos) * (4.0 * inv_dx * inv_dx)  # C = sum w v (x_i - x_p)^T / (dx^2/4)

        phase.v[p] = v_new
        phase.C[p] = C_new
        phase.update_deformation(p, C_new, dt)

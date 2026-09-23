import taichi as ti

@ti.kernel
def P2G(grid_m: ti.template(), grid_v: ti.template(),
        x:ti.template(), v:ti.template(), C:ti.template(), J:ti.template(), 
        inv_dx:float, dt:float, dx:float,
        E:float, p_mass:float, p_vol:float):

    # First, initialize fields
    for i,j in grid_m:

        grid_v[i,j] = [0.0, 0.0]
        grid_m[i,j] = 0.0

    for p in x:

        base = (x[p] * inv_dx - 0.5).cast(int) # left-bottom node from 3x3 stencil
        fx = x[p]*inv_dx - base

        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5) **2]

        stress = -dt * E*p_vol*(J[p] -1) * 4.0*inv_dx*inv_dx

        # Roll over all 9 stencil nodes
        for i, j in ti.static(ti.ndrange(3,3)):

            offset = ti.Vector([i, j])
            d_pos = (offset - fx) * dx
            weight = w[i].x * w[j].y

            grid_v[base + offset] += weight * (p_mass * (v[p] + C[p] @ d_pos) + stress*d_pos)
            grid_m[base + offset] += weight * p_mass


@ti.kernel
def G2P(grid_m: ti.template(), grid_v: ti.template(),
        x:ti.template(), v:ti.template(), C:ti.template(), J:ti.template(), 
        inv_dx:float, dt:float, dx:float,
        E:float, p_mass:float, p_vol:float):

    for p in x:

        base = (x[p] * inv_dx - 0.5).cast(int) # left-bottom node from 3x3 stencil
        fx = x[p]*inv_dx - base
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5) **2]

        v_new = ti.Vector.zero(ti.f32, 2)
        C_new = ti.Matrix.zero(ti.f32, 2, 2)

        for i,j in ti.static(ti.ndrange(3,3)):

            offset = ti.Vector([i,j])
            x_i, x_p = offset*dx, fx*dx
            d_pos =  x_i - x_p
            weight = w[i].x * w[j].y

            v_new += weight * grid_v[base + offset]
            C_new += weight * grid_v[base + offset].outer_product(x_i - x_p) * (4*inv_dx*inv_dx) # C = sum w v (x_i - x_p)^T / (dx^2/4)

        v[p] = v_new
        C[p] = C_new
        J[p] *= 1.0 + dt*C_new.trace()


@ti.func
def apply_external_forces(v: ti.template(), dt:float, g:float):

    v.y -= dt*g

@ti.func
def apply_BC(i:int, j:int, v:ti.template(), 
             bound:int, nx_grid:int,
             solid_val):

    # Slip condition + impervious walls
    if i < bound and v.x < 0:
        v.x = 0.0

    if i > nx_grid - bound and v.x > 0:
        v.x = 0.0

    if j < bound and v.y < 0:
        v.y = 0.0
    
    if j > nx_grid - bound and v.y > 0:
        v.y = 0.0

    # Obstacle handling
    if solid_val == 1:
        v.x, v.y = 0.0, 0.0

@ti.kernel
def grid_step(grid_m: ti.template(), grid_v: ti.template(), solid: ti.template(),
              dt:float, g:float, bound:int, nx_grid:int):

    for i, j in grid_m:

        if grid_m[i,j] > 0:

            grid_v[i,j] /= grid_m[i,j]
            apply_external_forces(grid_v[i,j], dt, g)
            apply_BC(i, j, grid_v[i,j], bound, nx_grid, solid[i,j])


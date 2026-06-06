import taichi as ti

gravity = ti.Vector([0.0, -9.81])

# Fluid mask: 1 if cell contains fluid, 0 otherwise
fluid_mask = None

@ti.kernel
def init(pos: ti.template(), radius: float, v: ti.template(), C: ti.template()):
    for particle in pos:
        # Create particles in a ball at the center of domain (0.5, 0.5)
        # Use rejection sampling or polar coordinates
        angle = 2.0 * ti.math.pi * ti.random()
        r = radius * ti.sqrt(ti.random())  # sqrt for uniform distribution
        
        center_x = 0.5
        center_y = 0.5
        
        pos[particle] = [center_x + r * ti.cos(angle), 
                         center_y + r * ti.sin(angle)]
        
        v[particle] = ti.Vector([0.0, 0.0])
        C[particle] = ti.Matrix.zero(float, 2, 2)

@ti.kernel
def clear_grid(grid_m: ti.template(), grid_v: ti.template()):
    for i, j in grid_m:
        grid_m[i, j] = 0
        grid_v[i, j] = ti.Vector([0.0, 0.0])


@ti.kernel
def apply_fluid_mask(grid_m: ti.template(), mask: ti.template()):
    """Build fluid mask: 1 if cell has mass (fluid), 0 otherwise"""
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            mask[i, j] = 1
        else:
            mask[i, j] = 0


# Particle to grid transfer (particle_to_grid)
"""
For each particle
    1) Find neighbouring cell
    2) Compute B-spline weight
    3) Transfer mass and momentum to grid
"""
p_mass = 1.0
@ti.kernel
def particle_to_grid(grid_m: ti.template(), grid_v: ti.template(),
                     pos: ti.template(), v: ti.template(), C: ti.template(), dx: float):
    for p in pos:

        base = (pos[p]/dx - 0.5).cast(int)
        fx = pos[p]/dx - base.cast(float)

        # B-spline weights for each dimension
        w_x = [0.5 * (1.5 - fx.x) ** 2,
               0.75 - (fx.x - 1.0) ** 2,
               0.5 * (fx.x - 0.5) ** 2]
        
        w_y = [0.5 * (1.5 - fx.y) ** 2,
               0.75 - (fx.y - 1.0) ** 2,
               0.5 * (fx.y - 0.5) ** 2]
        
        # Loop over neighbouring grid nodes (3x3)
        nb_grid = grid_v.shape[0]
        for i, j in ti.static(ti.ndrange(3, 3)):

            offset = ti.Vector([i, j])
            node = base + offset

            if 0 <= node.x < nb_grid and 0 <= node.y < nb_grid:
                # 2D weight: product of 1D weights
                weight = w_x[i] * w_y[j]
                dpos = (offset.cast(float) - fx) * dx

                momentum = v[p] + C[p] @ dpos
                grid_v[node] += weight * p_mass * momentum
                grid_m[node] += weight * p_mass


# Add gravity
@ti.kernel
def add_gravity(grid_m: ti.template(), grid_v: ti.template(),
                dt: float):
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] += gravity * dt



# Normalize grid velocity
@ti.kernel
def normalize_grid(grid_m: ti.template(), grid_v: ti.template()):
    for i, j in grid_v:
        if grid_m[i,j] > 0:
            grid_v[i, j] /= grid_m[i,j]





@ti.kernel
def grid_to_particle(grid_v: ti.template(), pos: ti.template(), v: ti.template(), C: ti.template(), dx: float):

    for p in pos:

        base = (pos[p]/dx - 0.5).cast(int)
        fx = pos[p]/dx - base.cast(float)

        # B-spline weights for each dimension
        w_x = [0.5 * (1.5 - fx.x) ** 2,
               0.75 - (fx.x - 1.0) ** 2,
               0.5 * (fx.x - 0.5) ** 2]
        
        w_y = [0.5 * (1.5 - fx.y) ** 2,
               0.75 - (fx.y - 1.0) ** 2,
               0.5 * (fx.y - 0.5) ** 2]

        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix.zero(float, 2, 2)
        nb_grid = grid_v.shape[0]

        for i, j in ti.static(ti.ndrange(3, 3)):

            offset = ti.Vector([i, j])
            node = base + offset

            if 0 <= node.x < nb_grid and 0 <= node.y < nb_grid:

                # 2D weight: product of 1D weights
                weight = w_x[i] * w_y[j]

                dpos = (offset.cast(float) - fx) * dx

                g_v = grid_v[node]

                new_v += weight * g_v

                new_C += (4.0 * weight * g_v.outer_product(dpos)/dx)

        v[p] = new_v
        C[p] = new_C


# Advection of particles
@ti.kernel
def advection_particles(pos: ti.template(), v: ti.template(), dt: float):

    for particle in pos:

        pos[particle] += dt * v[particle]

        # Avoid particle leaving domain
        pos[particle].x = ti.max(0.0, ti.min(0.999, pos[particle].x))
        pos[particle].y = ti.max(0.0, ti.min(0.999, pos[particle].y))
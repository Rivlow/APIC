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
def build_fluid_mask(grid_m: ti.template(), mask: ti.template()):
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
def compute_divergence(grid_m: ti.template(), grid_v: ti.template(), 
                       dx: float, dt: float, b: ti.template(), mask: ti.template()):
    """
    Compute divergence of velocity field: div(u) = ∂u/∂x + ∂v/∂y
    Only for fluid cells (where mask == 1), and only using fluid neighbors
    """

    nb_grid = grid_v.shape[0]

    for i, j in grid_m:
        if mask[i, j] == 1:  # Only compute for fluid cells
            dudx = 0.0
            dvdy = 0.0

            # ∂u/∂x: check both neighbors are fluid before central diff
            if i < nb_grid - 1 and i > 0 and mask[i+1, j] == 1 and mask[i-1, j] == 1:
                dudx = (grid_v[i+1, j].x - grid_v[i-1, j].x) / (2.0 * dx)
            elif i == 0 and nb_grid > 1 and mask[1, j] == 1:
                dudx = (grid_v[1, j].x - grid_v[0, j].x) / dx
            elif i == nb_grid - 1 and nb_grid > 1 and mask[i-1, j] == 1:
                dudx = (grid_v[i, j].x - grid_v[i-1, j].x) / dx

            # ∂v/∂y: check both neighbors are fluid before central diff
            if j < nb_grid - 1 and j > 0 and mask[i, j+1] == 1 and mask[i, j-1] == 1:
                dvdy = (grid_v[i, j+1].y - grid_v[i, j-1].y) / (2.0 * dx)
            elif j == 0 and nb_grid > 1 and mask[i, 1] == 1:
                dvdy = (grid_v[i, 1].y - grid_v[i, 0].y) / dx
            elif j == nb_grid - 1 and nb_grid > 1 and mask[i, j-1] == 1:
                dvdy = (grid_v[i, j].y - grid_v[i, j-1].y) / dx

            # RHS for Poisson equation: ∇²p = (1/dt) * div(u)
            b[i, j] = (dudx + dvdy) / dt
        else:
            b[i, j] = 0.0

@ti.kernel
def clear_pressure(p: ti.template(), Ax: ti.template(), b: ti.template()):
    """Initialize pressure fields to zero"""
    for i, j in p:
        p[i, j] = 0.0
        Ax[i, j] = 0.0
        b[i, j] = 0.0



@ti.kernel
@ti.kernel
def jacobi_iteration(grid_m: ti.template(), grid_v: ti.template(), p: ti.template(), p_new: ti.template(), b: ti.template(), dx: float, mask: ti.template()):

    nb_grid = grid_v.shape[0]

    for i, j in grid_m:
        if mask[i, j] == 1:  # Only iterate for fluid cells
            left = 0.0
            right = 0.0
            down = 0.0
            up = 0.0
            # left
            if i == 0:
                left = p[1, j] if mask[1, j] == 1 else 0.0
            else:
                left = p[i-1, j] if mask[i-1, j] == 1 else 0.0
            # right
            if i == nb_grid - 1:
                right = p[nb_grid - 2, j] if mask[nb_grid - 2, j] == 1 else 0.0
            else:
                right = p[i+1, j] if mask[i+1, j] == 1 else 0.0
            # down
            if j == 0:
                down = p[i, 1] if mask[i, 1] == 1 else 0.0
            else:
                down = p[i, j-1] if mask[i, j-1] == 1 else 0.0
            # up
            if j == nb_grid - 1:
                up = 0.0
            else:
                up = p[i, j+1] if mask[i, j+1] == 1 else 0.0
            
            p_new[i, j] = (left + right + down + up - dx*dx * b[i, j]) / 4.0
        else:
            p_new[i, j] = 0.0


@ti.kernel
def swap_pressure(p: ti.template(), p_new: ti.template()):
    """Copy p_new back to p"""
    for i, j in p:
        p[i, j] = p_new[i, j]


@ti.kernel
def apply_pressure_gradient(grid_m: ti.template(), grid_v: ti.template(), p: ti.template(), dx: float, dt: float, mask: ti.template()):

    nb_grid = grid_v.shape[0]
    for i, j in grid_v:
        if mask[i, j] == 1:  # Only apply gradient for fluid cells
            dpdx = 0.0
            dpdy = 0.0

            # ∂p/∂x: check both neighbors are fluid
            if i < nb_grid - 1 and i > 0 and mask[i+1, j] == 1 and mask[i-1, j] == 1:
                dpdx = (p[i+1, j] - p[i-1, j]) / (2.0 * dx)
            elif i == 0 and nb_grid > 1 and mask[1, j] == 1:
                dpdx = (p[1, j] - p[0, j]) / dx
            elif i == nb_grid - 1 and nb_grid > 1 and mask[i-1, j] == 1:
                dpdx = (p[i, j] - p[i-1, j]) / dx

            # ∂p/∂y: check both neighbors are fluid
            if j < nb_grid - 1 and j > 0 and mask[i, j+1] == 1 and mask[i, j-1] == 1:
                dpdy = (p[i, j+1] - p[i, j-1]) / (2.0 * dx)
            elif j == 0 and nb_grid > 1 and mask[i, 1] == 1:
                dpdy = (p[i, 1] - p[i, 0]) / dx
            elif j == nb_grid - 1 and nb_grid > 1 and mask[i, j-1] == 1:
                dpdy = (p[i, j] - p[i, j-1]) / dx

            grid_v[i, j] -= dt * ti.Vector([dpdx, dpdy])


def pressure_projection(grid_m, grid_v, p, p_new, b, dx, dt, mask):
    """Pressure projection using Jacobi iterations:
        1) Compute divergence: b = div(u)
        2) Solve ∇²p = (1/dt) * b using Jacobi iterations
        3) Update velocity: u = u - dt * ∇p
    Only affects fluid cells (mask == 1)
    """
    compute_divergence(grid_m, grid_v, dx, dt, b, mask)
    p.fill(0.0)
    for iteration in range(40):
        jacobi_iteration(grid_m, grid_v, p, p_new, b, dx, mask)
        swap_pressure(p, p_new)
       
    apply_pressure_gradient(grid_m, grid_v, p, dx, dt, mask)



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
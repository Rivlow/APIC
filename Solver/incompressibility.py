import taichi as ti


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


def ensure_incompressibility(grid_m, grid_v, p, p_new, b, dx, dt, mask):
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

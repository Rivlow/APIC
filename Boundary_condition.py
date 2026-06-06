import taichi as ti


# Apply BCs
@ti.kernel
def apply_boundary(grid_m: ti.template(), grid_v: ti.template()):

    nb_grid = grid_v.shape[0]
    
    for i, j in grid_m:

        if grid_m[i, j] > 0:

            if i < 3 and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0

            if i > nb_grid - 3 and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0

            if j < 3 and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0

            #if j > nb_grid - 3 and grid_v[i, j].y > 0:
            #    grid_v[i, j].y = 0
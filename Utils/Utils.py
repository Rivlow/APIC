import taichi as ti
from fields import N, nx, ny, px, py, is_fluid


@ti.func
def is_solid_cell(i: int, j: int) -> bool:
    return i < 0 or i >= nx or j < 0 or j >= ny


@ti.kernel
def compute_is_fluid(dx: ti.f32, dy: ti.f32):
    for i, j in is_fluid:
        is_fluid[i, j] = 0
    for p in range(N):
        i = int(px[p] / dx)
        j = int(py[p] / dy)
        if 0 <= i < nx and 0 <= j < ny:
            is_fluid[i, j] = 1

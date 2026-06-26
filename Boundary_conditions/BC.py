import taichi as ti
from fields import N, px, py, pu, pvy


@ti.kernel
def apply_BC(Lx: ti.f32, Ly: ti.f32):  # type: ignore[misc]
    for p in range(N):
        if px[p] < 0.0:
            px[p] = -px[p]
            pu[p] = -pu[p]
        elif px[p] > Lx:
            px[p] = 2 * Lx - px[p]
            pu[p] = -pu[p]

        if py[p] < 0.0:
            py[p] = -py[p]
            pvy[p] = -pvy[p]
        elif py[p] > Ly:
            py[p] = 2 * Ly - py[p]
            pvy[p] = -pvy[p]

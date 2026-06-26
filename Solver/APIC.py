import taichi as ti
from fields import N, nx, ny, px, py, pu, pvy, pC, u_face, v_face, mass_u, mass_v
from Solver.Interpolation import bspline_w


@ti.kernel
def normalize():
    for i, j in u_face:
        if mass_u[i, j] > 1e-12:
            u_face[i, j] /= mass_u[i, j]
    for i, j in v_face:
        if mass_v[i, j] > 1e-12:
            v_face[i, j] /= mass_v[i, j]


@ti.kernel
def P2G(dx: ti.f32, dy: ti.f32):  # type: ignore[misc]
    for i, j in u_face:
        u_face[i, j] = 0.0
        mass_u[i, j] = 0.0
    for i, j in v_face:
        v_face[i, j] = 0.0
        mass_v[i, j] = 0.0

    for p in range(N):
        x = px[p];  y = py[p]
        u = pu[p];  v = pvy[p]
        C = pC[p]

        # u-face  (staggered: ix = x/dx, jy = y/dy - 0.5)
        ix = x / dx
        jy = y / dy - 0.5
        i0 = int(ti.floor(ix - 0.5))
        j0 = int(ti.floor(jy - 0.5))
        fx = ix - i0   # ∈ [0.5, 1.5)
        fy = jy - j0
        wx = bspline_w(fx)
        wy = bspline_w(fy)
        for di in ti.static(range(3)):
            for dj in ti.static(range(3)):
                w  = wx[di] * wy[dj]
                ig = ti.max(0, ti.min(i0 + di, nx))
                jg = ti.max(0, ti.min(j0 + dj, ny - 1))
                delta_x = (di - fx) * dx
                delta_y = (dj - fy) * dy
                ti.atomic_add(u_face[ig, jg], w * (u + C[0, 0] * delta_x + C[0, 1] * delta_y))
                ti.atomic_add(mass_u[ig, jg], w)

        # v-face  (staggered: ix = x/dx - 0.5, jy = y/dy)
        ix = x / dx - 0.5
        jy = y / dy
        i0 = int(ti.floor(ix - 0.5))
        j0 = int(ti.floor(jy - 0.5))
        fx = ix - i0
        fy = jy - j0
        wx = bspline_w(fx)
        wy = bspline_w(fy)
        for di in ti.static(range(3)):
            for dj in ti.static(range(3)):
                w  = wx[di] * wy[dj]
                ig = ti.max(0, ti.min(i0 + di, nx - 1))
                jg = ti.max(0, ti.min(j0 + dj, ny))
                delta_x = (di - fx) * dx
                delta_y = (dj - fy) * dy
                ti.atomic_add(v_face[ig, jg], w * (v + C[1, 0] * delta_x + C[1, 1] * delta_y))
                ti.atomic_add(mass_v[ig, jg], w)


@ti.kernel
def G2P(dx: ti.f32, dy: ti.f32):  # type: ignore[misc]
    for p in range(N):
        x = px[p];  y = py[p]
        vx_new = ti.cast(0.0, ti.f32)
        vy_new = ti.cast(0.0, ti.f32)
        C_new  = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        # u-face
        ix = x / dx
        jy = y / dy - 0.5
        i0 = int(ti.floor(ix - 0.5))
        j0 = int(ti.floor(jy - 0.5))
        fx = ix - i0
        fy = jy - j0
        wx = bspline_w(fx)
        wy = bspline_w(fy)
        for di in ti.static(range(3)):
            for dj in ti.static(range(3)):
                w   = wx[di] * wy[dj]
                ig  = ti.max(0, ti.min(i0 + di, nx))
                jg  = ti.max(0, ti.min(j0 + dj, ny - 1))
                u_g = u_face[ig, jg]
                delta_x = (di - fx) * dx
                delta_y = (dj - fy) * dy
                vx_new      += w * u_g
                C_new[0, 0] += w * u_g * delta_x
                C_new[0, 1] += w * u_g * delta_y

        # v-face
        ix = x / dx - 0.5
        jy = y / dy
        i0 = int(ti.floor(ix - 0.5))
        j0 = int(ti.floor(jy - 0.5))
        fx = ix - i0
        fy = jy - j0
        wx = bspline_w(fx)
        wy = bspline_w(fy)
        for di in ti.static(range(3)):
            for dj in ti.static(range(3)):
                w   = wx[di] * wy[dj]
                ig  = ti.max(0, ti.min(i0 + di, nx - 1))
                jg  = ti.max(0, ti.min(j0 + dj, ny))
                v_g = v_face[ig, jg]
                delta_x = (di - fx) * dx
                delta_y = (dj - fy) * dy
                vy_new      += w * v_g
                C_new[1, 0] += w * v_g * delta_x
                C_new[1, 1] += w * v_g * delta_y

        pu[p]  = vx_new
        pvy[p] = vy_new
        # D^-1 = 4/h² — exact pour la quadratic B-spline (D = h²/4 * I constant)
        pC[p] = ti.Matrix([
            [C_new[0, 0] * 4.0 / (dx * dx), C_new[0, 1] * 4.0 / (dy * dy)],
            [C_new[1, 0] * 4.0 / (dx * dx), C_new[1, 1] * 4.0 / (dy * dy)],
        ])


@ti.kernel
def advection(dt: ti.f32):  # type: ignore[misc]
    for p in range(N):
        px[p] += dt * pu[p]
        py[p] += dt * pvy[p]

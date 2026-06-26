import taichi as ti
from fields import nx, ny, u_face, v_face, is_fluid, div_field, pressure, \
                   cg_r, cg_p, cg_Ap, cg_r2, cg_alpha
from Utils.Utils import is_solid_cell


@ti.kernel
def compute_divergence(dx: ti.f32, dy: ti.f32):  # type: ignore[misc]
    for i, j in div_field:
        u_L = ti.cast(0.0, ti.f32) if is_solid_cell(i - 1, j) else u_face[i, j]
        u_R = ti.cast(0.0, ti.f32) if is_solid_cell(i + 1, j) else u_face[i + 1, j]
        v_B = ti.cast(0.0, ti.f32) if is_solid_cell(i, j - 1) else v_face[i, j]
        v_T = ti.cast(0.0, ti.f32) if is_solid_cell(i, j + 1) else v_face[i, j + 1]
        div_field[i, j] = (u_R - u_L) / dx + (v_T - v_B) / dy


@ti.kernel
def init_cg(dx: ti.f32, dy: ti.f32, dt: ti.f32, rho: ti.f32):  # type: ignore[misc]
    for i, j in pressure:
        b_val = ti.cast(0.0, ti.f32)
        if is_fluid[i, j] == 1:
            b_val = -(rho * dx * dy / dt) * div_field[i, j]
        pressure[i, j] = 0.0
        cg_r[i, j]     = b_val
        cg_p[i, j]     = b_val  # x0=0 donc r0=b, p0=b


@ti.kernel
def laplacian_op(q_in: ti.template(), q_out: ti.template()):  # type: ignore[misc]
    for i, j in q_out:
        if is_fluid[i, j] == 1:
            diag    = ti.cast(0.0, ti.f32)
            offdiag = ti.cast(0.0, ti.f32)
            for di, dj in ti.static([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    diag += 1.0
                    if is_fluid[ni, nj] == 1:
                        offdiag += q_in[ni, nj]
            q_out[i, j] = diag * q_in[i, j] - offdiag
        else:
            q_out[i, j] = ti.cast(0.0, ti.f32)


@ti.kernel
def dot_product(a: ti.template(), b: ti.template(), result: ti.template()):  # type: ignore[misc]
    result[None] = 0.0
    for i, j in a:
        ti.atomic_add(result[None], a[i, j] * b[i, j])


@ti.kernel
def axpy(alpha: ti.f32, x: ti.template(), y: ti.template()):  # type: ignore[misc]
    for i, j in y:
        y[i, j] += alpha * x[i, j]


@ti.kernel
def update_p_dir(beta: ti.f32):  # type: ignore[misc]
    for i, j in cg_p:
        cg_p[i, j] = cg_r[i, j] + beta * cg_p[i, j]


def taichi_cg(tol=1e-6, max_iter=500):
    dot_product(cg_r, cg_r, cg_r2)
    r2_old = cg_r2[None]
    if r2_old < tol:
        return
    for _ in range(max_iter):
        laplacian_op(cg_p, cg_Ap)
        dot_product(cg_p, cg_Ap, cg_alpha)
        alpha = r2_old / cg_alpha[None]
        axpy( alpha, cg_p,  pressure)
        axpy(-alpha, cg_Ap, cg_r)
        dot_product(cg_r, cg_r, cg_r2)
        r2_new = cg_r2[None]
        if r2_new < tol:
            break
        update_p_dir(r2_new / r2_old)
        r2_old = r2_new


@ti.kernel
def apply_pressure_correction(dx: ti.f32, dy: ti.f32, dt: ti.f32, rho: ti.f32):  # type: ignore[misc]
    for i, j in u_face:
        if i == 0 or i == nx:
            u_face[i, j] = 0.0
        else:
            left_fluid  = is_fluid[i - 1, j] == 1
            right_fluid = is_fluid[i,     j] == 1
            if left_fluid or right_fluid:
                pL = pressure[i - 1, j] if left_fluid  else ti.cast(0.0, ti.f32)
                pR = pressure[i,     j] if right_fluid else ti.cast(0.0, ti.f32)
                u_face[i, j] -= dt / (rho * dx) * (pR - pL)
            else:
                u_face[i, j] = 0.0

    for i, j in v_face:
        if j == 0 or j == ny:
            v_face[i, j] = 0.0
        else:
            bottom_fluid = is_fluid[i, j - 1] == 1
            top_fluid    = is_fluid[i, j    ] == 1
            if bottom_fluid or top_fluid:
                pB = pressure[i, j - 1] if bottom_fluid else ti.cast(0.0, ti.f32)
                pT = pressure[i, j    ] if top_fluid    else ti.cast(0.0, ti.f32)
                v_face[i, j] -= dt / (rho * dy) * (pT - pB)
            else:
                v_face[i, j] = 0.0


def ensure_incompressibility(dx, dy, dt, rho=1.0):
    compute_divergence(dx, dy)
    init_cg(dx, dy, dt, rho)
    taichi_cg()
    apply_pressure_correction(dx, dy, dt, rho)

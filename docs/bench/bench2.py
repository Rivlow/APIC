"""Benchmark complémentaire : rendu décomposé, convergence CG von Kármán, prototypes de fusion de kernels."""
import os
import sys
import time

ROOT = r"C:\Users\lucas\Dev\Fun\APIC"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import taichi as ti

from ui.runner import SimulationRunner
from ui import kernels as K
from ui import kernels_inc as M
from Code_tuto.mpm_solid import kirchhoff_stress


def timeit(fn, reps=5, warm=1):
    for _ in range(warm):
        fn()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ti.sync()
    return (time.perf_counter() - t0) * 1000.0 / reps


# ---------------------------------------------------------------- prototype : une itération CG en un seul kernel
@ti.kernel
def cg_step_fused(q: ti.template(), r: ti.template(), pd: ti.template(), Ap: ti.template(),
                  ctype: ti.template(), cg: ti.template(), n: int):
    pAp = 0.0
    for i, j in ctype:
        if ctype[i, j] == M.FLUID:
            Ap[i, j] = M.apply_A(pd, ctype, i, j, n)
            pAp += pd[i, j] * Ap[i, j]
    cg[1] = pAp
    alpha = cg[0] / ti.max(cg[1], 1e-30)
    rr_new = 0.0
    for i, j in ctype:
        if ctype[i, j] == M.FLUID:
            q[i, j] += alpha * pd[i, j]
            r[i, j] -= alpha * Ap[i, j]
            rr_new += r[i, j] * r[i, j]
    cg[2] = rr_new
    beta = cg[2] / ti.max(cg[0], 1e-30)
    for i, j in ctype:
        if ctype[i, j] == M.FLUID:
            pd[i, j] = r[i, j] + beta * pd[i, j]
    cg[0] = cg[2]


# ---------------------------------------------------------------- prototype : sous-pas solide complet en un seul kernel
@ti.kernel
def solid_substep_fused(grid_m: ti.template(), grid_v: ti.template(), grid_e: ti.template(), grid_w: ti.template(),
                        cells: ti.template(), wall_type: ti.template(), wall_v: ti.template(), fp: ti.template(),
                        x: ti.template(), v: ti.template(), C: ti.template(), F: ti.template(),
                        D: ti.template(), broken: ti.template(),
                        inv_dx: float, dt: float, dx: float, mu: float, la: float, p_mass: float, p_vol: float,
                        k_res: float, g: float, bound: int, n: int, eps0: float, epsf: float, tau_D: float,
                        use_rupture: int):
    I = ti.Matrix.identity(ti.f32, 2)
    for i, j in grid_m:                                   # clear (remplace 2 fill + clear_eps)
        grid_m[i, j] = 0.0
        grid_v[i, j] = [0.0, 0.0]
        grid_e[i, j] = 0.0
        grid_w[i, j] = 0.0
    for p in x:                                           # P2G_solid
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base
        w = K.weights(fx)
        k = ti.max(1.0 - D[p], k_res)
        if broken[p] == 1:
            k = 1.0
        tau = kirchhoff_stress(F[p], k * mu, k * la)
        affine = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * tau + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            d_pos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ d_pos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:                                   # grid_update + add_accel
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * g
            if cells[i, j] & K.OBSTACLE:
                grid_v[i, j] = [0.0, 0.0]
            if i < bound:                                 # parois : même logique que kernels.grid_update
                t = wall_type[K.LEFT, j] if bound <= j < n - bound else K.WALL
                if t == K.INLET:
                    grid_v[i, j] = wall_v[K.LEFT, j]
                elif t == K.WALL and grid_v[i, j].x < 0:
                    grid_v[i, j].x = 0.0
            if i > n - bound:
                t = wall_type[K.RIGHT, j] if bound <= j < n - bound else K.WALL
                if t == K.INLET:
                    grid_v[i, j] = wall_v[K.RIGHT, j]
                elif t == K.WALL and grid_v[i, j].x > 0:
                    grid_v[i, j].x = 0.0
            if j < bound:
                t = wall_type[K.BOTTOM, i] if bound <= i < n - bound else K.WALL
                if t == K.INLET:
                    grid_v[i, j] = wall_v[K.BOTTOM, i]
                elif t == K.WALL and grid_v[i, j].y < 0:
                    grid_v[i, j].y = 0.0
            if j > n - bound:
                t = wall_type[K.TOP, i] if bound <= i < n - bound else K.WALL
                if t == K.INLET:
                    grid_v[i, j] = wall_v[K.TOP, i]
                elif t == K.WALL and grid_v[i, j].y > 0:
                    grid_v[i, j].y = 0.0
            grid_v[i, j] += dt * fp[i, j]
    for p in x:                                           # G2P_solid + scatter_eps
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base
        w = K.weights(fx)
        v_new = ti.Vector.zero(ti.f32, 2)
        C_new = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            d_pos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            v_new += weight * g_v
            C_new += weight * g_v.outer_product(d_pos) * (4.0 * inv_dx * inv_dx)
        v[p] = v_new
        C[p] = C_new
        F_new = (I + dt * C_new) @ F[p]
        if broken[p] == 1:
            U, sig, V = ti.svd(F_new)
            for d in ti.static(range(2)):
                sig[d, d] = ti.math.clamp(sig[d, d], 0.1, 1.0)
            F_new = U @ sig @ V.transpose()
        F[p] = F_new
        x[p] += dt * v[p]
        x[p] = ti.math.clamp(x[p], bound * dx, 1.0 - bound * dx)
        if broken[p] == 0:
            base2 = (x[p] * inv_dx - 0.5).cast(int)
            fx2 = x[p] * inv_dx - base2
            w2 = K.weights(fx2)
            U, sig, V = ti.svd(F[p])
            eps = ti.max(sig[0, 0], sig[1, 1]) - 1.0
            for i, j in ti.static(ti.ndrange(3, 3)):
                weight = w2[i].x * w2[j].y
                grid_e[base2 + ti.Vector([i, j])] += weight * eps
                grid_w[base2 + ti.Vector([i, j])] += weight
    for p in x:                                           # update_damage
        if broken[p] == 0:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base
            w = K.weights(fx)
            eps = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):
                node = base + ti.Vector([i, j])
                if grid_w[node] > 0:
                    eps += w[i].x * w[j].y * grid_e[node] / grid_w[node]
            D_new = ti.math.clamp((eps - eps0) / (epsf - eps0), 0.0, 1.0)
            D[p] = ti.max(D[p], ti.min(D_new, D[p] + dt / tau_D))
            if use_rupture == 1 and D[p] >= 1.0:
                broken[p] = 1
                U, sig, V = ti.svd(F[p])
                for d in ti.static(range(2)):
                    sig[d, d] = ti.math.clamp(sig[d, d], 0.1, 1.0)
                F[p] = U @ sig @ V.transpose()


out = {}

# ---- rendu décomposé (demo)
r = SimulationRunner.demo()
s = r.solver()
s.step()
out["arch"] = s.arch
res = s.p["res"]
out["render_total_ms"] = timeit(lambda: s.render(0, 0, 1.0, True, True), reps=10)
out["render_kernel_only_ms"] = timeit(lambda: K.render(s.img, res, 0.0, 0.0, 1.0, s.cells, s.wall_type, s.p["n"],
                                                        s.p["bound"], 1, 1,
                                                        s.x_f, s.alive, 1, 0.35, 0.65, 1.0, 1,
                                                        s.sc_f, 0, 1.0, s.x_s, s.col_s, 1, 1), reps=10)
out["img_to_numpy_ms"] = timeit(lambda: s.img.to_numpy(), reps=10)
out["render_scale64_ms"] = timeit(lambda: s.render(0.5, 0.5, 64.0, True, True), reps=3)
s.release()

# ---- pont : fusion du sous-pas solide
r = SimulationRunner.load("pont_incompressible.json")
s = r.solver()
for _ in range(3):
    s.step()
p = s.p
out["pont_ms_frame"] = timeit(lambda: s.step(), reps=5)
out["pont_solid_substep_ms"] = timeit(lambda: s._solid_substep(s.dt_solid, True), reps=30)
out["pont_solid_substep_fused_ms"] = timeit(
    lambda: solid_substep_fused(s.grid_m, s.grid_v, s.grid_e, s.grid_w, s.cells, s.wall_type, s.wall_v, s.fp,
                                s.x_s, s.v_s, s.C_s, s.F_s, s.D_s, s.broken_s,
                                s.inv_dx, s.dt_solid, s.dx, s.mu_s, s.la_s, s.p_mass_s, s.p_vol, p["k_res"],
                                p["gravity"], p["bound"], p["n"], p["eps0"], p["epsf"], s.tau_D, 1), reps=30)
out["pont_cg_pair_ms"] = timeit(lambda: (M.cg_apply(s.pd, s.Ap, s.ctype, s.cg, p["n"]),
                                          M.cg_update(s.q, s.r, s.pd, s.Ap, s.ctype, s.cg)), reps=100)
out["pont_cg_fused_ms"] = timeit(lambda: cg_step_fused(s.q, s.r, s.pd, s.Ap, s.ctype, s.cg, p["n"]), reps=100)
out["pont_project_ms"] = timeit(lambda: s._project(), reps=5)
s.release()

# ---- von Kármán : convergence CG et fusion
r = SimulationRunner.load("von_karman.json")
s = r.solver()
for _ in range(3):
    s.step()
p = s.p
out["vk_ms_frame"] = timeit(lambda: s.step(), reps=5)
out["vk_project_ms"] = timeit(lambda: s._project(), reps=5)
out["vk_cg_pair_ms"] = timeit(lambda: (M.cg_apply(s.pd, s.Ap, s.ctype, s.cg, p["n"]),
                                        M.cg_update(s.q, s.r, s.pd, s.Ap, s.ctype, s.cg)), reps=100)
out["vk_cg_fused_ms"] = timeit(lambda: cg_step_fused(s.q, s.r, s.pd, s.Ap, s.ctype, s.cg, p["n"]), reps=100)
s.step()
M.cg_init(s.q, s.r, s.pd, s.rhs, s.u, s.v, s.ctype, s.cg, s.dx, p["n"])
rr0 = float(M.cg_residual(s.cg))
hist = []
for k in range(400):
    M.cg_apply(s.pd, s.Ap, s.ctype, s.cg, p["n"])
    M.cg_update(s.q, s.r, s.pd, s.Ap, s.ctype, s.cg)
    hist.append(float(M.cg_residual(s.cg)))
hist = np.array(hist) / max(rr0, 1e-30)
def first_below(tol):
    idx = np.nonzero(hist < tol)[0]
    return int(idx[0]) + 1 if len(idx) else None
out["vk_cg_rel_at_150"] = float(hist[149])
out["vk_cg_it_1e-2"] = first_below(1e-2)
out["vk_cg_it_1e-4"] = first_below(1e-4)
out["vk_cg_it_1e-6"] = first_below(1e-6)
out["vk_fluid_cells"] = int(np.sum(s.ctype.to_numpy() == M.FLUID))
st = s.stats()
out["vk_div_max"] = st["div_max"]
s.release()

for k, v in out.items():
    print(f"{k:28s} {v}")

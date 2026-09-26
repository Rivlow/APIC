"""Micro-benchmark du solveur ui/ : où va le temps ? Lancer avec APIC_UI_ARCH=vulkan|cuda."""
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


def timeit(fn, reps=5, warm=1):
    for _ in range(warm):
        fn()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ti.sync()
    return (time.perf_counter() - t0) * 1000.0 / reps


out = {}

# ---- 1. scène compressible de référence (demo, n=250, 20 sous-pas)
r = SimulationRunner.demo()
s = r.solver()
s.step()
out["arch"] = s.arch
out["demo_ms_frame"] = timeit(lambda: s.step(), reps=10)
out["demo_ms_substep"] = out["demo_ms_frame"] / r.p["substeps"]
st = s.stats()
out["demo_n_fluid"] = st["n_fluid"]
out["demo_n_solid"] = st["n_solid"]
# rendu
out["render_scale1_ms"] = timeit(lambda: s.render(0, 0, 1.0, True, True), reps=10)
out["render_scale8_ms"] = timeit(lambda: s.render(0.4, 0.4, 8.0, True, True), reps=5)
out["render_scale64_ms"] = timeit(lambda: s.render(0.5, 0.5, 64.0, True, True), reps=2)
# lancement d'un kernel minuscule (surcoût pur)
out["tiny_launch_ms"] = timeit(lambda: K.init_solid_state(s.C_s, s.F_s, s.D_s, s.broken_s), reps=200)
# lecture GPU->CPU d'un scalaire
out["readback_scalar_ms"] = timeit(lambda: float(K.count_alive(s.alive)), reps=20)
# set_params (2 to_numpy de bc_v)
out["set_params_ms"] = timeit(lambda: s.set_params({"gravity": 9.81}), reps=10)
s.release()

# ---- 2. pont incompressible (n=160, 2 sous-pas, cg 150, solide sous-cyclé)
r = SimulationRunner.load("pont_incompressible.json")
s = r.solver()
s.step()
for _ in range(5):
    s.step()
out["pont_ms_frame"] = timeit(lambda: s.step(), reps=5)
n_in = max(1, int(np.ceil(s.dt / s.dt_solid)))
out["pont_n_in"] = n_in
out["pont_dt"] = s.dt
out["pont_dt_solid"] = s.dt_solid
out["pont_project_ms"] = timeit(lambda: s._project(), reps=5)
out["pont_solid_substep_ms"] = timeit(lambda: s._solid_substep(s.dt_solid, True), reps=20)
out["pont_cg_pair_ms"] = timeit(lambda: (M.cg_apply(s.pd, s.Ap, s.ctype, s.cg, s.p["n"]),
                                          M.cg_update(s.q, s.r, s.pd, s.Ap, s.ctype, s.cg)), reps=100)
out["pont_p2g_ms"] = timeit(lambda: M.mac_p2g(s.x_f, s.v_f, s.C_f, s.alive, s.u, s.v, s.mu, s.mv, s.inv_dx, s.dx), reps=20)
out["pont_g2p_ms"] = timeit(lambda: M.mac_g2p(s.x_f, s.v_f, s.C_f, s.alive, s.u, s.v, s.mu, s.mv, s.inv_dx, s.dx), reps=20)
# combien d'itérations CG suffisent réellement (lecture du résidu à chaque itération : mesure hors production)
s.step()
p = s.p
M.cg_init(s.q, s.r, s.pd, s.rhs, s.u, s.v, s.ctype, s.cg, s.dx, p["n"])
rr0 = float(M.cg_residual(s.cg))
hist = []
for k in range(p["cg_iters"]):
    M.cg_apply(s.pd, s.Ap, s.ctype, s.cg, p["n"])
    M.cg_update(s.q, s.r, s.pd, s.Ap, s.ctype, s.cg)
    hist.append(float(M.cg_residual(s.cg)))
hist = np.array(hist) / max(rr0, 1e-30)
def first_below(tol):
    idx = np.nonzero(hist < tol)[0]
    return int(idx[0]) + 1 if len(idx) else None
out["cg_rr0"] = rr0
out["cg_it_1e-4"] = first_below(1e-4)
out["cg_it_1e-6"] = first_below(1e-6)
out["cg_it_1e-8"] = first_below(1e-8)
out["cg_rr_final_rel"] = float(hist[-1])
nf = int(np.sum(s.ctype.to_numpy() == M.FLUID))
out["pont_fluid_cells"] = nf
out["pont_cells"] = p["n"] * p["n"]
s.release()

# ---- 3. von Kármán (n=256, plein, cg 150)
r = SimulationRunner.load("von_karman.json")
s = r.solver()
s.step()
for _ in range(3):
    s.step()
out["vk_ms_frame"] = timeit(lambda: s.step(), reps=5)
out["vk_project_ms"] = timeit(lambda: s._project(), reps=5)
out["vk_cg_pair_ms"] = timeit(lambda: (M.cg_apply(s.pd, s.Ap, s.ctype, s.cg, s.p["n"]),
                                        M.cg_update(s.q, s.r, s.pd, s.Ap, s.ctype, s.cg)), reps=100)
st = s.stats()
out["vk_n_fluid"] = st["n_fluid"]
out["vk_capacity"] = st["capacity"]
s.release()

# ---- 4. recompilation au rebuild (nouveau FieldsBuilder)
r = SimulationRunner.demo()
t0 = time.perf_counter(); s = r.solver(); s.step(1); ti.sync(); out["rebuild1_ms"] = (time.perf_counter() - t0) * 1000; s.release()
t0 = time.perf_counter(); s = r.solver(); s.step(1); ti.sync(); out["rebuild2_ms"] = (time.perf_counter() - t0) * 1000; s.release()

for k, v in out.items():
    print(f"{k:24s} {v}")

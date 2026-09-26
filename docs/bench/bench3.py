"""Variantes de copie GPU->CPU de l'image rendue (700x700x3 u8), et réductions."""
import os, sys, time
ROOT = r"C:\Users\lucas\Dev\Fun\APIC"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
import taichi as ti
from ui.solver import ensure_taichi

arch = ensure_taichi()
res = 700


def timeit(fn, reps=20, warm=2):
    for _ in range(warm):
        fn()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ti.sync()
    return (time.perf_counter() - t0) * 1000.0 / reps


img_vec_u8 = ti.Vector.field(3, ti.u8, (res, res))          # actuel
img_flat_u8 = ti.field(ti.u8, (res, res * 3))
img_i32 = ti.field(ti.i32, (res, res))                        # RGB packé
img_f32 = ti.Vector.field(3, ti.f32, (res, res))
nd_u8 = ti.ndarray(ti.u8, (res, res, 3))
nd_i32 = ti.ndarray(ti.i32, (res, res))


@ti.kernel
def fill_nd(a: ti.types.ndarray(dtype=ti.u8, ndim=3)):
    for i, j, k in a:
        a[i, j, k] = ti.u8((i + j + k) & 255)


@ti.kernel
def fill_nd32(a: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i, j in a:
        a[i, j] = (i + j) & 0xFFFFFF


out = {"arch": arch}
out["vec_u8_to_numpy_ms"] = timeit(lambda: img_vec_u8.to_numpy())
out["flat_u8_to_numpy_ms"] = timeit(lambda: img_flat_u8.to_numpy())
out["i32_to_numpy_ms"] = timeit(lambda: img_i32.to_numpy())
out["vec_f32_to_numpy_ms"] = timeit(lambda: img_f32.to_numpy())
out["ndarray_u8_to_numpy_ms"] = timeit(lambda: nd_u8.to_numpy())
out["ndarray_i32_to_numpy_ms"] = timeit(lambda: nd_i32.to_numpy())
fill_nd(nd_u8); fill_nd32(nd_i32)
out["fill_nd_u8_ms"] = timeit(lambda: fill_nd(nd_u8))

# ---- réductions : atomique global vs par ligne
n = 256
f = ti.field(ti.f32, (n, n))
part = ti.field(ti.f32, n)
acc = ti.field(ti.f32, 2)
f.fill(1.0)


@ti.kernel
def red_atomic():
    s = 0.0
    for i, j in f:
        s += f[i, j] * f[i, j]
    acc[0] = s


@ti.kernel
def red_rows():
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += f[i, j] * f[i, j]
        part[i] = s
    s2 = 0.0
    for i in range(n):
        s2 += part[i]
    acc[1] = s2


@ti.kernel
def stencil_only():
    for i, j in f:
        if 0 < i < n - 1 and 0 < j < n - 1:
            part[0] = 4 * f[i, j] - f[i + 1, j] - f[i - 1, j] - f[i, j + 1] - f[i, j - 1]


out["reduce_atomic_65k_ms"] = timeit(red_atomic, reps=100)
out["reduce_rows_65k_ms"] = timeit(red_rows, reps=100)
out["stencil_no_reduce_ms"] = timeit(stencil_only, reps=100)
red_atomic(); red_rows()
out["check"] = (float(acc[0]), float(acc[1]))
for k, v in out.items():
    print(f"{k:28s} {v}")

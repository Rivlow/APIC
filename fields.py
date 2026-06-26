import taichi as ti

ti.init(arch=ti.gpu)

N      = 100000
nx, ny = 50, 50
Lx, Ly = 1.0, 1.0
dx     = Lx / nx
dy     = Ly / ny
dt     = 0.05
g      = -9.81
Nsteps = 200

px  = ti.field(dtype=ti.f32, shape=N)
py  = ti.field(dtype=ti.f32, shape=N)
pu  = ti.field(dtype=ti.f32, shape=N)
pvy = ti.field(dtype=ti.f32, shape=N)
pC  = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N)

cg_r  = ti.field(dtype=ti.f32, shape=(nx, ny))
cg_p  = ti.field(dtype=ti.f32, shape=(nx, ny))
cg_Ap = ti.field(dtype=ti.f32, shape=(nx, ny))
cg_r2    = ti.field(dtype=ti.f32, shape=())
cg_alpha = ti.field(dtype=ti.f32, shape=())
cg_beta  = ti.field(dtype=ti.f32, shape=())

u_face    = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
v_face    = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
mass_u    = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
mass_v    = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
is_fluid  = ti.field(dtype=ti.i32, shape=(nx, ny))
div_field = ti.field(dtype=ti.f32, shape=(nx, ny))
pressure  = ti.field(dtype=ti.f32, shape=(nx, ny))

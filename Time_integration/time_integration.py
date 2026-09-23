import taichi as ti

CFL = 0.5

@ti.func
def compute_dt(v:ti.template(), dx:float, E:float, rho:float) -> float:

    # dt = CFL / max_p[(|u|+c_s)/dx + (|v|+c_s)/dy]
    c_s = ti.sqrt(E/rho)
    denom = 0.0
    for p in v:
        rate = (ti.abs(v[p][0]) + c_s)/dx + (ti.abs(v[p][1]) + c_s)/dx
        ti.atomic_max(denom, rate)
    return CFL/denom

@ti.kernel
def time_integration(x:ti.template(), v:ti.template(),
                     bound:int, dx:float,
                     E:float, rho:float) -> float:

    dt = compute_dt(v, dx, E, rho)
    for p in x:
        x[p] += dt * v[p]
        x[p] = ti.math.clamp(x[p], bound * dx, 1.0 - bound * dx)  # prevent for particle out of the box
    return dt

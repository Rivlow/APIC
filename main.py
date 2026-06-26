import taichi as ti
import numpy as np
from fields import N, Lx, Ly, dx, dy, dt, g, Nsteps, px, py, pu, pvy, is_fluid, mass_u

from Pressure_projection.incompressibility import ensure_incompressibility
from Utils.Utils import compute_is_fluid
from Solver.APIC import normalize, P2G, G2P, advection
from Boundary_conditions.BC import apply_BC


@ti.kernel
def apply_gravity(g: ti.f32, dt: ti.f32):  # type: ignore[misc]
    for p in range(N):
        pvy[p] += g * dt


def init_particles():
    np.random.seed(42)
    x_np  = (Lx / 2 + np.random.uniform(-0.3,  0.3,  N)).astype(np.float32)
    y_np  = (0.8 * Ly + np.random.uniform(-0.1, 0.1, N)).astype(np.float32)
    u_np  = np.random.uniform(-0.2, 0.2, N).astype(np.float32)
    vy_np = (-1.0 + np.random.uniform(-0.2, 0.2, N)).astype(np.float32)
    px.from_numpy(x_np)
    py.from_numpy(y_np)
    pu.from_numpy(u_np)
    pvy.from_numpy(vy_np)


def print_mass_stats(frame, px_np, py_np):
    total_grid_mass = mass_u.to_numpy().sum()
    fluid_cells     = is_fluid.to_numpy().sum()
    out_of_domain   = ((px_np < 0) | (px_np > Lx) | (py_np < 0) | (py_np > Ly)).sum()
    print(f"[frame {frame:4d}]  grid_mass={total_grid_mass:.1f}  fluid_cells={fluid_cells:4d}  hors_domaine={out_of_domain}")


def main_animate():

    init_particles()
    gui = ti.GUI("PIC 2D – splash avec surface libre", res=(600, 600))
    frame = 0

    while gui.running and frame < Nsteps:
        # 1) Gravité
        apply_gravity(g, dt)

        # 2) P2G
        P2G(dx, dy)
        normalize()

        # 3) Déterminer les cellules fluides
        compute_is_fluid(dx, dy)

        # 4) Projection incompressible avec surface libre
        ensure_incompressibility(dx, dy, dt)

        # 5) G2P
        G2P(dx, dy)

        # 6) Advection et rebonds
        advection(dt)
        apply_BC(Lx, Ly)

        # Affichage
        px_np  = px.to_numpy()
        py_np  = py.to_numpy()
        pos_np = np.stack([px_np, py_np], axis=1)

        if frame % 10 == 0:
            print_mass_stats(frame, px_np, py_np)
        gui.circles(pos_np, radius=2, color=0xFF4444)
        gui.text(content=f"t = {frame * dt:.3f} s", pos=(0.05, 0.95), color=0xFFFFFF, font_size=18)
        gui.show()
        frame += 1



if __name__ == "__main__":
    main_animate()

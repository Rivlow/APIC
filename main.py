
from Solver.APIC import clear_grid, P2G, G2P
from Solver.physics import grid_step
from Solver.walls import Walls, OBSTACLE
from Solver.Fluid.fluid import Fluid
from Solver.Time_integration.time_integration import time_integration

import numpy as np
import taichi as ti
ti.init(arch=ti.gpu)  

# Numerical params
nx_grid = 256
dx = 1/nx_grid
inv_dx = float(nx_grid)

render_substep = 10
bound = 3

dt = 2e-5

# Physical params
rho = 1.0
E = 400.0 # p = E(1-J), cs = sqrt(R/rho)
g = 9.81

p_vol = (0.5*dx)**2 # 4 particules par cellule (volume d'une particule)
p_mass = p_vol*rho

# Initial state: bloc of water
x0, x1 = 0.05, 0.45
y0, y1 = 0.3, 0.9

n_particles = int(4*(x1 - x0)*(y1 - y0)*nx_grid*nx_grid)

# Obstacle (cellules bloquées, masque numpy (n, n) indexé [i, j] = (x, y))
Ox, R = (0.55, 0.30), 0.1
Xc, Yc = np.meshgrid((np.arange(nx_grid) + 0.5) * dx, (np.arange(nx_grid) + 0.5) * dx, indexing="ij")
obstacle = (Xc - Ox[0])**2 + (Yc - Ox[1])**2 < R**2

# Conditions aux limites : segments sur les quatre murs (mur glissant partout par défaut)
walls = Walls()
# walls.set("left", "inlet", velocity=(1.0, 0.0), span=(0.3, 0.6))   # exemple : entrée sur le mur gauche

#-------------- Fields -----------------#
fluid = Fluid(n_particles, E, p_mass, p_vol)                   # x, v, C (APIC), J (dilatation rate)

grid_v = ti.Vector.field(2, ti.f32, (nx_grid, nx_grid))       # Momentum 
grid_m = ti.field(ti.f32, (nx_grid, nx_grid))                 # Mass 

cells = ti.field(ti.i32, (nx_grid, nx_grid))                  # bit OBSTACLE : noeud bloqué
cells.from_numpy(np.where(obstacle, OBSTACLE, 0).astype(np.int32))
wall_type, wall_v, wall_d, _ = walls.fields(nx_grid, bound, obstacle)   # table des parois (4, n)
Image = ti.Vector.field(3, ti.f32, (nx_grid, nx_grid))


@ti.kernel
def apply_IC():

    for p in fluid.x:

        fluid.x[p] = [x0 + ti.random()*(x1 - x0), y0 + ti.random()*(y1 - y0)]
        fluid.v[p] = [0.0, 0.0]

        fluid.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        fluid.J[p] = 1.0



@ti.kernel
def render(mode: ti.i32):
    for i, j in Image:
        if cells[i, j] & OBSTACLE:
            Image[i, j] = [0.35, 0.35, 0.35]
        else:
            val = 0.0
            if mode == 0:
                val = grid_m[i, j] / (dx * dx * rho) / 0.8   # masse volumique relative (1 = eau au repos), gain 1/0.8
            else:
                val = grid_v[i, j].norm() / 3.0        # vitesse (3 m/s -> couleur maximale)
            val = ti.math.clamp(val, 0.0, 1.0)
            if mode == 0:
                Image[i, j] = ti.Vector([0.02, 0.02, 0.08]) * (1 - val) + ti.Vector([0.35, 0.65, 1.0]) * val
            else:
                Image[i, j] = ti.Vector([0.02, 0.02, 0.08]) * (1 - val) + ti.Vector([1.0, 0.75, 0.2]) * val


# ---------------------------------------------------------------- Boucle principale
def main():
    window = ti.ui.Window("APIC 2D - eau quasi-incompressible", res=(512, 512))
    canvas = window.get_canvas()
    gui = window.get_gui()
    mode = 0
    t_sim = 0.0
    dt_cfl = 0.0
    apply_IC()
    while window.running:
        while window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                window.running = False
            elif window.event.key == ti.ui.SPACE:
                mode = 1 - mode
            elif window.event.key == 'r':
                apply_IC()
                t_sim = 0.0
        for _ in range(render_substep):

            clear_grid(grid_m, grid_v)
            P2G(fluid, grid_m, grid_v, inv_dx, dx, dt)

            grid_step(grid_m, grid_v, cells, wall_type, wall_v, wall_d,
                      dt, g, 0.0, bound, nx_grid)

            G2P(fluid, grid_v, inv_dx, dx, dt)

            dt_cfl = time_integration(fluid.x, fluid.v, bound, dx, E, rho)
            t_sim += dt_cfl

        render(mode)
        canvas.set_image(Image)
        gui.begin("APIC 2D", 0.02, 0.02, 0.50, 0.22)
        gui.text(f"dt = {dt_cfl:.3e} s    t_sim = {t_sim:.4f} s")
        gui.text("Affichage : " + ("masse volumique" if mode == 0 else "vitesse"))
        gui.text("ESPACE : basculer l'affichage")
        gui.text("R : reset      ESC : quitter")
        gui.end()
        window.show()


if __name__ == "__main__":
    main()

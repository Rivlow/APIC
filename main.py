
from APIC.APIC import *
from Time_integration.time_integration import *

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

# Obstacle 
Ox = ti.Vector([0.55, 0.30])
R = 0.1

#-------------- Fields -----------------#
x = ti.Vector.field(2, ti.f32, n_particles)
v = ti.Vector.field(2, ti.f32, n_particles)

C = ti.Matrix.field(2, 2, ti.f32, n_particles)                # affine matrix APIC
J = ti.field(ti.f32, n_particles)                             # dilatation rate

grid_v = ti.Vector.field(2, ti.f32, (nx_grid, nx_grid))       # Momentum 
grid_m = ti.field(ti.f32, (nx_grid, nx_grid))                 # Mass 

solid = ti.field(ti.f32, (nx_grid, nx_grid))                  # 1 if node in obstacle
Image = ti.Vector.field(3, ti.f32, (nx_grid, nx_grid))


@ti.kernel
def apply_IC():

    for p in x:

        x[p] = [x0 + ti.random()*(x1 - x0), y0 + ti.random()*(y1 - y0)]
        v[p] = [0.0, 0.0]

        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        J[p] = 1.0

    for i, j in solid:

        pos = ti.Vector([i, j])*dx

        in_circle = (pos - Ox).norm() < R

        solid[i,j] = 1 if in_circle else 0



@ti.kernel
def render(mode: ti.i32):
    for i, j in Image:
        if solid[i, j] == 1:
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

            P2G(grid_m, grid_v,
                x, v, C, J, 
                inv_dx, dt, dx,
                E, p_mass, p_vol)
            
            grid_step(grid_m, grid_v, solid,
                      dt, g, bound, nx_grid)

            G2P(grid_m, grid_v,
                x, v, C, J,
                inv_dx, dt, dx,
                E, p_mass, p_vol)

            dt_cfl = time_integration(x, v, bound, dx, E, rho)
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

# main_fsi.py -- Partie II du tutoriel : eau APIC + pont MPM sur une grille commune
#
# Un bloc d'eau tombe sur un tablier de pont encastré dans deux piliers ; le tablier fléchit,
# s'endommage, se rompt, et les morceaux tombent dans le bassin. Le fluide est le solveur APIC
# existant (APIC/APIC.py, inchangé) ; le solide est le module MpM/mpm_solid.py.
#
# Cycle par pas de temps (une seule grille grid_m / grid_v pour les deux phases) :
#   1. P2G        (fluide)  : remet la grille à zéro, dépose masse, quantité de mouvement, pression
#   2. P2G_solid  (solide)  : ajoute masse, quantité de mouvement, force élastique
#   3. grid_step            : vitesse = quantité de mouvement / masse, gravité, parois, piliers
#   4. G2P        (fluide)  : v, C, J
#   5. G2P_solid  (solide)  : v, C, F (écrêtage des particules rompues), advection
#   6. scatter_eps + update_damage : endommagement non local, à vitesse limitée, rupture
#   7. advect_fluid         : advection du fluide avec le même dt
#
# Touches : ESPACE couleur du solide (endommagement / déformation), R reset, ESC quitter.

import taichi as ti
from APIC.APIC import P2G, G2P, grid_step
from Code_tuto.mpm_solid import (P2G_solid, G2P_solid, clear_eps, scatter_eps, update_damage,
                                 init_beam, solid_colors, solid_stats)

ti.init(arch=ti.gpu)

# ---------------------------------------------------------------- grille
n_grid = 250
dx = 1.0 / n_grid
inv_dx = float(n_grid)
bound = 3
render_substep = 20

# ---------------------------------------------------------------- fluide (comme main.py)
rho_f = 1.0
E_f = 400.0                                  # p = E (1 - J), c_f = sqrt(E_f / rho_f) = 20
p_vol_f = (0.5 * dx) ** 2
p_mass_f = p_vol_f * rho_f

water_x0, water_x1, water_y0, water_y1 = 0.30, 0.70, 0.62, 0.95      # bloc qui tombe
pool_x0, pool_x1, pool_y0, pool_y1 = 0.03, 0.97, 0.03, 0.22           # bassin au repos
n_water = int(4 * (water_x1 - water_x0) * (water_y1 - water_y0) * n_grid * n_grid)
n_pool = int(4 * (pool_x1 - pool_x0) * (pool_y1 - pool_y0) * n_grid * n_grid)
n_fluid = n_water + n_pool

# ---------------------------------------------------------------- solide (comme MpM/demo_solid.py)
rho_s = 2.0
E_s = 3000.0
nu_s = 0.3
mu_s = E_s / (2 * (1 + nu_s))
la_s = E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s))
eps0, epsf = 0.05, 0.20
tau_D = 2e-3                                 # endommagement à vitesse limitée (dD <= dt / tau_D)
k_res = 1e-3                                 # raideur résiduelle
use_damage, use_rupture = 1, 1

p_spacing = 0.5 * dx
p_vol_s = p_spacing ** 2
p_mass_s = p_vol_s * rho_s

beam_x0, beam_x1 = 0.15, 0.85
beam_y0, beam_y1 = 0.40, 0.46
pillar_w = 0.06
n_px = int(round((beam_x1 - beam_x0) / p_spacing))
n_py = int(round((beam_y1 - beam_y0) / p_spacing))
n_solid = n_px * n_py

# ---------------------------------------------------------------- pas de temps commun
g = 9.81
c_f = (E_f / rho_f) ** 0.5
c_p = ((la_s + 2 * mu_s) / rho_s) ** 0.5
dt = 0.4 * dx / max(c_f, c_p)               # un seul dt pour les deux phases

# ---------------------------------------------------------------- champs fluide
x_f = ti.Vector.field(2, ti.f32, n_fluid)
v_f = ti.Vector.field(2, ti.f32, n_fluid)
C_f = ti.Matrix.field(2, 2, ti.f32, n_fluid)
J_f = ti.field(ti.f32, n_fluid)

# ---------------------------------------------------------------- champs solide
x_s = ti.Vector.field(2, ti.f32, n_solid)
v_s = ti.Vector.field(2, ti.f32, n_solid)
C_s = ti.Matrix.field(2, 2, ti.f32, n_solid)
F_s = ti.Matrix.field(2, 2, ti.f32, n_solid)
D_s = ti.field(ti.f32, n_solid)
broken_s = ti.field(ti.i32, n_solid)
col_s = ti.Vector.field(3, ti.f32, n_solid)

# ---------------------------------------------------------------- grille commune
grid_v = ti.Vector.field(2, ti.f32, (n_grid, n_grid))
grid_m = ti.field(ti.f32, (n_grid, n_grid))
grid_e = ti.field(ti.f32, (n_grid, n_grid))           # allongement lissé (endommagement non local du solide)
grid_w = ti.field(ti.f32, (n_grid, n_grid))
solid = ti.field(ti.f32, (n_grid, n_grid))            # 1 = pilier (masque d'obstacle du fluide)
img = ti.Vector.field(3, ti.f32, (n_grid, n_grid))


@ti.kernel
def init_fluid():
    for p in x_f:
        if p < n_water:
            x_f[p] = [water_x0 + ti.random() * (water_x1 - water_x0),
                      water_y0 + ti.random() * (water_y1 - water_y0)]
        else:
            x_f[p] = [pool_x0 + ti.random() * (pool_x1 - pool_x0),
                      pool_y0 + ti.random() * (pool_y1 - pool_y0)]
        v_f[p] = [0.0, 0.0]
        C_f[p] = ti.Matrix.zero(ti.f32, 2, 2)
        J_f[p] = 1.0
    for i, j in solid:
        pos = ti.Vector([i, j]) * dx
        left = beam_x0 <= pos.x <= beam_x0 + pillar_w
        right = beam_x1 - pillar_w <= pos.x <= beam_x1
        solid[i, j] = 1.0 if ((left or right) and pos.y <= beam_y1) else 0.0


@ti.kernel
def advect_fluid():
    for p in x_f:
        x_f[p] += dt * v_f[p]
        x_f[p] = ti.math.clamp(x_f[p], bound * dx, 1.0 - bound * dx)


@ti.kernel
def render_background():
    for i, j in img:
        img[i, j] = ti.Vector([0.35, 0.35, 0.35]) if solid[i, j] == 1 else ti.Vector([0.02, 0.02, 0.08])


def init_scene():
    init_fluid()
    init_beam(x_s, v_s, C_s, F_s, D_s, broken_s, beam_x0, beam_y0, n_px, p_spacing)


def substep():
    P2G(grid_m, grid_v, x_f, v_f, C_f, J_f, inv_dx, dt, dx, E_f, p_mass_f, p_vol_f)      # 1 (remet la grille à zéro)
    P2G_solid(grid_m, grid_v, x_s, v_s, C_s, F_s, D_s, broken_s,
              inv_dx, dt, dx, mu_s, la_s, p_mass_s, p_vol_s, k_res)                       # 2
    grid_step(grid_m, grid_v, solid, dt, g, bound, n_grid)                                # 3
    G2P(grid_m, grid_v, x_f, v_f, C_f, J_f, inv_dx, dt, dx, E_f, p_mass_f, p_vol_f)      # 4
    G2P_solid(grid_v, x_s, v_s, C_s, F_s, broken_s, inv_dx, dt, dx, bound)                # 5
    clear_eps(grid_e, grid_w)                                                             # 6
    scatter_eps(grid_e, grid_w, x_s, F_s, broken_s, inv_dx)
    update_damage(grid_e, grid_w, x_s, F_s, D_s, broken_s, inv_dx, dt,
                  eps0, epsf, tau_D, use_rupture)
    advect_fluid()                                                                        # 7


# ---------------------------------------------------------------- boucle principale
def main():
    window = ti.ui.Window("APIC + MPM : eau et pont", res=(700, 700))
    canvas = window.get_canvas()
    gui = window.get_gui()
    color_mode = 0
    t_sim = 0.0
    init_scene()
    render_background()

    while window.running:
        while window.get_event(ti.ui.PRESS):
            k = window.event.key
            if k == ti.ui.ESCAPE:
                window.running = False
            elif k == ti.ui.SPACE:
                color_mode = 1 - color_mode
            elif k == 'r':
                init_scene()
                t_sim = 0.0

        for _ in range(render_substep):
            substep()
            t_sim += dt

        stats = solid_stats(D_s, broken_s)
        solid_colors(F_s, D_s, broken_s, col_s, color_mode, epsf)
        canvas.set_image(img)
        canvas.circles(x_f, radius=0.5 * p_spacing, color=(0.35, 0.65, 1.0))
        canvas.circles(x_s, radius=0.6 * p_spacing, per_vertex_color=col_s)

        gui.begin("APIC + MPM", 0.02, 0.02, 0.46, 0.22)
        gui.text(f"dt = {dt:.2e}   t = {t_sim:.3f} s")
        gui.text(f"D max = {stats[1]:.2f}   rompues = {int(stats[0])}/{n_solid}")
        gui.text("couleur : " + ("endommagement" if color_mode == 0 else "deformation principale") + "  (ESPACE)")
        gui.text("R : reset      ESC : quitter")
        gui.end()
        window.show()


if __name__ == "__main__":
    main()

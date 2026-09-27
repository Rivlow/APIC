# test_cases/poutre_endommagement.py -- Poutre encastrée dans deux piliers, chargée par une gravité en rampe :
# élasticité corotationnelle, endommagement non local, rupture. (ancien Solver/MpM/main.py)
#
# Lancer :  python test_cases/poutre_endommagement.py                  (fenêtre interactive)
#           python test_cases/poutre_endommagement.py --headless 2000  (2000 pas, affiche l'état)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import taichi as ti  # noqa: E402

ti.init(arch=ti.gpu)

from Solver.APIC import clear_grid, P2G, G2P  # noqa: E402
from Solver.physics import grid_step, advect  # noqa: E402
from Solver.Solid.solid import Solid  # noqa: E402
from Solver.walls import OBSTACLE, Walls  # noqa: E402


# ---------------------------------------------------------------- paramètres numériques
n_grid = 256
dx = 1.0 / n_grid
inv_dx = float(n_grid)
bound = 3
render_substep = 20

# ---------------------------------------------------------------- paramètres physiques
rho_s = 2.0                          # masse volumique du solide
E_s = 30000.0                        # module d'Young
nu_s = 0.3                           # coefficient de Poisson
mu_s = E_s / (2 * (1 + nu_s))        # coefficients de Lamé
la_s = E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s))
g = 9.81

eps0 = 0.05                          # début de l'endommagement (allongement principal)
epsf = 0.20                          # rupture complète (petit = fragile, grand = ductile)
tau_D = 2e-3                         # temps caractéristique d'endommagement : D croît au plus de dt/tau_D par pas
k_res = 1e-3                         # raideur résiduelle (1-D >= k_res) pour éviter les particules sans raideur
damp = 20.0                          # amortissement de la vitesse de grille (1/s), dissipe l'énergie cinétique
ramp_rate = 2000.0                   # montée de la charge (en x g par seconde simulée), évite le choc initial

# Pas de temps explicite : dt < dx / c_p avec c_p = sqrt((la + 2 mu) / rho) (ondes de compression)
c_p = ((la_s + 2 * mu_s) / rho_s) ** 0.5
dt = 0.4 * dx / c_p

p_spacing = 0.5 * dx                 # 2 x 2 particules par cellule
p_vol = p_spacing ** 2
p_mass = p_vol * rho_s

# ---------------------------------------------------------------- géométrie : poutre + piliers
beam_x0, beam_x1 = 0.15, 0.85
beam_y0, beam_y1 = 0.55, 0.61
pillar_w = 0.06                      # les extrémités de la poutre sont noyées dans les piliers

n_px = int(round((beam_x1 - beam_x0) / p_spacing))
n_py = int(round((beam_y1 - beam_y0) / p_spacing))
n_solid = n_px * n_py

# piliers : obstacles (noeuds bloqués), masque (n, n) indexé [i, j] = (x, y) ; noeud (i, j) en (i dx, j dx)
_X, _Y = np.meshgrid(np.arange(n_grid) * dx, np.arange(n_grid) * dx, indexing="ij")
pillars = (((beam_x0 <= _X) & (_X <= beam_x0 + pillar_w)) | ((beam_x1 - pillar_w <= _X) & (_X <= beam_x1))) \
    & (_Y <= beam_y1)

# ---------------------------------------------------------------- conditions aux limites
walls = Walls()                      # quatre murs glissants (valeur par défaut) ; ex. walls.set("right", "outlet")

# ---------------------------------------------------------------- champs
solid = Solid(n_solid, mu_s, la_s, k_res, p_mass, p_vol)
col_s = ti.Vector.field(3, ti.f32, n_solid)

grid_v = ti.Vector.field(2, ti.f32, (n_grid, n_grid))
grid_m = ti.field(ti.f32, (n_grid, n_grid))
cells = ti.field(ti.i32, (n_grid, n_grid))            # bit OBSTACLE = pilier
cells.from_numpy(np.where(pillars, OBSTACLE, 0).astype(np.int32))
wall_type, wall_v, wall_d = walls.fields(n_grid, bound, pillars)
grid_e = ti.field(ti.f32, (n_grid, n_grid))           # allongement pondéré (endommagement non local)
grid_w = ti.field(ti.f32, (n_grid, n_grid))           # somme des poids associée

Image = ti.Vector.field(3, ti.f32, (n_grid, n_grid))


# ---------------------------------------------------------------- état initial
@ti.kernel
def init_beam():
    """Réseau régulier 2 x 2 par cellule dans le rectangle de la poutre."""
    for p in solid.x:
        i = p % n_px
        j = p // n_px
        solid.x[p] = [beam_x0 + (i + 0.5) * p_spacing, beam_y0 + (j + 0.5) * p_spacing]


def reset():
    init_beam()
    solid.reset_state()


# ---------------------------------------------------------------- pas de temps
def substep(load, use_damage, use_rupture):
    clear_grid(grid_m, grid_v)
    P2G(solid, grid_m, grid_v, inv_dx, dx, dt)
    grid_step(grid_m, grid_v, cells, wall_type, wall_v, wall_d, dt, load * g, damp, bound, n_grid)
    G2P(solid, grid_v, inv_dx, dx, dt)
    advect(solid, dt, bound, dx)
    if use_damage == 1:
        solid.damage_step(grid_e, grid_w, inv_dx, dt, eps0, epsf, tau_D, use_rupture)


def ramp(load, load_target):
    return min(load + ramp_rate * dt, load_target) if load < load_target else load_target


# ---------------------------------------------------------------- affichage
@ti.kernel
def render_background():
    for i, j in Image:
        Image[i, j] = ti.Vector([0.35, 0.35, 0.35]) if cells[i, j] & OBSTACLE else ti.Vector([0.02, 0.02, 0.08])


def main():
    window = ti.ui.Window("MPM 2D - poutre : élasticité, endommagement, rupture", res=(700, 700))
    canvas = window.get_canvas()
    gui = window.get_gui()
    model = 3                # 1 élastique, 2 endommagement, 3 endommagement + rupture
    color_mode = 0
    load_target = 1.0
    load = 0.0               # charge effective : rejoint load_target en rampe (évite le choc)
    t_sim = 0.0

    reset()
    render_background()

    while window.running:
        while window.get_event(ti.ui.PRESS):
            k = window.event.key
            if k == ti.ui.ESCAPE:
                window.running = False
            elif k == ti.ui.SPACE:
                color_mode = 1 - color_mode
            elif k == 'r':
                reset()
                t_sim = 0.0
                load = 0.0
            elif k in ('1', '2', '3'):
                model = int(k)

        use_damage = 1 if model >= 2 else 0
        use_rupture = 1 if model == 3 else 0
        for _ in range(render_substep):
            load = ramp(load, load_target)
            substep(load, use_damage, use_rupture)
            t_sim += dt

        solid.colors(col_s, color_mode, epsf)
        canvas.set_image(Image)
        canvas.circles(solid.x, radius=0.6 * p_spacing, per_vertex_color=col_s)

        gui.begin("MPM solide", 0.02, 0.02, 0.46, 0.30)
        load_target = gui.slider_float("charge (x g)", load_target, 0.0, 250.0)
        gui.text(f"modele : {['', 'elastique', 'endommagement', 'endommagement + rupture'][model]}   (touches 1/2/3)")
        gui.text(f"dt = {dt:.2e}   t = {t_sim:.3f} s")
        gui.text("couleur : " + ("endommagement" if color_mode == 0 else "deformation principale") + "  (ESPACE)")
        gui.text("R : reset      ESC : quitter")
        gui.end()
        window.show()


def run_headless(n_steps, load_target=150.0, model=3):
    """n_steps pas sans fenêtre ; retourne (positions, D) finales."""
    reset()
    load = 0.0
    for _ in range(n_steps):
        load = ramp(load, load_target)
        substep(load, 1 if model >= 2 else 0, 1 if model == 3 else 0)
    return solid.x.to_numpy(), solid.D.to_numpy()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--headless":
        n = int(sys.argv[2]) if len(sys.argv) >= 3 else 2000
        x, D = run_headless(n)
        s = solid.stats()
        print(f"{n} pas, t = {n * dt:.4f} s : rompues = {int(s[0])} / {n_solid}, D max = {s[1]:.3f}, "
              f"y min = {x[:, 1].min():.4f}")
    else:
        main()

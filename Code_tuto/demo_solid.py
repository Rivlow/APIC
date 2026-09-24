# MpM/demo_solid.py -- Partie I du tutoriel : une poutre MPM seule (sans fluide)
#
# Une poutre encastrée à ses deux extrémités dans des piliers rigides, soumise à une charge
# réglable (multiple de la gravité). Trois niveaux de modèle, sélectionnés au clavier :
#   1 : élastique pur          2 : + endommagement          3 : + rupture
#
# Lancer depuis la racine du projet :  python -m MpM.demo_solid
# Touches : 1/2/3 modèle, ESPACE bascule couleur (endommagement / déformation), R reset, ESC quitter.
# Le curseur "charge" de l'interface multiplie la gravité : augmentez-la jusqu'à la rupture.

import taichi as ti

try:
    from Code_tuto.mpm_solid import *
except ImportError:          # lancé directement depuis le dossier MpM/
    from mpm_solid import *

ti.init(arch=ti.gpu)

# ---------------------------------------------------------------- paramètres numériques
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
bound = 3
render_substep = 20

# ---------------------------------------------------------------- paramètres physiques
rho_s = 2.0                          # masse volumique du solide
E_s = 3000.0                         # module d'Young
nu_s = 0.3                           # coefficient de Poisson
mu_s = E_s / (2 * (1 + nu_s))                        # coefficients de Lamé
la_s = E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s))
g = 9.81

eps0 = 0.05                          # début de l'endommagement (allongement principal)
epsf = 0.20                          # rupture complète (petit = fragile, grand = ductile)

# Pas de temps explicite : dt < dx / c_p avec c_p = sqrt((la + 2 mu) / rho) (ondes de compression)
c_p = (( la_s + 2 * mu_s) / rho_s) ** 0.5
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

# ---------------------------------------------------------------- champs
x_s = ti.Vector.field(2, ti.f32, n_solid)
v_s = ti.Vector.field(2, ti.f32, n_solid)
C_s = ti.Matrix.field(2, 2, ti.f32, n_solid)
F_s = ti.Matrix.field(2, 2, ti.f32, n_solid)
D_s = ti.field(ti.f32, n_solid)
broken_s = ti.field(ti.i32, n_solid)
col_s = ti.Vector.field(3, ti.f32, n_solid)

grid_v = ti.Vector.field(2, ti.f32, (n_grid, n_grid))
grid_m = ti.field(ti.f32, (n_grid, n_grid))
mask = ti.field(ti.i32, (n_grid, n_grid))            # 1 = pilier (obstacle rigide)
img = ti.Vector.field(3, ti.f32, (n_grid, n_grid))


@ti.kernel
def init_mask():
    for i, j in mask:
        pos = ti.Vector([i, j]) * dx
        left = beam_x0 <= pos.x <= beam_x0 + pillar_w
        right = beam_x1 - pillar_w <= pos.x <= beam_x1
        mask[i, j] = 1 if ((left or right) and pos.y <= beam_y1) else 0


@ti.kernel
def clear_grid():
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0


@ti.kernel
def render_background():
    for i, j in img:
        img[i, j] = ti.Vector([0.35, 0.35, 0.35]) if mask[i, j] == 1 else ti.Vector([0.02, 0.02, 0.08])


def init_scene():
    init_beam(x_s, v_s, C_s, F_s, D_s, broken_s, beam_x0, beam_y0, n_px, p_spacing)
    init_mask()


def substep(load, use_damage, use_rupture):
    clear_grid()
    P2G_solid(grid_m, grid_v, x_s, v_s, C_s, F_s, D_s, broken_s,
              inv_dx, dt, dx, mu_s, la_s, p_mass, p_vol)
    grid_update(grid_m, grid_v, mask, dt, load * g, bound, n_grid)
    G2P_solid(grid_v, x_s, v_s, C_s, F_s, D_s, broken_s,
              inv_dx, dt, dx, bound, eps0, epsf, use_damage, use_rupture)


# ---------------------------------------------------------------- boucle principale
def main():
    window = ti.ui.Window("MPM 2D - poutre : élasticité, endommagement, rupture", res=(700, 700))
    canvas = window.get_canvas()
    gui = window.get_gui()
    model = 3                # 1 élastique, 2 endommagement, 3 endommagement + rupture
    color_mode = 0
    load = 1.0
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
            elif k in ('1', '2', '3'):
                model = int(k)

        use_damage = 1 if model >= 2 else 0
        use_rupture = 1 if model == 3 else 0
        for _ in range(render_substep):
            substep(load, use_damage, use_rupture)
            t_sim += dt

        stats = solid_stats(D_s, broken_s)
        solid_colors(F_s, D_s, broken_s, col_s, color_mode, epsf)
        canvas.set_image(img)
        canvas.circles(x_s, radius=0.6 * p_spacing, per_vertex_color=col_s)

        gui.begin("MPM solide", 0.02, 0.02, 0.46, 0.30)
        load = gui.slider_float("charge (x g)", load, 0.0, 6.0)
        gui.text(f"modele : {['', 'elastique', 'endommagement', 'endommagement + rupture'][model]}   (touches 1/2/3)")
        gui.text(f"dt = {dt:.2e}   t = {t_sim:.3f} s")
        gui.text(f"D max = {stats[1]:.2f}   rompues = {int(stats[0])}/{n_solid}")
        gui.text("couleur : " + ("endommagement" if color_mode == 0 else "deformation principale") + "  (ESPACE)")
        gui.text("R : reset      ESC : quitter")
        gui.end()
        window.show()


if __name__ == "__main__":
    main()

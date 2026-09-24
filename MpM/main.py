import taichi as ti

ti.init(arch=ti.gpu)


# ---------------------------------------------------------------- paramètres numériques
n_grid = 256
dx = 1.0 / n_grid
inv_dx = float(n_grid)
bound = 3
render_substep = 20

# ---------------------------------------------------------------- paramètres physiques
rho_s = 2.0                          # masse volumique du solide
E_s = 30000.0                         # module d'Young
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

x_s      = ti.Vector.field(2, ti.f32, n_solid)
v_s      = ti.Vector.field(2, ti.f32, n_solid)

C_s      = ti.Matrix.field(2, 2, ti.f32, n_solid)        # APIC matrix
F_s      = ti.Matrix.field(2, 2, ti.f32, n_solid)        # deformation gradient
D_s      = ti.field(ti.f32, n_solid)                     # Dammage ebtween [0, 1]

broken_s = ti.field(ti.i32, n_solid)                     # = 1 if broken
col_s    = ti.Vector.field(3, ti.f32, n_solid)

grid_v   = ti.Vector.field(2, ti.f32, (n_grid, n_grid))
grid_m   = ti.field(ti.f32, (n_grid, n_grid))
mask     = ti.field(ti.i32, (n_grid, n_grid))

Image = ti.Vector.field(3, ti.f32, (n_grid, n_grid))


@ti.func
def kirchhoff_stress(F, mu: float, la: float):

    U, sigma, V = ti.svd(F)
    R = U @ V.transpose()
    J = F.determinant()
    I = ti.Matrix.identity(ti.f32, 2)

    return 2.0*mu * (F - R) @ F.transpose() + la*J*(J-1.0)*I

@ti.kernel
def P2G_solid(grid_m:ti.template(), grid_v:ti.template(), x:ti.template(), v:ti.template(),
              C:ti.template(), F:ti.template(), D:ti.template(), broken:ti.template(),
              inv_dx:float, dt:float, dx:float,
              mu:float, la:float, p_mass:float, p_vol:float):
    
    for p in x:

        base = (x[p]*inv_dx - 0.5).cast(int)
        fx  = x[p]*inv_dx - base

        w = [0.5*(1.5-fx)**2,
             0.75 - (fx-1.0)**2,
             0.5*(fx-0.5)**2]

        k = 1.0 - D[p]       # effective stiffness (1-D)*E
        if broken[p] == 1.0:
            k = 1.0
        tau = kirchhoff_stress(F[p], k*mu, k*la)


        affine = (-dt*p_vol * 4.0*inv_dx*inv_dx) * tau + p_mass*C[p]

        for i,j in ti.static(ti.ndrange(3, 3)):

            offset = ti.Vector([i,j])
            d_pos = (offset - fx)*dx
            weight = w[i].x * w[j].y

            grid_m[base + offset] += weight * p_mass
            grid_v[base + offset] += weight * (p_mass*v[p] + affine@d_pos)

@ti.kernel
def grid_update(grid_m:ti.template(), grid_v:ti.template(), mask:ti.template(),
                dt:float, g:float, bound:int, n_grid:int):

    for i,j in grid_m:

        if grid_m[i,j] > 0:
            grid_v[i,j] /= grid_m[i,j]
            grid_v[i,j].y -= dt*g

            if i < bound and grid_v[i,j].x < 0:
                grid_v[i,j].x = 0.0
            if i > n_grid - bound and grid_v[i,j].x > 0:
                grid_v[i,j].x = 0.0
            if j < bound and grid_v[i,j].y < 0:
                grid_v[i,j].y = 0.0
            if j > n_grid - bound and grid_v[i,j].y > 0:
                grid_v[i,j].y = 0.0

            if mask[i,j] == 1:
                grid_v[i,j] = [0.0, 0.0]


@ti.kernel
def init_mask():

    for i,j in mask:

        pos = ti.Vector([i,j])*dx

        left = beam_x0 <= pos.x <= beam_x0 + pillar_w
        right = beam_x1 - pillar_w <= pos.x <= beam_x1

        mask[i,j] = 1 if ((left or right) and pos.y <= beam_y1) else 0

@ti.kernel
def G2P_solid(grid_v:ti.template(), x:ti.template(), v:ti.template(),
              C:ti.template(), F:ti.template(), D:ti.template(), broken:ti.template(),
              inv_dx:float, dt:float, dx:float, bound:int,
              eps0:float, epsf:float, use_damage:int, use_rupture:int):

    I = ti.Matrix.identity(ti.f32, 2)

    for p in x:

        base = (x[p]*inv_dx - 0.5).cast(int)
        fx = x[p]*inv_dx - base
        w = [0.5*(1.5-fx)**2,
             0.75 - (fx-1.0)**2,
             0.5*(fx-0.5)**2]
        v_new = ti.Vector.zero(ti.f32, 2)
        C_new = ti.Matrix.zero(ti.f32, 2, 2)

        for i,j in ti.static(ti.ndrange(3,3)):

            offset = ti.Vector([i,j])
            d_pos = (offset - fx)*dx
            weight = w[i].x * w[j].y

            v_new += weight*grid_v[base + offset]
            C_new += weight* grid_v[base + offset].outer_product(d_pos) * (4.0*inv_dx*inv_dx)

        v[p] = v_new
        C[p] = C_new

        # df/dt = grad(v) ~ C
        F_new = (I + dt*C_new) @ F[p]

        if broken[p] == 1:
            U, sigma, V = ti.svd(F_new)

            for d in ti.static(range(2)):
                sigma[d,d] = ti.math.clamp(sigma[d,d], 0.1, 1.0)
            F_new = U @ sigma @ V.transpose()

        elif use_damage == 1:
            U, sigma, V = ti.svd(F_new)
            
            eps = ti.max(sigma[0,0], sigma[1,1]) - 1.0
            D_new = ti.math.clamp((eps-eps0)/(epsf-eps0), 0.0, 1.0)

            D[p] = ti.max(D[p], D_new)

            if use_rupture == 1 and D[p] >= 1.0:

                broken[p] = 1
                F_new = I

        F[p] = F_new
        x[p] += dt*v[p]
        x[p] = ti.math.clamp(x[p], bound*dx, 1.0 - bound*dx)

@ti.kernel
def init_beam(x: ti.template(), v: ti.template(), C: ti.template(), F: ti.template(),
              D: ti.template(), broken: ti.template(),
              x0: float, y0: float, n_px: int, spacing: float):
    for p in x:
        i = p % n_px
        j = p // n_px
        x[p] = [x0 + (i + 0.5) * spacing, y0 + (j + 0.5) * spacing]
        v[p] = [0.0, 0.0]
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        F[p] = ti.Matrix.identity(ti.f32, 2)
        D[p] = 0.0
        broken[p] = 0

@ti.kernel
def clear_grid():
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0

def substep(load, use_damage, use_rupture):
    clear_grid()
    P2G_solid(grid_m, grid_v, x_s, v_s, C_s, F_s, D_s, broken_s,
    inv_dx, dt, dx, mu_s, la_s, p_mass, p_vol)
    grid_update(grid_m, grid_v, mask, dt, load * g, bound, n_grid)
    G2P_solid(grid_v, x_s, v_s, C_s, F_s, D_s, broken_s,
    inv_dx, dt, dx, bound, eps0, epsf, use_damage, use_rupture)

@ti.kernel
def render_background():
    for i, j in Image:
        Image[i, j] = ti.Vector([0.35, 0.35, 0.35]) if mask[i, j] == 1 else ti.Vector([0.02, 0.02, 0.08])


@ti.kernel
def solid_colors(F: ti.template(), D: ti.template(), broken: ti.template(),
                 col: ti.template(), mode: int, eps_scale: float):
    """Couleur d'affichage : mode 0 = endommagement (gris -> jaune, rouge si rompu),
    mode 1 = allongement principal max (bleu = compression, rouge = traction)."""

    for p in D:
        if mode == 0:
            if broken[p] == 1:
                col[p] = ti.Vector([0.90, 0.15, 0.15])
            else:
                col[p] = ti.Vector([0.75, 0.75, 0.75]) * (1 - D[p]) + ti.Vector([1.0, 0.85, 0.1]) * D[p]
        else:
            U, sig, V = ti.svd(F[p])
            eps = ti.max(sig[0, 0], sig[1, 1]) - 1.0
            t = ti.math.clamp(0.5 + 0.5 * eps / eps_scale, 0.0, 1.0)
            col[p] = ti.Vector([0.2, 0.4, 1.0]) * (1 - t) + ti.Vector([1.0, 0.25, 0.1]) * t

def main():
    window = ti.ui.Window("MPM 2D - poutre : élasticité, endommagement, rupture", res=(700, 700))
    canvas = window.get_canvas()
    gui = window.get_gui()
    model = 3                # 1 élastique, 2 endommagement, 3 endommagement + rupture
    color_mode = 0
    load = 1.0
    t_sim = 0.0
    
    init_beam(x_s, v_s, C_s, F_s, D_s, broken_s, beam_x0, beam_y0, n_px, p_spacing)
    init_mask()

    render_background()

    while window.running:
        while window.get_event(ti.ui.PRESS):
            k = window.event.key
            if k == ti.ui.ESCAPE:
                window.running = False
            elif k == ti.ui.SPACE:
                color_mode = 1 - color_mode
            elif k == 'r':
                init_beam(x_s, v_s, C_s, F_s, D_s, broken_s, beam_x0, beam_y0, n_px, p_spacing)
                init_mask()

                t_sim = 0.0
            elif k in ('1', '2', '3'):
                model = int(k)

        use_damage = 1 if model >= 2 else 0
        use_rupture = 1 if model == 3 else 0
        for _ in range(render_substep):
            substep(load, use_damage, use_rupture)
            t_sim += dt

        solid_colors(F_s, D_s, broken_s, col_s, color_mode, epsf)
        canvas.set_image(Image)
        canvas.circles(x_s, radius=0.6 * p_spacing, per_vertex_color=col_s)

        gui.begin("MPM solide", 0.02, 0.02, 0.46, 0.30)
        load = gui.slider_float("charge (x g)", load, 0.0, 250.0)
        gui.text(f"modele : {['', 'elastique', 'endommagement', 'endommagement + rupture'][model]}   (touches 1/2/3)")
        gui.text(f"dt = {dt:.2e}   t = {t_sim:.3f} s")
        gui.text("couleur : " + ("endommagement" if color_mode == 0 else "deformation principale") + "  (ESPACE)")
        gui.text("R : reset      ESC : quitter")
        gui.end()
        window.show()


if __name__ == "__main__":
    main()
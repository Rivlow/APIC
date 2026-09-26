# MpM/beton.py -- Poutre en béton armé en flexion 4 points, MLS-MPM 2D
#
# Inspiré de Cremona & Houde, "Modélisation déterministe et probabiliste du comportement
# mécanique simplifié des corps d'épreuve" (poutres de la Rance) :
#   - béton en compression : loi de Desayi/Neville, écrasement vers -0.004
#   - béton en traction    : linéaire jusqu'à fcr, puis fcr / (1 + sqrt(500 eps))  (raidissement)
#   - acier                : bilinéaire (élastique parfaitement plastique)
#   - résistances aléatoires par particule (lois lognormales, comme dans l'approche Monte-Carlo)
#   - essai de flexion 4 points sur appuis simples, portée 2 m, charges à 0.6 m des appuis
#
# Unités SI (m, kg, s, Pa). Le calcul est 2D en déformations planes : les efforts sont par mètre de
# profondeur ; on les multiplie par la largeur de la poutre b = 0.2 m pour retrouver des kN.
#
# Lancer :  python MpM/beton.py                 (fenêtre interactive)
#           python MpM/beton.py --headless 25   (course de vérin 25 mm, écrit beton_courbe.csv)

import sys
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, random_seed=1)

# ---------------------------------------------------------------- domaine (rectangle Lx x Ly)
Lx, Ly = 3.0, 0.625
nx, ny = 192, 40
dx = Lx / nx                          # 1.56 cm
inv_dx = 1.0 / dx
bound = 3
render_substep = 50

# ---------------------------------------------------------------- béton (poutre 421 de la Rance)
rho_c = 2400.0
E_c = 30.4e9                          # module d'Young
nu_c = 0.2
fc = 47.1e6                           # résistance en compression
fcr = 3.5e6                           # résistance en traction
cv_fc, cv_fcr = 0.11, 0.26            # coefficients de variation (écart type / moyenne, tableau 3)
eps_cu = 0.004                        # déformation d'écrasement
r_min = 1e-3                          # raideur résiduelle en traction (béton fissuré)
r_crush = 0.1                         # raideur résiduelle en compression (béton écrasé)
mu_c = E_c / (2 * (1 + nu_c))
la_c = E_c * nu_c / ((1 + nu_c) * (1 - 2 * nu_c))

# ---------------------------------------------------------------- acier d'armature
E_s = 230e9
fy = 309e6                            # limite d'élasticité

# ---------------------------------------------------------------- amortissement et chargement
damp = 100.0                          # amortissement de la vitesse de grille (1/s)
g = 9.81
t_ramp = 0.02                         # durée de la montée en charge (s)
b_depth = 0.2                         # largeur de la poutre (m) : passage N/m -> N

# ---------------------------------------------------------------- géométrie
beam_x0, beam_x1 = 0.3, 2.7           # la poutre fait 2.4 m
beam_y0 = 0.25
xs1, xs2 = 0.5, 2.5                   # appuis : portée 2.0 m
xp1, xp2 = 1.1, 1.9                   # charges à 0.6 m des appuis
hw = 0.05                             # demi-largeur des appuis et des vérins
h_target = 0.2

p_spacing = 0.5 * dx                  # 2 x 2 particules par cellule
p_vol = p_spacing ** 2
p_mass = p_vol * rho_c
n_px = int(round((beam_x1 - beam_x0) / p_spacing))
n_py = int(round(h_target / p_spacing))
n_solid = n_px * n_py
h_beam = n_py * p_spacing
beam_y1 = beam_y0 + h_beam
x_mid = 0.5 * (xs1 + xs2)

# lits d'armatures du type 4 de l'article : (distance à la base en m, aire totale en m2)
rebars = [(0.019, 4 * 113.1e-6), (0.072, 2 * 56.5e-6), (0.128, 2 * 56.5e-6), (0.181, 4 * 113.1e-6)]
phi_np = np.zeros(n_py, dtype=np.float32)
for dist, area in rebars:
    phi_np[int(dist / p_spacing)] += (area / b_depth) / p_spacing      # fraction volumique d'acier
phi_max = float(phi_np.max())

# pas de temps explicite : dt < dx / c avec la raideur axiale maximale (béton + acier)
c_max = ((la_c + 2 * mu_c + phi_max * E_s) / rho_c) ** 0.5
dt = 0.3 * dx / c_max

sl_cr = float(np.log(1 + cv_fcr ** 2) ** 0.5)      # écarts types des logarithmes (lognormale)
sl_fc = float(np.log(1 + cv_fc ** 2) ** 0.5)

# ---------------------------------------------------------------- champs
X0 = ti.Vector.field(2, ti.f32, n_solid)          # position initiale
Up = ti.Vector.field(2, ti.f32, n_solid)          # déplacement (x = X0 + Up, voir précision float32)
v_s = ti.Vector.field(2, ti.f32, n_solid)
C_s = ti.Matrix.field(2, 2, ti.f32, n_solid)      # matrice affine APIC
Fm = ti.Matrix.field(2, 2, ti.f32, n_solid)       # F - I (l'écart à l'identité garde la précision)
kap_t = ti.field(ti.f32, n_solid)                 # plus grande déformation de traction équivalente
kap_c = ti.field(ti.f32, n_solid)                 # plus grande déformation de compression équivalente
phi = ti.field(ti.f32, n_solid)                   # fraction volumique d'acier de la particule
eps_p = ti.field(ti.f32, n_solid)                 # déformation plastique de l'acier
fcr_p = ti.field(ti.f32, n_solid)                 # résistance en traction de la particule
fc_p = ti.field(ti.f32, n_solid)                  # résistance en compression de la particule
phi_row = ti.field(ti.f32, n_py)
phi_row.from_numpy(phi_np)

grid_v = ti.Vector.field(2, ti.f32, (nx, ny))
grid_m = ti.field(ti.f32, (nx, ny))
grid_et = ti.field(ti.f32, (nx, ny))              # déformation de traction lissée (non local)
grid_ec = ti.field(ti.f32, (nx, ny))
grid_w = ti.field(ti.f32, (nx, ny))
mask = ti.field(ti.i32, (nx, ny))                 # 1 = appui
react = ti.field(ti.f32, ())                      # réaction des vérins (N/m)

W, H = 1200, 250
Image = ti.Vector.field(3, ti.f32, (W, H))


# ---------------------------------------------------------------- lois de comportement
@ti.func
def r_tension(kap, fcr_i):
    """Raideur sécante en traction : 1 avant fissuration, puis fcr/(1+sqrt(500 eps)) / (E eps)."""
    r = 1.0
    if kap > fcr_i / E_c:
        r = ti.min(1.0, fcr_i / (1.0 + ti.sqrt(500.0 * kap)) / (E_c * kap))
    return ti.max(r, r_min)


@ti.func
def r_compression(kap, fc_i):
    """Raideur sécante en compression : loi de Desayi, puis écrasement progressif après eps_cu."""
    x = kap / (1.8 * fc_i / E_c)
    r = ti.min(1.0, (2.0 / 1.8) / (1.0 + x * x))
    if kap > eps_cu:
        t = ti.min((kap - eps_cu) / (0.5 * eps_cu), 1.0)
        r = r * (1.0 - t) + r_crush * t
    return ti.max(r, r_min)


@ti.func
def stress_kirchhoff(F, kt, kc, ph, ep, fcr_i, fc_i, nl: int):
    """Contrainte de Kirchhoff : béton (Hencky, raideurs sécantes par direction principale)
    + acier (barre uniaxiale suivant la direction x du matériau, élastique parfaitement plastique)."""
    U, sig, V = ti.svd(F)
    e1 = ti.log(ti.max(sig[0, 0], 0.05))
    e2 = ti.log(ti.max(sig[1, 1], 0.05))
    tr = e1 + e2
    s1 = 2.0 * mu_c * e1 + la_c * tr
    s2 = 2.0 * mu_c * e2 + la_c * tr
    if nl == 1:
        rt = r_tension(kt, fcr_i)
        rc = r_compression(kc, fc_i)
        if s1 > 0.0:
            s1 *= rt
        else:
            s1 *= rc
        if s2 > 0.0:
            s2 *= rt
        else:
            s2 *= rc
    S = ti.Matrix([[s1, 0.0], [0.0, s2]])
    tau = U @ S @ U.transpose()
    if ph > 0.0:
        n = ti.Vector([F[0, 0], F[1, 0]])
        lam = n.norm()
        n = n / lam
        ss = E_s * ti.log(lam)
        if nl == 1:
            ss = ti.math.clamp(E_s * (ti.log(lam) - ep), -fy, fy)
        tau = (1.0 - ph) * tau + ph * ss * n.outer_product(n)
    return tau


# ---------------------------------------------------------------- 1. particules -> grille
@ti.kernel
def P2G_solid(dt: float, nl: int):
    for p in X0:
        xp = X0[p] + Up[p]
        base = (xp * inv_dx - 0.5).cast(int)
        fx = xp * inv_dx - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

        F = ti.Matrix.identity(ti.f32, 2) + Fm[p]
        tau = stress_kirchhoff(F, kap_t[p], kap_c[p], phi[p], eps_p[p], fcr_p[p], fc_p[p], nl)
        affine = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * tau + p_mass * C_s[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            d_pos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_m[base + offset] += weight * p_mass
            grid_v[base + offset] += weight * (p_mass * v_s[p] + affine @ d_pos)


# ---------------------------------------------------------------- 2. grille : appuis, vérins
@ti.kernel
def grid_update(dt: float, g_eff: float, v_load: float, u_load: float):
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j] *= 1.0 - damp * dt
            grid_v[i, j].y -= dt * g_eff

            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0.0
            if i > nx - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0.0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0.0
            if j > ny - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0.0

            # appuis : contact glissant sans frottement (seule la composante normale est contrainte)
            if mask[i, j] == 1 and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0.0

            # vérins : plateaux glissants qui descendent à la vitesse v_load (composante normale imposée),
            # noeuds situés sous la face inférieure des plateaux
            pos = ti.Vector([i, j]) * dx
            if (ti.abs(pos.x - xp1) <= hw or ti.abs(pos.x - xp2) <= hw) and pos.y >= beam_y1 - u_load:
                if grid_v[i, j].y > -v_load:
                    react[None] += grid_m[i, j] * (grid_v[i, j].y + v_load) / dt    # force sur la poutre
                    grid_v[i, j].y = -v_load


@ti.kernel
def init_mask():
    for i, j in mask:
        pos = ti.Vector([i, j]) * dx
        support = (ti.abs(pos.x - xs1) <= hw or ti.abs(pos.x - xs2) <= hw) and pos.y <= beam_y0
        mask[i, j] = 1 if support else 0


# ---------------------------------------------------------------- 3. grille -> particules
@ti.kernel
def G2P_solid(dt: float):
    I = ti.Matrix.identity(ti.f32, 2)
    for p in X0:
        xp = X0[p] + Up[p]
        base = (xp * inv_dx - 0.5).cast(int)
        fx = xp * inv_dx - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

        v_new = ti.Vector.zero(ti.f32, 2)
        C_new = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            d_pos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            v_new += weight * grid_v[base + offset]
            C_new += weight * grid_v[base + offset].outer_product(d_pos) * (4.0 * inv_dx * inv_dx)
        v_s[p] = v_new
        C_s[p] = C_new

        # F = I + Fm, dF/dt = (grad v) F   ->   Fm <- Fm + dt C (I + Fm)
        Fm[p] += dt * C_new @ (I + Fm[p])

        Up[p] += dt * v_s[p]
        xn = X0[p] + Up[p]
        xc = ti.math.clamp(xn, ti.Vector([bound * dx, bound * dx]),
                           ti.Vector([Lx - bound * dx, Ly - bound * dx]))
        Up[p] += xc - xn


# ---------------------------------------------------------------- 4. endommagement non local
@ti.kernel
def clear_grid():
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0
        grid_et[i, j] = 0.0
        grid_ec[i, j] = 0.0
        grid_w[i, j] = 0.0


@ti.kernel
def scatter_eps():
    """Étale sur la grille les déformations équivalentes (contrainte principale élastique / E)."""
    for p in X0:
        xp = X0[p] + Up[p]
        base = (xp * inv_dx - 0.5).cast(int)
        fx = xp * inv_dx - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

        Us, sg, Vs = ti.svd(ti.Matrix.identity(ti.f32, 2) + Fm[p])
        e1 = ti.log(ti.max(sg[0, 0], 0.05))
        e2 = ti.log(ti.max(sg[1, 1], 0.05))
        s1 = 2.0 * mu_c * e1 + la_c * (e1 + e2)
        s2 = 2.0 * mu_c * e2 + la_c * (e1 + e2)
        et = ti.max(ti.max(s1, s2), 0.0) / E_c
        ec = ti.max(ti.max(-s1, -s2), 0.0) / E_c

        for i, j in ti.static(ti.ndrange(3, 3)):
            weight = w[i].x * w[j].y
            node = base + ti.Vector([i, j])
            grid_et[node] += weight * et
            grid_ec[node] += weight * ec
            grid_w[node] += weight


@ti.kernel
def update_state():
    """Historique de déformation (irréversible) du béton, plasticité de l'acier."""
    for p in X0:
        xp = X0[p] + Up[p]
        base = (xp * inv_dx - 0.5).cast(int)
        fx = xp * inv_dx - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

        et = 0.0
        ec = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            node = base + ti.Vector([i, j])
            if grid_w[node] > 0:
                weight = w[i].x * w[j].y
                et += weight * grid_et[node] / grid_w[node]
                ec += weight * grid_ec[node] / grid_w[node]
        kap_t[p] = ti.max(kap_t[p], et)
        kap_c[p] = ti.max(kap_c[p], ec)

        if phi[p] > 0.0:
            eps = ti.log(ti.Vector([Fm[p][0, 0] + 1.0, Fm[p][1, 0]]).norm())
            trial = E_s * (eps - eps_p[p])
            ss = ti.math.clamp(trial, -fy, fy)
            eps_p[p] += (trial - ss) / E_s                  # retour radial : écoulement plastique


# ---------------------------------------------------------------- initialisation, mesures
@ti.kernel
def init_beam():
    for p in X0:
        i = p % n_px
        j = p // n_px
        X0[p] = [beam_x0 + (i + 0.5) * p_spacing, beam_y0 + (j + 0.5) * p_spacing]
        Up[p] = [0.0, 0.0]
        v_s[p] = [0.0, 0.0]
        C_s[p] = ti.Matrix.zero(ti.f32, 2, 2)
        Fm[p] = ti.Matrix.zero(ti.f32, 2, 2)
        kap_t[p] = 0.0
        kap_c[p] = 0.0
        eps_p[p] = 0.0
        phi[p] = phi_row[j]
        # résistances aléatoires par particule : lognormales de moyenne fcr, fc
        fcr_p[p] = fcr * ti.min(ti.max(ti.exp(sl_cr * ti.randn() - 0.5 * sl_cr**2), 0.3), 3.0)
        fc_p[p] = fc * ti.min(ti.max(ti.exp(sl_fc * ti.randn() - 0.5 * sl_fc**2), 0.5), 2.0)


@ti.kernel
def bottom_mid_uy() -> ti.types.vector(2, ti.f32):
    """Somme et nombre des déplacements verticaux de la fibre inférieure à mi-portée."""
    s = 0.0
    n = 0.0
    for p in X0:
        if p // n_px == 0 and ti.abs(X0[p].x - x_mid) < 0.1:
            s += Up[p].y
            n += 1.0
    return ti.Vector([s, n])


@ti.kernel
def damage_counts() -> ti.types.vector(2, ti.f32):
    """Nombre de particules de béton fissurées (d_t > 0.9) et écrasées (eps_c > eps_cu)."""
    nt = 0.0
    nc = 0.0
    for p in X0:
        if phi[p] == 0.0:
            if 1.0 - r_tension(kap_t[p], fcr_p[p]) > 0.9:
                nt += 1.0
            if kap_c[p] > eps_cu:
                nc += 1.0
    return ti.Vector([nt, nc])


# ---------------------------------------------------------------- affichage
@ti.func
def particle_color(p: int, mode: int):
    col = ti.Vector([0.75, 0.75, 0.75])
    if mode == 0:
        if phi[p] > 0.0:
            col = ti.Vector([0.25, 0.45, 1.0])                           # armature élastique
            if ti.abs(eps_p[p]) > 1e-4:
                col = ti.Vector([1.0, 0.55, 0.10])                       # armature plastifiée
        else:
            d_t = 1.0 - r_tension(kap_t[p], fcr_p[p])
            d_c = ti.min(1.0, (1.0 - r_compression(kap_c[p], fc_p[p])) / 0.6)
            col = ti.Vector([0.75, 0.75, 0.75]) * (1 - d_t) + ti.Vector([1.0, 0.85, 0.1]) * d_t
            if d_t > 0.95:
                col = ti.Vector([0.90, 0.15, 0.15])                      # fissure
            col = col * (1 - d_c) + ti.Vector([0.3, 0.85, 0.95]) * d_c   # compression endommagée
            if kap_c[p] > eps_cu:
                col = ti.Vector([0.60, 0.20, 0.80])                      # écrasement
    else:
        Us, sg, Vs = ti.svd(ti.Matrix.identity(ti.f32, 2) + Fm[p])
        e1 = ti.log(ti.max(sg[0, 0], 0.05))
        e2 = ti.log(ti.max(sg[1, 1], 0.05))
        s1 = 2.0 * mu_c * e1 + la_c * (e1 + e2)
        s2 = 2.0 * mu_c * e2 + la_c * (e1 + e2)
        a = ti.math.clamp(ti.max(s1, s2) / fcr, 0.0, 1.0)
        b = ti.math.clamp(-ti.min(s1, s2) / fc, 0.0, 1.0)
        t = 0.5 + 0.5 * (a - b)
        col = ti.Vector([0.2, 0.4, 1.0]) * (1 - t) + ti.Vector([1.0, 0.25, 0.1]) * t
    return col


@ti.kernel
def render(mode: int, u_load: float):
    for i, j in Image:
        pos = ti.Vector([(i + 0.5) * Lx / W, (j + 0.5) * Ly / H])
        col = ti.Vector([0.02, 0.02, 0.08])
        if (ti.abs(pos.x - xs1) <= hw or ti.abs(pos.x - xs2) <= hw) and pos.y <= beam_y0:
            col = ti.Vector([0.35, 0.35, 0.35])
        if (ti.abs(pos.x - xp1) <= hw or ti.abs(pos.x - xp2) <= hw) and pos.y >= beam_y1 - u_load:
            col = ti.Vector([0.55, 0.55, 0.60])
        Image[i, j] = col
    for p in X0:
        xp = X0[p] + Up[p]
        ci = int(xp.x / Lx * W)
        cj = int(xp.y / Ly * H)
        col = particle_color(p, mode)
        for a, b in ti.static(ti.ndrange(4, 4)):            # carré de 4 x 4 pixels (pas de trous)
            ii = ci + a - 1
            jj = cj + b - 1
            if 0 <= ii < W and 0 <= jj < H:
                Image[ii, jj] = col


# ---------------------------------------------------------------- simulation
class State:
    t = 0.0
    u = 0.0          # course des vérins (m)


def reset():
    init_beam()
    init_mask()
    State.t = 0.0
    State.u = 0.0


def smoothstep(t):
    s = min(t / t_ramp, 1.0)
    return s * s * (3.0 - 2.0 * s)


def substep(v_load, model):
    """Un pas de temps. model : 1 élastique linéaire, 2 béton armé non linéaire."""
    s = smoothstep(State.t)
    v_eff = v_load * s
    nl = 1 if model >= 2 else 0
    clear_grid()
    P2G_solid(dt, nl)
    grid_update(dt, g * s, v_eff, State.u)
    G2P_solid(dt)
    if nl == 1:
        scatter_eps()
        update_state()
    State.t += dt
    State.u += v_eff * dt


def advance(n, v_load, model):
    """n pas de temps ; retourne (P en kN moyenné sur la série, flèche à mi-portée en mm)."""
    react[None] = 0.0
    for _ in range(n):
        substep(v_load, model)
    P = react[None] / n * b_depth / 1000.0
    sn = bottom_mid_uy()
    delta = -sn[0] / max(sn[1], 1.0) * 1000.0
    return P, delta


HEADER = "t_s,course_verin_mm,fleche_mm,P_kN,n_fissurees,n_ecrasees"


def run_headless(u_max_mm, v_load=0.2, model=2, csv="beton_courbe.csv"):
    reset()
    rows = []
    while State.u * 1000.0 < u_max_mm:
        P, delta = advance(render_substep, v_load, model)
        dc = damage_counts()
        rows.append((State.t, State.u * 1000.0, delta, P, dc[0], dc[1]))
    np.savetxt(csv, np.array(rows), delimiter=",", header=HEADER, comments="")
    return np.array(rows)


def main():
    window = ti.ui.Window("MPM 2D - poutre en béton armé, flexion 4 points", res=(W, H))
    canvas = window.get_canvas()
    gui = window.get_gui()
    model = 2
    color_mode = 0
    v_load = 0.2
    P_max = 0.0
    reset()
    hist = []

    while window.running:
        while window.get_event(ti.ui.PRESS):
            k = window.event.key
            if k == ti.ui.ESCAPE:
                window.running = False
            elif k == ti.ui.SPACE:
                color_mode = 1 - color_mode
            elif k == 'r':
                reset()
                P_max = 0.0
                hist = []
            elif k in ('1', '2'):
                model = int(k)
            elif k == 's' and hist:
                np.savetxt("beton_courbe.csv", np.array(hist), delimiter=",", header=HEADER, comments="")

        P, delta = advance(render_substep, v_load, model)
        P_max = max(P_max, P)
        dc = damage_counts()
        hist.append((State.t, State.u * 1000.0, delta, P, dc[0], dc[1]))

        render(color_mode, State.u)
        canvas.set_image(Image)

        gui.begin("Flexion 4 points", 0.01, 0.01, 0.42, 0.40)
        v_load = gui.slider_float("vitesse des vérins (m/s)", v_load, 0.05, 5.)
        gui.text(f"modèle : {['', 'élastique linéaire', 'béton armé non linéaire'][model]}   (touches 1/2)")
        gui.text(f"t = {State.t * 1000:.1f} ms   course = {State.u * 1000:.1f} mm")
        gui.text(f"charge P = {P:.1f} kN   (max {P_max:.1f} kN)")
        gui.text(f"flèche à mi-portée = {delta:.2f} mm")
        gui.text(f"béton fissuré = {int(dc[0])}   écrasé = {int(dc[1])}   (sur {n_solid})")
        gui.text("couleur : " + ("endommagement, acier" if color_mode == 0 else "contrainte principale") + "  (ESPACE)")
        gui.text("R : reset   S : sauver la courbe   ESC : quitter")
        gui.end()
        window.show()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--headless":
        umax = float(sys.argv[2]) if len(sys.argv) >= 3 else 25.0
        res = run_headless(umax)
        step = max(1, len(res) // 25)
        print(" t(ms)  course(mm)  fleche(mm)  P(kN)  fissurees  ecrasees")
        for r in res[::step]:
            print(f"{r[0]*1000:7.1f} {r[1]:9.2f} {r[2]:10.2f} {r[3]:8.1f} {int(r[4]):9d} {int(r[5]):9d}")
    else:
        main()

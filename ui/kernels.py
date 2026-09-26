"""Kernels Taichi du solveur : fluide APIC avec réservoir de particules, grille, émission, rendu.

Tous les arguments sont explicites (aucune globale). Les kernels solides (MLS-MPM, endommagement)
viennent de Code_tuto/mpm_solid.py, inchangé.

Grille `cells` (i32, n × n, indexée [i, j] = (x, y)) : bits  FLUID0 = 1 (eau initiale), SOLID0 = 2,
OBSTACLE = 4, INLET = 8, OUTLET = 16.  `bc_v` (vec2) : vitesse imposée sur les cellules d'entrée.

Réservoir fluide : capacité fixe, drapeau `alive`, particules mortes garées en (-1, -1) et ignorées
partout ; pile de slots libres (`free_stack`, `free_top[0]`) sur le GPU : une sortie y pousse les
particules détruites, une entrée y prend les slots des particules émises. Rien ne redescend au CPU.
"""
import taichi as ti

FLUID0, SOLID0, OBSTACLE, INLET, OUTLET = 1, 2, 4, 8, 16


@ti.func
def cell_of(xp, inv_dx: float, n: int):
    c = (xp * inv_dx).cast(int)
    return ti.Vector([ti.math.clamp(c.x, 0, n - 1), ti.math.clamp(c.y, 0, n - 1)])


@ti.func
def weights(fx):
    return [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]


# ---------------------------------------------------------------- initialisation
@ti.kernel
def init_pool(alive: ti.template(), x: ti.template(), C: ti.template(), J: ti.template(), n_init: int,
              free_stack: ti.template(), free_top: ti.template()):
    """Les n_init premières particules sont vivantes ; les autres sont garées et empilées comme libres."""
    for p in alive:
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        J[p] = 1.0
        if p < n_init:
            alive[p] = 1
        else:
            alive[p] = 0
            x[p] = [-1.0, -1.0]
            free_stack[p - n_init] = p
    free_top[0] = alive.shape[0] - n_init


@ti.kernel
def init_solid_state(C: ti.template(), F: ti.template(), D: ti.template(), broken: ti.template()):
    for p in D:
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        F[p] = ti.Matrix.identity(ti.f32, 2)
        D[p] = 0.0
        broken[p] = 0


@ti.kernel
def count_alive(alive: ti.template()) -> ti.i32:
    n = 0
    for p in alive:
        n += alive[p]
    return n


# ---------------------------------------------------------------- 1. particules -> grille (fluide)
@ti.kernel
def P2G_fluid(grid_m: ti.template(), grid_v: ti.template(),
              x: ti.template(), v: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
              inv_dx: float, dt: float, dx: float, E: float, p_mass: float, p_vol: float):
    """Remet la grille à zéro puis dépose masse, quantité de mouvement et pression (APIC/APIC.py + alive)."""
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0
    for p in x:
        if alive[p] == 1:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base
            w = weights(fx)
            stress = -dt * E * p_vol * (J[p] - 1) * 4.0 * inv_dx * inv_dx
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                d_pos = (offset - fx) * dx
                weight = w[i].x * w[j].y
                grid_v[base + offset] += weight * (p_mass * (v[p] + C[p] @ d_pos) + stress * d_pos)
                grid_m[base + offset] += weight * p_mass


# ---------------------------------------------------------------- 2. grille
@ti.kernel
def grid_update(grid_m: ti.template(), grid_v: ti.template(), cells: ti.template(), bc_v: ti.template(),
                dt: float, g: float, bound: int, n: int):
    """Quantité de mouvement -> vitesse, gravité, parois glissantes, obstacles, vitesse d'entrée imposée."""
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * g
            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0.0
            if i > n - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0.0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0.0
            if j > n - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0.0
            c = cells[i, j]
            if c & OBSTACLE:
                grid_v[i, j] = [0.0, 0.0]
            if c & INLET:
                grid_v[i, j] = bc_v[i, j]


# ---------------------------------------------------------------- 3. grille -> particules (fluide)
@ti.kernel
def G2P_fluid(grid_v: ti.template(),
              x: ti.template(), v: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
              cells: ti.template(), bc_v: ti.template(), inv_dx: float, dt: float, dx: float, n: int):
    """v, C, J depuis la grille ; dans une cellule d'entrée la particule garde la vitesse imposée."""
    for p in x:
        if alive[p] == 1:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base
            w = weights(fx)
            v_new = ti.Vector.zero(ti.f32, 2)
            C_new = ti.Matrix.zero(ti.f32, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                d_pos = (offset - fx) * dx
                weight = w[i].x * w[j].y
                v_new += weight * grid_v[base + offset]
                C_new += weight * grid_v[base + offset].outer_product(d_pos) * (4 * inv_dx * inv_dx)
            c = cell_of(x[p], inv_dx, n)
            if cells[c.x, c.y] & INLET:
                v_new = bc_v[c.x, c.y]
                C_new = ti.Matrix.zero(ti.f32, 2, 2)
            v[p] = v_new
            C[p] = C_new
            J[p] *= 1.0 + dt * C_new.trace()


# ---------------------------------------------------------------- 4. advection, sorties, comptage
@ti.kernel
def advect_fluid(x: ti.template(), v: ti.template(), alive: ti.template(),
                 cells: ti.template(), cell_count: ti.template(), inv_dx: float, dt: float, bound: int, dx: float,
                 n: int, free_stack: ti.template(), free_top: ti.template()):
    """Advection bornée à la bande [bound dx, 1 - bound dx] ; destruction dans les cellules de sortie ;
    comptage des particules par cellule d'entrée (pour l'émission)."""
    for i, j in cell_count:
        cell_count[i, j] = 0
    for p in x:
        if alive[p] == 1:
            x[p] += dt * v[p]
            x[p] = ti.math.clamp(x[p], bound * dx, 1.0 - bound * dx)
            c = cell_of(x[p], inv_dx, n)
            flags = cells[c.x, c.y]
            if flags & OUTLET:
                alive[p] = 0
                x[p] = [-1.0, -1.0]
                idx = ti.atomic_add(free_top[0], 1)
                free_stack[idx] = p
            elif flags & INLET:
                cell_count[c.x, c.y] += 1


# ---------------------------------------------------------------- 5. émission (par cellule d'entrée)
@ti.kernel
def emit(x: ti.template(), v: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
         cells: ti.template(), bc_v: ti.template(), cell_count: ti.template(), target: int,
         free_stack: ti.template(), free_top: ti.template(), dx: float, bound: int):
    """Complète chaque cellule d'entrée jusqu'à `target` particules (un thread par cellule)."""
    for i, j in cells:
        if cells[i, j] & INLET:
            deficit = ti.max(target - cell_count[i, j], 0)
            for _ in range(deficit):
                idx = ti.atomic_sub(free_top[0], 1) - 1
                if idx >= 0:
                    p = free_stack[idx]
                    xp = (ti.Vector([i, j]) + ti.Vector([ti.random(), ti.random()])) * dx
                    x[p] = ti.math.clamp(xp, bound * dx, 1.0 - bound * dx)
                    v[p] = bc_v[i, j]
                    C[p] = ti.Matrix.zero(ti.f32, 2, 2)
                    J[p] = 1.0
                    alive[p] = 1
                else:
                    ti.atomic_add(free_top[0], 1)     # plus de slot libre : on rend ce qu'on a pris


# ---------------------------------------------------------------- 6. quantité colorée du fluide
# modes : 0 uniforme, 1 |v|, 2 vx, 3 vy, 4 pression, 5 vorticité (1 : échelle 0..max ; 2-5 : ±max)
FLUID_MODES = ["uniforme", "vitesse |v|", "vx", "vy", "pression", "vorticité"]


@ti.func
def cmap_jet(t):
    """0..1 -> bleu, cyan, jaune, rouge."""
    r = ti.math.clamp(1.5 - ti.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = ti.math.clamp(1.5 - ti.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = ti.math.clamp(1.5 - ti.abs(4.0 * t - 1.0), 0.0, 1.0)
    return ti.Vector([r, g, b])


@ti.func
def cmap_signed(t):
    """0..1 (0.5 = zéro) -> bleu, blanc, rouge."""
    blue = ti.Vector([0.15, 0.35, 1.0])
    white = ti.Vector([0.95, 0.95, 0.95])
    red = ti.Vector([1.0, 0.2, 0.1])
    c = blue * (1.0 - 2.0 * t) + white * (2.0 * t)
    if t > 0.5:
        c = white * (2.0 - 2.0 * t) + red * (2.0 * t - 1.0)
    return c


@ti.kernel
def fluid_scalar_wc(mode: int, x: ti.template(), v: ti.template(), J: ti.template(), alive: ti.template(),
                    grid_v: ti.template(), sc: ti.template(), E: float, inv_dx: float, n: int):
    """Quantité par particule, mode faiblement compressible (pression = E (1 - J), vorticité sur la grille)."""
    for p in x:
        if alive[p] == 1:
            s = 0.0
            if mode == 1:
                s = v[p].norm()
            elif mode == 2:
                s = v[p].x
            elif mode == 3:
                s = v[p].y
            elif mode == 4:
                s = E * (1.0 - J[p])
            elif mode == 5:
                c = cell_of(x[p], inv_dx, n)
                i, j = ti.math.clamp(c.x, 1, n - 2), ti.math.clamp(c.y, 1, n - 2)
                s = (grid_v[i + 1, j].y - grid_v[i - 1, j].y - grid_v[i, j + 1].x + grid_v[i, j - 1].x) * 0.5 * inv_dx
            sc[p] = s


@ti.kernel
def fluid_scalar_inc(mode: int, x: ti.template(), v: ti.template(), alive: ti.template(),
                     u: ti.template(), vv: ti.template(), q: ti.template(), sc: ti.template(),
                     rho_over_dt: float, inv_dx: float, n: int):
    """Quantité par particule, mode incompressible (pression = q rho / dt, vorticité depuis les faces)."""
    for p in x:
        if alive[p] == 1:
            s = 0.0
            if mode == 1:
                s = v[p].norm()
            elif mode == 2:
                s = v[p].x
            elif mode == 3:
                s = v[p].y
            else:
                c = cell_of(x[p], inv_dx, n)
                i, j = ti.math.clamp(c.x, 1, n - 2), ti.math.clamp(c.y, 1, n - 2)
                if mode == 4:
                    s = q[i, j] * rho_over_dt
                elif mode == 5:
                    dvdx = ((vv[i + 1, j] + vv[i + 1, j + 1]) - (vv[i - 1, j] + vv[i - 1, j + 1])) * 0.25 * inv_dx
                    dudy = ((u[i, j + 1] + u[i + 1, j + 1]) - (u[i, j - 1] + u[i + 1, j - 1])) * 0.25 * inv_dx
                    s = dvdx - dudy
            sc[p] = s


@ti.kernel
def scalar_absmax(sc: ti.template(), alive: ti.template()) -> ti.f32:
    m = 0.0
    for p in sc:
        if alive[p] == 1:
            ti.atomic_max(m, ti.abs(sc[p]))
    return m


# ---------------------------------------------------------------- 7. rendu
@ti.kernel
def render(img: ti.template(), res: int, x0: float, y0: float, scale: float,
           cells: ti.template(), n: int, grid_on: int, tint_on: int,
           x_f: ti.template(), alive: ti.template(), has_fluid: int, fr: float, fg: float, fb: float, r_f: int,
           sc: ti.template(), fluid_mode: int, inv_smax: float,
           x_s: ti.template(), col_s: ti.template(), has_solid: int, r_s: int):
    """Image (res, res) u8 indexée [ligne, colonne], ligne 0 en haut (format QImage RGB888).

    Vue : x = x0 + (col + 0.5) / (res scale), y = y0 + (res - ligne - 0.5) / (res scale).
    Fond, obstacles, entrées, sorties, teintes des cellules initiales, maillage, puis particules.
    """
    k = 1.0 / (res * scale)
    cell_px = res * scale / n
    for row, col in img:
        xd = x0 + (col + 0.5) * k
        yd = y0 + (res - row - 0.5) * k
        c = ti.Vector([0.02, 0.02, 0.08])
        if 0.0 <= xd < 1.0 and 0.0 <= yd < 1.0:
            ci = ti.math.clamp(int(xd * n), 0, n - 1)
            cj = ti.math.clamp(int(yd * n), 0, n - 1)
            f = cells[ci, cj]
            if f & OBSTACLE:
                c = ti.Vector([0.35, 0.35, 0.35])
            elif tint_on == 1 and (f & FLUID0):
                c = ti.Vector([0.06, 0.10, 0.22])
            elif tint_on == 1 and (f & SOLID0):
                c = ti.Vector([0.22, 0.20, 0.08])
            if f & INLET:
                c = c * 0.5 + ti.Vector([0.15, 0.45, 0.20])
            if f & OUTLET:
                c = c * 0.5 + ti.Vector([0.45, 0.12, 0.12])
            if grid_on == 1 and cell_px >= 6.0:
                gx = (xd * n - ci) * cell_px
                gy = (yd * n - cj) * cell_px
                if gx < 1.0 or gy < 1.0:
                    c = c * 0.6 + ti.Vector([0.25, 0.25, 0.30])
        else:
            c = ti.Vector([0.10, 0.10, 0.12])
        img[row, col] = ti.cast(c * 255, ti.u8)

    for p in x_f:
        if has_fluid == 1 and alive[p] == 1:
            col = int((x_f[p].x - x0) / k)
            row = res - 1 - int((x_f[p].y - y0) / k)
            cp = ti.Vector([fr, fg, fb])
            if fluid_mode == 1:
                cp = cmap_jet(ti.math.clamp(sc[p] * inv_smax, 0.0, 1.0))
            elif fluid_mode > 1:
                cp = cmap_signed(ti.math.clamp(0.5 + 0.5 * sc[p] * inv_smax, 0.0, 1.0))
            for a, b in ti.ndrange((-r_f, r_f + 1), (-r_f, r_f + 1)):
                rr, cc = row + a, col + b
                if 0 <= rr < res and 0 <= cc < res:
                    img[rr, cc] = ti.cast(cp * 255, ti.u8)

    for p in x_s:
        if has_solid == 1:
            col = int((x_s[p].x - x0) / k)
            row = res - 1 - int((x_s[p].y - y0) / k)
            cs = col_s[p]
            for a, b in ti.ndrange((-r_s, r_s + 1), (-r_s, r_s + 1)):
                rr, cc = row + a, col + b
                if 0 <= rr < res and 0 <= cc < res:
                    img[rr, cc] = ti.cast(cs * 255, ti.u8)

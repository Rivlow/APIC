"""Kernels propres à l'UI : sans aucune globale, tous les champs et scalaires sont des arguments.

Les kernels de physique viennent de APIC/APIC.py et Code_tuto/mpm_solid.py, inchangés.
"""
import taichi as ti


@ti.kernel
def init_fluid_state(C: ti.template(), J: ti.template()):
    for p in J:
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        J[p] = 1.0


@ti.kernel
def init_solid_state(C: ti.template(), F: ti.template(), D: ti.template(), broken: ti.template()):
    for p in D:
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        F[p] = ti.Matrix.identity(ti.f32, 2)
        D[p] = 0.0
        broken[p] = 0


@ti.kernel
def advect_fluid(x: ti.template(), v: ti.template(), dt: float, bound: int, dx: float):
    """Copie paramétrée de main_fsi.advect_fluid."""
    for p in x:
        x[p] += dt * v[p]
        x[p] = ti.math.clamp(x[p], bound * dx, 1.0 - bound * dx)


@ti.kernel
def clear_grid(grid_m: ti.template(), grid_v: ti.template()):
    """Utilisé quand il n'y a pas de fluide (c'est P2G du fluide qui remet la grille à zéro sinon)."""
    for i, j in grid_m:
        grid_m[i, j] = 0.0
        grid_v[i, j] = [0.0, 0.0]


@ti.kernel
def render_background(bg: ti.template(), mask: ti.template()):
    """Fond du viewport : bleu nuit, gris sur les obstacles (copie de main_fsi.render_background)."""
    for i, j in bg:
        bg[i, j] = ti.Vector([0.35, 0.35, 0.35]) if mask[i, j] == 1 else ti.Vector([0.02, 0.02, 0.08])


# ---------------------------------------------------------------- rendu zéro copie (arch Vulkan)
@ti.kernel
def render_scene(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0),
                 res: int, bg: ti.template(), n_grid: int,
                 x_f: ti.template(), has_fluid: int, fr: float, fg: float, fb: float, r_f: int,
                 x_s: ti.template(), col_s: ti.template(), has_solid: int, r_s: int,
                 seg_a: ti.template(), seg_b: ti.template(), seg_c: ti.template(), n_seg: int):
    """Dessine toute la scène dans une texture RGBA8 que GGUI présente directement (aucun passage CPU).

    Les quatre boucles de premier niveau s'exécutent dans l'ordre : fond, fluide, solide, overlays.
    Les particules sont des carrés de (2r+1)² pixels ; les overlays des segments épais de 2 px.
    """
    for i, j in ti.ndrange(res, res):
        gi = ti.min(i * n_grid // res, n_grid - 1)
        gj = ti.min(j * n_grid // res, n_grid - 1)
        c = bg[gi, gj]
        tex.store(ti.Vector([i, j]), ti.Vector([c[0], c[1], c[2], 1.0]))

    # (les boucles struct-for doivent être au premier niveau du kernel : le test has_* est dedans)
    for p in x_f:
        if has_fluid == 1:
            ci = int(x_f[p].x * res)
            cj = int(x_f[p].y * res)
            for a, b in ti.ndrange((-r_f, r_f + 1), (-r_f, r_f + 1)):
                ii, jj = ci + a, cj + b
                if 0 <= ii < res and 0 <= jj < res:
                    tex.store(ti.Vector([ii, jj]), ti.Vector([fr, fg, fb, 1.0]))

    for p in x_s:
        if has_solid == 1:
            ci = int(x_s[p].x * res)
            cj = int(x_s[p].y * res)
            c = col_s[p]
            for a, b in ti.ndrange((-r_s, r_s + 1), (-r_s, r_s + 1)):
                ii, jj = ci + a, cj + b
                if 0 <= ii < res and 0 <= jj < res:
                    tex.store(ti.Vector([ii, jj]), ti.Vector([c[0], c[1], c[2], 1.0]))

    for s in range(n_seg):
        a = seg_a[s] * res
        b = seg_b[s] * res
        c = seg_c[s]
        n = int(ti.max(ti.abs(b.x - a.x), ti.abs(b.y - a.y))) + 1
        for k in range(n):
            q = a + (b - a) * (k / n)
            for da, db in ti.static(ti.ndrange(2, 2)):
                ii, jj = int(q.x) + da, int(q.y) + db
                if 0 <= ii < res and 0 <= jj < res:
                    tex.store(ti.Vector([ii, jj]), ti.Vector([c[0], c[1], c[2], 1.0]))


@ti.kernel
def clear_texture(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0),
                  res: int, r: float, g: float, b: float,
                  seg_a: ti.template(), seg_b: ti.template(), seg_c: ti.template(), n_seg: int):
    """Fond uni + overlays, quand la session n'est pas construite."""
    for i, j in ti.ndrange(res, res):
        tex.store(ti.Vector([i, j]), ti.Vector([r, g, b, 1.0]))
    for s in range(n_seg):
        a = seg_a[s] * res
        b2 = seg_b[s] * res
        c = seg_c[s]
        n = int(ti.max(ti.abs(b2.x - a.x), ti.abs(b2.y - a.y))) + 1
        for k in range(n):
            q = a + (b2 - a) * (k / n)
            for da, db in ti.static(ti.ndrange(2, 2)):
                ii, jj = int(q.x) + da, int(q.y) + db
                if 0 <= ii < res and 0 <= jj < res:
                    tex.store(ti.Vector([ii, jj]), ti.Vector([c[0], c[1], c[2], 1.0]))

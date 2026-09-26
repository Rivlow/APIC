# MpM/mpm_solid.py -- Solide MLS-MPM 2D : élasticité corotationnelle + endommagement + rupture
#
# Même grille et mêmes poids B-spline quadratiques que le fluide APIC (APIC/APIC.py).
# Chaque particule solide porte :
#   x, v, C   : position, vitesse, matrice affine APIC (comme le fluide)
#   F         : gradient de déformation (2x2), F = I au repos
#   D         : variable d'endommagement dans [0, 1]  (0 = intact, 1 = rompu)
#   broken    : 1 si la particule a perdu toute cohésion (rupture)
#
# Cycle par pas de temps (identique au fluide) :
#   1. P2G_solid    : dépôt masse + quantité de mouvement + force interne  -sum_p V_p tau_p grad(w_ip)
#   2. grid_update  : quantité de mouvement -> vitesse, gravité, parois, obstacles (= grid_step du fluide)
#   3. G2P_solid    : relecture v, C ; mise à jour de F (écrêtage des particules rompues) ; advection
#   4. scatter_eps + update_damage : endommagement NON LOCAL (allongement lissé par la grille) et à vitesse
#      limitée (dD <= dt / tau_D) ; c'est ce qui évite que la poutre éclate comme du verre à la rupture.

import taichi as ti


# ---------------------------------------------------------------- loi de comportement
@ti.func
def kirchhoff_stress(F, mu: float, la: float):
    """Contrainte de Kirchhoff tau = P F^T du modèle corotationnel (Stomakhin et al. 2012).

    P = 2 mu (F - R) + la J (J - 1) F^{-T}   ->   tau = 2 mu (F - R) F^T + la J (J - 1) I
    R est la rotation de la décomposition polaire F = R S, obtenue via la SVD F = U S V^T : R = U V^T.
    """
    U, sig, V = ti.svd(F)
    R = U @ V.transpose()
    J = F.determinant()
    I = ti.Matrix.identity(ti.f32, 2)
    return 2.0 * mu * (F - R) @ F.transpose() + la * J * (J - 1.0) * I


# ---------------------------------------------------------------- 1. Particules -> grille
@ti.kernel
def P2G_solid(grid_m: ti.template(), grid_v: ti.template(),
              x: ti.template(), v: ti.template(), C: ti.template(), F: ti.template(),
              D: ti.template(), broken: ti.template(),
              inv_dx: float, dt: float, dx: float,
              mu: float, la: float, p_mass: float, p_vol: float, k_res: float):
    """Ne remet PAS la grille à zéro : on peut l'appeler après le P2G du fluide (grille partagée)."""

    for p in x:

        base = (x[p] * inv_dx - 0.5).cast(int)   # noeud en bas à gauche du stencil 3x3
        fx = x[p] * inv_dx - base
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]

        # Raideur effective : (1 - D) E pour une particule endommagée, avec un plancher k_res
        # (une particule totalement sans raideur se comporterait comme de la poussière).
        # Une particule rompue garde sa raideur (elle résiste encore en compression)
        # mais son F est écrêté dans G2P pour ne jamais porter de traction.
        k = ti.max(1.0 - D[p], k_res)
        if broken[p] == 1:
            k = 1.0
        tau = kirchhoff_stress(F[p], k * mu, k * la)

        # Force interne MLS-MPM : dt * f_i = -dt * V_p * (4/dx^2) * w_ip * tau (x_i - x_p)
        # -> on range tout dans une matrice "affine" appliquée à dpos, comme pour le terme C_p.
        affine = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * tau + p_mass * C[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            d_pos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ d_pos)
            grid_m[base + offset] += weight * p_mass


# ---------------------------------------------------------------- 2. Mise à jour de la grille
@ti.kernel
def grid_update(grid_m: ti.template(), grid_v: ti.template(), mask: ti.template(),
                dt: float, g: float, damp: float, bound: int, n_grid: int):
    """Même rôle que grid_step() du fluide : vitesse, gravité, parois glissantes, obstacles (mask == 1).
    damp (1/s) : amortissement léger de la vitesse de grille, dissipe l'énergie cinétique."""

    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j] *= 1.0 - damp * dt
            grid_v[i, j].y -= dt * g
            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0.0
            if i > n_grid - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0.0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0.0
            if j > n_grid - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0.0
            if mask[i, j] == 1:
                grid_v[i, j] = [0.0, 0.0]


# ---------------------------------------------------------------- 3. Grille -> particules
@ti.kernel
def G2P_solid(grid_v: ti.template(),
              x: ti.template(), v: ti.template(), C: ti.template(), F: ti.template(),
              broken: ti.template(),
              inv_dx: float, dt: float, dx: float, bound: int):
    """v, C depuis la grille ; F <- (I + dt C) F ; écrêtage des particules rompues ; advection."""

    I = ti.Matrix.identity(ti.f32, 2)

    for p in x:

        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]

        v_new = ti.Vector.zero(ti.f32, 2)
        C_new = ti.Matrix.zero(ti.f32, 2, 2)

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            d_pos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            v_new += weight * g_v
            C_new += weight * g_v.outer_product(d_pos) * (4.0 * inv_dx * inv_dx)

        v[p] = v_new
        C[p] = C_new

        # --- gradient de déformation : dF/dt = (grad v) F, avec grad v ~ C
        F_new = (I + dt * C_new) @ F[p]

        if broken[p] == 1:
            # Rupture : matériau sans cohésion. On écrête les étirements principaux à 1
            # (aucune traction possible) et on borne la compression pour éviter l'inversion.
            U, sig, V = ti.svd(F_new)
            for d in ti.static(range(2)):
                sig[d, d] = ti.math.clamp(sig[d, d], 0.1, 1.0)
            F_new = U @ sig @ V.transpose()

        F[p] = F_new

        # --- advection
        x[p] += dt * v[p]
        x[p] = ti.math.clamp(x[p], bound * dx, 1.0 - bound * dx)


# ---------------------------------------------------------------- 4. Endommagement non local
@ti.kernel
def clear_eps(grid_e: ti.template(), grid_w: ti.template()):
    for i, j in grid_e:
        grid_e[i, j] = 0.0
        grid_w[i, j] = 0.0


@ti.kernel
def scatter_eps(grid_e: ti.template(), grid_w: ti.template(),
                x: ti.template(), F: ti.template(), broken: ti.template(), inv_dx: float):
    """Étale l'allongement principal des particules saines sur la grille (somme pondérée)."""

    for p in x:
        if broken[p] == 0:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base
            w = [0.5 * (1.5 - fx)**2,
                 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]

            U, sig, V = ti.svd(F[p])
            eps = ti.max(sig[0, 0], sig[1, 1]) - 1.0

            for i, j in ti.static(ti.ndrange(3, 3)):
                weight = w[i].x * w[j].y
                grid_e[base + ti.Vector([i, j])] += weight * eps
                grid_w[base + ti.Vector([i, j])] += weight


@ti.kernel
def update_damage(grid_e: ti.template(), grid_w: ti.template(),
                  x: ti.template(), F: ti.template(), D: ti.template(), broken: ti.template(),
                  inv_dx: float, dt: float,
                  eps0: float, epsf: float, tau_D: float, use_rupture: int):
    """Endommagement à partir de l'allongement LISSÉ (non local), à vitesse limitée (dD <= dt / tau_D).

      D_new = clamp((eps_nl - eps0) / (epsf - eps0), 0, 1)     D <- max(D, min(D_new, D + dt / tau_D))
    """

    for p in x:
        if broken[p] == 0:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base
            w = [0.5 * (1.5 - fx)**2,
                 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]

            eps = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):
                node = base + ti.Vector([i, j])
                if grid_w[node] > 0:
                    eps += w[i].x * w[j].y * grid_e[node] / grid_w[node]

            D_new = ti.math.clamp((eps - eps0) / (epsf - eps0), 0.0, 1.0)
            D[p] = ti.max(D[p], ti.min(D_new, D[p] + dt / tau_D))

            if use_rupture == 1 and D[p] >= 1.0:
                broken[p] = 1
                # on retire la traction tout de suite (au lieu de F = I) : la compression est conservée
                U, sig, V = ti.svd(F[p])
                for d in ti.static(range(2)):
                    sig[d, d] = ti.math.clamp(sig[d, d], 0.1, 1.0)
                F[p] = U @ sig @ V.transpose()


# ---------------------------------------------------------------- utilitaires
@ti.kernel
def init_beam(x: ti.template(), v: ti.template(), C: ti.template(), F: ti.template(),
              D: ti.template(), broken: ti.template(),
              x0: float, y0: float, n_px: int, spacing: float):
    """Place les particules sur un réseau régulier (2 x 2 par cellule) dans le rectangle
    [x0, x0 + n_px*spacing] x [y0, ...], au repos, intactes."""

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


@ti.kernel
def solid_stats(D: ti.template(), broken: ti.template()) -> ti.types.vector(2, ti.f32):
    """Retourne (nombre de particules rompues, D max)."""
    n_broken = 0
    D_max = 0.0
    for p in D:
        n_broken += broken[p]
        ti.atomic_max(D_max, D[p])
    return ti.Vector([float(n_broken), D_max])

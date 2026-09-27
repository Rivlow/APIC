# Solver/Solid/solid.py -- Solide élastique corotationnel avec endommagement D et rupture (broken).
#
# Endommagement NON LOCAL (allongement principal lissé par la grille) et à vitesse limitée (dD <= dt / tau_D),
# appliqué après G2P et l'advection par damage_step() ; c'est ce qui évite que la poutre éclate comme du verre.

import taichi as ti

from Solver.APIC import stencil


@ti.func
def kirchhoff_stress(F, mu: float, la: float):
    """Corotated Kirchhoff stress tau = 2 mu (F - R) F^T + la J (J - 1) I (Stomakhin et al. 2012).

    **Inputs**

    - `F` : mat2 f32 deformation gradient
    - `mu`, `la` : float Lamé parameters

    **Outputs**

    - mat2 f32
    """
    U, sig, V = ti.svd(F)
    R = U @ V.transpose()
    J = F.determinant()
    I = ti.Matrix.identity(ti.f32, 2)
    return 2.0 * mu * (F - R) @ F.transpose() + la * J * (J - 1.0) * I


@ti.data_oriented
class Solid:

    def __init__(self, n, mu, la, k_res, p_mass, p_vol):
        """Allocate particle fields (x, v, C, F, D, broken) and parameters.

        **Inputs**

        - `n` : int particle count
        - `mu`, `la`, `k_res` : float Lamé parameters, residual stiffness (tunable 0-D fields)
        - `p_mass`, `p_vol` : float

        **Note** : p_mass, p_vol are Python attributes baked at compile time.
        """

        self.x = ti.Vector.field(2, ti.f32, n)
        self.v = ti.Vector.field(2, ti.f32, n)
        self.C = ti.Matrix.field(2, 2, ti.f32, n)       # matrice affine APIC

        self.F = ti.Matrix.field(2, 2, ti.f32, n)       # gradient de déformation
        self.D = ti.field(ti.f32, n)                    # endommagement dans [0, 1]
        self.broken = ti.field(ti.i32, n)               # 1 si rompu

        # Constantes de compilation (attributs Python) : figées dans le kernel compilé.
        self.p_mass = p_mass
        self.p_vol = p_vol

        # Paramètres réglables : champs 0-D (un attribut Python serait figé à la compilation).
        self.mu = ti.field(ti.f32, shape=())
        self.la = ti.field(ti.f32, shape=())
        self.k_res = ti.field(ti.f32, shape=())         # raideur résiduelle minimale
        self.mu[None] = mu
        self.la[None] = la
        self.k_res[None] = k_res

    @ti.func
    def stress(self, p):
        """Damaged corotated stress, stiffness factor max(1 - D, k_res) (1 if broken).

        **Inputs**

        - `p` : int particle index

        **Outputs**

        - mat2 f32
        """
        # Raideur effective (1 - D), avec un plancher k_res. Une particule rompue garde sa raideur
        # (elle résiste en compression) mais son F est écrêté dans update_deformation (pas de traction).
        k = ti.max(1.0 - self.D[p], self.k_res[None])
        if self.broken[p] == 1:
            k = 1.0
        return kirchhoff_stress(self.F[p], k * self.mu[None], k * self.la[None])

    @ti.func
    def update_deformation(self, p, C_new, dt):
        """Update F <- (I + dt C) F ; broken particles: principal stretches clamped to [0.1, 1].

        **Inputs**

        - `p` : int particle index
        - `C_new` : mat2 f32 velocity gradient
        - `dt` : float

        **Outputs**

        - F[p] in place
        """
        F_new = (ti.Matrix.identity(ti.f32, 2) + dt * C_new) @ self.F[p]

        if self.broken[p] == 1:
            # Rupture : on écrête les étirements principaux à 1 (aucune traction)
            # et on borne la compression pour éviter l'inversion.
            U, sig, V = ti.svd(F_new)
            for d in ti.static(range(2)):
                sig[d, d] = ti.math.clamp(sig[d, d], 0.1, 1.0)
            F_new = U @ sig @ V.transpose()

        self.F[p] = F_new

    # ------------------------------------------------------------ état
    @ti.kernel
    def reset_state(self):
        """Reset particles to rest and undamaged.

        **Outputs**

        - v, C, F, D, broken reset in place

        **Note** : positions x are set by the case, not here.
        """
        for p in self.x:
            self.v[p] = [0.0, 0.0]
            self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            self.F[p] = ti.Matrix.identity(ti.f32, 2)
            self.D[p] = 0.0
            self.broken[p] = 0

    # ------------------------------------------------------------ endommagement non local
    def damage_step(self, grid_e, grid_w, inv_dx: float, dt: float,
                    eps0: float, epsf: float, tau_D: float, use_rupture: int):
        """Non-local damage update: clear scratch grids, scatter strain, update D.

        **Inputs**

        - `grid_e`, `grid_w` : f32 field (nx, ny), scratch (weighted stretch, weight sum)
        - `inv_dx`, `dt`, `eps0`, `epsf`, `tau_D` : float
        - `use_rupture` : int (1 = allow rupture)

        **Outputs**

        - D, broken, F updated in place ; grid_e, grid_w overwritten
        """
        grid_e.fill(0.0)
        grid_w.fill(0.0)
        self.scatter_eps(grid_e, grid_w, inv_dx)
        self.update_damage(grid_e, grid_w, inv_dx, dt, eps0, epsf, tau_D, use_rupture)

    @ti.kernel
    def scatter_eps(self, grid_e: ti.template(), grid_w: ti.template(), inv_dx: float):
        """Scatter max principal stretch of intact particles on the grid.

        **Inputs**

        - `grid_e`, `grid_w` : f32 field (nx, ny)
        - `inv_dx` : float

        **Outputs**

        - grid_e (weighted stretch), grid_w (weight sum) accumulated in place

        **Note** : does not clear the grid.
        """
        for p in self.x:
            if self.broken[p] == 0:
                base, fx, w = stencil(self.x[p], inv_dx)
                U, sig, V = ti.svd(self.F[p])
                eps = ti.max(sig[0, 0], sig[1, 1]) - 1.0
                for i, j in ti.static(ti.ndrange(3, 3)):
                    weight = w[i, 0] * w[j, 1]
                    grid_e[base + ti.Vector([i, j])] += weight * eps
                    grid_w[base + ti.Vector([i, j])] += weight

    @ti.kernel
    def update_damage(self, grid_e: ti.template(), grid_w: ti.template(), inv_dx: float, dt: float,
                      eps0: float, epsf: float, tau_D: float, use_rupture: int):
        """Update damage from grid-smoothed stretch: D <- max(D, min(D_new, D + dt / tau_D)).

        **Inputs**

        - `grid_e`, `grid_w` : f32 field (nx, ny)
        - `inv_dx`, `dt`, `eps0`, `epsf`, `tau_D` : float (D_new = clamp((eps - eps0) / (epsf - eps0), 0, 1))
        - `use_rupture` : int (1: D >= 1 breaks the particle)

        **Outputs**

        - D, broken, F in place (newly broken: tension removed from F)
        """
        for p in self.x:
            if self.broken[p] == 0:
                base, fx, w = stencil(self.x[p], inv_dx)
                eps = 0.0
                for i, j in ti.static(ti.ndrange(3, 3)):
                    node = base + ti.Vector([i, j])
                    if grid_w[node] > 0:
                        eps += w[i, 0] * w[j, 1] * grid_e[node] / grid_w[node]

                D_new = ti.math.clamp((eps - eps0) / (epsf - eps0), 0.0, 1.0)
                self.D[p] = ti.max(self.D[p], ti.min(D_new, self.D[p] + dt / tau_D))

                if use_rupture == 1 and self.D[p] >= 1.0:
                    self.broken[p] = 1
                    # on retire la traction tout de suite (au lieu de F = I) : la compression est conservée
                    U, sig, V = ti.svd(self.F[p])
                    for d in ti.static(range(2)):
                        sig[d, d] = ti.math.clamp(sig[d, d], 0.1, 1.0)
                    self.F[p] = U @ sig @ V.transpose()

    # ------------------------------------------------------------ affichage, diagnostics
    @ti.kernel
    def colors(self, col: ti.template(), mode: int, eps_scale: float):
        """Particle display colors.

        **Inputs**

        - `col` : vec3 f32 field (N,)
        - `mode` : int: 0 damage (grey -> yellow, red if broken), 1 max principal stretch (blue -> red)
        - `eps_scale` : float, stretch mapped to full color

        **Outputs**

        - col written in place
        """
        for p in self.D:
            if mode == 0:
                if self.broken[p] == 1:
                    col[p] = ti.Vector([0.90, 0.15, 0.15])
                else:
                    col[p] = ti.Vector([0.75, 0.75, 0.75]) * (1 - self.D[p]) + ti.Vector([1.0, 0.85, 0.1]) * self.D[p]
            else:
                U, sig, V = ti.svd(self.F[p])
                eps = ti.max(sig[0, 0], sig[1, 1]) - 1.0
                t = ti.math.clamp(0.5 + 0.5 * eps / eps_scale, 0.0, 1.0)
                col[p] = ti.Vector([0.2, 0.4, 1.0]) * (1 - t) + ti.Vector([1.0, 0.25, 0.1]) * t

    @ti.kernel
    def stats(self) -> ti.types.vector(2, ti.f32):
        """Damage diagnostics.

        **Outputs**

        - vec2 f32 (broken particle count, max D)
        """
        n_broken = 0
        D_max = 0.0
        for p in self.D:
            n_broken += self.broken[p]
            ti.atomic_max(D_max, self.D[p])
        return ti.Vector([float(n_broken), D_max])

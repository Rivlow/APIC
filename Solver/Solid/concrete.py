# Solver/Solid/concrete.py -- Béton armé (Cremona & Houde, poutres de la Rance) pour MLS-MPM 2D.
#
#   - béton en compression : loi de Desayi/Neville, écrasement vers -eps_cu
#   - béton en traction    : linéaire jusqu'à fcr, puis fcr / (1 + sqrt(500 eps))  (raidissement)
#   - acier                : barre uniaxiale suivant x du matériau, élastique parfaitement plastique,
#                            mélangée au béton par fraction volumique phi
#   - résistances fcr_p, fc_p par particule (tirées par le cas de calcul)
#
# Précision float32 : les déplacements (~mm sur un domaine de mètres) et les déformations (~1e-4) sont
# plus petits que l'ulp de x et de F. On stocke donc X0 + Up (position initiale + déplacement) et Fm = F - I.
# x = X0 + Up n'est qu'un cache, lu par les transferts APIC : il doit être rafraîchi (refresh_x) après
# chaque modification de Up.

import taichi as ti

from Solver.APIC import stencil


@ti.data_oriented
class ReinforcedConcrete:

    def __init__(self, n, p_mass, p_vol,
                 E_c, nu_c, eps_cu, r_min, r_crush, E_s, fy):
        """Allocate particle fields and material constants.

        **Inputs**

        - `n` : int particle count
        - `p_mass`, `p_vol` : float
        - `E_c`, `nu_c` : float concrete Young modulus, Poisson ratio
        - `eps_cu`, `r_min`, `r_crush` : float crushing strain, residual tension / compression stiffness
        - `E_s`, `fy` : float steel modulus, yield stress

        **Note** : scalars are Python attributes baked at compile time; only `nonlinear` (0-D i32) is tunable.
        """

        self.x = ti.Vector.field(2, ti.f32, n)          # cache X0 + Up (transferts APIC)
        self.X0 = ti.Vector.field(2, ti.f32, n)         # position initiale
        self.Up = ti.Vector.field(2, ti.f32, n)         # déplacement
        self.v = ti.Vector.field(2, ti.f32, n)
        self.C = ti.Matrix.field(2, 2, ti.f32, n)       # matrice affine APIC
        self.Fm = ti.Matrix.field(2, 2, ti.f32, n)      # F - I

        self.kap_t = ti.field(ti.f32, n)                # plus grande déformation de traction équivalente
        self.kap_c = ti.field(ti.f32, n)                # plus grande déformation de compression équivalente
        self.phi = ti.field(ti.f32, n)                  # fraction volumique d'acier
        self.eps_p = ti.field(ti.f32, n)                # déformation plastique de l'acier
        self.fcr_p = ti.field(ti.f32, n)                # résistance en traction de la particule
        self.fc_p = ti.field(ti.f32, n)                 # résistance en compression de la particule

        # Constantes de compilation (attributs Python) : figées dans le kernel compilé.
        self.p_mass = p_mass
        self.p_vol = p_vol
        self.E_c = E_c
        self.mu_c = E_c / (2 * (1 + nu_c))
        self.la_c = E_c * nu_c / ((1 + nu_c) * (1 - 2 * nu_c))
        self.eps_cu = eps_cu
        self.r_min = r_min                              # raideur résiduelle en traction (béton fissuré)
        self.r_crush = r_crush                          # raideur résiduelle en compression (béton écrasé)
        self.E_s = E_s
        self.fy = fy

        # 0 = élastique linéaire, 1 = béton armé non linéaire (modifiable à chaud)
        self.nonlinear = ti.field(ti.i32, shape=())

    # ------------------------------------------------------------ lois de comportement
    @ti.func
    def r_tension(self, kap, fcr_i):
        """Secant tension stiffness ratio: 1 before cracking, then fcr / (1 + sqrt(500 kap)) / (E_c kap).

        **Inputs**

        - `kap` : float max equivalent tension strain
        - `fcr_i` : float tensile strength

        **Outputs**

        - float in [r_min, 1]
        """
        r = 1.0
        if kap > fcr_i / self.E_c:
            r = ti.min(1.0, fcr_i / (1.0 + ti.sqrt(500.0 * kap)) / (self.E_c * kap))
        return ti.max(r, self.r_min)

    @ti.func
    def r_compression(self, kap, fc_i):
        """Secant compression stiffness ratio: Desayi law, then progressive crushing past eps_cu.

        **Inputs**

        - `kap` : float max equivalent compression strain
        - `fc_i` : float compressive strength

        **Outputs**

        - float >= r_min
        """
        x = kap / (1.8 * fc_i / self.E_c)
        r = ti.min(1.0, (2.0 / 1.8) / (1.0 + x * x))
        if kap > self.eps_cu:
            t = ti.min((kap - self.eps_cu) / (0.5 * self.eps_cu), 1.0)
            r = r * (1.0 - t) + self.r_crush * t
        return ti.max(r, self.r_min)

    @ti.func
    def principal_stresses(self, p):
        """Elastic (Hencky) principal stresses of the concrete.

        **Inputs**

        - `p` : int particle index

        **Outputs**

        - `s1`, `s2` : float
        - `U` : mat2 f32 principal frame (SVD)
        """
        U, sg, V = ti.svd(ti.Matrix.identity(ti.f32, 2) + self.Fm[p])
        e1 = ti.log(ti.max(sg[0, 0], 0.05))
        e2 = ti.log(ti.max(sg[1, 1], 0.05))
        s1 = 2.0 * self.mu_c * e1 + self.la_c * (e1 + e2)
        s2 = 2.0 * self.mu_c * e2 + self.la_c * (e1 + e2)
        return s1, s2, U

    @ti.func
    def stress(self, p):
        """Kirchhoff stress: concrete (Hencky, secant stiffness per principal direction) mixed with a steel bar
        along material x (elastic-perfectly plastic, volume fraction phi).

        **Inputs**

        - `p` : int particle index

        **Outputs**

        - mat2 f32
        """
        s1, s2, U = self.principal_stresses(p)
        if self.nonlinear[None] == 1:
            rt = self.r_tension(self.kap_t[p], self.fcr_p[p])
            rc = self.r_compression(self.kap_c[p], self.fc_p[p])
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
        ph = self.phi[p]
        if ph > 0.0:
            F = ti.Matrix.identity(ti.f32, 2) + self.Fm[p]
            n = ti.Vector([F[0, 0], F[1, 0]])
            lam = n.norm()
            n = n / lam
            ss = self.E_s * ti.log(lam)
            if self.nonlinear[None] == 1:
                ss = ti.math.clamp(self.E_s * (ti.log(lam) - self.eps_p[p]), -self.fy, self.fy)
            tau = (1.0 - ph) * tau + ph * ss * n.outer_product(n)
        return tau

    @ti.func
    def update_deformation(self, p, C_new, dt):
        """Update Fm = F - I: Fm <- Fm + dt C (I + Fm).

        **Inputs**

        - `p` : int particle index
        - `C_new` : mat2 f32 velocity gradient
        - `dt` : float

        **Outputs**

        - Fm[p] in place
        """
        # F = I + Fm, dF/dt = (grad v) F   ->   Fm <- Fm + dt C (I + Fm)
        self.Fm[p] += dt * C_new @ (ti.Matrix.identity(ti.f32, 2) + self.Fm[p])

    # ------------------------------------------------------------ position
    @ti.kernel
    def refresh_x(self):
        """Refresh the position cache x = X0 + Up.

        **Outputs**

        - x written in place

        **Note** : must be called after every change of Up.
        """
        for p in self.x:
            self.x[p] = self.X0[p] + self.Up[p]

    # ------------------------------------------------------------ historique (non local)
    def history_step(self, grid_et, grid_ec, grid_w, inv_dx: float):
        """Update irreversible concrete strain history (grid-smoothed) and steel plasticity.

        **Inputs**

        - `grid_et`, `grid_ec`, `grid_w` : f32 field (nx, ny), scratch
        - `inv_dx` : float

        **Outputs**

        - kap_t, kap_c, eps_p updated in place ; grids overwritten
        """
        grid_et.fill(0.0)
        grid_ec.fill(0.0)
        grid_w.fill(0.0)
        self.scatter_eps(grid_et, grid_ec, grid_w, inv_dx)
        self.update_state(grid_et, grid_ec, grid_w, inv_dx)

    @ti.kernel
    def scatter_eps(self, grid_et: ti.template(), grid_ec: ti.template(), grid_w: ti.template(), inv_dx: float):
        """Scatter equivalent strains (principal elastic stress / E_c) on the grid.

        **Inputs**

        - `grid_et`, `grid_ec`, `grid_w` : f32 field (nx, ny)
        - `inv_dx` : float

        **Outputs**

        - grid_et (tension), grid_ec (compression), grid_w (weight sum) accumulated in place

        **Note** : does not clear the grid.
        """
        for p in self.x:
            base, fx, w = stencil(self.x[p], inv_dx)
            s1, s2, U = self.principal_stresses(p)
            et = ti.max(ti.max(s1, s2), 0.0) / self.E_c
            ec = ti.max(ti.max(-s1, -s2), 0.0) / self.E_c
            for i, j in ti.static(ti.ndrange(3, 3)):
                weight = w[i, 0] * w[j, 1]
                node = base + ti.Vector([i, j])
                grid_et[node] += weight * et
                grid_ec[node] += weight * ec
                grid_w[node] += weight

    @ti.kernel
    def update_state(self, grid_et: ti.template(), grid_ec: ti.template(), grid_w: ti.template(), inv_dx: float):
        """Update strain history from grid-smoothed strains and steel plastic strain (return mapping).

        **Inputs**

        - `grid_et`, `grid_ec`, `grid_w` : f32 field (nx, ny)
        - `inv_dx` : float

        **Outputs**

        - kap_t, kap_c (running max), eps_p updated in place
        """
        for p in self.x:
            base, fx, w = stencil(self.x[p], inv_dx)
            et = 0.0
            ec = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):
                node = base + ti.Vector([i, j])
                if grid_w[node] > 0:
                    weight = w[i, 0] * w[j, 1]
                    et += weight * grid_et[node] / grid_w[node]
                    ec += weight * grid_ec[node] / grid_w[node]
            self.kap_t[p] = ti.max(self.kap_t[p], et)
            self.kap_c[p] = ti.max(self.kap_c[p], ec)

            if self.phi[p] > 0.0:
                eps = ti.log(ti.Vector([self.Fm[p][0, 0] + 1.0, self.Fm[p][1, 0]]).norm())
                trial = self.E_s * (eps - self.eps_p[p])
                ss = ti.math.clamp(trial, -self.fy, self.fy)
                self.eps_p[p] += (trial - ss) / self.E_s      # retour radial : écoulement plastique

    # ------------------------------------------------------------ diagnostics
    @ti.kernel
    def damage_counts(self) -> ti.types.vector(2, ti.f32):
        """Count cracked (d_t > 0.9) and crushed (kap_c > eps_cu) plain-concrete particles.

        **Outputs**

        - vec2 f32 (cracked, crushed)
        """
        nt = 0.0
        nc = 0.0
        for p in self.x:
            if self.phi[p] == 0.0:
                if 1.0 - self.r_tension(self.kap_t[p], self.fcr_p[p]) > 0.9:
                    nt += 1.0
                if self.kap_c[p] > self.eps_cu:
                    nc += 1.0
        return ti.Vector([nt, nc])

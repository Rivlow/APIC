"""Fluide incompressible : grille décalée (MAC) + projection de pression par gradient conjugué sur GPU, 2D ou 3D.

Grille n = (nx, ny[, nz]) de cellules carrées de côté dx. Faces de l'axe a : champ uf[a] de forme n + e_a, la face
I est en (I + 1/2 - e_a / 2) dx (position entière sur l'axe a, milieu de cellule sur les autres) ; mf[a] = poids
de transfert (0 = face sans particule). Cellules `ctype` (n) :
AIR = 0 (pression nulle), FLUID = 1 (inconnue), SOLID = 2 (obstacle, entrée, bande de paroi : vitesse
imposée, Neumann), MOVING = 3 (occupée par le solide MPM). Les cellules de sortie sont de l'air : pression
imposée (Dirichlet), écoulement libre.

On résout A q = b avec q = dt p / rho, A = laplacien positif à 2·dim + 1 points sur les cellules fluides
(deg = nombre de voisins non solides), b = -dx div u, puis u -= grad q.
Frottement aux parois : faces tangentielles dans une cellule solide = valeur fantôme (1 - 2 beta) u_fluide
(beta = 0 glissant, 1 adhérent : vitesse moyenne nulle sur la paroi), voir bc. Viscosité implicite sur
les faces avant la projection (3 bis). Correction de densité sur les positions après l'advection (4 bis).
Le gradient conjugué vit sur le GPU : `cg` = [rr, pAp, rr_new] ; α et β sont calculés dans les kernels.
Tout est porté par la classe MAC (data_oriented) : listes de champs de faces indexées par axe (ti.static).
"""
import taichi as ti

from Solver.boundary import (FRAMES, INLET, OBSTACLE, OUTLET, band_bc, band_cell,  # noqa: F401
                             in_band, outlet_q)
from ui.kernels import cell_of, rotor_sdf, rotor_velocity, stencil3, weights, wprod

AIR, FLUID, SOLID, MOVING, ROTOR = 0, 1, 2, 3, 4   # MOVING : solide MPM ; ROTOR : obstacle tournant (vitesse imposée)
VISC_OFF, VISC_FREE, VISC_FIXED = 0, 1, 2
# décalage (en cellules) des faces de l'axe a : 0 sur a, 0.5 sur les autres axes ; centres des cellules ; axes
# tangents à a ; ordre de priorité des murs pour le frottement d'une face de l'axe a (tangents puis a)
OFFS = {d: tuple(tuple(0.0 if c == a else 0.5 for c in range(d)) for a in range(d)) for d in (2, 3)}
CENTER = {d: (0.5,) * d for d in (2, 3)}
TANG = {d: tuple(tuple(c for c in range(d) if c != a) for a in range(d)) for d in (2, 3)}
BETA_ORDER = {d: tuple(TANG[d][a] + (a,) for a in range(d)) for d in (2, 3)}


@ti.func
def stagger(xp, inv_dx: float, off):
    """Base node and fraction on a grid staggered by `off` cells.

    **Inputs**

    - `xp` : vec dim f32 position
    - `inv_dx` : float
    - `off` : vec dim f32 offset (cells)

    **Outputs**

    - (ivec dim base, vec dim fx)
    """
    base = (xp * inv_dx - off - 0.5).cast(int)
    fx = xp * inv_dx - off - base
    return base, fx


@ti.func
def apply_A(q, ct, I):
    """Apply the (2 dim + 1)-point Laplacian (air: 0, solid: excluded) at cell I.

    **Inputs**

    - `q` : f32 field (n)
    - `ct` : i32 field (n) cell types
    - `I` : ivec dim

    **Outputs**

    - float (A q)_I
    """
    n = ti.static(ct.shape)
    dim = ti.static(len(n))
    deg = 0
    s = 0.0
    for a in ti.static(range(dim)):
        for sg in ti.static((1, -1)):
            J = I + sg * ti.Vector.unit(dim, a, ti.i32)
            if 0 <= J[a] < n[a]:
                t = ct[J]
                if t < SOLID:                         # solide fixe ou mobile : exclu (Neumann)
                    deg += 1
                    if t == FLUID:
                        s += q[J]
    return deg * q[I] - s


@ti.func
def _cell_grad(f0, fm, fp, okm, okp, inv_dx):
    """Central difference at a cell, one-sided if a neighbor is not fluid, 0 if none is."""
    g = 0.0
    if okm and okp:
        g = (fp - fm) * 0.5 * inv_dx
    elif okp:
        g = (fp - f0) * inv_dx
    elif okm:
        g = (f0 - fm) * inv_dx
    return g


@ti.func
def _spawn(x: ti.template(), vel: ti.template(), C: ti.template(), J: ti.template(), alive: ti.template(),
           free_stack: ti.template(), free_top: ti.template(), xp, vp):
    """Pop a free slot and spawn one particle (no-op when the pool is full).

    **Inputs**

    - pool fields (cap,) ; `xp`, `vp` : vec dim f32 position / velocity

    **Outputs**

    - pool fields written in place
    """
    idx = ti.atomic_sub(free_top[0], 1) - 1
    if idx >= 0:
        p = free_stack[idx]
        x[p] = xp
        vel[p] = vp
        C[p] = ti.Matrix.zero(ti.f32, xp.n, xp.n)
        J[p] = 1.0
        alive[p] = 1
    else:
        ti.atomic_add(free_top[0], 1)                 # plus de slot libre : on rend ce qu'on a pris


@ti.data_oriented
class MAC:
    def __init__(self, fb, shape):
        """Allocate the staggered grid, the cell fields and the CG / viscosity buffers.

        **Inputs**

        - `fb` : ti.FieldsBuilder   solver field tree (finalized by the caller)
        - `shape` : tuple[int, ...] cells per axis (2 or 3)

        **Outputs**

        - fields: uf[a], mf[a], df[a] (faces) ; ctype, q, r, pd, Ap, rhs, dens, phi, bvol, nu_c, fp (cells) ;
          cg, cg_visc (3,) ; viscosity per face axis: ft, nuf, vr, vpd, vAp, vrhs
        """
        self.n = tuple(int(v) for v in shape)
        self.dim = len(self.n)
        ax = ti.ij if self.dim == 2 else ti.ijk
        self.uf, self.mf, self.df = [], [], []
        self.ft, self.nuf, self.vr, self.vpd, self.vAp, self.vrhs = [], [], [], [], [], []
        for a in range(self.dim):
            fs = tuple(v + (1 if c == a else 0) for c, v in enumerate(self.n))
            u, m, d = ti.field(ti.f32), ti.field(ti.f32), ti.field(ti.f32)
            ft, nuf = ti.field(ti.i32), ti.field(ti.f32)
            vr, vpd, vAp, vrhs = (ti.field(ti.f32) for _ in range(4))
            fb.dense(ax, fs).place(u, m, d, ft, nuf, vr, vpd, vAp, vrhs)
            self.uf.append(u)
            self.mf.append(m)
            self.df.append(d)
            self.ft.append(ft)
            self.nuf.append(nuf)
            self.vr.append(vr)
            self.vpd.append(vpd)
            self.vAp.append(vAp)
            self.vrhs.append(vrhs)
        self.ctype = ti.field(ti.i32)
        self.q, self.r, self.pd, self.Ap, self.rhs = (ti.field(ti.f32) for _ in range(5))
        self.dens, self.phi, self.bvol, self.nu_c = (ti.field(ti.f32) for _ in range(4))
        self.fp = ti.Vector.field(self.dim, ti.f32)   # accélération de pression sur les nœuds du solide
        # condition limite de chaque cellule (calculée par classify, lue par bc) : type (WALL / INLET / OUTLET),
        # vitesse imposée, frottement d'une cellule solide vu par une face de chaque axe
        self.cbc = ti.field(ti.i32)
        self.cvel = ti.Vector.field(self.dim, ti.f32)
        self.cbeta = ti.Vector.field(self.dim, ti.f32)
        fb.dense(ax, self.n).place(self.ctype, self.q, self.r, self.pd, self.Ap, self.rhs, self.dens, self.phi,
                                   self.bvol, self.nu_c, self.fp, self.cbc, self.cvel, self.cbeta)
        self.cg, self.cg_visc = ti.field(ti.f32), ti.field(ti.f32)
        fb.dense(ti.i, 3).place(self.cg, self.cg_visc)
        # obstacle tournant : distance signée à l'angle 0, centre ; couple de pression (diagnostic)
        self.rsdf = ti.field(ti.f32)
        fb.dense(ax, self.n).place(self.rsdf)
        self.rc = ti.Vector.field(self.dim, ti.f32)
        self.torque = ti.Vector.field(3, ti.f32)
        fb.place(self.rc, self.torque)

    # ------------------------------------------------------------ 1. particules -> faces (APIC)
    def p2g(self, x, vel, C, alive, inv_dx: float, dx: float) -> None:
        """Transfer particles to MAC faces (APIC), one kernel per face axis.

        **Inputs**

        - `x`, `vel` : vec dim f32 field (cap,) positions / velocities
        - `C` : mat dim f32 field (cap,) affine matrices
        - `alive` : i32 field (cap,) 1 = live particle
        - `inv_dx`, `dx` : float

        **Outputs**

        - uf, mf written in place

        **Note** : split by axis to keep each 3D kernel small (compile time), 2 extra launches.
        """
        for a in range(self.dim):
            self._p2g_axis(x, vel, C, alive, inv_dx, dx, a)

    @ti.kernel
    def _p2g_axis(self, x: ti.template(), vel: ti.template(), C: ti.template(), alive: ti.template(),
                  inv_dx: float, dx: float, a: ti.template()):
        """P2G on the faces of axis a (clear, scatter, normalize)."""
        dim = ti.static(self.dim)
        for I in ti.grouped(self.uf[a]):
            self.uf[a][I] = 0.0
            self.mf[a][I] = 0.0
        for p in x:
            if alive[p] == 1:
                off = ti.Vector(OFFS[dim][a])
                base, fx = stagger(x[p], inv_dx, off)
                w = weights(fx)
                for o in ti.static(stencil3(dim)):
                    node = base + o
                    d = (node + off) * dx - x[p]
                    wt = wprod(w, o)
                    val = vel[p][a]
                    for c in ti.static(range(dim)):
                        val += C[p][a, c] * d[c]
                    self.uf[a][node] += wt * val
                    self.mf[a][node] += wt
        for I in ti.grouped(self.uf[a]):
            if self.mf[a][I] > 0:
                self.uf[a][I] /= self.mf[a][I]

    # ------------------------------------------------------------ 2. type des cellules
    @ti.kernel
    def cell_bc(self, wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(),
                wall_f: ti.template(), bound: int, beta_o: float):
        """Boundary condition of every cell, read by bc: type, imposed velocity, friction per face axis.

        **Inputs**

        - `wall_type`, `wall_d` : i32 field (2 dim, na, nb) ; `wall_v` : vec dim f32 ; `wall_f` : f32
        - `bound` : int ; `beta_o` : float obstacle friction

        **Outputs**

        - cbc, cvel, cbeta written in place

        **Note** : static between rebuilds except for time-dependent inlet velocities (called every substep) ; kept
        out of bc so that the 3D bc kernel stays small (compile time).
        """
        n = ti.static(self.n)
        for I in ti.grouped(self.cbc):
            t, vel = band_bc(wall_type, wall_v, wall_d, I, n, bound)
            self.cbc[I] = t
            self.cvel[I] = vel
            for a in ti.static(range(self.dim)):
                self.cbeta[I][a] = self.solid_beta(wall_f, I, bound, beta_o, a)

    @ti.kernel
    def classify(self, cells: ti.template(), wall_type: ti.template(), wall_v: ti.template(),
                 wall_d: ti.template(), x: ti.template(), alive: ti.template(), x_s: ti.template(), has_solid: int,
                 inv_dx: float, bound: int, free_surface: int):
        """Classify cells (air/fluid/solid/moving).

        **Inputs**

        - `cells` : i32 field (n) bit flags
        - `wall_type`, `wall_d` : i32 field (2 dim, na, nb) ; `wall_v` : vec dim f32 field
        - `x` : vec dim f32 field (cap,) fluid positions ; `alive` : i32 field (cap,)
        - `x_s` : vec dim f32 field (Ns,) solid positions ; `has_solid` : int
        - `inv_dx` : float ; `bound`, `free_surface` : int

        **Outputs**

        - ctype written in place

        **Note** : outlet wall cells are AIR; free_surface = 0 makes every non-solid cell FLUID.
        """
        n = ti.static(self.n)
        for I in ti.grouped(self.ctype):
            f = cells[I]
            side, _ka, _kb = band_cell(I, n, bound, wall_d)
            wall = in_band(I, n, bound) or side >= 0
            if wall:
                self.ctype[I] = AIR if self.cbc[I] == OUTLET else SOLID
            elif f & OBSTACLE:
                self.ctype[I] = SOLID
            elif free_surface == 0:
                self.ctype[I] = FLUID
            else:
                self.ctype[I] = AIR
        for p in x_s:
            if has_solid == 1:
                c = cell_of(x_s[p], inv_dx, n)
                if self.ctype[c] < SOLID:
                    self.ctype[c] = MOVING
        for p in x:
            if alive[p] == 1:
                c = cell_of(x[p], inv_dx, n)
                side, _ka, _kb = band_cell(c, n, bound, wall_d)
                wall = in_band(c, n, bound) or side >= 0
                if self.ctype[c] == AIR and not wall:
                    self.ctype[c] = FLUID

    @ti.kernel
    def rotor_classify(self, axis: ti.types.vector(3, ti.f32), theta: float, inv_dx: float):
        """Mark the cells covered by the rotor at angle theta (after classify ; walls and obstacles win).

        **Inputs**

        - `axis` : vec3 unit ; `theta` : float rad ; `inv_dx` : float

        **Outputs**

        - ctype = ROTOR where the rotated rotor distance is negative
        """
        center = self.rc[None]
        for I in ti.grouped(self.ctype):
            if self.ctype[I] < SOLID:
                pc = (I.cast(ti.f32) + 0.5) / inv_dx
                phi, _g = rotor_sdf(self.rsdf, pc, center, axis, theta, inv_dx)
                if phi < 0.0:
                    self.ctype[I] = ROTOR

    @ti.kernel
    def rotor_torque(self, rho_over_dt: float, dx: float):
        """Pressure torque of the fluid on the rotor about its center (diagnostic, read by stats).

        **Inputs**

        - `rho_over_dt` : float (p = q rho / dt) ; `dx` : float

        **Outputs**

        - torque[None] (3,) N m (per unit depth in 2D, along z)
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        center = self.rc[None]
        self.torque[None] = ti.Vector([0.0, 0.0, 0.0])
        area = dx ** (dim - 1)
        for I in ti.grouped(self.ctype):
            if self.ctype[I] == FLUID:
                for a in ti.static(range(dim)):
                    for sg in ti.static((1, -1)):
                        J = I + sg * ti.Vector.unit(dim, a, ti.i32)
                        if 0 <= J[a] < n[a]:
                            if self.ctype[J] == ROTOR:
                                # force de pression sur la face du rotor : -p n_rotor dA, n_rotor = -sg e_a
                                pf = (I.cast(ti.f32) + 0.5 + 0.5 * sg * ti.Vector.unit(dim, a, ti.f32)) * dx
                                f = ti.Vector.unit(dim, a, ti.f32) * (sg * self.q[I] * rho_over_dt * area)
                                r = pf - center
                                if ti.static(dim == 3):
                                    self.torque[None] += r.cross(f)
                                else:
                                    self.torque[None][2] += r[0] * f[1] - r[1] * f[0]

    @ti.func
    def solid_beta(self, wall_f: ti.template(), C, bound: int, beta_o: float, a: ti.template()) -> float:
        """Friction of solid cell C next to a tangential face of axis a: wall friction in the band, else beta_o.

        **Inputs**

        - `wall_f` : f32 field (2 dim, na, nb) wall friction
        - `C` : ivec dim cell ; `bound` : int ; `beta_o` : float obstacle friction
        - `a` : static face axis

        **Outputs**

        - float beta

        **Note** : in a corner, the walls whose normal is not a win (the face is tangential to them), in axis
        order, then axis a.
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        beta = beta_o
        found = False
        for b in ti.static(BETA_ORDER[dim][a]):
            if not found:
                if C[b] < bound or C[b] >= n[b] - bound:
                    found = True
                    for s in ti.static((2 * b, 2 * b + 1)):
                        plus = ti.static(s % 2)
                        on = C[b] >= n[b] - bound if ti.static(plus) else C[b] < bound
                        if on:
                            ts = ti.static(FRAMES[dim][s][2])
                            kb = 0
                            if ti.static(dim == 3):
                                kb = C[ts[1]]
                            beta = wall_f[s, C[ts[0]], kb]
        return beta

    @ti.kernel
    def bc(self, wall_type: ti.template(), wall_v: ti.template(), wall_d: ti.template(), wall_f: ti.template(),
           grid_v: ti.template(), grid_m: ti.template(), dt: float, g: float, beta_o: float, bound: int,
           with_gravity: int, dx: float, axis: ti.types.vector(3, ti.f32), omega: float):
        """Apply gravity and face BCs (walls, inlets, moving solid, rotor, friction ghosts).

        **Inputs**

        - `wall_type`, `wall_d` : i32 field (2 dim, na, nb) ; `wall_v` : vec dim f32 ; `wall_f` : f32
        - `grid_v` : vec dim f32 field (n) solid node velocities ; `grid_m` : f32 field (n) solid node masses
        - `dt`, `g`, `beta_o` : float ; `bound`, `with_gravity` : int
        - `dx` : float ; `axis` : vec3 unit, `omega` : float rad/s (rotor, center in rc)

        **Outputs**

        - uf written in place
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        for I in ti.grouped(self.uf[1]):
            if with_gravity == 1:
                self.uf[1][I] -= dt * g
        for a in ti.static(range(dim)):
            ea = ti.Vector.unit(dim, a, ti.i32)
            ts = ti.static(TANG[dim][a])
            for I in ti.grouped(self.uf[a]):             # face entre les cellules I - e_a et I
                Il = I - ea
                Il[a] = ti.math.clamp(Il[a], 0, n[a] - 1)
                Ir = I
                Ir[a] = ti.math.clamp(Ir[a], 0, n[a] - 1)
                tl, tr = self.ctype[Il], self.ctype[Ir]
                if tl >= SOLID or tr >= SOLID:
                    val = 0.0
                    if self.cbc[Il] == INLET:
                        val = self.cvel[Il][a]
                    elif self.cbc[Ir] == INLET:
                        val = self.cvel[Ir][a]
                    elif tl == ROTOR or tr == ROTOR:            # paroi tournante : vitesse rigide au centre de la face
                        pf = (I.cast(ti.f32) + ti.Vector(OFFS[dim][a])) * dx
                        val = rotor_velocity(pf, self.rc[None], axis, omega)[a]
                    elif tl == MOVING or tr == MOVING:
                        # nœuds de la face : Ir + {0, 1} sur les axes tangents (2^(dim-1) nœuds)
                        sm, sv = 0.0, 0.0
                        for o in ti.static(ti.grouped(ti.ndrange(*([2] * (dim - 1))))):
                            N = Ir
                            for k in ti.static(range(dim - 1)):
                                N[ts[k]] = ti.min(N[ts[k]] + o[k], n[ts[k]] - 1)
                            m = grid_m[N]
                            sm += m
                            sv += grid_v[N][a] * m
                        if sm > 0:
                            val = sv / sm
                    elif tl == SOLID and tr == SOLID and 0 < I[a] < n[a]:
                        # face tangentielle : voisine le long d'un axe tangent, ses deux cellules non solides et au
                        # moins une fluide. Jamais une face air-air (bande de sortie : jamais projetée, elle ne porte
                        # que la moyenne des particules) : les fantômes d'un coin 3D (sortie x fond x arrière) la
                        # recopiaient, G2P / P2G la réamplifiaient et le calcul divergeait.
                        found = False
                        for t in ti.static(ts):
                            for sg in ti.static((1, -1)):
                                J = I + sg * ti.Vector.unit(dim, t, ti.i32)
                                if not found and 0 <= J[t] < n[t]:
                                    cl, cr = self.ctype[J - ea], self.ctype[J]
                                    if cl < SOLID and cr < SOLID and (cl == FLUID or cr == FLUID):
                                        found = True
                                        val = (1.0 - 2.0 * self.cbeta[Ir][a]) * self.uf[a][J]
                    self.uf[a][I] = val

    @ti.kernel
    def pressure_force(self, grid_m: ti.template(), inv_dx: float, coef: float):
        """Pressure acceleration -grad p / rho_s on solid grid nodes.

        **Inputs**

        - `grid_m` : f32 field (n) solid node masses
        - `inv_dx`, `coef` : float

        **Outputs**

        - fp written in place
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        for I in ti.grouped(self.fp):
            acc = ti.Vector.zero(ti.f32, dim)
            if grid_m[I] > 0:
                for o in ti.static(ti.grouped(ti.ndrange(*([2] * dim)))):   # cellules autour du nœud
                    Cc = ti.math.clamp(I - 1 + o, 0, ti.Vector(n) - 1)
                    qv = self.q[Cc] if self.ctype[Cc] == FLUID else 0.0
                    for c in ti.static(range(dim)):
                        acc[c] += qv if ti.static(o[c] == 1) else -qv
                acc *= coef * inv_dx / 2 ** (dim - 1)
            self.fp[I] = acc

    @ti.kernel
    def add_accel(self, grid_v: ti.template(), grid_m: ti.template(), dt: float):
        """Add fp dt to grid velocities of massive nodes.

        **Inputs**

        - `grid_v` : vec dim f32 field (n) ; `grid_m` : f32 field (n) ; `dt` : float

        **Outputs**

        - grid_v written in place
        """
        for I in ti.grouped(grid_v):
            if grid_m[I] > 0:
                grid_v[I] += dt * self.fp[I]

    # ------------------------------------------------------------ 3. pression (gradient conjugué)
    @ti.kernel
    def particle_density(self, x: ti.template(), alive: ti.template(), inv_dx: float, ppc: int):
        """rho / rho0 at cell centers : boundary volume map + quadratic B-spline weights of the particles / ppc^dim.

        **Inputs**

        - `x` : vec dim f32 field (cap,) positions ; `alive` : i32 field (cap,)
        - `inv_dx` : float ; `ppc` : int

        **Outputs**

        - dens written in place (1 = nominal density, walls included)

        **Note** : as SPlisHSPlasH volume maps, the boundary counts in the density : no truncated kernel near a
        wall, and fluid pressed against it becomes overdense, so the density projection pushes it away. The MPM
        solid is not counted (tested: the fluid next to it then looks overdense and is pushed away violently).
        """
        dim = ti.static(self.dim)
        for I in ti.grouped(self.dens):
            self.dens[I] = self.bvol[I]
        inv = 1.0 / ppc ** dim
        for p in x:
            if alive[p] == 1:
                base, fx = stagger(x[p], inv_dx, ti.Vector(CENTER[dim]))   # centres des cellules
                w = weights(fx)
                for o in ti.static(stencil3(dim)):
                    self.dens[base + o] += wprod(w, o) * inv

    @ti.kernel
    def cg_init(self, wall_type: ti.template(), wall_d: ti.template(), wall_p: ti.template(), dx: float,
                dt: float, inv_rho: float, bound: int):
        """Solve pressure: CG init (rhs, Dirichlet air cells, r, p, rr).

        **Inputs**

        - `wall_type`, `wall_d` : i32 field (2 dim, na, nb) ; `wall_p` : f32 field (outlet pressure)
        - `dx`, `dt`, `inv_rho` : float ; `bound` : int

        **Outputs**

        - q, r, pd, rhs, cg[0] written in place

        **Note** : warm start from previous q.
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        for I in ti.grouped(self.ctype):
            if self.ctype[I] != FLUID:
                self.q[I] = 0.0
                if self.ctype[I] == AIR:
                    self.q[I] = outlet_q(wall_type, wall_d, wall_p, I, n, bound, dt, inv_rho)
        for I in ti.grouped(self.ctype):
            if self.ctype[I] == FLUID:
                div = 0.0
                for a in ti.static(range(dim)):
                    div += self.uf[a][I + ti.Vector.unit(dim, a, ti.i32)] - self.uf[a][I]
                b = -dx * div
                for a in ti.static(range(dim)):
                    for sg in ti.static((1, -1)):
                        J = I + sg * ti.Vector.unit(dim, a, ti.i32)
                        if 0 <= J[a] < n[a]:
                            if self.ctype[J] == AIR:
                                b += self.q[J]
                self.rhs[I] = b
            else:
                self.rhs[I] = 0.0
        rr = 0.0                                         # réduction locale (pas d'atomiques sur une seule case)
        for I in ti.grouped(self.ctype):
            if self.ctype[I] == FLUID:
                self.r[I] = self.rhs[I] - apply_A(self.q, self.ctype, I)
                self.pd[I] = self.r[I]
                rr += self.r[I] * self.r[I]
            else:
                self.r[I] = 0.0
                self.pd[I] = 0.0
        self.cg[0] = rr

    @ti.kernel
    def cg_apply(self):
        """Solve pressure: CG step A p and pAp.

        **Outputs**

        - Ap, cg[1] written in place
        """
        pAp = 0.0
        for I in ti.grouped(self.ctype):
            if self.ctype[I] == FLUID:
                self.Ap[I] = apply_A(self.pd, self.ctype, I)
                pAp += self.pd[I] * self.Ap[I]
        self.cg[1] = pAp

    @ti.kernel
    def cg_update(self, x: ti.template()):
        """Solve pressure: CG update of x (q or phi), r, p.

        **Inputs**

        - `x` : f32 field (n) unknown

        **Outputs**

        - x, r, pd, cg written in place

        **Note** : alpha, beta computed on GPU (no read-back).
        """
        alpha = self.cg[0] / ti.max(self.cg[1], 1e-30)
        rr_new = 0.0
        for I in ti.grouped(self.ctype):
            if self.ctype[I] == FLUID:
                x[I] += alpha * self.pd[I]
                self.r[I] -= alpha * self.Ap[I]
                rr_new += self.r[I] * self.r[I]
        self.cg[2] = rr_new
        beta = self.cg[2] / ti.max(self.cg[0], 1e-30)
        for I in ti.grouped(self.ctype):
            if self.ctype[I] == FLUID:
                self.pd[I] = self.r[I] + beta * self.pd[I]
        self.cg[0] = self.cg[2]

    # ------------------------------------------------------------ 3 bis. viscosité implicite (faces)
    # Diffusion implicite (I - dt div(nu grad)) u* = u, résolue séparément sur les faces de chaque axe avant la
    # projection. nu par cellule = nu0 + nu_t, nu_t = (C_s dx)^2 |S| (Smagorinsky, cellules fluides seulement) :
    # nul dans un écoulement uniforme, fort dans les zones cisaillées (rouleau d'un ressaut). nu d'une face =
    # moyenne de ses deux cellules ; couplage entre deux faces voisines = k (nu_f + nu_n) / 2, k = dt / dx² : SPD.
    # Type de face `ft` : VISC_FREE (inconnue : aucune cellule voisine solide, au moins une fluide), VISC_FIXED
    # (Dirichlet : une cellule solide, valeur posée par bc, fantômes de frottement compris), VISC_OFF (air ou
    # hors domaine : Neumann, surface libre sans cisaillement). CG à itérations fixes, sans lecture GPU -> CPU.
    @ti.kernel
    def visc_nu(self, nu0: float, l2: float, inv_dx: float):
        """Viscosity per cell: nu0 + Smagorinsky nu_t = l2 |S| on fluid cells, |S| = sqrt(2 S:S).

        **Inputs**

        - `nu0` : float molecular viscosity (m²/s) ; `l2` : float (C_s dx)² (m²) ; `inv_dx` : float

        **Outputs**

        - nu_c written in place

        **Note** : cross derivatives from cell-centered velocities of fluid neighbors only (no air/solid garbage
        at the free surface).
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        for I in ti.grouped(self.nu_c):
            nu = nu0
            if l2 > 0 and self.ctype[I] == FLUID:
                g = ti.Matrix.zero(ti.f32, dim, dim)     # g[a, b] = d u_b / d x_a
                for a in ti.static(range(dim)):
                    ea = ti.Vector.unit(dim, a, ti.i32)
                    Im = ti.max(I - ea, 0)
                    Ip = ti.min(I + ea, ti.Vector(n) - 1)
                    okm = I[a] > 0 and self.ctype[Im] == FLUID
                    okp = I[a] < n[a] - 1 and self.ctype[Ip] == FLUID
                    for b in ti.static(range(dim)):
                        eb = ti.Vector.unit(dim, b, ti.i32)
                        if ti.static(a == b):
                            g[a, b] = (self.uf[b][I + eb] - self.uf[b][I]) * inv_dx
                        else:
                            u0 = 0.5 * (self.uf[b][I] + self.uf[b][I + eb])
                            um = 0.5 * (self.uf[b][Im] + self.uf[b][Im + eb])
                            up = 0.5 * (self.uf[b][Ip] + self.uf[b][Ip + eb])
                            g[a, b] = _cell_grad(u0, um, up, okm, okp, inv_dx)
                s2 = 0.0
                for a in ti.static(range(dim)):
                    for b in ti.static(range(dim)):
                        sab = 0.5 * (g[a, b] + g[b, a])
                        s2 += 2.0 * sab * sab
                nu += l2 * ti.sqrt(s2)
            self.nu_c[I] = nu

    @ti.kernel
    def visc_classify(self, a: ti.template()):
        """Viscosity: classify faces of axis a (free unknown, fixed Dirichlet, off) and face viscosity.

        **Inputs**

        - `a` : static face axis

        **Outputs**

        - ft[a], nuf[a] written in place
        """
        n = ti.static(self.n)
        ea = ti.Vector.unit(self.dim, a, ti.i32)
        for I in ti.grouped(self.ft[a]):
            Il = I - ea
            Il[a] = ti.math.clamp(Il[a], 0, n[a] - 1)
            Ir = I
            Ir[a] = ti.math.clamp(Ir[a], 0, n[a] - 1)
            ta, tb = self.ctype[Il], self.ctype[Ir]
            t = VISC_OFF
            if ta >= SOLID or tb >= SOLID:
                t = VISC_FIXED
            elif ta == FLUID or tb == FLUID:
                t = VISC_FREE
            self.ft[a][I] = t
            self.nuf[a][I] = 0.5 * (self.nu_c[Il] + self.nu_c[Ir])

    @ti.func
    def visc_A(self, x: ti.template(), a: ti.template(), I, k):
        """Apply (I - dt div(nu grad)) at free face I of axis a (fixed neighbors: moved to the rhs, off: Neumann).

        **Inputs**

        - `x` : f32 field, face grid of axis a ; `a` : static axis ; `I` : ivec dim ; `k` : float dt / dx²

        **Outputs**

        - float (A x)_I
        """
        dim = ti.static(self.dim)
        fs = ti.static(x.shape)
        diag = 1.0
        s = 0.0
        for b in ti.static(range(dim)):
            for sg in ti.static((1, -1)):
                J = I + sg * ti.Vector.unit(dim, b, ti.i32)
                if 0 <= J[b] < fs[b]:
                    t = self.ft[a][J]
                    if t != VISC_OFF:
                        c = k * 0.5 * (self.nuf[a][I] + self.nuf[a][J])
                        diag += c
                        if t == VISC_FREE:
                            s += c * x[J]
        return diag * x[I] - s

    @ti.kernel
    def visc_cg_init(self, a: ti.template(), k: float):
        """Viscosity: CG init on faces of axis a (rhs = u + coupling-weighted fixed neighbors, warm start x = u).

        **Inputs**

        - `a` : static axis ; `k` : float dt / dx²

        **Outputs**

        - vr[a], vpd[a], vrhs[a], cg_visc[0] written in place
        """
        dim = ti.static(self.dim)
        w = ti.static(self.uf[a])
        fs = ti.static(w.shape)
        for I in ti.grouped(w):
            if self.ft[a][I] == VISC_FREE:
                b = w[I]
                for c in ti.static(range(dim)):
                    for sg in ti.static((1, -1)):
                        J = I + sg * ti.Vector.unit(dim, c, ti.i32)
                        if 0 <= J[c] < fs[c]:
                            if self.ft[a][J] == VISC_FIXED:
                                b += k * 0.5 * (self.nuf[a][I] + self.nuf[a][J]) * w[J]
                self.vrhs[a][I] = b
        rr = 0.0
        for I in ti.grouped(w):
            if self.ft[a][I] == VISC_FREE:
                self.vr[a][I] = self.vrhs[a][I] - self.visc_A(w, a, I, k)
                self.vpd[a][I] = self.vr[a][I]
                rr += self.vr[a][I] * self.vr[a][I]
            else:
                self.vr[a][I] = 0.0
                self.vpd[a][I] = 0.0
        self.cg_visc[0] = rr

    @ti.kernel
    def visc_apply(self, a: ti.template(), k: float):
        """Viscosity: CG step A p and pAp on faces of axis a.

        **Inputs**

        - `a` : static axis ; `k` : float dt / dx²

        **Outputs**

        - vAp[a], cg_visc[1] written in place
        """
        pAp = 0.0
        for I in ti.grouped(self.vpd[a]):
            if self.ft[a][I] == VISC_FREE:
                self.vAp[a][I] = self.visc_A(self.vpd[a], a, I, k)
                pAp += self.vpd[a][I] * self.vAp[a][I]
        self.cg_visc[1] = pAp

    @ti.kernel
    def visc_update(self, a: ti.template()):
        """Viscosity: CG update of the face velocities of axis a, r, p.

        **Inputs**

        - `a` : static axis

        **Outputs**

        - uf[a], vr[a], vpd[a], cg_visc written in place

        **Note** : alpha, beta computed on GPU (no read-back).
        """
        alpha = self.cg_visc[0] / ti.max(self.cg_visc[1], 1e-30)
        rr_new = 0.0
        for I in ti.grouped(self.uf[a]):
            if self.ft[a][I] == VISC_FREE:
                self.uf[a][I] += alpha * self.vpd[a][I]
                self.vr[a][I] -= alpha * self.vAp[a][I]
                rr_new += self.vr[a][I] * self.vr[a][I]
        self.cg_visc[2] = rr_new
        beta = self.cg_visc[2] / ti.max(self.cg_visc[0], 1e-30)
        for I in ti.grouped(self.uf[a]):
            if self.ft[a][I] == VISC_FREE:
                self.vpd[a][I] = self.vr[a][I] + beta * self.vpd[a][I]
        self.cg_visc[0] = self.cg_visc[2]

    # ------------------------------------------------------------ 4 bis. projection de densité (positions)
    # Deuxième projection, séparée de celle des vitesses (comme DFSPH : solveur de divergence + solveur de densité
    # constante ; Kugelstadt et al. 2019 sur grille). La projection de vitesse rend u à divergence nulle sur la
    # grille, mais la vitesse interpolée aux particules ne l'est pas près de la surface libre : les particules se
    # tassent et le volume d'eau vu par la grille diminue. On corrige donc les POSITIONS, sans toucher aux vitesses
    # (pas d'énergie injectée) : dilatation div(dx_p) = e = rho/rho0 - 1, soit dx_p = -grad phi avec
    # A phi = dx² e (même laplacien que la pression, air et sorties : phi = 0, solides exclus).
    @ti.kernel
    def density_cg_init(self, kappa: float, dx: float):
        """Density projection: CG init with rhs kappa dx^2 (rho/rho0 - 1), phi = 0.

        **Inputs**

        - `kappa`, `dx` : float

        **Outputs**

        - phi, r, pd, rhs, cg[0] written in place

        **Note** : surface cells correct overdensity only.
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        for I in ti.grouped(self.ctype):
            self.phi[I] = 0.0
            self.rhs[I] = 0.0
            if self.ctype[I] == FLUID:
                # bord : voisin d'air (cellule partiellement remplie) ou de solide (noyau de densité tronqué, pas de
                # particules de l'autre côté) -> densité sous-estimée, on n'y corrige que la surdensité
                surface = False
                for a in ti.static(range(dim)):
                    for sg in ti.static((1, -1)):
                        J = I + sg * ti.Vector.unit(dim, a, ti.i32)
                        if 0 <= J[a] < n[a]:
                            if self.ctype[J] == AIR or self.ctype[J] >= MOVING:   # parois : dans la volume map
                                surface = True
                e = self.dens[I] - 1.0
                if surface:
                    e = ti.max(e, 0.0)
                self.rhs[I] = kappa * dx * dx * e
        rr = 0.0
        for I in ti.grouped(self.ctype):
            self.r[I] = self.rhs[I]
            self.pd[I] = self.rhs[I]
            rr += self.rhs[I] * self.rhs[I]
        self.cg[0] = rr

    @ti.kernel
    def density_gradient(self, dx: float):
        """Face displacements -grad phi (0 next to solids).

        **Inputs**

        - `dx` : float

        **Outputs**

        - df written in place
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        for a in ti.static(range(dim)):
            ea = ti.Vector.unit(dim, a, ti.i32)
            for I in ti.grouped(self.df[a]):
                val = 0.0
                if 0 < I[a] < n[a]:
                    tl, tr = self.ctype[I - ea], self.ctype[I]
                    if tl < SOLID and tr < SOLID and (tl == FLUID or tr == FLUID):
                        val = -(self.phi[I] - self.phi[I - ea]) / dx
                self.df[a][I] = val

    @ti.kernel
    def density_shift(self, x: ti.template(), alive: ti.template(), inv_dx: float, dx: float, bound: int):
        """Shift particle positions by the interpolated face displacement.

        **Inputs**

        - `x` : vec dim f32 field (cap,) positions ; `alive` : i32 field (cap,)
        - `inv_dx`, `dx` : float ; `bound` : int

        **Outputs**

        - x written in place

        **Note** : shift clamped to 0.5 dx; moves into solid cells rejected; velocities untouched.
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        lo = ti.Vector.zero(ti.f32, dim) + bound * dx
        hi = (ti.Vector(n) - bound).cast(ti.f32) * dx
        for p in x:
            if alive[p] == 1:
                dsp = ti.Vector.zero(ti.f32, dim)
                for a in ti.static(range(dim)):
                    base, fx = stagger(x[p], inv_dx, ti.Vector(OFFS[dim][a]))
                    w = weights(fx)
                    sa = 0.0
                    for o in ti.static(stencil3(dim)):
                        sa += wprod(w, o) * self.df[a][base + o]
                    dsp[a] = sa
                d = ti.math.clamp(dsp, -0.5 * dx, 0.5 * dx)
                xn = ti.math.clamp(x[p] + d, lo, hi)
                if self.ctype[cell_of(xn, inv_dx, n)] < SOLID:
                    x[p] = xn

    # ------------------------------------------------------------ 5. projection
    @ti.kernel
    def project(self, dx: float):
        """Subtract grad q on faces between non-solid cells (one fluid).

        **Inputs**

        - `dx` : float

        **Outputs**

        - uf written in place
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        for a in ti.static(range(dim)):
            ea = ti.Vector.unit(dim, a, ti.i32)
            for I in ti.grouped(self.uf[a]):
                if 0 < I[a] < n[a]:
                    tl, tr = self.ctype[I - ea], self.ctype[I]
                    if tl < SOLID and tr < SOLID and (tl == FLUID or tr == FLUID):
                        self.uf[a][I] -= (self.q[I] - self.q[I - ea]) / dx

    # ------------------------------------------------------------ 6. faces -> particules (APIC)
    def g2p(self, x, vel, C, alive, inv_dx: float, dx: float) -> None:
        """Transfer face velocities to particles (APIC), one kernel per face axis.

        **Inputs**

        - `x`, `vel` : vec dim f32 field (cap,) positions / velocities
        - `C` : mat dim f32 field (cap,) affine matrices ; `alive` : i32 field (cap,)
        - `inv_dx`, `dx` : float

        **Outputs**

        - vel, C written in place

        **Note** : component kept unchanged when no face has weight > 0 ; split by axis (compile time).
        """
        for a in range(self.dim):
            self._g2p_axis(x, vel, C, alive, inv_dx, dx, a)

    @ti.kernel
    def _g2p_axis(self, x: ti.template(), vel: ti.template(), C: ti.template(), alive: ti.template(),
                  inv_dx: float, dx: float, a: ti.template()):
        """G2P of the velocity component a and row a of C.

        **Note** : faces between two air cells of an outlet (wall band or glued obstacle face) are skipped: never
        projected, they only hold the particles' own APIC extrapolation, which fed back through G2P / P2G and blew
        up at 3D outlet edges (outlet x bottom x back, slip walls). On such a truncated stencil C falls back to 0
        (PIC for this component): C = 4/dx² Σ w u d is the affine fit only on the full stencil.
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        ea = ti.Vector.unit(dim, a, ti.i32)
        k = 4.0 * inv_dx * inv_dx
        for p in x:
            if alive[p] == 1:
                off = ti.Vector(OFFS[dim][a])
                base, fx = stagger(x[p], inv_dx, off)
                w = weights(fx)
                su, sw, cv = 0.0, 0.0, ti.Vector.zero(ti.f32, dim)
                for o in ti.static(stencil3(dim)):
                    node = base + o
                    Cl = ti.math.clamp(node - ea, 0, ti.Vector(n) - 1)
                    Cr = ti.math.clamp(node, 0, ti.Vector(n) - 1)
                    outlet_air = self.ctype[Cl] == AIR and self.ctype[Cr] == AIR and (
                        self.cbc[Cl] == OUTLET or self.cbc[Cr] == OUTLET)
                    if self.mf[a][node] > 0 and not outlet_air:
                        wt = wprod(w, o)
                        d = (node + off) * dx - x[p]
                        su += wt * self.uf[a][node]
                        cv += wt * self.uf[a][node] * d
                        sw += wt
                if sw > 0:
                    vel[p][a] = su / sw
                    # C = 4/dx² Σ w u d n'est l'ajustement affine que sur un stencil complet (Σ w d d^T = dx²/4 I) :
                    # stencil tronqué (faces sans poids, air-air) -> gradient biaisé qui s'amplifie (arêtes 3D) ;
                    # on repasse alors en PIC pour cette composante
                    full = 1.0 if sw > 0.95 else 0.0
                    for c in ti.static(range(dim)):
                        C[p][a, c] = cv[c] / sw * k * full

    # ------------------------------------------------------------ 7. sortie à pression imposée : émission si l'eau rentre
    @ti.kernel
    def emit_pressure_outlet(self, x: ti.template(), vel: ti.template(), C: ti.template(), J: ti.template(),
                             alive: ti.template(), wall_type: ti.template(), wall_d: ti.template(),
                             wall_p: ti.template(), acc: ti.template(), ppc_face: float, free_stack: ti.template(),
                             free_top: ti.template(), dt: float, dx: float, bound: int):
        """Emit particles by flux at pressure outlets (p > 0) with inflow.

        **Inputs**

        - pool fields `x`, `vel`, `C`, `J`, `alive`, `free_stack`, `free_top`
        - `wall_type`, `wall_d` : i32 field (2 dim, na, nb) ; `wall_p` : f32 field (outlet pressure)
        - `acc` : f32 field (2 dim, na, nb) fractional particle carry
        - `ppc_face` : float ppc^dim (particles per wall cell per unit of v_n dt / dx)
        - `dt`, `dx` : float ; `bound` : int

        **Outputs**

        - pool fields, acc written in place
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        for side, ka, kb in wall_type:
            for s in ti.static(range(2 * dim)):
                a, plus, ts = ti.static(FRAMES[dim][s])
                ok = side == s and wall_type[side, ka, kb] == OUTLET and wall_p[side, ka, kb] > 0.0 \
                    and bound <= ka < n[ts[0]] - bound
                if ti.static(dim == 3):
                    ok = ok and bound <= kb < n[ts[1]] - bound
                if ok:
                    d = wall_d[side, ka, kb]
                    A = ti.Vector.zero(ti.i32, dim)          # première cellule libre devant le mur
                    A[ts[0]] = ka
                    if ti.static(dim == 3):
                        A[ts[1]] = kb
                    Fc = A                                   # face du mur
                    nrm = ti.Vector.zero(ti.f32, dim)        # normale rentrante
                    vn = 0.0
                    if ti.static(plus):
                        A[a] = n[a] - d - 1
                        Fc[a] = n[a] - d
                        nrm[a] = -1.0
                        vn = -self.uf[a][Fc]
                    else:
                        A[a] = d
                        Fc[a] = d
                        nrm[a] = 1.0
                        vn = self.uf[a][Fc]
                    if self.ctype[A] == FLUID and vn > 0.0:
                        acc[side, ka, kb] += ppc_face * vn * dt / dx
                        count = int(acc[side, ka, kb])
                        acc[side, ka, kb] -= count
                        wall = (A.cast(ti.f32) + 0.5) * dx - 0.5 * dx * nrm     # point du mur
                        for _ in range(count):
                            xp = wall + nrm * (ti.random() * vn * dt)
                            for t in ti.static(ts):
                                xp[t] += (ti.random() - 0.5) * dx
                            _spawn(x, vel, C, J, alive, free_stack, free_top, xp, nrm * vn)

    # ------------------------------------------------------------ diagnostics et couleurs
    @ti.kernel
    def divergence_max(self, dx: float) -> ti.f32:
        """Max |div u| over fluid cells (diagnostic).

        **Inputs**

        - `dx` : float

        **Outputs**

        - f32 max
        """
        dim = ti.static(self.dim)
        m = 0.0
        for I in ti.grouped(self.ctype):
            if self.ctype[I] == FLUID:
                div = 0.0
                for a in ti.static(range(dim)):
                    div += self.uf[a][I + ti.Vector.unit(dim, a, ti.i32)] - self.uf[a][I]
                ti.atomic_max(m, ti.abs(div) / dx)
        return m

    @ti.kernel
    def fluid_scalar(self, mode: int, x: ti.template(), v: ti.template(), alive: ti.template(), sc: ti.template(),
                     rho_over_dt: float, rho0: float, inv_dx: float):
        """Per-particle display scalar, incompressible mode (mode ids of ui.kernels.fluid_modes).

        **Inputs**

        - `mode` : int ; `x`, `v` : vec dim f32 field (cap,) ; `alive` : i32 field (cap,) ; `sc` : f32 field (cap,)
        - `rho_over_dt`, `rho0`, `inv_dx` : float

        **Outputs**

        - sc written in place (dens must be up to date for the density mode, nu_c for the viscosity mode)
        """
        n = ti.static(self.n)
        dim = ti.static(self.dim)
        for p in x:
            if alive[p] == 1:
                s = 0.0
                if mode == 1:
                    s = v[p].norm()
                elif mode == 2:
                    s = v[p][0]
                elif mode == 3:
                    s = v[p][1]
                elif mode == 8:
                    if ti.static(dim == 3):
                        s = v[p][2]
                else:
                    c = cell_of(x[p], inv_dx, n)
                    ci = ti.math.clamp(c, 1, ti.Vector(n) - 2)
                    if mode == 4:
                        s = self.q[ci] * rho_over_dt
                    elif mode == 5:
                        g = ti.Matrix.zero(ti.f32, dim, dim)                 # g[a, b] = d u_b / d x_a
                        for a in ti.static(range(dim)):
                            ea = ti.Vector.unit(dim, a, ti.i32)
                            for b in ti.static(range(dim)):
                                eb = ti.Vector.unit(dim, b, ti.i32)
                                up = 0.5 * (self.uf[b][ci + ea] + self.uf[b][ci + ea + eb])
                                um = 0.5 * (self.uf[b][ci - ea] + self.uf[b][ci - ea + eb])
                                g[a, b] = (up - um) * 0.5 * inv_dx
                        if ti.static(dim == 2):
                            s = g[0, 1] - g[1, 0]
                        else:
                            s = ti.Vector([g[1, 2] - g[2, 1], g[2, 0] - g[0, 2], g[0, 1] - g[1, 0]]).norm()
                    elif mode == 6:
                        s = rho0 * (self.dens[c] - 1.0)
                    elif mode == 7:
                        s = self.nu_c[c]
                sc[p] = s


@ti.kernel
def cg_residual(cg: ti.template()) -> ti.f32:
    """Current CG residual rr.

    **Inputs**

    - `cg` : f32 field (3,) [rr, pAp, rr_new]

    **Outputs**

    - f32 cg[0]
    """
    return cg[0]


@ti.kernel
def max_speed(vel: ti.template(), alive: ti.template()) -> ti.f32:
    """Max particle speed.

    **Inputs**

    - `vel` : vec dim f32 field (cap,) velocities
    - `alive` : i32 field (cap,) 1 = live particle

    **Outputs**

    - f32 max |vel|
    """
    m = 0.0
    for p in vel:
        if alive[p] == 1:
            ti.atomic_max(m, vel[p].norm())
    return m

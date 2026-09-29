"""Gradient conjugué préconditionné par multigrille (MGPCG, McAdams et al. 2010) pour le laplacien des cellules.

Même opérateur que kernels_inc.apply_A (5 points non mis à l'échelle, air : Dirichlet 0, solide : exclu) sur une
hiérarchie de grilles n, n/2, n/4... Cellule grossière : AIR si un enfant est de l'air, sinon FLUID si un enfant
est fluide, sinon SOLID. Restriction = somme des 4 enfants du résidu (l'opérateur non mis à l'échelle grossit de
4 par niveau), prolongation = injection constante (transposée de la restriction). Lissage Gauss-Seidel rouge-noir,
rouge puis noir à la descente, noir puis rouge à la remontée : V-cycle symétrique, donc préconditionneur SPD.

Le coût est dominé par les lancements de kernels (~0.1 ms chacun) : une itération PCG = 2 L + 1 kernels (un par
niveau et par phase du V-cycle, plus A p et la direction). Ne pas dérouler tout le V-cycle dans un seul kernel :
la compilation Taichi explose avec le nombre de boucles (7 niveaux : plus de 10 min). Aucune lecture GPU -> CPU.
Le vecteur `cg` contient [r·z, pAp, r·z nouveau].
"""
import taichi as ti

from ui.kernels_inc import AIR, FLUID, SOLID, apply_A


@ti.func
def _zero(x):
    """Zero field x."""
    for I in ti.grouped(x):
        x[I] = 0.0


@ti.func
def _gs_cell(x, b, ct, i, j):
    """Gauss-Seidel update of fluid cell (i, j) for A x = b (air neighbors = 0, solids excluded)."""
    if ct[i, j] == FLUID:
        deg = 0
        s = 0.0
        for di, dj in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
            ni, nj = i + di, j + dj
            if 0 <= ni < x.shape[0] and 0 <= nj < x.shape[1]:
                t = ct[ni, nj]
                if t < SOLID:
                    deg += 1
                    if t == FLUID:
                        s += x[ni, nj]
        if deg > 0:
            x[i, j] = (b[i, j] + s) / deg


@ti.func
def _smooth(x, b, ct, color: ti.template()):
    """One Gauss-Seidel half sweep on cells with (i + j) % 2 == color (A x = b, air neighbors = 0).

    **Inputs**

    - `x`, `b` : f32 fields, level grid ; `ct` : i32 field, level cell types
    - `color` : 0 red, 1 black

    **Outputs**

    - x updated in place on fluid cells of that color
    """
    for i, j in x:
        if (i + j) % 2 == color:
            _gs_cell(x, b, ct, i, j)


@ti.func
def _restrict_types(ctf, ctc):
    """Coarse cell types from the 4 children (air wins, then fluid, else solid)."""
    for I, J in ctc:
        air, fluid = 0, 0
        for a, c in ti.static(ti.ndrange(2, 2)):
            i, j = 2 * I + a, 2 * J + c
            if i < ctf.shape[0] and j < ctf.shape[1]:
                t = ctf[i, j]
                if t == AIR:
                    air = 1
                elif t == FLUID:
                    fluid = 1
        ctc[I, J] = AIR if air == 1 else (FLUID if fluid == 1 else SOLID)


@ti.func
def _restrict(xf, bf, ctf, bc, ctc):
    """Coarse rhs = sum of the fine residuals b - A x over the 4 children."""
    for I, J in bc:
        s = 0.0
        if ctc[I, J] == FLUID:
            for a, c in ti.static(ti.ndrange(2, 2)):
                i, j = 2 * I + a, 2 * J + c
                if i < xf.shape[0] and j < xf.shape[1]:
                    if ctf[i, j] == FLUID:
                        s += bf[i, j] - apply_A(xf, ctf, i, j, xf.shape[0], xf.shape[1])
        bc[I, J] = s


@ti.func
def _prolong(xf, ctf, xc):
    """Add the coarse correction to the fine fluid cells (constant injection)."""
    for i, j in xf:
        if ctf[i, j] == FLUID:
            xf[i, j] += xc[i // 2, j // 2]


@ti.data_oriented
class MGPCG:
    def __init__(self, fb, ctype, r, pd, Ap, cg, nx: int, ny: int, smooth: int = 2, coarse: int = 10,
                 max_coarse: int = 16, max_levels: int = 10):
        """Allocate the multigrid hierarchy on the solver's field builder.

        **Inputs**

        - `fb` : ti.FieldsBuilder   solver field tree (finalized by the caller)
        - `ctype`, `r`, `pd`, `Ap` : fields (nx, ny) cell types, CG residual, direction, A direction
        - `cg` : f32 field (3,) CG scalars
        - `nx`, `ny` : int fine grid
        - `smooth` : int red-black sweeps before and after each level
        - `coarse` : int symmetric sweep pairs on the coarsest level (serial)
        - `max_coarse`, `max_levels` : int coarsen until the coarsest level has <= max_coarse cells

        **Outputs**

        - self.L levels, self.z preconditioned residual (nx, ny)
        """
        shapes = [(nx, ny)]
        while len(shapes) < max_levels and shapes[-1][0] * shapes[-1][1] > max_coarse and min(shapes[-1]) >= 2:
            a, b = shapes[-1]
            shapes.append(((a + 1) // 2, (b + 1) // 2))
        self.L, self.shapes = len(shapes), shapes
        self.smooth, self.coarse = smooth, coarse
        self.z = ti.field(ti.f32)
        fb.dense(ti.ij, shapes[0]).place(self.z)
        self.ct, self.x, self.b = [ctype], [self.z], [r]
        for s in shapes[1:]:
            ct, x, b = ti.field(ti.i32), ti.field(ti.f32), ti.field(ti.f32)
            fb.dense(ti.ij, s).place(ct, x, b)
            self.ct.append(ct)
            self.x.append(x)
            self.b.append(b)
        self.r, self.pd, self.Ap, self.cg = r, pd, Ap, cg

    @ti.kernel
    def _down(self, l: ti.template()):
        """V-cycle descent at level l: x_l = 0, pre-smoothing (red then black), restriction of the residual."""
        _zero(self.x[l])
        for _ in ti.static(range(self.smooth)):
            _smooth(self.x[l], self.b[l], self.ct[l], 0)
            _smooth(self.x[l], self.b[l], self.ct[l], 1)
        _restrict(self.x[l], self.b[l], self.ct[l], self.b[l + 1], self.ct[l + 1])

    @ti.kernel
    def _coarse(self):
        """Coarsest level (<= max_coarse cells): `coarse` symmetric Gauss-Seidel sweeps from zero, in one serial loop.

        **Note** : a serial loop instead of unrolled parallel sweeps: 40 unrolled sweeps took 25 s to compile. Serial
        cost ~0.3 us per cell update (256 cells x 20 sweeps = 1.6 ms), hence the tiny coarsest level.
        """
        x, b, ct = ti.static(self.x[self.L - 1], self.b[self.L - 1], self.ct[self.L - 1])
        _zero(x)
        n0, n1 = ti.static(x.shape[0], x.shape[1])
        ti.loop_config(serialize=True)
        for k in range(2 * self.coarse * n0 * n1):     # balayage avant puis arrière, alternés : symétrique
            s, c = k // (n0 * n1), k % (n0 * n1)
            if s % 2 == 1:
                c = n0 * n1 - 1 - c
            _gs_cell(x, b, ct, c // n1, c % n1)

    @ti.kernel
    def _up(self, l: ti.template()):
        """V-cycle ascent at level l: prolongation of the coarse correction, post-smoothing (black then red)."""
        _prolong(self.x[l], self.ct[l], self.x[l + 1])
        for _ in ti.static(range(self.smooth)):
            _smooth(self.x[l], self.b[l], self.ct[l], 1)
            _smooth(self.x[l], self.b[l], self.ct[l], 0)

    def _vcycle(self) -> None:
        """z = M r : one symmetric V-cycle from zero (2 L - 1 small kernels: fast to compile at any depth)."""
        for l in range(self.L - 1):
            self._down(l)
        self._coarse()
        for l in reversed(range(self.L - 1)):
            self._up(l)

    @ti.kernel
    def _types(self):
        """Coarse cell types of every level from the fine ctype."""
        for l in ti.static(range(self.L - 1)):
            _restrict_types(self.ct[l], self.ct[l + 1])

    def init(self) -> None:
        """After cg_init / density_cg_init (r = b - A x): coarse types, z = M r, p = z, cg[0] = r·z.

        **Outputs**

        - coarse ctype, z, pd, cg[0] written in place
        """
        self._types()
        self._vcycle()
        self._init_dir()

    def iterate(self, x) -> None:
        """One PCG iteration on unknown x (pressure q or density potential phi).

        **Inputs**

        - `x` : f32 field (nx, ny) unknown, updated in place

        **Outputs**

        - x, r, pd, Ap, z, cg written in place

        **Note** : alpha, beta computed on GPU (no read-back); 2 L + 1 kernel launches.
        """
        self._step(x)
        self._vcycle()
        self._dir()

    @ti.func
    def _remove_null(self):
        """No air cell (closed full domain, pure Neumann): the Laplacian is singular, remove the mean of z.

        **Note** : without it the V-cycle feeds the constant mode into the CG, which blows up to NaN.
        """
        ct = ti.static(self.ct[0])
        n_air, n_f, sz = 0, 0, 0.0
        for i, j in ct:
            if ct[i, j] == AIR:
                n_air += 1
            elif ct[i, j] == FLUID:
                n_f += 1
                sz += self.z[i, j]
        shift = 0.0
        if n_air == 0:
            shift = sz / ti.max(n_f, 1)
        for i, j in ct:
            if ct[i, j] == FLUID:
                self.z[i, j] -= shift

    @ti.kernel
    def _init_dir(self):
        """p = z, cg[0] = r·z."""
        self._remove_null()
        rz = 0.0
        for i, j in self.z:
            if self.ct[0][i, j] == FLUID:
                self.pd[i, j] = self.z[i, j]
                rz += self.r[i, j] * self.z[i, j]
            else:
                self.pd[i, j] = 0.0
        self.cg[0] = rz

    @ti.kernel
    def _step(self, x: ti.template()):
        """PCG: Ap, alpha = r·z / pAp, x += alpha p, r -= alpha Ap."""
        ct = ti.static(self.ct[0])
        pAp = 0.0
        for i, j in ct:
            if ct[i, j] == FLUID:
                self.Ap[i, j] = apply_A(self.pd, ct, i, j, ct.shape[0], ct.shape[1])
                pAp += self.pd[i, j] * self.Ap[i, j]
        self.cg[1] = pAp
        alpha = 0.0                                    # convergé au bruit f32 près (pAp <= 0) : on n'avance plus
        if self.cg[1] > 1e-30 and self.cg[0] > 0.0:
            alpha = self.cg[0] / self.cg[1]
        for i, j in ct:
            if ct[i, j] == FLUID:
                x[i, j] += alpha * self.pd[i, j]
                self.r[i, j] -= alpha * self.Ap[i, j]

    @ti.kernel
    def _dir(self):
        """PCG: rz_new = r·z, beta = rz_new / rz, p = z + beta p."""
        self._remove_null()
        ct = ti.static(self.ct[0])
        rz = 0.0
        for i, j in ct:
            if ct[i, j] == FLUID:
                rz += self.r[i, j] * self.z[i, j]
        self.cg[2] = rz
        beta = 0.0
        if self.cg[0] > 1e-30 and self.cg[2] > 0.0:
            beta = self.cg[2] / self.cg[0]
        for i, j in ct:
            if ct[i, j] == FLUID:
                self.pd[i, j] = self.z[i, j] + beta * self.pd[i, j]
        self.cg[0] = self.cg[2]

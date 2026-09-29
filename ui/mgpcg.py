"""Gradient conjugué préconditionné par multigrille (MGPCG, McAdams et al. 2010) pour le laplacien des cellules,
2D ou 3D.

Même opérateur que kernels_inc.apply_A (2·dim + 1 points non mis à l'échelle, air : Dirichlet 0, solide : exclu) sur
une hiérarchie de grilles n, n/2, n/4... Cellule grossière : AIR si un enfant est de l'air, sinon FLUID si un enfant
est fluide, sinon SOLID. Restriction = somme des 2^dim enfants du résidu × 2^(2-dim) : l'opérateur non mis à
l'échelle grossit de 4 par niveau quelle que soit la dimension, la somme des enfants de 2^dim (d'où 0.5 en 3D).
Prolongation = injection constante (transposée de la restriction, au facteur près). Lissage Gauss-Seidel
rouge-noir, rouge puis noir à la descente, noir puis rouge à la remontée : V-cycle symétrique, donc
préconditionneur SPD.

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
def _gs_cell(x, b, ct, I):
    """Gauss-Seidel update of fluid cell I for A x = b (air neighbors = 0, solids excluded)."""
    n = ti.static(x.shape)
    dim = ti.static(len(n))
    if ct[I] == FLUID:
        deg = 0
        s = 0.0
        for a in ti.static(range(dim)):
            for sg in ti.static((1, -1)):
                J = I + sg * ti.Vector.unit(dim, a, ti.i32)
                if 0 <= J[a] < n[a]:
                    t = ct[J]
                    if t < SOLID:
                        deg += 1
                        if t == FLUID:
                            s += x[J]
        if deg > 0:
            x[I] = (b[I] + s) / deg


@ti.func
def _smooth(x, b, ct, color: ti.template()):
    """One Gauss-Seidel half sweep on cells with (sum of indices) % 2 == color (A x = b, air neighbors = 0).

    **Inputs**

    - `x`, `b` : f32 fields, level grid ; `ct` : i32 field, level cell types
    - `color` : 0 red, 1 black

    **Outputs**

    - x updated in place on fluid cells of that color
    """
    for I in ti.grouped(x):
        if I.sum() % 2 == color:
            _gs_cell(x, b, ct, I)


@ti.func
def _restrict_types(ctf, ctc):
    """Coarse cell types from the 2^dim children (air wins, then fluid, else solid)."""
    nf = ti.static(ctf.shape)
    dim = ti.static(len(nf))
    for Ic in ti.grouped(ctc):
        air, fluid = 0, 0
        for o in ti.static(ti.grouped(ti.ndrange(*([2] * dim)))):
            If = 2 * Ic + o
            inside = True
            for c in ti.static(range(dim)):
                inside = inside and If[c] < nf[c]
            if inside:
                t = ctf[If]
                if t == AIR:
                    air = 1
                elif t == FLUID:
                    fluid = 1
        ctc[Ic] = AIR if air == 1 else (FLUID if fluid == 1 else SOLID)


@ti.func
def _residual(xf, bf, ctf, rf):
    """Fine residual rf = b - A x on fluid cells (0 elsewhere) : one apply_A per cell."""
    for I in ti.grouped(rf):
        v = 0.0
        if ctf[I] == FLUID:
            v = bf[I] - apply_A(xf, ctf, I)
        rf[I] = v


@ti.func
def _restrict(rf, bc, ctc):
    """Coarse rhs = 2^(2-dim) × sum of the fine residuals over the 2^dim children.

    **Note** : the residual is precomputed (_residual) : inlining apply_A for each of the 8 children made the 3D
    kernel take ~25 s to compile.
    """
    nf = ti.static(rf.shape)
    dim = ti.static(len(nf))
    for Ic in ti.grouped(bc):
        s = 0.0
        if ctc[Ic] == FLUID:
            for o in ti.static(ti.grouped(ti.ndrange(*([2] * dim)))):
                If = ti.min(2 * Ic + o, ti.Vector(nf) - 1)
                inside = True
                for c in ti.static(range(dim)):
                    inside = inside and 2 * Ic[c] + o[c] < nf[c]
                if inside:
                    s += rf[If]
        bc[Ic] = s * ti.static(2.0 ** (2 - dim))


@ti.func
def _prolong(xf, ctf, xc):
    """Add the coarse correction to the fine fluid cells (constant injection)."""
    for I in ti.grouped(xf):
        if ctf[I] == FLUID:
            xf[I] += xc[I // 2]


@ti.data_oriented
class MGPCG:
    def __init__(self, fb, ctype, r, pd, Ap, cg, shape, smooth: int = 2, coarse: int = 10,
                 max_coarse: int = 16, max_levels: int = 10):
        """Allocate the multigrid hierarchy on the solver's field builder.

        **Inputs**

        - `fb` : ti.FieldsBuilder   solver field tree (finalized by the caller)
        - `ctype`, `r`, `pd`, `Ap` : fields (shape) cell types, CG residual, direction, A direction
        - `cg` : f32 field (3,) CG scalars
        - `shape` : tuple[int, ...] fine grid, 2 or 3 axes
        - `smooth` : int red-black sweeps before and after each level
        - `coarse` : int symmetric sweep pairs on the coarsest level (serial)
        - `max_coarse`, `max_levels` : int coarsen until the coarsest level has <= max_coarse cells

        **Outputs**

        - self.L levels, self.z preconditioned residual (shape)
        """
        shapes = [tuple(int(v) for v in shape)]
        while len(shapes) < max_levels and _cells(shapes[-1]) > max_coarse and min(shapes[-1]) >= 2:
            shapes.append(tuple((v + 1) // 2 for v in shapes[-1]))
        ax = ti.ij if len(shapes[0]) == 2 else ti.ijk
        self.L, self.shapes = len(shapes), shapes
        self.smooth, self.coarse = smooth, coarse
        self.z = ti.field(ti.f32)
        self.rs = [ti.field(ti.f32) for _ in shapes[:-1]]   # résidu de chaque niveau (sauf le plus grossier)
        fb.dense(ax, shapes[0]).place(self.z)
        if len(shapes) > 1:
            fb.dense(ax, shapes[0]).place(self.rs[0])
        self.ct, self.x, self.b = [ctype], [self.z], [r]
        for li, s in enumerate(shapes[1:], start=1):
            ct, x, b = ti.field(ti.i32), ti.field(ti.f32), ti.field(ti.f32)
            fb.dense(ax, s).place(ct, x, b)
            if li < len(shapes) - 1:
                fb.dense(ax, s).place(self.rs[li])
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
        _residual(self.x[l], self.b[l], self.ct[l], self.rs[l])
        _restrict(self.rs[l], self.b[l + 1], self.ct[l + 1])

    @ti.kernel
    def _coarse(self):
        """Coarsest level (<= max_coarse cells): `coarse` symmetric Gauss-Seidel sweeps from zero, in one serial loop.

        **Note** : a serial loop instead of unrolled parallel sweeps: 40 unrolled sweeps took 25 s to compile. Serial
        cost ~0.3 us per cell update (256 cells x 20 sweeps = 1.6 ms), hence the tiny coarsest level.
        """
        x, b, ct = ti.static(self.x[self.L - 1], self.b[self.L - 1], self.ct[self.L - 1])
        _zero(x)
        n = ti.static(x.shape)
        dim = ti.static(len(n))
        nc = ti.static(_cells(n))
        ti.loop_config(serialize=True)
        for k in range(2 * self.coarse * nc):          # balayage avant puis arrière, alternés : symétrique
            s, c = k // nc, k % nc
            if s % 2 == 1:
                c = nc - 1 - c
            I = ti.Vector.zero(ti.i32, dim)             # indice linéaire -> (i, j[, k]), dernier axe le plus rapide
            for a in ti.static(range(dim)):
                stride = ti.static(_cells(n[a + 1:]))
                I[a] = (c // stride) % n[a]
            _gs_cell(x, b, ct, I)

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

        - `x` : f32 field (shape) unknown, updated in place

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
        for I in ti.grouped(ct):
            if ct[I] == AIR:
                n_air += 1
            elif ct[I] == FLUID:
                n_f += 1
                sz += self.z[I]
        shift = 0.0
        if n_air == 0:
            shift = sz / ti.max(n_f, 1)
        for I in ti.grouped(ct):
            if ct[I] == FLUID:
                self.z[I] -= shift

    @ti.kernel
    def _init_dir(self):
        """p = z, cg[0] = r·z."""
        self._remove_null()
        rz = 0.0
        for I in ti.grouped(self.z):
            if self.ct[0][I] == FLUID:
                self.pd[I] = self.z[I]
                rz += self.r[I] * self.z[I]
            else:
                self.pd[I] = 0.0
        self.cg[0] = rz

    @ti.kernel
    def _step(self, x: ti.template()):
        """PCG: Ap, alpha = r·z / pAp, x += alpha p, r -= alpha Ap."""
        ct = ti.static(self.ct[0])
        pAp = 0.0
        for I in ti.grouped(ct):
            if ct[I] == FLUID:
                self.Ap[I] = apply_A(self.pd, ct, I)
                pAp += self.pd[I] * self.Ap[I]
        self.cg[1] = pAp
        alpha = 0.0                                    # convergé au bruit f32 près (pAp <= 0) : on n'avance plus
        if self.cg[1] > 1e-30 and self.cg[0] > 0.0:
            alpha = self.cg[0] / self.cg[1]
        for I in ti.grouped(ct):
            if ct[I] == FLUID:
                x[I] += alpha * self.pd[I]
                self.r[I] -= alpha * self.Ap[I]

    @ti.kernel
    def _dir(self):
        """PCG: rz_new = r·z, beta = rz_new / rz, p = z + beta p."""
        self._remove_null()
        ct = ti.static(self.ct[0])
        rz = 0.0
        for I in ti.grouped(ct):
            if ct[I] == FLUID:
                rz += self.r[I] * self.z[I]
        self.cg[2] = rz
        beta = 0.0
        if self.cg[0] > 1e-30 and self.cg[2] > 0.0:
            beta = self.cg[2] / self.cg[0]
        for I in ti.grouped(ct):
            if ct[I] == FLUID:
                self.pd[I] = self.z[I] + beta * self.pd[I]
        self.cg[0] = self.cg[2]


def _cells(shape) -> int:
    """Number of cells of a grid shape (1 for an empty tuple)."""
    m = 1
    for v in shape:
        m *= int(v)
    return m

"""SimulationRunner : définit une simulation (paramètres + matrices nx × ny + parois), la sauve en JSON, la lance.

    from ui.runner import SimulationRunner
    r = SimulationRunner(Lx=4.0, Ly=1.0, nx=256, gravity=9.81)   # boîte 4 m × 1 m, dx = Lx / nx, ny déduit
    X, Y = r.centers()                      # centres des cellules (m), (nx, ny), X[i, j] = (i + 0.5) dx
    r.set_fluid(Y < 0.2)                    # masque numpy (nx, ny) : cellules pleines d'eau
    r.set_solid(r.rect(2.2, 0.2, 2.3, 0.55))
    r.set_obstacle(r.circle(1.4, 0.28, 0.05))
    r.set_wall("left", "inlet", velocity=(3.0, 0.0), span=(0.25, 0.45))   # condition limite sur un mur (m, m/s)
    r.set_wall("right", "outlet")           # tout le mur droit
    r.set_wall("bottom", "wall", friction=1.0)   # fond adhérent (0 = glissant, défaut)
    r.save("canal.json")                    # paramètres + matrices + parois, un seul fichier
    s = r.run(150, callback=lambda s, k: print(k, s.stats()["n_fluid"]))   # sans fenêtre
    r.show()                                # interface Qt

Conventions : unités physiques SI (m, s, kg, Pa). Boîte Lx × Ly (m), cellules carrées dx = Lx / nx, ny = Ly / dx
arrondi (r.Ly est la hauteur réelle de la grille), matrices indexées [i, j] = (x, y) comme les champs Taichi
(cells[:, 0] est la rangée du bas). Anciens paramètres sans dimensions (n, ou nx + ny) : domaine normalisé
Lx = nx / max(nx, ny), Ly = ny / max(nx, ny), comme avant. Un seul matériau fluide et un seul solide (paramètres plats).

Les conditions aux limites n'existent que sur les parois du domaine : une zone de mur (côté, étendue le long du
mur en mètres, type entrée ou sortie, vitesse) ; la paroi par défaut est un mur glissant. Un obstacle collé à
une entrée / sortie en devient la surface : sa face de même normale que le mur porte la condition (voir
Solver/walls.py).

3D (dim=3) : boîte Lx × Ly × Lz, y vertical (gravité), z profondeur ; matrices (nx, ny, nz), vitesses à 3
composantes ; rect / circle deviennent des extrusions sur toute la profondeur, box / sphere / cylinder sont 3D ;
zones de mur rectangulaires, span=((a0, a1), (b0, b1)) sur les deux axes tangents (ordre croissant : (y, z) pour
left / right, (x, z) pour bottom / top, (x, y) pour back / front).

    r = SimulationRunner(dim=3, Lx=2.0, Ly=1.0, Lz=0.5, nx=128, incompressible=True)
    r.set_fluid(r.box((0.1, 0.05, 0.05), (0.8, 0.6, 0.45)))
    r.add_prim("sphere", "obstacle", center=(1.4, 0.3, 0.25), radius=0.12)     # primitive éditable dans l'UI
    r.add_prim("mesh", "obstacle", path="piece.stl", scale=0.001, translate=(1.0, 0.4, 0.25))
    r.set_wall("left", "inlet", velocity=(1.0, 0, 0), span=((0.1, 0.4), (0.1, 0.4)))

Primitives (self.prims) : boîtes, sphères, cylindres, maillages, appliqués dans l'ordre PAR-DESSUS les matrices
(la dernière l'emporte) ; c'est la scène que l'UI 3D édite. masks() rend les matrices effectives.
"""
from __future__ import annotations

import base64
import json

import numpy as np

from Solver.walls import SIDES, WallTable, Walls, frame, span_box  # noqa: F401  (SIDES réexporté pour ui.ui ; numpy seul)

MATERIALS = ("fluid", "solid", "obstacle", "clear", "rotor")
PRIM_KINDS = ("box", "sphere", "cylinder", "mesh")


class SimulationRunner:
    PARAMS = {
        "dim": 2, "Lx": 1.0, "Ly": 1.0, "Lz": 1.0, "nx": 128, "bound": 3, "ppc": 2, "cfl": 0.4, "gravity": 9.81, "substeps": 20, "seed": 0,
        "capacity": 0, "res": 700,
        "fluid_rho": 1.0, "fluid_E": 400.0, "fluid_nu": 0.0, "smagorinsky": 0.0, "visc_iters": 20,
        "solid_rho": 2.0, "solid_E": 3000.0, "solid_nu": 0.3,
        "eps0": 0.05, "epsf": 0.2, "tau_D": 2e-3, "k_res": 1e-3,
        "use_damage": True, "use_rupture": True, "color_mode": 0,
        "incompressible": False, "cg_iters": 150, "multigrid": True, "free_surface": True, "volume_correction": 1.0,
        "density_iters": 20, "obstacle_friction": 1.0,
    }
    LABELS = {
        "dim": "Dimension (2 ou 3)", "Lz": "Boîte : Lz (m, 3D)",
        "Lx": "Boîte : Lx (m)", "Ly": "Boîte : Ly (m)", "nx": "Grille : nx (cellules en x ; ny déduit)", "bound": "Cellules de bord", "ppc": "Particules / côté de cellule",
        "cfl": "CFL", "gravity": "Gravité g", "substeps": "Sous-pas par image", "seed": "Graine",
        "capacity": "Capacité fluide (0 = auto)", "res": "Résolution du rendu",
        "fluid_rho": "Fluide : densité", "fluid_E": "Fluide : raideur E",
        "fluid_nu": "Fluide : viscosité ν (m²/s, incompressible)", "visc_iters": "Viscosité : itérations CG",
        "smagorinsky": "Fluide : Smagorinsky C_s (ν_t = (C_s dx)² |S|, 0 = aucun, incompressible)",
        "solid_rho": "Solide : densité", "solid_E": "Solide : module E", "solid_nu": "Solide : Poisson ν",
        "eps0": "ε0 (début endommagement)", "epsf": "εf (rupture)", "tau_D": "τ_D", "k_res": "Raideur résiduelle",
        "use_damage": "Endommagement", "use_rupture": "Rupture", "color_mode": "Couleur solide",
        "incompressible": "Fluide incompressible", "cg_iters": "CG : nombre d'itérations ",
        "multigrid": "CG : préconditionneur multigrille",
        "free_surface": "Surface libre (cellule vide : p = 0)",
        "volume_correction": "Correction de densité (positions, 0 = aucune)",
        "density_iters": "Correction de densité : itérations CG",
        "obstacle_friction": "Obstacles : frottement β (0 glissant, 1 adhérent)",
    }
    STRUCTURAL = ("dim", "Lx", "Ly", "Lz", "nx", "bound", "ppc", "capacity", "seed", "res", "incompressible")
    BOOL = ("fluid", "solid", "obstacle")
    FLOAT = ("vx0", "vy0", "vz0")

    def __init__(self, **params):
        """Create a simulation with default parameters overridden by `params`, empty matrices and walls.

        **Inputs**

        - **params  parameter overrides (keys of PARAMS: Lx, Ly in m, nx ; legacy n or nx + ny without
          dimensions = normalized domain Lx = nx / max(nx, ny), Ly = ny / max(nx, ny))
        """
        if "n" in params:                           # grille carrée (anciens scripts / fichiers) : n = nx = ny
            n = int(params.pop("n"))
            params.setdefault("nx", n)
            params.setdefault("ny", n)
        if "ny" in params:                          # ny donné : sans dimensions, domaine normalisé comme avant
            ny = int(params.pop("ny"))
            nx = int(params.get("nx", self.PARAMS["nx"]))
            if "Lx" not in params and "Ly" not in params:
                params["Lx"], params["Ly"] = nx / max(nx, ny), ny / max(nx, ny)
            else:
                params.setdefault("Ly", ny * params.get("Lx", self.PARAMS["Lx"]) / nx)
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise KeyError(f"paramètres inconnus : {sorted(unknown)} (connus : {sorted(self.PARAMS)})")
        self.p = {**self.PARAMS, **params}
        if int(self.p["dim"]) not in (2, 3):
            raise ValueError(f"dim doit valoir 2 ou 3 (reçu {self.p['dim']})")
        shape = self.shape
        self.m = {k: np.zeros(shape, bool) for k in self.BOOL}
        self.m.update({k: np.zeros(shape, np.float32) for k in self.FLOAT})
        self._walls = Walls()                       # zones de paroi, la dernière l'emporte
        self.consts: dict[str, float] = {}          # constantes des conditions limites données par expression
        self.prims: list[dict] = []                 # primitives (appliquées par-dessus self.m, voir masks())
        self._mesh_cache: dict = {}                 # voxelisations de maillages (clé : paramètres + grille)

    @property
    def walls(self) -> list[dict]:
        """Wall segments.

        **Outputs**

        - list[dict]   editable in place
        """
        return self._walls.segments

    @walls.setter
    def walls(self, segments: list[dict]) -> None:
        """Replace the wall segments.

        **Inputs**

        - `segments` : list[dict]
        """
        self._walls.segments = segments

    # ------------------------------------------------------------ géométrie
    @property
    def nx(self) -> int:
        """Number of cells in x (int)."""
        return int(self.p["nx"])

    @property
    def ny(self) -> int:
        """Number of cells in y, round(Ly / dx) (int)."""
        return max(1, int(round(float(self.p["Ly"]) / self.dx)))

    @property
    def dim(self) -> int:
        """Dimension, 2 or 3 (int)."""
        return int(self.p["dim"])

    @property
    def nz(self) -> int:
        """Number of cells in z, round(Lz / dx) in 3D, 1 in 2D (int)."""
        return max(1, int(round(float(self.p["Lz"]) / self.dx))) if self.dim == 3 else 1

    @property
    def shape(self) -> tuple:
        """Grid shape (nx, ny) or (nx, ny, nz) (tuple)."""
        return (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)

    @property
    def n(self) -> int:
        """Longest side in cells (int)."""
        return max(self.shape)

    @property
    def dx(self) -> float:
        """Cell size Lx / nx, m (float)."""
        return float(self.p["Lx"]) / self.nx

    @property
    def Lx(self) -> float:
        """Box width nx dx, m (float)."""
        return self.nx * self.dx

    @property
    def Ly(self) -> float:
        """Box height ny dx, m (float) ; may differ slightly from p["Ly"] (rounded to whole cells)."""
        return self.ny * self.dx

    @property
    def Lz(self) -> float:
        """Box depth nz dx, m (float, 3D) ; may differ slightly from p["Lz"] (rounded to whole cells)."""
        return self.nz * self.dx

    @property
    def extent(self) -> tuple:
        """Box size per axis, m (tuple of dim floats)."""
        return tuple(n * self.dx for n in self.shape)

    @property
    def band(self) -> float:
        """Wall band thickness.

        **Outputs**

        - `float` : bound dx

        **Note** : particles stay in [band, Lx - band] × [band, Ly - band].
        """
        return self.p["bound"] * self.dx

    def centers(self) -> tuple:
        """Cell center coordinates.

        **Outputs**

        - `X`, `Y`[, `Z`] : np.ndarray f64 (shape)   X[i, j(, k)] = (i + 0.5) dx
        """
        axes = [(np.arange(n) + 0.5) * self.dx for n in self.shape]
        return tuple(np.meshgrid(*axes, indexing="ij"))

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Mask of cells whose center lies in a rectangle (3D: extruded over the whole depth).

        **Inputs**

        - `x0`, `y0`, `x1`, `y1` : float   corners in m (any order)

        **Outputs**

        - np.ndarray bool (shape)
        """
        C = self.centers()
        X, Y = C[0], C[1]
        return (X >= min(x0, x1)) & (X <= max(x0, x1)) & (Y >= min(y0, y1)) & (Y <= max(y0, y1))

    def circle(self, cx: float, cy: float, r: float) -> np.ndarray:
        """Mask of cells whose center lies in a disc (3D: cylinder along z over the whole depth).

        **Inputs**

        - `cx`, `cy`, `r` : float   center and radius in m

        **Outputs**

        - np.ndarray bool (shape)
        """
        C = self.centers()
        return (C[0] - cx) ** 2 + (C[1] - cy) ** 2 <= r * r

    def box(self, lo, hi) -> np.ndarray:
        """Mask of cells whose center lies in an axis-aligned box.

        **Inputs**

        - `lo`, `hi` : (dim,) float   corners in m (any order)

        **Outputs**

        - np.ndarray bool (shape)
        """
        C = self.centers()
        m = np.ones(self.shape, bool)
        for a in range(self.dim):
            m &= (C[a] >= min(lo[a], hi[a])) & (C[a] <= max(lo[a], hi[a]))
        return m

    def sphere(self, center, radius: float) -> np.ndarray:
        """Mask of cells whose center lies in a ball (disc in 2D).

        **Inputs**

        - `center` : (dim,) float m ; `radius` : float m

        **Outputs**

        - np.ndarray bool (shape)
        """
        C = self.centers()
        return sum((C[a] - center[a]) ** 2 for a in range(self.dim)) <= radius * radius

    def cylinder(self, p0, p1, radius: float) -> np.ndarray:
        """Mask of cells whose center lies in a finite cylinder of axis p0 -> p1 (3D).

        **Inputs**

        - `p0`, `p1` : (3,) float axis end points, m ; `radius` : float m

        **Outputs**

        - np.ndarray bool (shape)
        """
        C = np.stack(self.centers(), axis=-1)
        p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
        ax = p1 - p0
        L2 = max(float(ax @ ax), 1e-30)
        t = ((C - p0) @ ax) / L2
        d = C - p0 - t[..., None] * ax
        return (t >= 0) & (t <= 1) & (np.einsum("...i,...i", d, d) <= radius * radius)

    def _mask(self, mask) -> np.ndarray:
        """Validate a cell mask.

        **Inputs**

        - `mask` : array-like (nx, ny)

        **Outputs**

        - np.ndarray bool (nx, ny)   (ValueError on wrong shape)
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"masque {mask.shape} attendu {self.shape}")
        return mask

    def _set_vel(self, m: dict, mask: np.ndarray, velocity) -> None:
        """Write an initial velocity (2 or 3 components) into vx0 / vy0 / vz0 of matrices m."""
        for k, v in zip(self.FLOAT[:self.dim], list(velocity) + [0.0] * 3):
            m[k][mask] = v

    # ------------------------------------------------------------ définition (intérieur)
    def set_fluid(self, mask, velocity=(0.0, 0.0)) -> None:
        """Mark cells as fluid (clears solid/obstacle there).

        **Inputs**

        - `mask` : np.ndarray bool (nx, ny)
        - `velocity` : tuple[float, float]   initial velocity

        **Outputs**

        - self.m updated
        """
        mask = self._mask(mask)
        self.m["fluid"][mask], self.m["solid"][mask], self.m["obstacle"][mask] = True, False, False
        self._set_vel(self.m, mask, velocity)

    def set_solid(self, mask, velocity=(0.0, 0.0)) -> None:
        """Mark cells as solid (clears fluid/obstacle there).

        **Inputs**

        - `mask` : np.ndarray bool (nx, ny)
        - `velocity` : tuple[float, float]   initial velocity

        **Outputs**

        - self.m updated
        """
        mask = self._mask(mask)
        self.m["solid"][mask], self.m["fluid"][mask], self.m["obstacle"][mask] = True, False, False
        self._set_vel(self.m, mask, velocity)

    def set_obstacle(self, mask) -> None:
        """Mark cells as obstacle (clears fluid/solid there).

        **Inputs**

        - `mask` : np.ndarray bool (nx, ny)

        **Outputs**

        - self.m updated
        """
        mask = self._mask(mask)
        self.m["obstacle"][mask], self.m["fluid"][mask], self.m["solid"][mask] = True, False, False

    def set_velocity(self, mask, velocity) -> None:
        """Set the initial velocity of particles seeded in these cells.

        **Inputs**

        - `mask` : np.ndarray bool (nx, ny)
        - `velocity` : tuple[float, float]

        **Outputs**

        - self.m["vx0"], self.m["vy0"] updated
        """
        mask = self._mask(mask)
        self._set_vel(self.m, mask, velocity)

    def clear(self, mask) -> None:
        """Clear all materials and initial velocity in these cells.

        **Inputs**

        - `mask` : np.ndarray bool (nx, ny)

        **Outputs**

        - self.m updated
        """
        mask = self._mask(mask)
        for k in self.BOOL:
            self.m[k][mask] = False
        for k in self.FLOAT:
            self.m[k][mask] = 0.0

    # ------------------------------------------------------------ définition (parois)
    def set_wall(self, side: str, kind: str, velocity=(0.0, 0.0), span=None, pressure=None,
                 friction=None) -> None:
        """Set a wall boundary segment.

        **Inputs**

        - `side` : str                   left / right / bottom / top (/ back / front in 3D)
        - `kind` : str                   wall / inlet / outlet
        - `velocity` : tuple of dim float | str   inlet velocity ; str = expression of x, y, z, t
        - `span` : (a, b) in 2D, ((a0, a1), (b0, b1)) in 3D (tangent axes in increasing order), m ;
          None : whole wall
        - `pressure` : float | str | None    outlet pressure (incompressible only; None = 0, free outlet) ;
          str = expression of x, y, t, e.g. "rho*g*(H - y)" (constants: rho, g, pi, Lx, Ly, dx, self.consts)
        - `friction` : float | None          wall beta in [0, 1] (0 slip, 1 no-slip)

        **Outputs**

        - self._walls updated (new segment overrides overlapped ones)

        **Note** : span clipped to the wall size.
        """
        self._walls.set(side, kind, velocity, self._span(side, span), pressure, friction)

    def _span(self, side: str, span):
        """Zone extent clipped to the wall size (None : whole wall).

        **Inputs**

        - `side` : str ; `span` : (a, b) | ((a0, a1), (b0, b1)) | None

        **Outputs**

        - (a, b) in 2D, ((a0, a1), (b0, b1)) in 3D
        """
        if side not in SIDES[:2 * self.dim]:
            raise ValueError(f"côté {side!r} inconnu en {self.dim}D (attendu : {SIDES[:2 * self.dim]})")
        ext = [self.extent[t] for t in frame(SIDES.index(side), self.dim)[2]]
        box = [[0.0, e] for e in ext] if span is None else span_box(span)
        if len(box) != len(ext):
            raise ValueError(f"span {span!r} : {len(ext)} intervalle(s) attendu(s) en {self.dim}D")
        box = [(min(max(iv[0], 0.0), e), min(max(iv[1], 0.0), e)) for iv, e in zip(box, ext)]
        return box[0] if self.dim == 2 else tuple(box)

    def clear_wall(self, side: str, span=None) -> None:
        """Reset part of a wall to the default slip wall.

        **Inputs**

        - `side` : str                   left / right / bottom / top (/ back / front)
        - `span` : zone extent (see set_wall) | None (whole wall)

        **Outputs**

        - self._walls updated
        """
        self._walls.clear(side, self._span(side, span))

    def set_constants(self, **consts) -> None:
        """Define constants usable in boundary expressions (e.g. set_constants(H=0.333, U=1.2)).

        **Inputs**

        - `**consts` : float values by name

        **Outputs**

        - self.consts updated
        """
        self.consts.update({k: float(v) for k, v in consts.items()})

    def all_constants(self) -> dict:
        """Constants available in boundary expressions.

        **Outputs**

        - dict[str, float] : rho, g (from params), pi, Lx, Ly, dx, then self.consts

        **Note** : rho and g are taken when the solver is built (Reset), not updated live.
        """
        return {"rho": float(self.p["fluid_rho"]), "g": float(self.p["gravity"]), "pi": float(np.pi),
                "Lx": self.Lx, "Ly": self.Ly, "Lz": self.Lz if self.dim == 3 else 0.0, "dx": self.dx, **self.consts}

    def wall_table(self) -> WallTable:
        """Rasterize wall segments (expressions without t evaluated here).

        **Outputs**

        - `WallTable` : type, v, depth, pressure, friction, expr per wall cell (2 dim, na, nb[, …]), sources

        **Note** : an obstacle glued to an inlet/outlet carries it on its inward face (primitives included).
        """
        return self._walls.table(self.shape, self.p["bound"], self.masks()["obstacle"], self.all_constants(),
                                 self.dx)

    def resize(self, nx: int, ny: int | None = None) -> None:
        """Change the resolution, box size unchanged.

        **Inputs**

        - `nx` : int          new cells in x
        - `ny` : int | None   new cells in y (then Ly = ny dx) ; None : ny follows from Ly

        **Outputs**

        - self.m and self.p updated (see regrid)
        """
        self.p["nx"] = int(nx)
        if ny is not None:
            self.p["Ly"] = int(ny) * self.dx
        self.regrid()

    def regrid(self) -> None:
        """Resample the matrices to the current grid (after a change of Lx, Ly or nx).

        **Outputs**

        - self.m resampled to (nx, ny) (nearest cell center, same physical position)

        **Note** : walls are in metres and unchanged ; matrices keep their physical position when the box grows.
        """
        old = self.m["fluid"].shape                    # taille réelle des matrices (p peut déjà avoir changé)
        new = self.shape
        if old == new:
            return
        if len(old) != len(new):                       # changement de dimension : on repart de matrices vides
            self.m = {k: np.zeros(new, v.dtype) for k, v in self.m.items()}
            return
        idx = [np.minimum((np.arange(n) * o / n).astype(int), o - 1) for o, n in zip(old, new)]
        self.m = {k: np.ascontiguousarray(v[np.ix_(*idx)]) for k, v in self.m.items()}

    # ------------------------------------------------------------ fichiers
    def to_dict(self) -> dict:
        """Serialize to a JSON-ready dict.

        **Outputs**

        - `dict` : version, params, matrices (base64, empty ones omitted), walls
        """
        mats = {}
        for k, a in self.m.items():
            if not a.any():
                continue                                   # matrices vides omises
            raw = np.packbits(a.ravel()) if a.dtype == bool else a.astype(np.float32).ravel()
            mats[k] = {"dtype": str(a.dtype), "shape": list(a.shape),
                       "data": base64.b64encode(np.ascontiguousarray(raw).tobytes()).decode("ascii")}
        return {"version": 5, "params": dict(self.p), "matrices": mats,
                "walls": [dict(w) for w in self.walls], "consts": dict(self.consts),
                "prims": [dict(q) for q in self.prims]}

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationRunner":
        """Build a runner from a serialized dict.

        **Inputs**

        - `d` : dict   as produced by to_dict (v2 inlet/outlet matrices migrated to walls)

        **Outputs**

        - SimulationRunner
        """
        params = {k: v for k, v in d.get("params", {}).items() if k in cls.PARAMS or k in ("n", "ny")}
        r = cls(**params)
        shape = r.shape
        size = int(np.prod(shape))
        legacy = {}
        for k, spec in d.get("matrices", {}).items():
            buf = base64.b64decode(spec["data"])
            if spec["dtype"] == "bool":
                a = np.unpackbits(np.frombuffer(buf, np.uint8))[:size].astype(bool)
            else:
                a = np.frombuffer(buf, np.float32)
            a = a.reshape(shape)
            if k in r.m:
                r.m[k] = np.ascontiguousarray(a.astype(r.m[k].dtype))
            elif k in ("inlet", "outlet", "inlet_vx", "inlet_vy"):
                legacy[k] = a
        r.walls = [dict(w) for w in d.get("walls", [])]
        r.consts = {k: float(v) for k, v in d.get("consts", {}).items()}
        r.prims = [dict(q) for q in d.get("prims", [])]
        if legacy:
            r._migrate_legacy(legacy)
        return r

    def _migrate_legacy(self, legacy: dict) -> None:
        """Convert v2 inlet/outlet cell bands along the walls into wall segments.

        **Inputs**

        - `legacy` : dict of np.ndarray (nx, ny)   inlet, outlet (bool), inlet_vx, inlet_vy (f32)

        **Outputs**

        - self._walls updated
        """
        nx, ny, bound = self.nx, self.ny, self.p["bound"]
        n = self.n
        depth = bound + 4
        zeros = np.zeros((nx, ny), np.float32)
        inlet = legacy.get("inlet", np.zeros((nx, ny), bool))
        outlet = legacy.get("outlet", np.zeros((nx, ny), bool))
        ivx, ivy = legacy.get("inlet_vx", zeros), legacy.get("inlet_vy", zeros)
        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        dist = {"left": I, "right": nx - 1 - I, "bottom": J, "top": ny - 1 - J}
        owner = np.argmin(np.stack(list(dist.values())), axis=0)   # mur le plus proche (gauche/droite gagnent les égalités)
        for s, (side, d) in enumerate(dist.items()):  # chaque cellule de bord appartient à un seul mur
            zone = (d < depth) & (owner == s)
            axis = 0 if side in ("left", "right") else 1
            for kind, mask in (("outlet", outlet), ("inlet", inlet)):
                sel_side = mask & zone
                along = sel_side.any(axis=axis)
                ln = len(along)
                k = 0
                while k < ln:
                    if not along[k]:
                        k += 1
                        continue
                    k0 = k
                    while k < ln and along[k]:
                        k += 1
                    run = sel_side & ((J >= k0) & (J < k) if axis == 0 else (I >= k0) & (I < k))
                    perp = d[run]
                    if k - k0 <= int(perp.max() - perp.min() + 1) and k - k0 < depth:
                        continue                      # fragment de coin d'une bande perpendiculaire
                    vel = (0.0, 0.0)
                    if kind == "inlet":
                        vel = (float(ivx[run].mean()), float(ivy[run].mean()))
                    self.set_wall(side, kind, velocity=vel, span=(k0 * self.dx, k * self.dx))

    def save(self, path: str) -> None:
        """Save params, matrices and walls to a JSON file.

        **Inputs**

        - `path` : str
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "SimulationRunner":
        """Load a runner from a JSON file.

        **Inputs**

        - `path` : str

        **Outputs**

        - SimulationRunner
        """
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def equals(self, other: "SimulationRunner") -> bool:
        """Compare params, matrices and walls.

        **Inputs**

        - `other` : SimulationRunner

        **Outputs**

        - bool
        """
        return (self.p == other.p and all(np.array_equal(self.m[k], other.m[k]) for k in self.m)
                and self.walls == other.walls and self.consts == other.consts and self.prims == other.prims)

    # ------------------------------------------------------------ exécution
    def solver(self):
        """Build a Solver for this simulation.

        **Outputs**

        - `Solver` : (allocates Taichi fields)
        """
        from ui.solver import Solver
        return Solver({**self.p, "ny": self.ny, "nz": self.nz}, self.masks(), self.wall_table(),
                      self.all_constants(), self.rotor())

    # ------------------------------------------------------------ primitives (scène éditable, 3D surtout)
    def add_prim(self, kind: str, material: str, velocity=(0.0, 0.0, 0.0), **params) -> dict:
        """Append a primitive applied on top of the matrices (the last one wins).

        **Inputs**

        - `kind` : str in box / sphere / cylinder / mesh
        - `material` : str in fluid / solid / obstacle / clear / rotor (rigid obstacle turning at imposed speed)
        - `velocity` : (dim,) float initial velocity (fluid / solid)
        - **params : box lo, hi ; sphere center, radius ; cylinder p0, p1, radius ; mesh path, scale, rotate (deg),
          translate (m, where the pivot lands), pivot (file units, None = mesh center), fill (bool) ; rotor : rpm
          (tr/min, sign = direction), axis ((3,) direction, default y), center (m, default: primitive center)

        **Outputs**

        - dict the stored primitive (editable in place)
        """
        if kind not in PRIM_KINDS:
            raise ValueError(f"primitive {kind!r} inconnue (attendu : {PRIM_KINDS})")
        if material not in MATERIALS:
            raise ValueError(f"matériau {material!r} inconnu (attendu : {MATERIALS})")
        prim = {"kind": kind, "material": material, "velocity": [float(v) for v in velocity],
                **{k: (list(v) if isinstance(v, (tuple, list, np.ndarray)) else v) for k, v in params.items()}}
        self.prims.append(prim)
        return prim

    def prim_mask(self, prim: dict) -> np.ndarray:
        """Cells covered by a primitive.

        **Inputs**

        - `prim` : dict (see add_prim)

        **Outputs**

        - np.ndarray bool (shape)
        """
        k = prim["kind"]
        if k == "box":
            return self.box(prim["lo"], prim["hi"])
        if k == "sphere":
            return self.sphere(prim["center"], prim["radius"])
        if k == "cylinder":
            if self.dim == 2:
                return self.sphere(prim["p0"][:2], prim["radius"])
            return self.cylinder(prim["p0"], prim["p1"], prim["radius"])
        if k == "mesh":
            return self._mesh_mask(prim)
        raise ValueError(f"primitive {k!r} inconnue")

    def mesh_triangles(self, prim: dict) -> np.ndarray:
        """Triangles of a mesh primitive after its transform (m).

        **Inputs**

        - `prim` : dict mesh primitive

        **Outputs**

        - np.ndarray (T, 3, 3)
        """
        from Solver import mesh
        key = ("tri", prim["path"], repr(prim.get("scale", 1.0)), repr(prim.get("rotate", [0, 0, 0])),
               repr(prim.get("translate", [0, 0, 0])), repr(prim.get("pivot")))
        if key not in self._mesh_cache:
            raw = self._mesh_cache.setdefault(("raw", prim["path"]), None)
            if raw is None:
                raw = self._mesh_cache[("raw", prim["path"])] = mesh.load_mesh(prim["path"])
            self._mesh_cache[key] = mesh.transform(raw, prim.get("scale", 1.0), prim.get("rotate", (0, 0, 0)),
                                                   prim.get("translate", (0, 0, 0)), prim.get("pivot"))
        return self._mesh_cache[key]

    def _mesh_mask(self, prim: dict) -> np.ndarray:
        """Voxelized mesh primitive (cached per parameters and grid).

        **Inputs**

        - `prim` : dict mesh primitive

        **Outputs**

        - np.ndarray bool (shape)
        """
        from Solver import mesh
        if self.dim != 3:
            raise ValueError("un maillage n'a de sens qu'en 3D (dim=3)")
        key = ("vox", prim["path"], repr(prim.get("scale", 1.0)), repr(prim.get("rotate", [0, 0, 0])),
               repr(prim.get("translate", [0, 0, 0])), repr(prim.get("pivot")), bool(prim.get("fill", True)),
               self.shape, self.dx)
        if key not in self._mesh_cache:
            self._mesh_cache[key] = mesh.voxelize(self.mesh_triangles(prim), self.shape, self.dx,
                                                  fill=bool(prim.get("fill", True)))
        return self._mesh_cache[key]

    def prim_center(self, prim: dict) -> np.ndarray:
        """Center of a primitive (rotor axis point by default).

        **Inputs**

        - `prim` : dict

        **Outputs**

        - np.ndarray (dim,) m
        """
        k = prim["kind"]
        if "center" in prim and k != "sphere":
            c = prim["center"]
        elif k == "box":
            c = 0.5 * (np.asarray(prim["lo"], float) + np.asarray(prim["hi"], float))
        elif k == "sphere":
            c = prim["center"]
        elif k == "cylinder":
            c = 0.5 * (np.asarray(prim["p0"], float) + np.asarray(prim["p1"], float))
        else:
            c = prim.get("translate", [0.0, 0.0, 0.0])
        return np.asarray(c, np.float64)[:self.dim]

    def rotor(self):
        """The rotating obstacle of the scene (first primitive of material rotor), at angle 0.

        **Outputs**

        - dict (mask bool (shape), center (dim,) m, axis (3,) unit, omega rad/s) | None

        **Note** : one rotor per scene ; its cells are cleared in masks() (no particle seeded inside).
        """
        rot = [q for q in self.prims if q["material"] == "rotor"]
        if not rot:
            return None
        prim = rot[0]
        axis = np.asarray(prim.get("axis", [0.0, 1.0, 0.0]), np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return {"mask": self.prim_mask(prim), "center": self.prim_center(prim), "axis": axis,
                "omega": float(prim.get("rpm", 60.0)) * 2.0 * np.pi / 60.0}

    def masks(self) -> dict:
        """Effective matrices: self.m with the primitives applied in order.

        **Outputs**

        - dict of np.ndarray (shape) : fluid, solid, obstacle (bool), vx0, vy0, vz0 (f32)

        **Note** : a rotor clears its initial cells (it is handled by the solver as a moving obstacle).
        """
        if not self.prims:
            return self.m
        m = {k: v.copy() for k, v in self.m.items()}
        for prim in self.prims:
            mask = self.prim_mask(prim)
            mat = prim["material"]
            for k in self.BOOL:
                m[k][mask] = k == mat
            for k in self.FLOAT:
                m[k][mask] = 0.0
            if mat in ("fluid", "solid"):
                self._set_vel(m, mask, prim.get("velocity", ()))
        return m

    def run(self, frames: int, substeps: int | None = None, callback=None):
        """Run headless for a number of frames.

        **Inputs**

        - `frames` : int
        - `substeps` : int | None             substeps per frame (None: p["substeps"])
        - `callback` : callable(Solver, int)  called after each frame

        **Outputs**

        - Solver
        """
        s = self.solver()
        for k in range(frames):
            s.step(substeps)
            if callback is not None:
                callback(s, k)
        return s

    def show(self, path: str | None = None, vtk_dir: str | None = None, vtk_every: int | None = None) -> int:
        """Open the Qt GUI on this simulation (blocking).

        **Inputs**

        - `path` : str | None   file path shown / used for saving
        - `vtk_dir` : str | None   VTK export folder (export on from the start) ; `vtk_every` : int | None steps

        **Outputs**

        - `int` : Qt exit code
        """
        import sys

        from PySide6.QtWidgets import QApplication

        from ui.ui import UI
        app = QApplication.instance() or QApplication(sys.argv)
        win = UI(self, path=path, vtk_dir=vtk_dir, vtk_every=vtk_every)
        win.show()
        return app.exec()

    # ------------------------------------------------------------ scène de référence
    @classmethod
    def demo(cls) -> "SimulationRunner":
        """Build the reference scene: water block falling on a deck held by two pillars.

        **Outputs**

        - SimulationRunner
        """
        r = cls(n=250)
        r.set_fluid(r.rect(0.03, 0.03, 0.97, 0.22))
        r.set_fluid(r.rect(0.30, 0.62, 0.70, 0.95))
        r.set_solid(r.rect(0.15, 0.40, 0.85, 0.46))
        r.set_obstacle(r.rect(0.15, 0.0, 0.21, 0.40))
        r.set_obstacle(r.rect(0.79, 0.0, 0.85, 0.40))
        return r

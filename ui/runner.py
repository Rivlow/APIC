"""SimulationRunner : définit une simulation (paramètres + matrices nx × ny + parois), la sauve en JSON, la lance.

    from ui.runner import SimulationRunner
    r = SimulationRunner(nx=256, ny=64, gravity=9.81)   # ou n=128 : grille carrée
    X, Y = r.centers()                      # centres des cellules, (nx, ny), X[i, j] = (i + 0.5) dx
    r.set_fluid(Y < 0.2)                    # masque numpy (nx, ny) : cellules pleines d'eau
    r.set_solid(r.rect(0.55, 0.2, 0.58, 0.55))
    r.set_obstacle(r.circle(0.35, 0.28, 0.05))
    r.set_wall("left", "inlet", velocity=(3.0, 0.0), span=(0.25, 0.45))   # condition limite sur un mur
    r.set_wall("right", "outlet")           # tout le mur droit
    r.set_wall("bottom", "wall", friction=1.0)   # fond adhérent (0 = glissant, défaut)
    r.save("canal.json")                    # paramètres + matrices + parois, un seul fichier
    s = r.run(150, callback=lambda s, k: print(k, s.stats()["n_fluid"]))   # sans fenêtre
    r.show()                                # interface Qt

Conventions : cellules carrées dx = 1 / max(nx, ny), domaine [0, nx dx] × [0, ny dx] (inclus dans le carré unité,
r.Lx × r.Ly), matrices indexées [i, j] = (x, y) comme les champs Taichi (cells[:, 0] est la rangée du bas). Un seul matériau fluide et un seul solide (paramètres plats).

Les conditions aux limites n'existent que sur les quatre parois du domaine : un segment de mur
(côté, étendue le long du mur en unités domaine, type entrée ou sortie, vitesse) ; la paroi par
défaut est un mur glissant. Un obstacle collé à une entrée / sortie en devient la surface : sa face de
même normale que le mur porte la condition (voir Solver/walls.py).
"""
from __future__ import annotations

import base64
import json

import numpy as np

from Solver.walls import SIDES, WallTable, Walls  # noqa: F401  (SIDES réexporté pour ui.ui ; numpy seul, sans Taichi)


class SimulationRunner:
    PARAMS = {
        "nx": 128, "ny": 128, "bound": 3, "ppc": 2, "cfl": 0.4, "gravity": 9.81, "substeps": 20, "seed": 0,
        "capacity": 0, "res": 700,
        "fluid_rho": 1.0, "fluid_E": 400.0,
        "solid_rho": 2.0, "solid_E": 3000.0, "solid_nu": 0.3,
        "eps0": 0.05, "epsf": 0.2, "tau_D": 2e-3, "k_res": 1e-3,
        "use_damage": True, "use_rupture": True, "color_mode": 0,
        "incompressible": False, "cg_iters": 150, "free_surface": True, "volume_correction": 1.0,
        "density_iters": 20, "obstacle_friction": 1.0,
    }
    LABELS = {
        "nx": "Grille : nx (cellules en x)", "ny": "Grille : ny (cellules en y)", "bound": "Cellules de bord", "ppc": "Particules / côté de cellule",
        "cfl": "CFL", "gravity": "Gravité g", "substeps": "Sous-pas par image", "seed": "Graine",
        "capacity": "Capacité fluide (0 = auto)", "res": "Résolution du rendu",
        "fluid_rho": "Fluide : densité", "fluid_E": "Fluide : raideur E",
        "solid_rho": "Solide : densité", "solid_E": "Solide : module E", "solid_nu": "Solide : Poisson ν",
        "eps0": "ε0 (début endommagement)", "epsf": "εf (rupture)", "tau_D": "τ_D", "k_res": "Raideur résiduelle",
        "use_damage": "Endommagement", "use_rupture": "Rupture", "color_mode": "Couleur solide",
        "incompressible": "Fluide incompressible", "cg_iters": "CG : nombre d'itérations ",
        "free_surface": "Surface libre (cellule vide : p = 0)",
        "volume_correction": "Correction de densité (positions, 0 = aucune)",
        "density_iters": "Correction de densité : itérations CG",
        "obstacle_friction": "Obstacles : frottement β (0 glissant, 1 adhérent)",
    }
    STRUCTURAL = ("nx", "ny", "bound", "ppc", "capacity", "seed", "res", "incompressible")
    BOOL = ("fluid", "solid", "obstacle")
    FLOAT = ("vx0", "vy0")

    def __init__(self, **params):
        """Create a simulation with default parameters overridden by `params`, empty matrices and walls.

        **Inputs**

        - **params  parameter overrides (keys of PARAMS; n = square grid nx = ny)
        """
        if "n" in params:                           # grille carrée (et anciens fichiers) : n = nx = ny
            n = int(params.pop("n"))
            params.setdefault("nx", n)
            params.setdefault("ny", n)
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise KeyError(f"paramètres inconnus : {sorted(unknown)} (connus : {sorted(self.PARAMS)})")
        self.p = {**self.PARAMS, **params}
        shape = (self.nx, self.ny)
        self.m = {k: np.zeros(shape, bool) for k in self.BOOL}
        self.m.update({k: np.zeros(shape, np.float32) for k in self.FLOAT})
        self._walls = Walls()                       # segments de paroi, le dernier l'emporte

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
        """Number of cells in y (int)."""
        return int(self.p["ny"])

    @property
    def n(self) -> int:
        """Cells per domain unit.

        **Outputs**

        - `int` : max(nx, ny) (dx = 1 / n)
        """
        return max(self.nx, self.ny)

    @property
    def dx(self) -> float:
        """Cell size 1 / max(nx, ny) (float)."""
        return 1.0 / self.n

    @property
    def Lx(self) -> float:
        """Domain width nx dx (float)."""
        return self.nx * self.dx

    @property
    def Ly(self) -> float:
        """Domain height ny dx (float)."""
        return self.ny * self.dx

    @property
    def band(self) -> float:
        """Wall band thickness.

        **Outputs**

        - `float` : bound dx

        **Note** : particles stay in [band, Lx - band] × [band, Ly - band].
        """
        return self.p["bound"] * self.dx

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell center coordinates.

        **Outputs**

        - `X`, `Y` : np.ndarray f64 (nx, ny)   X[i, j] = (i + 0.5) dx
        """
        cx = (np.arange(self.nx) + 0.5) * self.dx
        cy = (np.arange(self.ny) + 0.5) * self.dx
        return np.meshgrid(cx, cy, indexing="ij")

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Mask of cells whose center lies in a rectangle.

        **Inputs**

        - `x0`, `y0`, `x1`, `y1` : float   corners in domain units (any order)

        **Outputs**

        - np.ndarray bool (nx, ny)
        """
        X, Y = self.centers()
        return (X >= min(x0, x1)) & (X <= max(x0, x1)) & (Y >= min(y0, y1)) & (Y <= max(y0, y1))

    def circle(self, cx: float, cy: float, r: float) -> np.ndarray:
        """Mask of cells whose center lies in a disc.

        **Inputs**

        - `cx`, `cy`, `r` : float   center and radius in domain units

        **Outputs**

        - np.ndarray bool (nx, ny)
        """
        X, Y = self.centers()
        return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r

    def _mask(self, mask) -> np.ndarray:
        """Validate a cell mask.

        **Inputs**

        - `mask` : array-like (nx, ny)

        **Outputs**

        - np.ndarray bool (nx, ny)   (ValueError on wrong shape)
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.nx, self.ny):
            raise ValueError(f"masque {mask.shape} attendu ({self.nx}, {self.ny})")
        return mask

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
        self.m["vx0"][mask], self.m["vy0"][mask] = velocity

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
        self.m["vx0"][mask], self.m["vy0"][mask] = velocity

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
        self.m["vx0"][mask], self.m["vy0"][mask] = velocity

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
    def set_wall(self, side: str, kind: str, velocity=(0.0, 0.0), span=(0.0, 1.0), pressure=None,
                 friction=None) -> None:
        """Set a wall boundary segment.

        **Inputs**

        - `side` : str                   left / right / bottom / top
        - `kind` : str                   wall / inlet / outlet
        - `velocity` : tuple[float, float]   inlet velocity
        - `span` : tuple[float, float]   extent along the wall, domain units
        - `pressure` : float | None          outlet pressure (incompressible only; None = 0, free outlet)
        - `friction` : float | None          wall beta in [0, 1] (0 slip, 1 no-slip)

        **Outputs**

        - self._walls updated (new segment overrides overlapped ones)

        **Note** : span clipped to wall length (Ly for left/right, Lx for bottom/top).
        """
        ext = self.Ly if side in ("left", "right") else self.Lx
        span = (min(max(float(span[0]), 0.0), ext), min(max(float(span[1]), 0.0), ext))
        self._walls.set(side, kind, velocity, span, pressure, friction)

    def clear_wall(self, side: str, span=(0.0, 1.0)) -> None:
        """Reset part of a wall to the default slip wall.

        **Inputs**

        - `side` : str                   left / right / bottom / top
        - `span` : tuple[float, float]   extent along the wall, domain units

        **Outputs**

        - self._walls updated
        """
        self._walls.clear(side, span)

    def wall_table(self) -> WallTable:
        """Rasterize wall segments.

        **Outputs**

        - `WallTable` : type, v, depth, pressure, friction per wall cell (4, max(nx, ny)[, 2])

        **Note** : an obstacle glued to an inlet/outlet carries it on its inward face.
        """
        return self._walls.table(self.nx, self.ny, self.p["bound"], self.m["obstacle"])

    def resize(self, nx: int, ny: int | None = None) -> None:
        """Change the grid size.

        **Inputs**

        - `nx` : int          new cells in x
        - `ny` : int | None   new cells in y (None: square nx × nx)

        **Outputs**

        - self.m and self.p["nx"], self.p["ny"] updated

        **Note** : resamples matrices (nearest neighbor), walls unchanged (domain units).
        """
        ny = nx if ny is None else ny
        ox, oy = self.m["fluid"].shape                 # taille réelle des matrices (p peut déjà avoir changé)
        ix = np.minimum((np.arange(nx) * ox / nx).astype(int), ox - 1)
        iy = np.minimum((np.arange(ny) * oy / ny).astype(int), oy - 1)
        self.m = {k: np.ascontiguousarray(v[ix][:, iy]) for k, v in self.m.items()}
        self.p["nx"], self.p["ny"] = int(nx), int(ny)

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
        return {"version": 3, "params": dict(self.p), "matrices": mats,
                "walls": [dict(w) for w in self.walls]}

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationRunner":
        """Build a runner from a serialized dict.

        **Inputs**

        - `d` : dict   as produced by to_dict (v2 inlet/outlet matrices migrated to walls)

        **Outputs**

        - SimulationRunner
        """
        params = {k: v for k, v in d.get("params", {}).items() if k in cls.PARAMS or k == "n"}
        r = cls(**params)
        nx, ny = r.nx, r.ny
        legacy = {}
        for k, spec in d.get("matrices", {}).items():
            buf = base64.b64decode(spec["data"])
            if spec["dtype"] == "bool":
                a = np.unpackbits(np.frombuffer(buf, np.uint8))[:nx * ny].astype(bool)
            else:
                a = np.frombuffer(buf, np.float32)
            a = a.reshape(nx, ny)
            if k in r.m:
                r.m[k] = np.ascontiguousarray(a.astype(r.m[k].dtype))
            elif k in ("inlet", "outlet", "inlet_vx", "inlet_vy"):
                legacy[k] = a
        r.walls = [dict(w) for w in d.get("walls", [])]
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
                    self.set_wall(side, kind, velocity=vel, span=(k0 / n, k / n))

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
                and self.walls == other.walls)

    # ------------------------------------------------------------ exécution
    def solver(self):
        """Build a Solver for this simulation.

        **Outputs**

        - `Solver` : (allocates Taichi fields)
        """
        from ui.solver import Solver
        return Solver(self.p, self.m, self.wall_table())

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

    def show(self, path: str | None = None) -> int:
        """Open the Qt GUI on this simulation (blocking).

        **Inputs**

        - `path` : str | None   file path shown / used for saving

        **Outputs**

        - `int` : Qt exit code
        """
        import sys

        from PySide6.QtWidgets import QApplication

        from ui.ui import UI
        app = QApplication.instance() or QApplication(sys.argv)
        win = UI(self, path=path)
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

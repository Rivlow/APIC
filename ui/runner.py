"""SimulationRunner : définit une simulation (paramètres + matrices n × n + parois), la sauve en JSON, la lance.

    from ui.runner import SimulationRunner
    r = SimulationRunner(n=128, gravity=9.81)
    X, Y = r.centers()                      # centres des cellules, (n, n), X[i, j] = (i + 0.5) dx
    r.set_fluid(Y < 0.2)                    # masque numpy (n, n) : cellules pleines d'eau
    r.set_solid(r.rect(0.55, 0.2, 0.58, 0.55))
    r.set_obstacle(r.circle(0.35, 0.28, 0.05))
    r.set_wall("left", "inlet", velocity=(3.0, 0.0), span=(0.25, 0.45))   # condition limite sur un mur
    r.set_wall("right", "outlet")           # tout le mur droit
    r.save("canal.json")                    # paramètres + matrices + parois, un seul fichier
    s = r.run(150, callback=lambda s, k: print(k, s.stats()["n_fluid"]))   # sans fenêtre
    r.show()                                # interface Qt

Conventions : domaine [0, 1]², matrices indexées [i, j] = (x, y) comme les champs Taichi
(cells[:, 0] est la rangée du bas). Un seul matériau fluide et un seul solide (paramètres plats).

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
        "n": 128, "bound": 3, "ppc": 2, "cfl": 0.4, "gravity": 9.81, "substeps": 20, "seed": 0,
        "capacity": 0, "res": 700,
        "fluid_rho": 1.0, "fluid_E": 400.0,
        "solid_rho": 2.0, "solid_E": 3000.0, "solid_nu": 0.3,
        "eps0": 0.05, "epsf": 0.2, "tau_D": 2e-3, "k_res": 1e-3,
        "use_damage": True, "use_rupture": True, "color_mode": 0,
        "incompressible": False, "cg_iters": 150, "free_surface": True, "volume_correction": 1.0,
        "density_iters": 20,
    }
    LABELS = {
        "n": "Grille n × n", "bound": "Cellules de bord", "ppc": "Particules / côté de cellule",
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
    }
    STRUCTURAL = ("n", "bound", "ppc", "capacity", "seed", "res", "incompressible")
    BOOL = ("fluid", "solid", "obstacle")
    FLOAT = ("vx0", "vy0")

    def __init__(self, **params):
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise KeyError(f"paramètres inconnus : {sorted(unknown)} (connus : {sorted(self.PARAMS)})")
        self.p = {**self.PARAMS, **params}
        n = self.p["n"]
        self.m = {k: np.zeros((n, n), bool) for k in self.BOOL}
        self.m.update({k: np.zeros((n, n), np.float32) for k in self.FLOAT})
        self._walls = Walls()                       # segments de paroi, le dernier l'emporte

    @property
    def walls(self) -> list[dict]:
        """Segments de paroi (liste de dicts, modifiable en place ; voir Solver/boundary.py)."""
        return self._walls.segments

    @walls.setter
    def walls(self, segments: list[dict]) -> None:
        self._walls.segments = segments

    # ------------------------------------------------------------ géométrie
    @property
    def n(self) -> int:
        return self.p["n"]

    @property
    def dx(self) -> float:
        return 1.0 / self.p["n"]

    @property
    def band(self) -> float:
        """Épaisseur de la bande de paroi : les particules restent dans [band, 1 - band]."""
        return self.p["bound"] * self.dx

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        c = (np.arange(self.n) + 0.5) * self.dx
        return np.meshgrid(c, c, indexing="ij")

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        X, Y = self.centers()
        return (X >= min(x0, x1)) & (X <= max(x0, x1)) & (Y >= min(y0, y1)) & (Y <= max(y0, y1))

    def circle(self, cx: float, cy: float, r: float) -> np.ndarray:
        X, Y = self.centers()
        return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r

    def _mask(self, mask) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n, self.n):
            raise ValueError(f"masque {mask.shape} attendu ({self.n}, {self.n})")
        return mask

    # ------------------------------------------------------------ définition (intérieur)
    def set_fluid(self, mask, velocity=(0.0, 0.0)) -> None:
        mask = self._mask(mask)
        self.m["fluid"][mask], self.m["solid"][mask], self.m["obstacle"][mask] = True, False, False
        self.m["vx0"][mask], self.m["vy0"][mask] = velocity

    def set_solid(self, mask, velocity=(0.0, 0.0)) -> None:
        mask = self._mask(mask)
        self.m["solid"][mask], self.m["fluid"][mask], self.m["obstacle"][mask] = True, False, False
        self.m["vx0"][mask], self.m["vy0"][mask] = velocity

    def set_obstacle(self, mask) -> None:
        mask = self._mask(mask)
        self.m["obstacle"][mask], self.m["fluid"][mask], self.m["solid"][mask] = True, False, False

    def set_velocity(self, mask, velocity) -> None:
        """Vitesse initiale des particules semées dans ces cellules."""
        mask = self._mask(mask)
        self.m["vx0"][mask], self.m["vy0"][mask] = velocity

    def clear(self, mask) -> None:
        mask = self._mask(mask)
        for k in self.BOOL:
            self.m[k][mask] = False
        for k in self.FLOAT:
            self.m[k][mask] = 0.0

    # ------------------------------------------------------------ définition (parois)
    def set_wall(self, side: str, kind: str, velocity=(0.0, 0.0), span=(0.0, 1.0), pressure=None) -> None:
        """Condition limite sur un mur : side dans left/right/bottom/top, kind dans wall/inlet/outlet,
        span = étendue le long du mur en unités domaine (x pour bottom/top, y pour left/right).
        pressure (sortie seulement) : pression imposée p, uniforme sur le segment (mode incompressible) ;
        None = 0 (sortie libre). Un nouveau segment remplace les anciens là où il les recouvre."""
        self._walls.set(side, kind, velocity, span, pressure)

    def clear_wall(self, side: str, span=(0.0, 1.0)) -> None:
        self._walls.clear(side, span)

    def wall_table(self) -> WallTable:
        """Rastérisation des segments : type, v, depth, pressure (voir Solver.walls.WallTable) ; un obstacle collé à une
        entrée / sortie en porte la condition sur sa face qui regarde l'intérieur du domaine."""
        return self._walls.table(self.n, self.p["bound"], self.m["obstacle"])

    def resize(self, n: int) -> None:
        """Change la grille en rééchantillonnant les matrices (plus proche voisin) ; les parois sont en
        unités domaine et ne changent pas."""
        old = self.m["fluid"].shape[0]                # taille réelle des matrices (p["n"] peut déjà avoir changé)
        idx = np.minimum((np.arange(n) * old / n).astype(int), old - 1)
        self.m = {k: np.ascontiguousarray(v[idx][:, idx]) for k, v in self.m.items()}
        self.p["n"] = int(n)

    # ------------------------------------------------------------ fichiers
    def to_dict(self) -> dict:
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
        params = {k: v for k, v in d.get("params", {}).items() if k in cls.PARAMS}
        r = cls(**params)
        n = r.n
        legacy = {}
        for k, spec in d.get("matrices", {}).items():
            buf = base64.b64decode(spec["data"])
            if spec["dtype"] == "bool":
                a = np.unpackbits(np.frombuffer(buf, np.uint8))[:n * n].astype(bool)
            else:
                a = np.frombuffer(buf, np.float32)
            a = a.reshape(n, n)
            if k in r.m:
                r.m[k] = np.ascontiguousarray(a.astype(r.m[k].dtype))
            elif k in ("inlet", "outlet", "inlet_vx", "inlet_vy"):
                legacy[k] = a
        r.walls = [dict(w) for w in d.get("walls", [])]
        if legacy:
            r._migrate_legacy(legacy)
        return r

    def _migrate_legacy(self, legacy: dict) -> None:
        """Fichiers v2 : les bandes de cellules d'entrée / sortie collées à un mur deviennent des segments."""
        n, bound = self.n, self.p["bound"]
        depth = bound + 4
        zeros = np.zeros((n, n), np.float32)
        inlet = legacy.get("inlet", np.zeros((n, n), bool))
        outlet = legacy.get("outlet", np.zeros((n, n), bool))
        ivx, ivy = legacy.get("inlet_vx", zeros), legacy.get("inlet_vy", zeros)
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        dist = {"left": I, "right": n - 1 - I, "bottom": J, "top": n - 1 - J}
        owner = np.argmin(np.stack(list(dist.values())), axis=0)   # mur le plus proche (gauche/droite gagnent les égalités)
        for s, (side, d) in enumerate(dist.items()):  # chaque cellule de bord appartient à un seul mur
            zone = (d < depth) & (owner == s)
            axis = 0 if side in ("left", "right") else 1
            for kind, mask in (("outlet", outlet), ("inlet", inlet)):
                sel_side = mask & zone
                along = sel_side.any(axis=axis)
                k = 0
                while k < n:
                    if not along[k]:
                        k += 1
                        continue
                    k0 = k
                    while k < n and along[k]:
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "SimulationRunner":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def equals(self, other: "SimulationRunner") -> bool:
        return (self.p == other.p and all(np.array_equal(self.m[k], other.m[k]) for k in self.m)
                and self.walls == other.walls)

    # ------------------------------------------------------------ exécution
    def solver(self):
        from ui.solver import Solver
        return Solver(self.p, self.m, self.wall_table())

    def run(self, frames: int, substeps: int | None = None, callback=None):
        """Calcule `frames` images sans fenêtre ; callback(solver, k) après chaque image. Renvoie le solveur."""
        s = self.solver()
        for k in range(frames):
            s.step(substeps)
            if callback is not None:
                callback(s, k)
        return s

    def show(self, path: str | None = None) -> int:
        """Ouvre l'interface Qt sur cette simulation (bloquant)."""
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
        """La scène de main_fsi.py : bassin, bloc d'eau qui tombe sur un tablier tenu par deux piliers."""
        r = cls(n=250)
        r.set_fluid(r.rect(0.03, 0.03, 0.97, 0.22))
        r.set_fluid(r.rect(0.30, 0.62, 0.70, 0.95))
        r.set_solid(r.rect(0.15, 0.40, 0.85, 0.46))
        r.set_obstacle(r.rect(0.15, 0.0, 0.21, 0.40))
        r.set_obstacle(r.rect(0.79, 0.0, 0.85, 0.40))
        return r

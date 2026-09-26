"""SimulationRunner : définit une simulation (paramètres + matrices n × n), la sauve en JSON, la lance.

    from ui.runner import SimulationRunner
    r = SimulationRunner(n=128, gravity=9.81)
    X, Y = r.centers()                      # centres des cellules, (n, n), X[i, j] = (i + 0.5) dx
    r.set_fluid(Y < 0.2)                    # masque numpy (n, n) : cellules pleines d'eau
    r.set_solid(r.rect(0.55, 0.2, 0.58, 0.55))
    r.set_obstacle(r.circle(0.35, 0.28, 0.05))
    r.set_inlet((X < 0.05) & (Y > 0.25) & (Y < 0.45), velocity=(3.0, 0.0))
    r.set_outlet(X > 0.96)
    r.save("canal.json")                    # paramètres + matrices, un seul fichier
    s = r.run(150, callback=lambda s, k: print(k, s.stats()["n_fluid"]))   # sans fenêtre
    r.show()                                # interface Qt

Conventions : domaine [0, 1]², matrices indexées [i, j] = (x, y) comme les champs Taichi
(cells[:, 0] est la rangée du bas). Un seul matériau fluide et un seul solide (paramètres plats).
"""
from __future__ import annotations

import base64
import json

import numpy as np


class SimulationRunner:
    PARAMS = {
        "n": 128, "bound": 3, "ppc": 2, "cfl": 0.4, "gravity": 9.81, "substeps": 20, "seed": 0,
        "capacity": 0, "res": 700,
        "fluid_rho": 1.0, "fluid_E": 400.0,
        "solid_rho": 2.0, "solid_E": 3000.0, "solid_nu": 0.3,
        "eps0": 0.05, "epsf": 0.2, "tau_D": 2e-3, "k_res": 1e-3,
        "use_damage": True, "use_rupture": True, "color_mode": 0,
    }
    LABELS = {
        "n": "Grille n × n", "bound": "Cellules de bord", "ppc": "Particules / côté de cellule",
        "cfl": "CFL", "gravity": "Gravité g", "substeps": "Sous-pas par image", "seed": "Graine",
        "capacity": "Capacité fluide (0 = auto)", "res": "Résolution du rendu",
        "fluid_rho": "Fluide : densité", "fluid_E": "Fluide : raideur E",
        "solid_rho": "Solide : densité", "solid_E": "Solide : module E", "solid_nu": "Solide : Poisson ν",
        "eps0": "ε0 (début endommagement)", "epsf": "εf (rupture)", "tau_D": "τ_D", "k_res": "Raideur résiduelle",
        "use_damage": "Endommagement", "use_rupture": "Rupture", "color_mode": "Couleur solide (0 dégât, 1 déformation)",
    }
    STRUCTURAL = ("n", "bound", "ppc", "capacity", "seed", "res")
    BOOL = ("fluid", "solid", "obstacle", "inlet", "outlet")
    FLOAT = ("vx0", "vy0", "inlet_vx", "inlet_vy")

    def __init__(self, **params):
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise KeyError(f"paramètres inconnus : {sorted(unknown)} (connus : {sorted(self.PARAMS)})")
        self.p = {**self.PARAMS, **params}
        n = self.p["n"]
        self.m = {k: np.zeros((n, n), bool) for k in self.BOOL}
        self.m.update({k: np.zeros((n, n), np.float32) for k in self.FLOAT})

    # ------------------------------------------------------------ géométrie
    @property
    def n(self) -> int:
        return self.p["n"]

    @property
    def dx(self) -> float:
        return 1.0 / self.p["n"]

    @property
    def band(self) -> float:
        """Épaisseur de la bande de paroi : les particules restent dans [band, 1 - band] ; une entrée ou
        une sortie doit avoir des cellules au-delà."""
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

    # ------------------------------------------------------------ définition
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

    def set_inlet(self, mask, velocity) -> None:
        """Entrée : vitesse imposée + émission jusqu'à ppc² particules par cellule (hors bande de paroi)."""
        mask = self._mask(mask)
        self.m["inlet"][mask], self.m["outlet"][mask] = True, False
        self.m["inlet_vx"][mask], self.m["inlet_vy"][mask] = velocity

    def set_outlet(self, mask) -> None:
        """Sortie : les particules fluides qui y entrent sont détruites (hors bande de paroi)."""
        mask = self._mask(mask)
        self.m["outlet"][mask], self.m["inlet"][mask] = True, False

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

    def resize(self, n: int) -> None:
        """Change la grille en rééchantillonnant toutes les matrices (plus proche voisin)."""
        old = self.n
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
        return {"version": 2, "params": dict(self.p), "matrices": mats}

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationRunner":
        params = {k: v for k, v in d.get("params", {}).items() if k in cls.PARAMS}
        r = cls(**params)
        n = r.n
        for k, spec in d.get("matrices", {}).items():
            if k not in r.m:
                continue
            buf = base64.b64decode(spec["data"])
            if spec["dtype"] == "bool":
                a = np.unpackbits(np.frombuffer(buf, np.uint8))[:n * n].astype(bool)
            else:
                a = np.frombuffer(buf, np.float32)
            r.m[k] = np.ascontiguousarray(a.reshape(n, n).astype(r.m[k].dtype))
        return r

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "SimulationRunner":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def equals(self, other: "SimulationRunner") -> bool:
        return self.p == other.p and all(np.array_equal(self.m[k], other.m[k]) for k in self.m)

    # ------------------------------------------------------------ exécution
    def solver(self):
        from ui.solver import Solver
        return Solver(self.p, self.m)

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

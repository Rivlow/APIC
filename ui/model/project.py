"""Modèle de projet : dataclasses pures, sérialisables en JSON.

Chaque champ éditable porte des métadonnées (scope, label, bornes, choix...) exploitées par le
panneau de propriétés. Le `scope` dit quoi faire quand la valeur change :
  structural : réallouer les champs Taichi (rebuild au prochain Reset / Play / Step)
  initial    : re-semer les particules (reset)
  runtime    : appliquer immédiatement (scalaires passés aux kernels)
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar

STRUCTURAL = "structural"
INITIAL = "initial"
RUNTIME = "runtime"

ROLES = ["fluid", "solid", "obstacle"]
COLOR_MODES = ["damage", "strain"]


def param(default=None, *, scope=RUNTIME, label="", lo=None, hi=None, step=None,
          decimals=None, choices=None, readonly=False, hidden=False, factory=None):
    meta = {"scope": scope, "label": label, "lo": lo, "hi": hi, "step": step,
            "decimals": decimals, "choices": choices, "readonly": readonly, "hidden": hidden}
    if factory is not None:
        return field(default_factory=factory, metadata=meta)
    return field(default=default, metadata=meta)


def meta_of(obj, name: str) -> dict:
    for f in fields(obj):
        if f.name == name:
            return dict(f.metadata)
    return {}


def scope_of(obj, name: str) -> str:
    return meta_of(obj, name).get("scope", RUNTIME)


def new_id() -> str:
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------- domaine / solveur
@dataclass
class Domain:
    n_grid: int = param(250, scope=STRUCTURAL, label="Grille n × n", lo=32, hi=1024)
    bound: int = param(3, scope=STRUCTURAL, label="Cellules de bord", lo=1, hi=10)
    size: float = param(1.0, label="Taille du domaine", readonly=True)


@dataclass
class SolverSettings:
    cfl: float = param(0.4, label="CFL", lo=0.05, hi=0.9, step=0.05, decimals=2)
    substeps_per_frame: int = param(20, label="Sous-pas par image", lo=1, hi=500)
    gravity: float = param(9.81, label="Gravité g", lo=-100.0, hi=100.0, step=0.1, decimals=2)
    use_damage: bool = param(True, label="Endommagement")
    use_rupture: bool = param(True, label="Rupture")
    color_mode: str = param("damage", label="Couleur solide", choices=COLOR_MODES)
    ppc_side: int = param(2, scope=STRUCTURAL, label="Particules par côté de cellule", lo=1, hi=4)
    seed: int = param(0, scope=INITIAL, label="Graine aléatoire", lo=0, hi=10**9)
    viewport_res: int = param(700, label="Résolution viewport (au démarrage)", lo=200, hi=2000)


# ---------------------------------------------------------------- matériaux
@dataclass
class FluidMaterial:
    TAG: ClassVar[str] = "fluid"
    name: str = param("water", label="Nom")
    rho: float = param(1.0, label="Densité ρ", lo=0.01, hi=1e4, step=0.1, decimals=3)
    E: float = param(400.0, label="Raideur E", lo=1.0, hi=1e7, step=10.0, decimals=1)
    color: list = param(factory=lambda: [0.35, 0.65, 1.0], label="Couleur")


@dataclass
class SolidMaterial:
    TAG: ClassVar[str] = "solid"
    name: str = param("beam", label="Nom")
    rho: float = param(2.0, label="Densité ρ", lo=0.01, hi=1e4, step=0.1, decimals=3)
    E: float = param(3000.0, label="Module E", lo=1.0, hi=1e8, step=100.0, decimals=1)
    nu: float = param(0.3, label="Poisson ν", lo=0.0, hi=0.49, step=0.01, decimals=3)
    eps0: float = param(0.05, label="ε0 (début endommagement)", lo=0.0, hi=5.0, step=0.01, decimals=4)
    epsf: float = param(0.20, label="εf (rupture)", lo=0.001, hi=5.0, step=0.01, decimals=4)
    tau_D: float = param(2e-3, label="τ_D (vitesse d'endommagement)", lo=1e-6, hi=1.0, step=1e-4, decimals=6)
    k_res: float = param(1e-3, label="Raideur résiduelle", lo=0.0, hi=1.0, step=1e-4, decimals=6)


# ---------------------------------------------------------------- géométrie
@dataclass
class Rect:
    TAG: ClassVar[str] = "rect"
    id: str = param(factory=new_id, hidden=True)
    name: str = param("Rect", label="Nom")
    role: str = param("fluid", scope=STRUCTURAL, label="Rôle", choices=ROLES)
    material: str = param("", scope=STRUCTURAL, label="Matériau")
    x0: float = param(0.4, scope=STRUCTURAL, label="x0", lo=0.0, hi=1.0, step=0.01, decimals=3)
    y0: float = param(0.4, scope=STRUCTURAL, label="y0", lo=0.0, hi=1.0, step=0.01, decimals=3)
    x1: float = param(0.6, scope=STRUCTURAL, label="x1", lo=0.0, hi=1.0, step=0.01, decimals=3)
    y1: float = param(0.6, scope=STRUCTURAL, label="y1", lo=0.0, hi=1.0, step=0.01, decimals=3)

    def contains(self, x: float, y: float) -> bool:
        return (min(self.x0, self.x1) <= x <= max(self.x0, self.x1)
                and min(self.y0, self.y1) <= y <= max(self.y0, self.y1))

    def translate(self, dx: float, dy: float) -> None:
        self.x0 += dx
        self.x1 += dx
        self.y0 += dy
        self.y1 += dy

    def area(self) -> float:
        return abs(self.x1 - self.x0) * abs(self.y1 - self.y0)


@dataclass
class Circle:
    TAG: ClassVar[str] = "circle"
    id: str = param(factory=new_id, hidden=True)
    name: str = param("Circle", label="Nom")
    role: str = param("obstacle", scope=STRUCTURAL, label="Rôle", choices=ROLES)
    material: str = param("", scope=STRUCTURAL, label="Matériau")
    cx: float = param(0.5, scope=STRUCTURAL, label="Centre x", lo=0.0, hi=1.0, step=0.01, decimals=3)
    cy: float = param(0.5, scope=STRUCTURAL, label="Centre y", lo=0.0, hi=1.0, step=0.01, decimals=3)
    r: float = param(0.1, scope=STRUCTURAL, label="Rayon", lo=0.001, hi=1.0, step=0.01, decimals=3)

    def contains(self, x: float, y: float) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2

    def translate(self, dx: float, dy: float) -> None:
        self.cx += dx
        self.cy += dy

    def area(self) -> float:
        return math.pi * self.r ** 2


# ---------------------------------------------------------------- conditions
@dataclass
class RegionVelocity:
    shape_id: str = param("", hidden=True)
    vx: float = param(0.0, scope=INITIAL, label="Vitesse initiale vx", lo=-100.0, hi=100.0, step=0.1, decimals=3)
    vy: float = param(0.0, scope=INITIAL, label="Vitesse initiale vy", lo=-100.0, hi=100.0, step=0.1, decimals=3)


@dataclass
class InitialConditions:
    entries: list = field(default_factory=list)


@dataclass
class BoundaryConditions:
    walls: str = param("slip", scope=STRUCTURAL, label="Parois du domaine", choices=["slip"])


# ---------------------------------------------------------------- projet
@dataclass
class Project:
    name: str = param("Nouveau projet", label="Nom du projet")
    version: int = param(1, hidden=True)
    domain: Domain = field(default_factory=Domain)
    materials: list = field(default_factory=list)
    shapes: list = field(default_factory=list)
    initial_conditions: InitialConditions = field(default_factory=InitialConditions)
    boundary_conditions: BoundaryConditions = field(default_factory=BoundaryConditions)
    solver: SolverSettings = field(default_factory=SolverSettings)

    # ---- accès
    def shape_by_id(self, sid: str):
        for s in self.shapes:
            if s.id == sid:
                return s
        return None

    def materials_of(self, kind: str) -> list:
        return [m for m in self.materials if m.TAG == kind]

    def material_for(self, shape):
        """Matériau d'une forme : celui nommé s'il a le bon type, sinon le premier du bon type."""
        if shape.role not in ("fluid", "solid"):
            return None
        cands = self.materials_of(shape.role)
        for m in cands:
            if m.name == shape.material:
                return m
        return cands[0] if cands else None

    def used_material(self, kind: str):
        """Le matériau utilisé par les formes de ce rôle (v1 : un seul par phase)."""
        for s in self.shapes:
            if s.role == kind:
                return self.material_for(s)
        return None

    def fluid_material(self):
        return self.used_material("fluid")

    def solid_material(self):
        return self.used_material("solid")

    def shapes_of(self, role: str) -> list:
        return [s for s in self.shapes if s.role == role]

    # ---- conditions initiales
    def sync_ic(self) -> None:
        """Une entrée de vitesse par forme fluide/solide, dans l'ordre des formes ; purge les orphelines."""
        old = {e.shape_id: e for e in self.initial_conditions.entries}
        self.initial_conditions.entries = [
            old.get(s.id) or RegionVelocity(shape_id=s.id)
            for s in self.shapes if s.role in ("fluid", "solid")
        ]

    def velocity_of(self, sid: str) -> tuple[float, float]:
        for e in self.initial_conditions.entries:
            if e.shape_id == sid:
                return e.vx, e.vy
        return 0.0, 0.0

    # ---- validation
    def validate(self) -> list[str]:
        errors = []
        for kind in ("fluid", "solid"):
            used = {id(self.material_for(s)) for s in self.shapes if s.role == kind}
            used.discard(id(None))
            if len(used) > 1:
                errors.append(f"v1 : un seul matériau {kind} utilisable à la fois")
            if self.shapes_of(kind) and not self.materials_of(kind):
                errors.append(f"aucun matériau {kind} défini")
        if not self.shapes:
            errors.append("aucune forme")
        return errors

    # ---- JSON
    def to_dict(self) -> dict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls(
            name=d.get("name", "Projet"),
            version=d.get("version", 1),
            domain=_construct(Domain, d.get("domain", {})),
            materials=[_material_from(m) for m in d.get("materials", [])],
            shapes=[_shape_from(s) for s in d.get("shapes", [])],
            initial_conditions=InitialConditions(
                entries=[_construct(RegionVelocity, e)
                         for e in d.get("initial_conditions", {}).get("entries", [])]),
            boundary_conditions=_construct(BoundaryConditions, d.get("boundary_conditions", {})),
            solver=_construct(SolverSettings, d.get("solver", {})),
        )
        p.sync_ic()
        return p

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _to_jsonable(obj: Any):
    if is_dataclass(obj):
        d = {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
        tag = getattr(type(obj), "TAG", None)
        if tag:
            d["type"] = tag
        return d
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _construct(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def _material_from(d: dict):
    return _construct(FluidMaterial if d.get("type") == "fluid" else SolidMaterial, d)


def _shape_from(d: dict):
    return _construct(Circle if d.get("type") == "circle" else Rect, d)


# ---------------------------------------------------------------- scène par défaut (main_fsi.py)
def default_project() -> Project:
    p = Project(name="Eau et pont (main_fsi)")
    p.materials = [FluidMaterial(name="water"), SolidMaterial(name="beam")]
    p.shapes = [
        Rect(name="Pool", role="fluid", material="water", x0=0.03, y0=0.03, x1=0.97, y1=0.22),
        Rect(name="Water block", role="fluid", material="water", x0=0.30, y0=0.62, x1=0.70, y1=0.95),
        Rect(name="Beam", role="solid", material="beam", x0=0.15, y0=0.40, x1=0.85, y1=0.46),
        Rect(name="Pillar left", role="obstacle", x0=0.15, y0=0.0, x1=0.21, y1=0.46),
        Rect(name="Pillar right", role="obstacle", x0=0.79, y0=0.0, x1=0.85, y1=0.46),
    ]
    p.sync_ic()
    return p

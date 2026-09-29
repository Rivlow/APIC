# Solver/walls.py -- Définition des conditions aux limites (numpy seul, importable sans Taichi).
#
# Grille nx × ny de cellules carrées de côté dx (en mètres, SI) : le domaine est [0, nx dx] × [0, ny dx].
# Une condition limite est un SEGMENT de mur : côté (left, right, bottom, top), étendue le long du mur en mètres
# (y pour left/right, x pour bottom/top), type et paramètres :
#   - wall   : mur imperméable (composante entrante annulée) -- partout par défaut. Frottement beta dans [0, 1] :
#              0 = glissant (défaut), 1 = adhérent (no-slip), entre les deux = vitesse tangentielle réduite.
#   - inlet  : vitesse imposée
#   - outlet : sortie à pression imposée p (Pa, unités de la simulation), uniforme sur le segment ; 0 par défaut
#              (sortie libre). La projection incompressible y fixe la pression (Dirichlet) : c'est le gradient
#              de pression qui fait sortir (ou rentrer) l'eau. Un profil (hydrostatique...) = plusieurs segments.
# Jamais de bande de cellules intérieure. Un obstacle collé au mur sur une entrée / sortie devient lui-même la
# condition limite sur sa face de même normale que le mur (la face qui regarde l'intérieur du domaine) : la
# profondeur wall_d (en cellules depuis le bord) y vaut la première couche libre après l'obstacle, au lieu de
# `bound`. Ses autres faces restent un obstacle ordinaire. Sur un mur, l'obstacle reste un obstacle (son
# frottement est le paramètre global `obstacle_friction`).
#
# Walls (liste de segments, sérialisable en JSON) -> WallTable (4, max(nx, ny)) -> champs Taichi wall_type, wall_v,
# wall_d, wall_p, wall_f, appliqués sur la grille par Solver/boundary.py.

from typing import NamedTuple

import numpy as np

from Solver import bc_expr

SIDES = ("left", "right", "bottom", "top")
LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3
WALL, INLET, OUTLET = 0, 1, 2
WALL_TYPES = {"wall": WALL, "inlet": INLET, "outlet": OUTLET}
OBSTACLE = 4                                  # bit de `cells` : nœud bloqué


def side_length(side: int, nx: int, ny: int) -> int:
    """Number of cells along a wall.

    **Inputs**

    - `side` : int in LEFT/RIGHT/BOTTOM/TOP
    - `nx`, `ny` : int

    **Outputs**

    - int, ny for left / right, nx for bottom / top
    """
    return ny if side in (LEFT, RIGHT) else nx


class WallTable(NamedTuple):
    """Table rastérisée, indexée [côté, cellule le long du mur] ; (4, max(nx, ny)), au-delà de la longueur du mur :
    mur glissant."""
    type: np.ndarray                          # int32 : WALL / INLET / OUTLET
    v: np.ndarray                             # (…, 2) float32 : vitesse imposée (entrée)
    depth: np.ndarray                         # int32 : position du mur en cellules depuis le bord
    pressure: np.ndarray                      # float32 : pression imposée sur une sortie (0 = libre)
    friction: np.ndarray                      # float32 : frottement beta d'un mur (0 glissant, 1 adhérent)
    expr: np.ndarray                          # (…, 3) int32 : expression dépendant de t pour (vx, vy, p), -1 = aucune
    sources: tuple                            # expressions dépendant de t (indice = expr), réévaluées sur le GPU


def _value(v):
    """Normalize a boundary value: number -> float, expression -> validated str.

    **Inputs**

    - `v` : float | int | str

    **Outputs**

    - float | str ; ValueError if the expression is not allowed
    """
    if bc_expr.is_expr(v):
        bc_expr.check(v)
        return v.strip()
    return float(v)


class Walls:
    """Segments de paroi ; le dernier posé l'emporte là où il recouvre les autres.

        walls = Walls()
        walls.set("left", "inlet", velocity=(0.5, 0.0), span=(0.25, 0.75))
        walls.set("right", "outlet", pressure=50.0, span=(0.0, 0.3))   # pression imposée sur un segment
        walls.set("bottom", "wall", friction=1.0)                      # fond adhérent
        wall_type, wall_v, wall_d, wall_p, wall_f = walls.fields(nx, ny, bound, obstacle_mask)
    """

    def __init__(self, segments=None):
        """Store a copy of the segments.

        **Inputs**

        - `segments` : list of segment dicts (JSON form) or None
        """
        self.segments: list[dict] = [dict(w) for w in (segments or [])]

    def set(self, side: str, kind: str, velocity=(0.0, 0.0), span=(0.0, 1.0), pressure=None,
            friction=None) -> None:
        """Add a wall segment, overriding what it overlaps.

        **Inputs**

        - `side` : str in left/right/bottom/top
        - `kind` : str in wall/inlet/outlet
        - `velocity` : (2,) float | str, inlet velocity ; str = expression of x, y, t (see Solver/bc_expr.py)
        - `span` : (2,) float, extent along the wall in domain units
        - `pressure` : float | str or None, outlet only (None = 0, free outlet) ; str = expression of x, y, t
        - `friction` : float in [0, 1] or None, wall only (0 slip, 1 no-slip)

        **Outputs**

        - self.segments updated ; ValueError on invalid arguments
        """
        if side not in SIDES:
            raise ValueError(f"côté {side!r} inconnu (attendu : {SIDES})")
        if kind not in WALL_TYPES:
            raise ValueError(f"type {kind!r} inconnu (attendu : {tuple(WALL_TYPES)})")
        if pressure is not None and kind != "outlet":
            raise ValueError("pressure (pression imposée) n'a de sens que pour une sortie")
        if friction is not None and kind != "wall":
            raise ValueError("friction (frottement) n'a de sens que pour un mur")
        if friction is not None and not 0.0 <= float(friction) <= 1.0:
            raise ValueError("friction doit être dans [0, 1] (0 glissant, 1 adhérent)")
        a, b = sorted((float(span[0]), float(span[1])))
        a = max(a, 0.0)                               # borné à la longueur du mur par le runner
        if b <= a:
            return
        kept = []
        for w in self.segments:
            if w["side"] != side or w["span"][1] <= a or w["span"][0] >= b:
                kept.append(w)
                continue
            if w["span"][0] < a:                      # morceau restant avant
                kept.append({**w, "span": [w["span"][0], a]})
            if w["span"][1] > b:                      # morceau restant après
                kept.append({**w, "span": [b, w["span"][1]]})
        if kind != "wall":
            seg = {"side": side, "span": [a, b], "type": kind,
                   "velocity": [_value(velocity[0]), _value(velocity[1])]}
            if pressure is not None:
                seg["pressure"] = _value(pressure)
            kept.append(seg)
        elif friction:                                # mur glissant = défaut : pas de segment
            kept.append({"side": side, "span": [a, b], "type": "wall", "friction": float(friction)})
        self.segments = kept

    def clear(self, side: str, span=(0.0, 1.0)) -> None:
        """Reset a span of a side to the default slip wall.

        **Inputs**

        - `side` : str in left/right/bottom/top
        - `span` : (2,) float, domain units

        **Outputs**

        - self.segments updated
        """
        self.set(side, "wall", span=span)

    def table(self, nx: int, ny: int, bound: int, obstacle=None, consts=None, dx=None) -> WallTable:
        """Rasterize wall segments into per-cell tables.

        **Inputs**

        - `nx`, `ny`, `bound` : int
        - `obstacle` : np.ndarray bool (nx, ny) or None
        - `consts` : dict[str, float] or None, constants usable in expressions (pi, Lx, Ly, dx always defined)
        - `dx` : float or None, cell size (m) ; None = 1 / max(nx, ny) (normalized domain)

        **Outputs**

        - WallTable of np.ndarray (4, nm) (v: (4, nm, 2)), nm = max(nx, ny)

        **Note** : corners (k in the transverse band) are always walls; an obstacle glued to an inlet / outlet carries it.
        """
        nm = max(nx, ny)                              # longueur des tables
        dx = 1.0 / nm if dx is None else float(dx)    # cellule k couvre [k dx, (k + 1) dx]
        wtype = np.zeros((4, nm), np.int32)
        wvel = np.zeros((4, nm, 2), np.float32)
        wpress = np.zeros((4, nm), np.float32)
        wfric = np.zeros((4, nm), np.float32)
        wexpr = np.full((4, nm, 3), -1, np.int32)
        pending = []                                  # expressions : (côté, k0, k1, canal 0 vx / 1 vy / 2 p, source)
        for w in self.segments:
            s = SIDES.index(w["side"])
            ln = side_length(s, nx, ny)
            k0 = max(int(np.floor(w["span"][0] / dx + 1e-9)), 0)
            k1 = min(int(np.ceil(w["span"][1] / dx - 1e-9)), ln)
            if w["type"] == "wall":                   # frottement : y compris dans les coins
                if k1 > k0:
                    wfric[s, k0:k1] = w.get("friction", 0.0)
                continue
            k0, k1 = max(k0, bound), min(k1, ln - bound)
            if k1 <= k0:
                continue
            wtype[s, k0:k1] = WALL_TYPES[w["type"]]
            vals = list(w.get("velocity", [0.0, 0.0])) + [w.get("pressure", 0.0)]
            for ch, val in enumerate(vals):
                if bc_expr.is_expr(val):
                    pending.append((s, k0, k1, ch, val))
                    val = 0.0
                if ch < 2:
                    wvel[s, k0:k1, ch] = val
                    wexpr[s, k0:k1, ch] = -1
                else:
                    wpress[s, k0:k1] = val
                    wexpr[s, k0:k1, 2] = -1
        for s in range(4):
            ln = side_length(s, nx, ny)
            wtype[s, :bound] = WALL
            wtype[s, ln - bound:] = WALL
        wdepth = np.full((4, nm), bound, np.int32)
        if obstacle is not None:
            wdepth = self._depth(np.asarray(obstacle, dtype=bool), wtype, nx, ny, bound)
        # expressions : sans t, évaluées ici une fois ; avec t, numérotées pour le kernel GPU (valeur initiale à t = 0)
        cst = {"pi": np.pi, "Lx": nx * dx, "Ly": ny * dx, "dx": dx, **(consts or {})}
        sources = []
        for s, k0, k1, ch, src in pending:
            if wtype[s, k0] != (INLET if ch < 2 else OUTLET):   # segment recouvert par un autre
                continue
            ks = np.arange(k0, k1)
            d = wdepth[s, ks]
            along = (ks + 0.5) * dx
            x = {LEFT: d * dx, RIGHT: (nx - d) * dx}.get(s, along)
            y = {BOTTOM: d * dx, TOP: (ny - d) * dx}.get(s, along)
            val = bc_expr.evaluate(src, x, y, 0.0, cst)
            if ch < 2:
                wvel[s, ks, ch] = val
            else:
                wpress[s, ks] = val
            if bc_expr.uses_time(src):
                if src not in sources:
                    sources.append(src)
                wexpr[s, ks, ch] = sources.index(src)
        wexpr[wtype != INLET, 0:2] = -1
        wexpr[wtype != OUTLET, 2] = -1
        wpress[wtype != OUTLET] = 0.0
        return WallTable(wtype, wvel, wdepth, wpress, wfric, wexpr, tuple(sources))

    @staticmethod
    def _depth(o: np.ndarray, wtype: np.ndarray, nx: int, ny: int, bound: int) -> np.ndarray:
        """Wall depth per inlet / outlet cell, pushed past an obstacle touching the wall.

        **Inputs**

        - `o` : np.ndarray bool (nx, ny) obstacle mask
        - `wtype` : np.ndarray int32 (4, nm)
        - `nx`, `ny`, `bound` : int

        **Outputs**

        - np.ndarray int32 (4, nm), depth in cells from the edge (first free layer, >= bound)

        **Note** : touching = within the band or the first usable cell; depth never beyond mid-domain.
        """
        # couches [côté][couche depuis le bord, cellule le long du mur]
        layers = [o, o[::-1, :], o.T, o.T[::-1, :]]
        wdepth = np.full((4, max(nx, ny)), bound, np.int32)
        for s in range(4):
            L = layers[s]
            d_max = L.shape[0] // 2 - 1               # jamais au-delà du milieu du domaine
            ln = side_length(s, nx, ny)
            for k in np.nonzero(wtype[s, :ln] != WALL)[0]:
                col = L[:, k]
                touch = np.nonzero(col[:bound + 1])[0]
                if touch.size == 0:
                    continue
                d = int(touch[0])
                while d < d_max and col[d]:
                    d += 1
                wdepth[s, k] = max(d, bound)
        return wdepth

    def fields(self, nx: int, ny: int, bound: int, obstacle=None, consts=None, dx=None):
        """Rasterize and upload the wall table to Taichi fields (once).

        **Inputs**

        - `nx`, `ny`, `bound` : int
        - `obstacle` : np.ndarray bool (nx, ny) or None
        - `consts` : dict[str, float] or None, constants for expressions
        - `dx` : float or None, cell size ; None = 1 / max(nx, ny)

        **Outputs**

        - `wall_type`, `wall_d` : i32 field (4, nm)
        - `wall_v` : vec2 f32 field (4, nm)
        - `wall_p`, `wall_f` : f32 field (4, nm)

        **Note** : time-dependent expressions are frozen at t = 0 here (only ui.solver.Solver re-evaluates them).
        """
        import taichi as ti                           # import local : ce module reste utilisable sans Taichi
        t = self.table(nx, ny, bound, obstacle, consts, dx)
        nm = max(nx, ny)
        wall_type = ti.field(ti.i32, (4, nm))
        wall_v = ti.Vector.field(2, ti.f32, (4, nm))
        wall_d = ti.field(ti.i32, (4, nm))
        wall_p = ti.field(ti.f32, (4, nm))
        wall_f = ti.field(ti.f32, (4, nm))
        wall_type.from_numpy(t.type)
        wall_v.from_numpy(t.v)
        wall_d.from_numpy(t.depth)
        wall_p.from_numpy(t.pressure)
        wall_f.from_numpy(t.friction)
        return wall_type, wall_v, wall_d, wall_p, wall_f

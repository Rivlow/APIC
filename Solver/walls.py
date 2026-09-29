# Solver/walls.py -- Définition des conditions aux limites (numpy seul, importable sans Taichi).
#
# Grille de cellules carrées de côté dx (en mètres, SI), nx × ny (2D) ou nx × ny × nz (3D) : le domaine est
# [0, nx dx] × [0, ny dx] (× [0, nz dx]). Axes : x horizontal, y vertical (gravité), z profondeur.
# Faces du domaine : left / right (x), bottom / top (y), back / front (z, 3D seulement) ; axe normal = côté // 2,
# axes tangents = les autres axes dans l'ordre croissant (2D : un seul ; 3D : deux).
# Une condition limite est une ZONE de mur : côté, étendue le long du mur en mètres -- un intervalle (a, b) en 2D
# (y pour left/right, x pour bottom/top), un rectangle ((a0, a1), (b0, b1)) en 3D (sur les deux axes tangents, dans
# l'ordre croissant : (y, z) pour left/right, (x, z) pour bottom/top, (x, y) pour back/front) -- type et paramètres :
#   - wall   : mur imperméable (composante entrante annulée) -- partout par défaut. Frottement beta dans [0, 1] :
#              0 = glissant (défaut), 1 = adhérent (no-slip), entre les deux = vitesse tangentielle réduite.
#   - inlet  : vitesse imposée (une vitesse sortante en fait une sortie à vitesse imposée, voir boundary.exit_at)
#   - outlet : sortie à pression imposée p (Pa, unités de la simulation), uniforme sur la zone ; 0 par défaut
#              (sortie libre). La projection incompressible y fixe la pression (Dirichlet) : c'est le gradient
#              de pression qui fait sortir (ou rentrer) l'eau. Un profil (hydrostatique...) = une expression.
# Jamais de bande de cellules intérieure. Un obstacle collé au mur sur une entrée / sortie devient lui-même la
# condition limite sur sa face de même normale que le mur (la face qui regarde l'intérieur du domaine) : la
# profondeur wall_d (en cellules depuis le bord) y vaut la première couche libre après l'obstacle, au lieu de
# `bound`. Ses autres faces restent un obstacle ordinaire. Sur un mur, l'obstacle reste un obstacle (son
# frottement est le paramètre global `obstacle_friction`).
#
# Walls (liste de zones, sérialisable en JSON) -> WallTable (2·dim, na, nb) -> champs Taichi wall_type, wall_v,
# wall_d, wall_p, wall_f, appliqués sur la grille par Solver/boundary.py. Indices [côté, ka, kb] : ka, kb = cellule
# le long du premier / second axe tangent (kb = 0 en 2D, nb = 1).

from typing import NamedTuple

import numpy as np

from Solver import bc_expr

SIDES = ("left", "right", "bottom", "top", "back", "front")
LEFT, RIGHT, BOTTOM, TOP, BACK, FRONT = 0, 1, 2, 3, 4, 5
WALL, INLET, OUTLET = 0, 1, 2
WALL_TYPES = {"wall": WALL, "inlet": INLET, "outlet": OUTLET}
OBSTACLE = 4                                  # bit de `cells` : nœud bloqué
AXES = "xyz"


def frame(side: int, dim: int) -> tuple:
    """Axes of a domain face.

    **Inputs**

    - `side` : int in 0 .. 2 dim - 1
    - `dim` : int 2 or 3

    **Outputs**

    - (a, plus, tangents) : normal axis, 1 for the far face (right / top / front), tuple of tangent axes
    """
    a = side // 2
    return a, side % 2, tuple(b for b in range(dim) if b != a)


def table_shape(shape) -> tuple:
    """Wall-table shape for a grid.

    **Inputs**

    - `shape` : tuple[int, ...] cells per axis (2 or 3)

    **Outputs**

    - (2 dim, na, nb) : na / nb = longest first / second tangent over the faces (nb = 1 in 2D)
    """
    dim = len(shape)
    ts = [frame(s, dim)[2] for s in range(2 * dim)]
    na = max(shape[t[0]] for t in ts)
    nb = max(shape[t[1]] for t in ts) if dim == 3 else 1
    return 2 * dim, na, nb


def side_length(side: int, nx: int, ny: int) -> int:
    """Number of cells along a wall (2D).

    **Inputs**

    - `side` : int in LEFT/RIGHT/BOTTOM/TOP
    - `nx`, `ny` : int

    **Outputs**

    - int, ny for left / right, nx for bottom / top
    """
    return ny if side in (LEFT, RIGHT) else nx


class WallTable(NamedTuple):
    """Table rastérisée, indexée [côté, ka, kb] ; (2 dim, na, nb), au-delà de la longueur du mur : mur glissant."""
    type: np.ndarray                          # int32 : WALL / INLET / OUTLET
    v: np.ndarray                             # (…, dim) float32 : vitesse imposée (entrée)
    depth: np.ndarray                         # int32 : position du mur en cellules depuis le bord
    pressure: np.ndarray                      # float32 : pression imposée sur une sortie (0 = libre)
    friction: np.ndarray                      # float32 : frottement beta d'un mur (0 glissant, 1 adhérent)
    expr: np.ndarray                          # (…, dim + 1) int32 : expression de t pour (v…, p), -1 = aucune
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


def span_box(span) -> list:
    """Normalize a zone extent to a list of [lo, hi] intervals (one per tangent axis).

    **Inputs**

    - `span` : (a, b) 2D interval or ((a0, a1), (b0, b1)) 3D rectangle

    **Outputs**

    - list[list[float]] sorted intervals, lower bounds clipped at 0
    """
    if np.ndim(span) == 1:
        span = [span]
    return [[max(min(float(s[0]), float(s[1])), 0.0), max(float(s[0]), float(s[1]))] for s in span]


def _box_minus(w: list, r: list) -> list:
    """Parts of box w outside box r (same dimension).

    **Inputs**

    - `w`, `r` : list[[lo, hi]] per axis

    **Outputs**

    - list of boxes covering w \\ r (disjoint)
    """
    if any(w[k][1] <= r[k][0] or w[k][0] >= r[k][1] for k in range(len(w))):
        return [w]
    out, cur = [], [list(iv) for iv in w]
    for k in range(len(w)):                           # tranches sous / sur r le long de chaque axe
        if cur[k][0] < r[k][0]:
            out.append([list(iv) for iv in cur[:k]] + [[cur[k][0], r[k][0]]] + [list(iv) for iv in cur[k + 1:]])
        if cur[k][1] > r[k][1]:
            out.append([list(iv) for iv in cur[:k]] + [[r[k][1], cur[k][1]]] + [list(iv) for iv in cur[k + 1:]])
        cur[k] = [max(cur[k][0], r[k][0]), min(cur[k][1], r[k][1])]
    return out


class Walls:
    """Zones de paroi ; la dernière posée l'emporte là où elle recouvre les autres.

        walls = Walls()
        walls.set("left", "inlet", velocity=(0.5, 0.0), span=(0.25, 0.75))            # 2D : intervalle
        walls.set("right", "outlet", pressure=50.0, span=(0.0, 0.3))
        walls.set("left", "inlet", velocity=(0.5, 0, 0), span=((0.2, 0.4), (0.1, 0.3)))   # 3D : rectangle (y, z)
        walls.set("bottom", "wall", friction=1.0)                      # fond adhérent
        table = walls.table((nx, ny[, nz]), bound, obstacle_mask, consts, dx)
    """

    def __init__(self, segments=None):
        """Store a copy of the zones.

        **Inputs**

        - `segments` : list of zone dicts (JSON form) or None
        """
        self.segments: list[dict] = [dict(w) for w in (segments or [])]

    @staticmethod
    def box(w: dict) -> list:
        """Extent of a stored zone as [lo, hi] intervals.

        **Inputs**

        - `w` : zone dict ("span" flat in 2D, nested in 3D)

        **Outputs**

        - list[[lo, hi]]
        """
        return span_box(w["span"])

    def set(self, side: str, kind: str, velocity=(0.0, 0.0), span=(0.0, 1.0), pressure=None,
            friction=None) -> None:
        """Add a wall zone, overriding what it overlaps.

        **Inputs**

        - `side` : str in SIDES
        - `kind` : str in wall/inlet/outlet
        - `velocity` : (dim,) float | str, inlet velocity ; str = expression of x, y, z, t (see Solver/bc_expr.py)
        - `span` : (a, b) in 2D, ((a0, a1), (b0, b1)) in 3D, extent along the tangent axes in domain units
        - `pressure` : float | str or None, outlet only (None = 0, free outlet) ; str = expression of x, y, z, t
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
        r = span_box(span)
        if any(iv[1] <= iv[0] for iv in r):
            return
        flat = len(r) == 1                            # 2D : intervalle simple (format JSON d'origine)

        def pack(box):
            return list(box[0]) if flat else [list(iv) for iv in box]

        kept = []
        for w in self.segments:
            wb = self.box(w)
            if w["side"] != side or len(wb) != len(r):
                kept.append(w)
                continue
            for piece in _box_minus(wb, r):
                kept.append({**w, "span": pack(piece)})
        if kind != "wall":
            seg = {"side": side, "span": pack(r), "type": kind, "velocity": [_value(c) for c in velocity]}
            if pressure is not None:
                seg["pressure"] = _value(pressure)
            kept.append(seg)
        elif friction:                                # mur glissant = défaut : pas de zone
            kept.append({"side": side, "span": pack(r), "type": "wall", "friction": float(friction)})
        self.segments = kept

    def clear(self, side: str, span=(0.0, 1.0)) -> None:
        """Reset a zone of a side to the default slip wall.

        **Inputs**

        - `side` : str in SIDES
        - `span` : zone extent, domain units

        **Outputs**

        - self.segments updated
        """
        self.set(side, "wall", span=span)

    def table(self, shape, bound: int, obstacle=None, consts=None, dx=None) -> WallTable:
        """Rasterize wall zones into per-cell tables.

        **Inputs**

        - `shape` : tuple[int, ...] cells per axis, (nx, ny) or (nx, ny, nz)
        - `bound` : int
        - `obstacle` : np.ndarray bool (shape) or None
        - `consts` : dict[str, float] or None, constants usable in expressions (pi, Lx, Ly, Lz, dx always defined)
        - `dx` : float or None, cell size (m) ; None = 1 / max(shape) (normalized domain)

        **Outputs**

        - WallTable of np.ndarray (2 dim, na, nb) (v: (…, dim), expr: (…, dim + 1))

        **Note** : corners (a tangent index in the transverse band) are always walls; an obstacle glued to an
        inlet / outlet carries it.
        """
        shape = tuple(int(n) for n in shape)
        dim = len(shape)
        ns, na, nb = table_shape(shape)
        dx = 1.0 / max(shape) if dx is None else float(dx)   # cellule k couvre [k dx, (k + 1) dx]
        wtype = np.zeros((ns, na, nb), np.int32)
        wvel = np.zeros((ns, na, nb, dim), np.float32)
        wpress = np.zeros((ns, na, nb), np.float32)
        wfric = np.zeros((ns, na, nb), np.float32)
        wexpr = np.full((ns, na, nb, dim + 1), -1, np.int32)
        pending = []                                  # expressions : (côté, tranches, canal (v… puis p), source)
        for w in self.segments:
            s = SIDES.index(w["side"])
            if s >= ns:
                continue                              # face z d'un fichier 3D ouvert en 2D
            _, _, ts = frame(s, dim)
            box = self.box(w)
            if len(box) != len(ts):
                continue
            lens = [shape[t] for t in ts]
            ks = [(max(int(np.floor(iv[0] / dx + 1e-9)), 0), min(int(np.ceil(iv[1] / dx - 1e-9)), ln))
                  for iv, ln in zip(box, lens)]
            if w["type"] == "wall":                   # frottement : y compris dans les coins
                if all(k1 > k0 for k0, k1 in ks):
                    wfric[(s, *_slices(ks, dim))] = w.get("friction", 0.0)
                continue
            ks = [(max(k0, bound), min(k1, ln - bound)) for (k0, k1), ln in zip(ks, lens)]
            if any(k1 <= k0 for k0, k1 in ks):
                continue
            sl = (s, *_slices(ks, dim))
            wtype[sl] = WALL_TYPES[w["type"]]
            vel = list(w.get("velocity", [])) + [0.0] * dim
            vals = vel[:dim] + [w.get("pressure", 0.0)]
            for ch, val in enumerate(vals):
                if bc_expr.is_expr(val):
                    pending.append((s, ks, ch, val))
                    val = 0.0
                if ch < dim:
                    wvel[sl + (ch,)] = val
                else:
                    wpress[sl] = val
                wexpr[sl + (ch,)] = -1
        for s in range(ns):                           # coins : toujours des murs
            _, _, ts = frame(s, dim)
            for k, t in enumerate(ts):
                idx = [slice(None), slice(None)]
                idx[k] = slice(0, bound)
                wtype[(s, *idx)] = WALL
                idx[k] = slice(shape[t] - bound, None)
                wtype[(s, *idx)] = WALL
        wdepth = np.full((ns, na, nb), bound, np.int32)
        if obstacle is not None:
            wdepth = self._depth(np.asarray(obstacle, dtype=bool), wtype, bound)
        # expressions : sans t, évaluées ici une fois ; avec t, numérotées pour le kernel GPU (valeur initiale à t = 0)
        cst = {"pi": np.pi, "Lx": shape[0] * dx, "Ly": shape[1] * dx, "Lz": (shape[2] if dim == 3 else 0) * dx,
               "dx": dx, **(consts or {})}
        sources = []
        for s, ks, ch, src in pending:
            a, plus, ts = frame(s, dim)
            sl = (s, *_slices(ks, dim))
            want = INLET if ch < dim else OUTLET
            ka = np.arange(ks[0][0], ks[0][1])
            kb = np.arange(ks[1][0], ks[1][1]) if dim == 3 else np.zeros(1, int)
            KA, KB = np.meshgrid(ka, kb, indexing="ij")
            keep = wtype[s, KA, KB] == want           # zone recouverte par une autre
            d = wdepth[s, KA, KB]
            pos = [np.zeros(KA.shape), np.zeros(KA.shape), np.zeros(KA.shape)]
            pos[a] = (shape[a] - d) * dx if plus else d * dx
            pos[ts[0]] = (KA + 0.5) * dx
            if dim == 3:
                pos[ts[1]] = (KB + 0.5) * dx
            val = bc_expr.evaluate(src, pos[0], pos[1], pos[2], 0.0, cst)
            if ch < dim:
                wvel[sl + (ch,)] = np.where(keep, val, wvel[sl + (ch,)])
            else:
                wpress[sl] = np.where(keep, val, wpress[sl])
            if bc_expr.uses_time(src):
                if src not in sources:
                    sources.append(src)
                wexpr[sl + (ch,)] = np.where(keep, sources.index(src), wexpr[sl + (ch,)])
        wexpr[..., :dim][wtype != INLET] = -1
        wexpr[..., dim][wtype != OUTLET] = -1
        wpress[wtype != OUTLET] = 0.0
        return WallTable(wtype, wvel, wdepth, wpress, wfric, wexpr, tuple(sources))

    @staticmethod
    def _depth(o: np.ndarray, wtype: np.ndarray, bound: int) -> np.ndarray:
        """Wall depth per inlet / outlet cell, pushed past an obstacle touching the wall.

        **Inputs**

        - `o` : np.ndarray bool (shape) obstacle mask
        - `wtype` : np.ndarray int32 (2 dim, na, nb)
        - `bound` : int

        **Outputs**

        - np.ndarray int32 (2 dim, na, nb), depth in cells from the edge (first free layer, >= bound)

        **Note** : touching = within the band or the first usable cell; depth never beyond mid-domain.
        """
        dim = o.ndim
        wdepth = np.full(wtype.shape, bound, np.int32)
        for s in range(2 * dim):
            a, plus, ts = frame(s, dim)
            L = np.moveaxis(o, a, 0)                   # [couche depuis le bord, tangentes...]
            if plus:
                L = L[::-1]
            if dim == 2:
                L = L[:, :, None]
            n_a, la, lb = L.shape
            act = wtype[s, :la, :lb] != WALL
            if not act.any():
                continue
            d_max = n_a // 2 - 1                       # jamais au-delà du milieu du domaine
            touch = L[:bound + 1].any(axis=0)
            d0 = np.argmax(L[:bound + 1], axis=0)      # première couche d'obstacle
            lay = np.arange(n_a)[:, None, None]
            filled = L | (lay < d0[None])               # couches avant l'obstacle comptées pleines : on les franchit
            free = np.where((~filled).any(axis=0), np.argmax(~filled, axis=0), n_a)
            d = np.minimum(free, d_max)
            sel = act & touch
            wdepth[s, :la, :lb][sel] = np.maximum(d[sel], bound)
        return wdepth

    def fields(self, nx: int, ny: int, bound: int, obstacle=None, consts=None, dx=None):
        """Rasterize and upload the wall table to Taichi fields (once, 2D legacy scripts).

        **Inputs**

        - `nx`, `ny`, `bound` : int
        - `obstacle` : np.ndarray bool (nx, ny) or None
        - `consts` : dict[str, float] or None, constants for expressions
        - `dx` : float or None, cell size ; None = 1 / max(nx, ny)

        **Outputs**

        - `wall_type`, `wall_d` : i32 field (4, nm, 1)
        - `wall_v` : vec2 f32 field (4, nm, 1)
        - `wall_p`, `wall_f` : f32 field (4, nm, 1)

        **Note** : time-dependent expressions are frozen at t = 0 here (only ui.solver.Solver re-evaluates them).
        """
        import taichi as ti                           # import local : ce module reste utilisable sans Taichi
        t = self.table((nx, ny), bound, obstacle, consts, dx)
        shp = t.type.shape
        wall_type = ti.field(ti.i32, shp)
        wall_v = ti.Vector.field(2, ti.f32, shp)
        wall_d = ti.field(ti.i32, shp)
        wall_p = ti.field(ti.f32, shp)
        wall_f = ti.field(ti.f32, shp)
        wall_type.from_numpy(t.type)
        wall_v.from_numpy(t.v)
        wall_d.from_numpy(t.depth)
        wall_p.from_numpy(t.pressure)
        wall_f.from_numpy(t.friction)
        return wall_type, wall_v, wall_d, wall_p, wall_f


def _slices(ks, dim: int) -> tuple:
    """Index slices (ka, kb) of a zone in a wall table.

    **Inputs**

    - `ks` : list of (k0, k1) per tangent axis
    - `dim` : int

    **Outputs**

    - (slice, slice) ; kb = slice(0, 1) in 2D
    """
    return slice(*ks[0]), (slice(*ks[1]) if dim == 3 else slice(0, 1))

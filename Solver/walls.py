# Solver/walls.py -- Définition des conditions aux limites (numpy seul, importable sans Taichi).
#
# Grille nx × ny de cellules carrées, dx = 1 / max(nx, ny) : le domaine est [0, nx dx] × [0, ny dx] (unités domaine).
# Une condition limite est un SEGMENT de mur : côté (left, right, bottom, top), étendue le long du mur en unités
# domaine (y pour left/right, x pour bottom/top), type et paramètres :
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

SIDES = ("left", "right", "bottom", "top")
LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3
WALL, INLET, OUTLET = 0, 1, 2
WALL_TYPES = {"wall": WALL, "inlet": INLET, "outlet": OUTLET}
OBSTACLE = 4                                  # bit de `cells` : nœud bloqué


def side_length(side: int, nx: int, ny: int) -> int:
    """Nombre de cellules le long du mur : ny pour gauche / droite, nx pour bas / haut."""
    return ny if side in (LEFT, RIGHT) else nx


class WallTable(NamedTuple):
    """Table rastérisée, indexée [côté, cellule le long du mur] ; (4, max(nx, ny)), au-delà de la longueur du mur :
    mur glissant."""
    type: np.ndarray                          # int32 : WALL / INLET / OUTLET
    v: np.ndarray                             # (…, 2) float32 : vitesse imposée (entrée)
    depth: np.ndarray                         # int32 : position du mur en cellules depuis le bord
    pressure: np.ndarray                      # float32 : pression imposée sur une sortie (0 = libre)
    friction: np.ndarray                      # float32 : frottement beta d'un mur (0 glissant, 1 adhérent)


class Walls:
    """Segments de paroi ; le dernier posé l'emporte là où il recouvre les autres.

        walls = Walls()
        walls.set("left", "inlet", velocity=(0.5, 0.0), span=(0.25, 0.75))
        walls.set("right", "outlet", pressure=50.0, span=(0.0, 0.3))   # pression imposée sur un segment
        walls.set("bottom", "wall", friction=1.0)                      # fond adhérent
        wall_type, wall_v, wall_d, wall_p, wall_f = walls.fields(nx, ny, bound, obstacle_mask)
    """

    def __init__(self, segments=None):
        self.segments: list[dict] = [dict(w) for w in (segments or [])]

    def set(self, side: str, kind: str, velocity=(0.0, 0.0), span=(0.0, 1.0), pressure=None,
            friction=None) -> None:
        """side dans left/right/bottom/top, kind dans wall/inlet/outlet, span en unités domaine le long du mur.
        pressure (sortie seulement) : pression imposée p, uniforme sur le segment ; None = 0 (sortie libre).
        friction (mur seulement) : beta dans [0, 1], 0 = glissant (défaut), 1 = adhérent."""
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
        a, b = max(a, 0.0), min(b, 1.0)
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
                   "velocity": [float(velocity[0]), float(velocity[1])]}
            if pressure is not None:
                seg["pressure"] = float(pressure)
            kept.append(seg)
        elif friction:                                # mur glissant = défaut : pas de segment
            kept.append({"side": side, "span": [a, b], "type": "wall", "friction": float(friction)})
        self.segments = kept

    def clear(self, side: str, span=(0.0, 1.0)) -> None:
        self.set(side, "wall", span=span)

    def table(self, nx: int, ny: int, bound: int, obstacle=None) -> WallTable:
        """Rastérisation (voir WallTable). obstacle : masque (nx, ny) bool ; un obstacle collé à une entrée / sortie
        en porte la condition (depth). Les coins (dans la bande transversale) sont toujours des murs."""
        nm = max(nx, ny)                              # dx = 1 / nm : cellule k couvre [k dx, (k + 1) dx]
        wtype = np.zeros((4, nm), np.int32)
        wvel = np.zeros((4, nm, 2), np.float32)
        wpress = np.zeros((4, nm), np.float32)
        wfric = np.zeros((4, nm), np.float32)
        for w in self.segments:
            s = SIDES.index(w["side"])
            ln = side_length(s, nx, ny)
            k0 = max(int(np.floor(w["span"][0] * nm + 1e-9)), 0)
            k1 = min(int(np.ceil(w["span"][1] * nm - 1e-9)), ln)
            if w["type"] == "wall":                   # frottement : y compris dans les coins
                if k1 > k0:
                    wfric[s, k0:k1] = w.get("friction", 0.0)
                continue
            k0, k1 = max(k0, bound), min(k1, ln - bound)
            if k1 <= k0:
                continue
            wtype[s, k0:k1] = WALL_TYPES[w["type"]]
            wvel[s, k0:k1] = w.get("velocity", [0.0, 0.0])
            wpress[s, k0:k1] = w.get("pressure", 0.0)
        for s in range(4):
            ln = side_length(s, nx, ny)
            wtype[s, :bound] = WALL
            wtype[s, ln - bound:] = WALL
        wpress[wtype != OUTLET] = 0.0
        wdepth = np.full((4, nm), bound, np.int32)
        if obstacle is not None:
            wdepth = self._depth(np.asarray(obstacle, dtype=bool), wtype, nx, ny, bound)
        return WallTable(wtype, wvel, wdepth, wpress, wfric)

    @staticmethod
    def _depth(o: np.ndarray, wtype: np.ndarray, nx: int, ny: int, bound: int) -> np.ndarray:
        """Pour chaque cellule d'entrée / sortie le long d'un mur : si un obstacle touche le mur (dans la bande
        ou la première cellule utilisable), le mur recule jusqu'à la première couche libre après lui."""
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

    def fields(self, nx: int, ny: int, bound: int, obstacle=None):
        """Table rastérisée copiée dans cinq champs Taichi (wall_type, wall_v, wall_d, wall_p, wall_f), une fois."""
        import taichi as ti                           # import local : ce module reste utilisable sans Taichi
        t = self.table(nx, ny, bound, obstacle)
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

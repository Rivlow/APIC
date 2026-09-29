"""Interface Qt 3D : Viewport3D (image rendue sur le GPU par ui/render3d.py, caméra orbitale, sélection de zones de
paroi et de primitives) et PrimPanel (liste et édition des primitives de la scène).

Souris dans la vue 3D : bouton gauche glissé = rotation autour de la cible, bouton droit ou milieu = déplacement,
molette = zoom, F = vue entière, Échap = désélection. Clic gauche sans glisser : sur une primitive = la
sélectionner ; sur une face du fond de la boîte = tout le mur. Maj + glisser sur une face = un rectangle de paroi,
aimanté aux cellules. Superpositions dessinées par QPainter (projection numpy, aucune donnée GPU) : arêtes de la
boîte, zones de paroi (vert entrée, rouge sortie, jaune mur à frottement), sélection, primitives en fil de fer.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from Solver.walls import SIDES, frame
from ui.render3d import Camera, ray_box
from ui.runner import MATERIALS, SimulationRunner

WALL_RGB = {"inlet": (60, 200, 90), "outlet": (230, 70, 70), "wall": (230, 190, 60)}
MAT_RGB = {"fluid": (90, 160, 255), "solid": (230, 200, 90), "obstacle": (200, 200, 200), "clear": (255, 120, 255),
           "rotor": (230, 150, 70)}
MAT_LABELS = {"fluid": "eau", "solid": "solide", "obstacle": "obstacle", "clear": "effacer",
              "rotor": "rotor (tourne)"}
KIND_LABELS = {"box": "boîte", "sphere": "sphère", "cylinder": "cylindre", "mesh": "maillage"}
SIDE_LABELS = {"left": "gauche", "right": "droite", "bottom": "bas", "top": "haut", "back": "arrière",
               "front": "avant"}


def _vec(text: str, n: int = 3) -> list:
    """Parse 'a, b, c' into n floats (ValueError otherwise)."""
    vals = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if len(vals) != n:
        raise ValueError(f"{n} valeurs attendues : {text!r}")
    return vals


def _txt(v) -> str:
    """Format a vector as 'a, b, c'."""
    return ", ".join(f"{float(c):.4g}" for c in v)


def face_rect(runner: SimulationRunner, side: int, box) -> np.ndarray:
    """3D corners of a rectangle on a domain face.

    **Inputs**

    - `runner` : SimulationRunner (3D) ; `side` : int ; `box` : [[a0, a1], [b0, b1]] m along the tangent axes

    **Outputs**

    - np.ndarray (4, 3)
    """
    a, plus, ts = frame(side, 3)
    ext = runner.extent
    pts = np.zeros((4, 3))
    for k, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        pts[k, a] = ext[a] if plus else 0.0
        pts[k, ts[0]] = box[0][u]
        pts[k, ts[1]] = box[1][v]
    return pts


def prim_segments(runner: SimulationRunner, prim: dict) -> list:
    """Wireframe of a primitive as 3D segments.

    **Inputs**

    - `runner` : SimulationRunner ; `prim` : dict

    **Outputs**

    - list of ((3,), (3,)) segments
    """
    def box_edges(lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c = [np.array([hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]])
             for i in range(8)]
        return [(c[i], c[j]) for i in range(8) for j in range(8) if i < j and bin(i ^ j).count("1") == 1]

    def circle(center, u, v, r, n=40):
        th = np.linspace(0, 2 * np.pi, n + 1)
        pts = [center + r * (np.cos(t) * u + np.sin(t) * v) for t in th]
        return list(zip(pts[:-1], pts[1:]))

    k = prim["kind"]
    if k == "box":
        return box_edges(prim["lo"], prim["hi"])
    if k == "sphere":
        c, r = np.asarray(prim["center"], float), float(prim["radius"])
        e = np.eye(3)
        return circle(c, e[0], e[1], r) + circle(c, e[0], e[2], r) + circle(c, e[1], e[2], r)
    if k == "cylinder":
        p0, p1, r = np.asarray(prim["p0"], float), np.asarray(prim["p1"], float), float(prim["radius"])
        ax = p1 - p0
        ax = ax / max(np.linalg.norm(ax), 1e-12)
        u = np.cross(ax, [0.0, 1.0, 0.0] if abs(ax[1]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(ax, u)
        segs = circle(p0, u, v, r) + circle(p1, u, v, r)
        for t in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
            d = r * (np.cos(t) * u + np.sin(t) * v)
            segs.append((p0 + d, p1 + d))
        return segs
    if k == "mesh":
        try:
            tri = runner.mesh_triangles(prim)
        except Exception:
            return []
        return box_edges(tri.reshape(-1, 3).min(axis=0), tri.reshape(-1, 3).max(axis=0))
    return []


def ray_prim(runner: SimulationRunner, prim: dict, o, d) -> float:
    """Distance along a ray to a primitive (bounding shape for cylinders and meshes).

    **Inputs**

    - `runner` : SimulationRunner ; `prim` : dict ; `o`, `d` : (3,) ray

    **Outputs**

    - float t (inf if missed)
    """
    k = prim["kind"]
    if k == "sphere":
        c, r = np.asarray(prim["center"], float), float(prim["radius"])
        oc = o - c
        b = oc @ d
        disc = b * b - (oc @ oc - r * r)
        if disc < 0:
            return np.inf
        t = -b - np.sqrt(disc)
        return t if t > 0 else np.inf
    if k == "box":
        lo, hi = np.asarray(prim["lo"], float), np.asarray(prim["hi"], float)
    elif k == "cylinder":
        p0, p1, r = np.asarray(prim["p0"], float), np.asarray(prim["p1"], float), float(prim["radius"])
        lo, hi = np.minimum(p0, p1) - r, np.maximum(p0, p1) + r
    else:
        try:
            v = runner.mesh_triangles(prim).reshape(-1, 3)
        except Exception:
            return np.inf
        lo, hi = v.min(axis=0), v.max(axis=0)
    t0, t1 = ray_box(o - np.minimum(lo, hi), d, np.abs(hi - lo))
    return t0 if t0 <= t1 and t1 > 0 else np.inf


class Viewport3D(QWidget):
    wallSelectionChanged = Signal(object)      # (side, (ka0, ka1), (kb0, kb1)) cellules incluses, ou None
    primSelected = Signal(object)              # indice de primitive ou None
    hoverChanged = Signal(str)
    viewChanged = Signal()

    def __init__(self, runner: SimulationRunner, parent=None):
        """Create the 3D viewport on a runner (camera framing the box, no selection).

        **Inputs**

        - `runner` : SimulationRunner (dim 3) ; `parent` : QWidget | None
        """
        super().__init__(parent)
        self.runner = runner
        self.cam = Camera.fit(runner.extent)
        self.sel = None                                   # compatibilité avec la vue 2D (pas de boîte de cellules)
        self.wsel = None
        self.prim_sel = None
        self.hover_face = None
        self._buf = self._qimg = None
        self._press = None                                # (position, bouton, Maj)
        self._last = None
        self._moved = False
        self._wstart = None                               # (side, ka, kb) au début d'un rectangle de paroi
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(300, 300)

    def sizeHint(self) -> QSize:
        """Preferred widget size."""
        return QSize(800, 700)

    # ------------------------------------------------------------ image et caméra
    def set_image(self, buf: np.ndarray) -> None:
        """Display a rendered image (keeps a reference: QImage does not own the memory)."""
        self._buf = np.ascontiguousarray(buf)
        h, w, _ = self._buf.shape
        self._qimg = QImage(self._buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.update()

    def render_view(self) -> tuple:
        """Camera and image size for the solver (window aspect, at most res pixels on the long side).

        **Outputs**

        - (Camera, (width, height))
        """
        W, H = max(self.width(), 1), max(self.height(), 1)
        s = min(1.0, self.runner.p["res"] / max(W, H))
        return self.cam, (max(1, int(W * s)), max(1, int(H * s)))

    def reset_view(self) -> None:
        """Frame the whole box."""
        self.cam = Camera.fit(self.runner.extent, self.cam.yaw, self.cam.pitch)
        self.viewChanged.emit()
        self.update()

    def resizeEvent(self, event) -> None:
        """Re-render at the new aspect ratio."""
        self.viewChanged.emit()
        super().resizeEvent(event)

    # ------------------------------------------------------------ sélection
    def face_at(self, pos):
        """Far box face under the cursor (the one drawn behind the particles).

        **Inputs**

        - `pos` : QPointF

        **Outputs**

        - (side int, ka int, kb int) cells clamped to the usable zone, or None
        """
        W, H = max(self.width(), 1), max(self.height(), 1)
        o, d = self.cam.ray(pos.x(), pos.y(), W, H)
        ext = np.array(self.runner.extent)
        t0, t1 = ray_box(o, d, ext)
        if t0 > t1 or t1 <= 0:
            return None
        p = o + t1 * d
        dist = [abs(p[a]) if s % 2 == 0 else abs(p[a] - ext[a]) for s in range(6) for a in [s // 2]]
        side = int(np.argmin(dist))
        _, _, ts = frame(side, 3)
        n, b = self.runner.shape, self.runner.p["bound"]
        ka = int(np.clip(int(p[ts[0]] / self.runner.dx), b, n[ts[0]] - b - 1))
        kb = int(np.clip(int(p[ts[1]] / self.runner.dx), b, n[ts[1]] - b - 1))
        return side, ka, kb

    def prim_at(self, pos):
        """Nearest primitive under the cursor (index or None)."""
        W, H = max(self.width(), 1), max(self.height(), 1)
        o, d = self.cam.ray(pos.x(), pos.y(), W, H)
        best, idx = np.inf, None
        for i, prim in enumerate(self.runner.prims):
            t = ray_prim(self.runner, prim, o, d)
            if t < best:
                best, idx = t, i
        return idx

    def _emit_wsel(self) -> None:
        self.wallSelectionChanged.emit(self.wsel)
        self.update()

    # ------------------------------------------------------------ souris / clavier
    def wheelEvent(self, event) -> None:
        """Zoom toward the target."""
        self.cam.zoom(1.2 ** (event.angleDelta().y() / 120.0))
        self.viewChanged.emit()
        self.update()

    def mousePressEvent(self, event) -> None:
        """Start an orbit, a pan or (Shift + left on a face) a wall rectangle."""
        self.setFocus()
        pos = event.position()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self._press, self._last, self._moved = (pos, event.button(), shift), pos, False
        self._wstart = None
        if event.button() == Qt.MouseButton.LeftButton and shift:
            f = self.face_at(pos)
            if f is not None:
                self._wstart = f
                self.wsel = (SIDES[f[0]], (f[1], f[1]), (f[2], f[2]))
                self._emit_wsel()

    def mouseMoveEvent(self, event) -> None:
        """Orbit / pan / grow the wall rectangle ; hover text."""
        pos = event.position()
        if self._press is not None:
            d = pos - self._last
            self._last = pos
            if abs(d.x()) + abs(d.y()) > 0:
                self._moved = True
            btn = self._press[1]
            if self._wstart is not None:
                f = self.face_at(pos)
                if f is not None and f[0] == self._wstart[0]:
                    s, a0, b0 = self._wstart
                    self.wsel = (SIDES[s], (min(a0, f[1]), max(a0, f[1])), (min(b0, f[2]), max(b0, f[2])))
                    self._emit_wsel()
            elif btn == Qt.MouseButton.LeftButton:
                self.cam.orbit(d.x(), d.y())
                self.viewChanged.emit()
                self.update()
            elif btn in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
                self.cam.pan(d.x(), d.y(), max(self.height(), 1))
                self.viewChanged.emit()
                self.update()
        f = self.face_at(pos)
        hover = None if f is None else f[0]
        if hover != self.hover_face:
            self.hover_face = hover
            self.update()
        if f is not None:
            s, ka, kb = f
            _, _, ts = frame(s, 3)
            ax = "xyz"
            dx = self.runner.dx
            self.hoverChanged.emit(f"paroi {SIDE_LABELS[SIDES[s]]}  {ax[ts[0]]} = {(ka + 0.5) * dx:.4g} m  "
                                   f"{ax[ts[1]]} = {(kb + 0.5) * dx:.4g} m  (cellule {ka}, {kb})")
        else:
            self.hoverChanged.emit("")

    def mouseReleaseEvent(self, event) -> None:
        """A click without motion selects a primitive, or a whole wall face."""
        if self._press is not None and not self._moved and self._press[1] == Qt.MouseButton.LeftButton \
                and self._wstart is None:
            pos = self._press[0]
            idx = self.prim_at(pos)
            if idx is not None:
                self.prim_sel = idx
                self.primSelected.emit(idx)
            else:
                f = self.face_at(pos)
                if f is not None:
                    s = f[0]
                    _, _, ts = frame(s, 3)
                    n, b = self.runner.shape, self.runner.p["bound"]
                    self.wsel = (SIDES[s], (b, n[ts[0]] - b - 1), (b, n[ts[1]] - b - 1))
                else:
                    self.wsel = None
                self._emit_wsel()
            self.update()
        self._press = self._wstart = None

    def keyPressEvent(self, event) -> None:
        """F: frame the box ; Escape: clear selections."""
        if event.key() == Qt.Key.Key_F:
            self.reset_view()
        elif event.key() == Qt.Key.Key_Escape:
            self.wsel = None
            self.prim_sel = None
            self.primSelected.emit(None)
            self._emit_wsel()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------ dessin
    def _poly(self, pts3):
        """Projected polygon (None if any vertex is behind the camera)."""
        px, z = self.cam.project(pts3, max(self.width(), 1), max(self.height(), 1))
        if (z <= 1e-6).any():
            return None
        return QPolygonF([QPointF(float(x), float(y)) for x, y in px])

    def _lines(self, painter: QPainter, segs, pen: QPen) -> None:
        """Draw projected 3D segments (both ends in front of the camera)."""
        if not segs:
            return
        a = np.array([s[0] for s in segs])
        b = np.array([s[1] for s in segs])
        W, H = max(self.width(), 1), max(self.height(), 1)
        pa, za = self.cam.project(a, W, H)
        pb, zb = self.cam.project(b, W, H)
        painter.setPen(pen)
        for i in range(len(segs)):
            if za[i] > 1e-6 and zb[i] > 1e-6:
                painter.drawLine(QPointF(*pa[i]), QPointF(*pb[i]))

    def paintEvent(self, event) -> None:
        """Draw the rendered image and the overlays (box, wall zones, selection, primitives)."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(16, 16, 24))
        if self._qimg is not None:
            painter.drawImage(self.rect(), self._qimg)
        r = self.runner
        ext = np.array(r.extent)
        # arêtes de la boîte
        corners = [np.array([ext[0] if i & 1 else 0.0, ext[1] if i & 2 else 0.0, ext[2] if i & 4 else 0.0])
                   for i in range(8)]
        edges = [(corners[i], corners[j]) for i in range(8) for j in range(8) if i < j and bin(i ^ j).count("1") == 1]
        self._lines(painter, edges, QPen(QColor(200, 200, 215, 150), 1.0))
        # face survolée
        if self.hover_face is not None:
            poly = self._poly(face_rect(r, self.hover_face, [[0, ext[t]] for t in frame(self.hover_face, 3)[2]]))
            if poly is not None:
                painter.setPen(QPen(QColor(255, 255, 255, 200), 2.0))
                painter.setBrush(QColor(255, 255, 255, 25))
                painter.drawPolygon(poly)
        # zones de paroi (telles que définies)
        for w in r.walls:
            box = w["span"] if np.ndim(w["span"]) == 2 else None
            if box is None or w["side"] not in SIDES:
                continue
            poly = self._poly(face_rect(r, SIDES.index(w["side"]), box))
            if poly is None:
                continue
            c = WALL_RGB[w["type"]]
            painter.setPen(QPen(QColor(*c), 1.5))
            painter.setBrush(QColor(c[0], c[1], c[2], 90))
            painter.drawPolygon(poly)
        # sélection de paroi
        if self.wsel is not None:
            side, (a0, a1), (b0, b1) = self.wsel
            dx = r.dx
            poly = self._poly(face_rect(r, SIDES.index(side), [[a0 * dx, (a1 + 1) * dx], [b0 * dx, (b1 + 1) * dx]]))
            if poly is not None:
                painter.setPen(QPen(QColor(255, 255, 255), 1.8, Qt.PenStyle.DashLine))
                painter.setBrush(QColor(255, 255, 255, 60))
                painter.drawPolygon(poly)
        # primitives en fil de fer
        for i, prim in enumerate(r.prims):
            c = MAT_RGB.get(prim["material"], (255, 255, 255))
            width = 2.2 if i == self.prim_sel else 1.0
            alpha = 255 if i == self.prim_sel else 170
            color = QColor(255, 255, 120, alpha) if i == self.prim_sel else QColor(c[0], c[1], c[2], alpha)
            self._lines(painter, prim_segments(r, prim), QPen(color, width))
        painter.end()


class PrimPanel(QWidget):
    changed = Signal()                          # la scène a changé (Reset nécessaire)
    selected = Signal(object)                   # primitive sélectionnée dans la liste (indice ou None)

    def __init__(self, runner: SimulationRunner, parent=None):
        """Primitive list with add / delete / reorder buttons and an editor for the selected one.

        **Inputs**

        - `runner` : SimulationRunner ; `parent` : QWidget | None
        """
        super().__init__(parent)
        self.runner = runner
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_row)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, kind in (("+ Boîte", "box"), ("+ Sphère", "sphere"), ("+ Cylindre", "cylinder"),
                           ("+ Maillage…", "mesh")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=kind: self.add(k))
            row.addWidget(b)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        for text, fn in (("Supprimer", self._delete), ("Monter", lambda: self._move(-1)),
                         ("Descendre", lambda: self._move(1))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row2.addWidget(b)
        lay.addLayout(row2)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        lay.addWidget(self.form_host)
        hint = QLabel("La dernière primitive l'emporte ; appliquées par-dessus les matrices. Clic dans la vue : "
                      "sélection.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.refresh()

    def set_runner(self, runner: SimulationRunner) -> None:
        """Switch to another runner."""
        self.runner = runner
        self.refresh()

    def refresh(self, keep: int | None = None) -> None:
        """Refill the list (and select `keep`)."""
        self.list.blockSignals(True)
        self.list.clear()
        for i, p in enumerate(self.runner.prims, start=1):
            name = KIND_LABELS[p["kind"]]
            if p["kind"] == "mesh":
                name += " " + os.path.basename(p["path"])
            self.list.addItem(f"{i}. {name} · {MAT_LABELS[p['material']]}")
        self.list.blockSignals(False)
        if keep is not None and 0 <= keep < self.list.count():
            self.list.setCurrentRow(keep)
        else:
            self._on_row(-1)

    def select(self, idx) -> None:
        """Select a primitive from the viewport."""
        self.list.setCurrentRow(-1 if idx is None else idx)

    # ------------------------------------------------------------ actions
    def add(self, kind: str) -> None:
        """Add a primitive centered in the box (mesh: file dialog, several files = one assembly)."""
        ext = np.array(self.runner.extent)
        c = 0.5 * ext
        m = float(ext.min())
        if kind == "box":
            self.runner.add_prim("box", "obstacle", lo=list(c - 0.1 * m), hi=list(c + 0.1 * m))
        elif kind == "sphere":
            self.runner.add_prim("sphere", "obstacle", center=list(c), radius=0.1 * m)
        elif kind == "cylinder":
            self.runner.add_prim("cylinder", "obstacle", p0=list(c - [0, 0, 0.25 * ext[2]]),
                                 p1=list(c + [0, 0, 0.25 * ext[2]]), radius=0.08 * m)
        else:
            paths, _ = QFileDialog.getOpenFileNames(self, "Importer un maillage (plusieurs fichiers : un assemblage)",
                                                    "", "Maillage (*.stl *.obj)")
            if not paths:
                return
            from Solver import mesh
            try:
                raws = [mesh.load_mesh(p) for p in paths]
            except Exception as exc:
                QMessageBox.critical(self, "Maillage", str(exc))
                return
            v = np.concatenate([t.reshape(-1, 3) for t in raws])
            lo, hi = v.min(axis=0), v.max(axis=0)
            pivot = 0.5 * (lo + hi)                          # pivot commun : les pièces restent assemblées
            scale = 0.5 * m / max(float((hi - lo).max()), 1e-12)
            for p in paths:
                self.runner.add_prim("mesh", "obstacle", path=p, scale=scale, rotate=[0.0, 0.0, 0.0],
                                     translate=list(c), pivot=list(pivot), fill=True)
        self.refresh(len(self.runner.prims) - 1)
        self.changed.emit()

    def _delete(self) -> None:
        i = self.list.currentRow()
        if 0 <= i < len(self.runner.prims):
            del self.runner.prims[i]
            self.refresh(min(i, len(self.runner.prims) - 1))
            self.changed.emit()

    def _move(self, step: int) -> None:
        i = self.list.currentRow()
        j = i + step
        if 0 <= i < len(self.runner.prims) and 0 <= j < len(self.runner.prims):
            pr = self.runner.prims
            pr[i], pr[j] = pr[j], pr[i]
            self.refresh(j)
            self.changed.emit()

    # ------------------------------------------------------------ éditeur
    def _on_row(self, i: int) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        if not 0 <= i < len(self.runner.prims):
            self.selected.emit(None)
            return
        prim = self.runner.prims[i]
        mat = QComboBox()
        for m in MATERIALS:
            mat.addItem(MAT_LABELS[m], m)
        mat.setCurrentIndex(MATERIALS.index(prim["material"]))
        mat.currentIndexChanged.connect(lambda _=0: self._set(i, "material", mat.currentData()))
        self.form.addRow("Matériau", mat)
        fields = {"box": (("lo", "coin min (m)"), ("hi", "coin max (m)")),
                  "sphere": (("center", "centre (m)"),),
                  "cylinder": (("p0", "axe : début (m)"), ("p1", "axe : fin (m)")),
                  "mesh": (("translate", "position du pivot (m)"), ("rotate", "rotation x, y, z (°)"))}[prim["kind"]]
        for key, label in fields:
            ed = QLineEdit(_txt(prim[key]))
            ed.editingFinished.connect(lambda k=key, e=ed: self._set_vec(i, k, e))
            self.form.addRow(label, ed)
        if prim["kind"] in ("sphere", "cylinder"):
            sp = QDoubleSpinBox()
            sp.setDecimals(4)
            sp.setRange(1e-4, 1e4)
            sp.setValue(float(prim["radius"]))
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda v: self._set(i, "radius", float(v)))
            self.form.addRow("rayon (m)", sp)
        if prim["kind"] == "mesh":
            sp = QDoubleSpinBox()
            sp.setDecimals(6)
            sp.setRange(1e-9, 1e6)
            sp.setValue(float(prim.get("scale", 1.0)))
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda v: self._set(i, "scale", float(v)))
            self.form.addRow("échelle (m / unité)", sp)
            fill = QCheckBox("plein (sinon coque)")
            fill.setChecked(bool(prim.get("fill", True)))
            fill.toggled.connect(lambda v: self._set(i, "fill", bool(v)))
            self.form.addRow("", fill)
            self.form.addRow("fichier", QLabel(os.path.basename(prim["path"])))
        if prim["material"] == "rotor":                 # obstacle tournant : vitesse imposée, axe
            sp = QDoubleSpinBox()
            sp.setDecimals(1)
            sp.setRange(-1e5, 1e5)
            sp.setValue(float(prim.get("rpm", 60.0)))
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda v: self._set(i, "rpm", float(v)))
            self.form.addRow("vitesse (tr/min, signe = sens)", sp)
            ed = QLineEdit(_txt(prim.get("axis", [0.0, 1.0, 0.0])))
            ed.editingFinished.connect(lambda e=ed: self._set_vec(i, "axis", e))
            self.form.addRow("axe de rotation", ed)
        else:
            ed = QLineEdit(_txt(list(prim.get("velocity", [0, 0, 0])) + [0.0] * (3 - len(prim.get("velocity", [])))))
            ed.editingFinished.connect(lambda e=ed: self._set_vec(i, "velocity", e))
            self.form.addRow("vitesse initiale (m/s)", ed)
        self.selected.emit(i)

    def _set(self, i: int, key: str, value) -> None:
        if 0 <= i < len(self.runner.prims) and self.runner.prims[i].get(key) != value:
            self.runner.prims[i][key] = value
            if key == "material":
                self.refresh(i)                               # libellé + champs (rotor : vitesse, axe)
            self.changed.emit()

    def _set_vec(self, i: int, key: str, editor: QLineEdit) -> None:
        try:
            value = _vec(editor.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Primitive", str(exc))
            editor.setText(_txt(self.runner.prims[i][key]))
            return
        self._set(i, key, value)

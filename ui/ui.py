"""Interface Qt : Viewport (image rendue sur le GPU, zoom, pan, sélection de cellules et de parois) et UI.

Souris dans le viewport : molette = zoom autour du curseur, bouton droit ou milieu = déplacement de la
vue, F = vue entière, Échap = désélection. Bouton gauche :
  - à l'intérieur du domaine : boîte de sélection de cellules (eau, solide, obstacle, vitesse, effacer) ;
  - près d'un bord ou dans la bande de paroi, le mur s'éclaire : clic = tout le mur, glisser le long du
    mur = un segment, aimanté aux frontières de cellules ; les segments existants (vert entrée, rouge
    sortie) ont deux poignées à étirer. Les conditions aux limites n'existent que sur les quatre parois.
Toute modification de la scène ou d'un paramètre structurel demande un Reset (bandeau orange).
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
                               QFrame, QGridLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QSpinBox, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ui.kernels import FLUID_MODES
from ui.runner import SIDES, SimulationRunner

INT_RANGES = {"nx": (8, 4096), "ny": (8, 4096), "bound": (1, 16), "ppc": (1, 4), "substeps": (1, 1000), "seed": (0, 10**9),
              "capacity": (0, 5_000_000), "res": (200, 2000), "color_mode": (0, 1)}
SIDE_LABELS = {"left": "gauche", "right": "droite", "bottom": "bas", "top": "haut"}
FLOAT_RANGES = {"obstacle_friction": (0.0, 1.0), "volume_correction": (0.0, 2.0)}
WALL_COLORS = {"inlet": QColor(60, 200, 90), "outlet": QColor(230, 70, 70), "wall": QColor(230, 190, 60)}
SNAP_PX = 12                                   # distance au bord (px) qui fait passer en mode paroi
HANDLE_PX = 7


class Viewport(QWidget):
    selectionChanged = Signal(object)          # (i0, j0, i1, j1) cellules incluses, ou None
    wallSelectionChanged = Signal(object)      # (side, k0, k1) cellules le long du mur incluses, ou None
    wallEdited = Signal()                      # une poignée de segment a été déplacée (runner.walls modifié)
    hoverChanged = Signal(str)

    def __init__(self, runner: SimulationRunner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.x0, self.y0, self.scale = 0.0, 0.0, 1.0     # vue = [x0, x0 + 1/scale] × [y0, y0 + 1/scale]
        self.sel = None
        self.wsel = None
        self.hover_wall = None                            # (side, k) pendant le survol d'un mur
        self._buf = None
        self._qimg = None
        self._drag = None                                 # cellule de départ de la boîte de sélection
        self._wdrag = None                                # (side, k) de départ d'un segment de mur
        self._hdrag = None                                # (indice du segment, extrémité 0/1) : poignée tirée
        self._moved = False
        self._pan = None                                  # dernière position souris pendant le pan
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(300, 300)
        self.setAutoFillBackground(True)

    @property
    def n(self) -> int:
        """max(nx, ny) : cellules par unité domaine (dx = 1 / n)."""
        return self.runner.n

    @property
    def nx(self) -> int:
        return self.runner.nx

    @property
    def ny(self) -> int:
        return self.runner.ny

    def side_len(self, side: str) -> int:
        """Cellules le long du mur : ny à gauche / droite, nx en bas / haut."""
        return self.ny if side in ("left", "right") else self.nx

    def side_extent(self, side: str) -> float:
        """Longueur du mur en unités domaine."""
        return self.side_len(side) / self.n

    @property
    def bound(self) -> int:
        return self.runner.p["bound"]

    # ------------------------------------------------------------ image
    def set_image(self, buf: np.ndarray) -> None:
        self._buf = np.ascontiguousarray(buf)             # QImage ne possède pas la mémoire : garder une référence
        h, w, _ = self._buf.shape
        self._qimg = QImage(self._buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(700, 700)

    # ------------------------------------------------------------ transformations
    def _square(self):
        side = min(self.width(), self.height())
        return (self.width() - side) // 2, (self.height() - side) // 2, side

    def px_to_dom(self, pos) -> tuple[float, float]:
        ox, oy, side = self._square()
        sx, sy = (pos.x() - ox) / side, (pos.y() - oy) / side
        return self.x0 + sx / self.scale, self.y0 + (1.0 - sy) / self.scale

    def dom_to_px(self, x: float, y: float) -> tuple[float, float]:
        ox, oy, side = self._square()
        return ox + (x - self.x0) * self.scale * side, oy + (1.0 - (y - self.y0) * self.scale) * side

    def px_per_unit(self) -> float:
        return self._square()[2] * self.scale

    def cell_at(self, pos) -> tuple[int, int]:
        x, y = self.px_to_dom(pos)
        return int(np.clip(int(x * self.n), 0, self.nx - 1)), int(np.clip(int(y * self.n), 0, self.ny - 1))

    def wall_at(self, pos):
        """(side, k) si le curseur est à moins de SNAP_PX d'un bord du domaine ou dans la bande de paroi
        (hors coins), sinon None. k = cellule le long du mur, bornée à la zone utilisable."""
        x, y = self.px_to_dom(pos)
        ppu = self.px_per_unit()
        n, b = self.n, self.bound
        Lx, Ly = self.nx / n, self.ny / n
        cands = {"left": abs(x) * ppu, "right": abs(Lx - x) * ppu, "bottom": abs(y) * ppu, "top": abs(Ly - y) * ppu}
        side = min(cands, key=cands.get)
        band = b / n
        in_band = {"left": 0.0 <= x < band, "right": Lx - band < x <= Lx,
                   "bottom": 0.0 <= y < band, "top": Ly - band < y <= Ly}[side]
        if cands[side] > SNAP_PX and not in_band:
            return None
        along = y if side in ("left", "right") else x
        if not (-SNAP_PX / ppu <= along <= self.side_extent(side) + SNAP_PX / ppu):
            return None
        k = int(np.clip(int(along * n), b, self.side_len(side) - b - 1))
        return side, k

    def _clamp_view(self) -> None:
        span = 1.0 / self.scale
        self.x0 = float(np.clip(self.x0, 0.0, 1.0 - span))
        self.y0 = float(np.clip(self.y0, 0.0, 1.0 - span))

    def reset_view(self) -> None:
        self.x0, self.y0, self.scale = 0.0, 0.0, 1.0
        self.update()

    # ------------------------------------------------------------ segments de paroi (géométrie écran)
    def _edge_point(self, side: str, along: float, depth_px: float = 0.0) -> QPointF:
        """Point du bord `side` à la coordonnée `along` (unités domaine), décalé de depth_px vers l'intérieur."""
        if side == "left":
            px, py = self.dom_to_px(0.0, along)
            return QPointF(px + depth_px, py)
        if side == "right":
            px, py = self.dom_to_px(self.nx / self.n, along)
            return QPointF(px - depth_px, py)
        if side == "bottom":
            px, py = self.dom_to_px(along, 0.0)
            return QPointF(px, py - depth_px)
        px, py = self.dom_to_px(along, self.ny / self.n)
        return QPointF(px, py + depth_px)

    def _segment_rect(self, side: str, a: float, b: float, thick: float, inset: float = 0.0) -> QRectF:
        p0, p1 = self._edge_point(side, a, inset), self._edge_point(side, b, inset + thick)
        return QRectF(p0, p1).normalized()

    def _handle_at(self, pos):
        """(indice du segment, extrémité) si le curseur est sur une poignée d'un segment existant."""
        for idx, w in enumerate(self.runner.walls):
            for end in (0, 1):
                c = self._edge_point(w["side"], w["span"][end], 5.0)
                if abs(c.x() - pos.x()) <= HANDLE_PX and abs(c.y() - pos.y()) <= HANDLE_PX:
                    return idx, end
        return None

    def _snap_along(self, side: str, pos) -> float:
        x, y = self.px_to_dom(pos)
        along = y if side in ("left", "right") else x
        n, b = self.n, self.bound
        k = int(np.clip(round(along * n), b, self.side_len(side) - b))
        return k / n

    # ------------------------------------------------------------ souris / clavier
    def wheelEvent(self, event) -> None:
        xc, yc = self.px_to_dom(event.position())
        new_scale = float(np.clip(self.scale * 1.25 ** (event.angleDelta().y() / 120.0), 1.0, 64.0))
        self.x0 = xc - (xc - self.x0) * self.scale / new_scale       # le point sous le curseur reste fixe
        self.y0 = yc - (yc - self.y0) * self.scale / new_scale
        self.scale = new_scale
        self._clamp_view()
        self.update()

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            self._moved = False
            handle = self._handle_at(event.position())
            wall = self.wall_at(event.position())
            if handle is not None:
                self._hdrag = handle
                self.sel = None
                self.selectionChanged.emit(None)
            elif wall is not None:
                side, k = wall
                self._wdrag = (side, k)
                self.wsel = (side, k, k)
                self.sel = None
                self.selectionChanged.emit(None)
                self.wallSelectionChanged.emit(self.wsel)
            else:
                i, j = self.cell_at(event.position())
                self._drag = (i, j)
                self.sel = (i, j, i, j)
                self.wsel = None
                self.wallSelectionChanged.emit(None)
                self.selectionChanged.emit(self.sel)
            self.update()
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan = event.position()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._drag is not None:
            i, j = self.cell_at(pos)
            i0, j0 = self._drag
            self.sel = (min(i0, i), min(j0, j), max(i0, i), max(j0, j))
            self.selectionChanged.emit(self.sel)
            self.update()
        elif self._wdrag is not None:
            side, k0 = self._wdrag
            x, y = self.px_to_dom(pos)
            along = y if side in ("left", "right") else x
            k = int(np.clip(int(along * self.n), self.bound, self.side_len(side) - self.bound - 1))
            if k != k0:
                self._moved = True
            self.wsel = (side, min(k0, k), max(k0, k))
            self.wallSelectionChanged.emit(self.wsel)
            self.update()
        elif self._hdrag is not None:
            idx, end = self._hdrag
            w = self.runner.walls[idx]
            v = self._snap_along(w["side"], pos)
            other = w["span"][1 - end]
            if (end == 0 and v < other) or (end == 1 and v > other):
                w["span"][end] = v
                self._moved = True
                self.update()
        elif self._pan is not None:
            _, _, side = self._square()
            d = pos - self._pan
            self._pan = pos
            self.x0 -= d.x() / (side * self.scale)
            self.y0 += d.y() / (side * self.scale)
            self._clamp_view()
            self.update()
        # survol : mur ou cellule
        wall = self.wall_at(pos) if self._drag is None else None
        if wall != self.hover_wall:
            self.hover_wall = wall
            self.update()
        x, y = self.px_to_dom(pos)
        if wall is not None:
            side, k = wall
            axis = "y" if side in ("left", "right") else "x"
            self.hoverChanged.emit(f"paroi {SIDE_LABELS[side]}  {axis} = {(k + 0.5) / self.n:.3f}  (cellule {k})")
        elif 0.0 <= x < self.nx / self.n and 0.0 <= y < self.ny / self.n:
            self.hoverChanged.emit(f"cellule ({int(x * self.n)}, {int(y * self.n)})  x={x:.3f} y={y:.3f}")

    def mouseReleaseEvent(self, event) -> None:
        if self._wdrag is not None and not self._moved:          # clic simple : tout le mur
            side, _ = self._wdrag
            self.wsel = (side, self.bound, self.side_len(side) - self.bound - 1)
            self.wallSelectionChanged.emit(self.wsel)
            self.update()
        if self._hdrag is not None and self._moved:
            self.wallEdited.emit()
        self._drag = self._wdrag = self._hdrag = self._pan = None

    def leaveEvent(self, event) -> None:
        if self.hover_wall is not None:
            self.hover_wall = None
            self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F:
            self.reset_view()
        elif event.key() == Qt.Key.Key_Escape:
            self.sel = None
            self.wsel = None
            self.selectionChanged.emit(None)
            self.wallSelectionChanged.emit(None)
            self.update()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------ dessin
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(16, 16, 24))
        ox, oy, side = self._square()
        clip = QRectF(ox, oy, side, side)
        if self._qimg is not None:
            painter.drawImage(QRect(ox, oy, side, side), self._qimg)
        painter.setClipRect(clip)
        n, b = self.n, self.bound
        thick = max(4.0, min(10.0, b / n * self.px_per_unit()))

        # segments de paroi de la scène (état courant du runner, même avant Reset)
        wtype = self.runner.wall_table()[0]
        for s, sname in enumerate(SIDES):
            k = b
            ln = self.side_len(sname)
            while k < ln - b:
                t = wtype[s, k]
                if t == 0:
                    k += 1
                    continue
                k0 = k
                while k < ln - b and wtype[s, k] == t:
                    k += 1
                col = WALL_COLORS["inlet" if t == 1 else "outlet"]
                painter.fillRect(self._segment_rect(sname, k0 / n, k / n, thick), QColor(col.red(), col.green(), col.blue(), 170))
        for w in self.runner.walls:                         # contour + poignées du segment tel que défini
            col = WALL_COLORS[w["type"]]
            painter.setPen(QPen(col, 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._segment_rect(w["side"], w["span"][0], w["span"][1], thick))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(col, 1.5))
            for end in (0, 1):
                c = self._edge_point(w["side"], w["span"][end], 5.0)
                painter.drawRect(QRectF(c.x() - 3.5, c.y() - 3.5, 7.0, 7.0))

        # mur survolé : tout le bord s'éclaire
        if self.hover_wall is not None and self._drag is None:
            sname = self.hover_wall[0]
            ext = self.side_extent(sname)
            painter.fillRect(self._segment_rect(sname, 0.0, ext, thick + 3, -1.5), QColor(255, 255, 255, 70))
            painter.setPen(QPen(QColor(255, 255, 255, 230), 2.5))
            painter.drawLine(self._edge_point(sname, 0.0), self._edge_point(sname, ext))

        # sélection de mur
        if self.wsel is not None:
            sname, k0, k1 = self.wsel
            r = self._segment_rect(sname, k0 / n, (k1 + 1) / n, thick + 4, -2.0)
            painter.fillRect(r, QColor(255, 255, 255, 90))
            painter.setPen(QPen(QColor(255, 255, 255), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r)

        # sélection de cellules
        if self.sel is not None:
            i0, j0, i1, j1 = self.sel
            px0, py1 = self.dom_to_px(i0 / n, j0 / n)
            px1, py0 = self.dom_to_px((i1 + 1) / n, (j1 + 1) / n)
            r = QRectF(px0, py0, px1 - px0, py1 - py0).intersected(clip)
            painter.fillRect(r, QColor(255, 255, 255, 40))
            painter.setPen(QPen(QColor(255, 255, 255), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r)
        painter.end()


class UI(QMainWindow):
    def __init__(self, runner: SimulationRunner, path: str | None = None):
        super().__init__()
        self.runner = runner
        self.path = path
        self.playing = False
        self.dirty = False
        self._frames = 0
        self._editors: dict[str, QWidget] = {}

        self.viewport = Viewport(runner)
        self.banner = QLabel("Modifié — Reset (R) pour appliquer")
        self.banner.setStyleSheet("background: #d0781a; color: white; padding: 4px; font-weight: bold;")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.hide()
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.banner)
        lay.addWidget(self.viewport, 1)
        self.setCentralWidget(central)

        self._build_params_dock()
        self._build_scene_dock()
        self._build_toolbar()
        self._build_statusbar()

        self.viewport.selectionChanged.connect(self._on_selection)
        self.viewport.wallSelectionChanged.connect(self._on_wall_selection)
        self.viewport.wallEdited.connect(self._mark_dirty)
        self.viewport.hoverChanged.connect(self.lbl_hover.setText)

        self.solver = None                                # construit après l'affichage de la fenêtre
        self.statusBar().showMessage("Semis et compilation des kernels…")
        QTimer.singleShot(0, self._rebuild)
        self._update_title()

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        secs = os.environ.get("APIC_UI_AUTOQUIT")        # diagnostic sans intervention
        if secs:
            QTimer.singleShot(int(float(secs) * 1000), self._autoquit)
            QTimer.singleShot(200, self._autoplay)

    # ------------------------------------------------------------ construction
    def _make_editor(self, key: str):
        default, value = SimulationRunner.PARAMS[key], self.runner.p[key]
        if isinstance(default, bool):
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v, k=key: self._on_param(k, bool(v)))
        elif isinstance(default, int):
            w = QSpinBox()
            w.setRange(*INT_RANGES.get(key, (0, 10**9)))
            w.setValue(int(value))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, k=key: self._on_param(k, int(v)))
        else:
            w = QDoubleSpinBox()
            w.setDecimals(6 if abs(default) < 0.01 else 3)
            w.setRange(*FLOAT_RANGES.get(key, (-1e6, 1e9)))
            w.setSingleStep(abs(default) / 10 if default else 0.1)
            w.setValue(float(value))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, k=key: self._on_param(k, float(v)))
        self._editors[key] = w
        return w

    def _build_params_dock(self) -> None:
        """Arbre de paramètres rangé par dossiers ; les dossiers « État initial » et « Conditions aux
        limites » résument la scène courante (mis à jour à chaque modification)."""
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Paramètre", "Valeur"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setStretchLastSection(True)

        def folder(parent, name):
            item = QTreeWidgetItem(parent or self.tree, [name])
            item.setExpanded(True)
            item.setFirstColumnSpanned(True) if parent is None else None
            return item

        def add_params(parent, keys):
            for key in keys:
                label = SimulationRunner.LABELS.get(key, key) + (" *" if key in SimulationRunner.STRUCTURAL else "")
                item = QTreeWidgetItem(parent, [label])
                self.tree.setItemWidget(item, 1, self._make_editor(key))

        add_params(folder(None, "Domaine"), ["nx", "ny", "bound", "ppc", "res", "seed", "capacity"])
        add_params(folder(None, "Solveur"), ["cfl", "substeps", "gravity", "incompressible", "free_surface",
                                             "cg_iters", "volume_correction", "density_iters", "obstacle_friction",
                                             "use_damage", "use_rupture", "color_mode"])
        mats = folder(None, "Matériaux")
        add_params(folder(mats, "Fluide"), ["fluid_rho", "fluid_E"])
        add_params(folder(mats, "Solide"), ["solid_rho", "solid_E", "solid_nu", "eps0", "epsf", "tau_D", "k_res"])
        self.item_ic = folder(None, "État initial")
        self.item_bc = folder(None, "Conditions aux limites (parois)")
        note = QTreeWidgetItem(self.tree, ["* : appliqué au Reset"])
        note.setFirstColumnSpanned(True)
        self._refresh_summaries()

        dock = QDockWidget("Paramètres", self)
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [360], Qt.Orientation.Horizontal)

    def _refresh_summaries(self) -> None:
        """Dossiers « État initial » et « Conditions aux limites » d'après le runner."""
        m, p, r = self.runner.m, self.runner.p, self.runner
        for item in (self.item_ic, self.item_bc):
            item.takeChildren()

        def row(parent, name, text):
            QTreeWidgetItem(parent, [name, text])

        def vel_groups(mask, vx, vy):
            """Regroupe les cellules d'un masque par vitesse (vx, vy) identique."""
            if not mask.any():
                return []
            pairs = np.stack([vx[mask], vy[mask]], axis=1)
            uniq, counts = np.unique(pairs, axis=0, return_counts=True)
            return [((float(a), float(b)), int(c)) for (a, b), c in zip(uniq, counts)]

        n_f, n_s = int(m["fluid"].sum()), int(m["solid"].sum())
        row(self.item_ic, "Fluide", f"{n_f} cellules" if n_f else "aucune")
        for (vx, vy), c in vel_groups(m["fluid"], m["vx0"], m["vy0"]):
            if vx or vy:
                row(self.item_ic, "   vitesse", f"({vx:g}, {vy:g}) sur {c} cellules")
        row(self.item_ic, "Solide", f"{n_s} cellules" if n_s else "aucune")
        for (vx, vy), c in vel_groups(m["solid"], m["vx0"], m["vy0"]):
            if vx or vy:
                row(self.item_ic, "   vitesse", f"({vx:g}, {vy:g}) sur {c} cellules")
        n_o = int(m["obstacle"].sum())
        row(self.item_ic, "Obstacles", f"{n_o} cellules" if n_o else "aucun")

        row(self.item_bc, "Par défaut", f"mur glissant, bande de {p['bound']} cellules ; "
                                        f"obstacles : frottement β = {p['obstacle_friction']:g}")
        tab = r.wall_table()
        wdepth = tab.depth
        if not r.walls:
            row(self.item_bc, "Segments", "aucun (clic sur un bord du domaine)")
        for k, w in enumerate(r.walls, start=1):
            s = SIDES.index(w["side"])
            axis = "y" if w["side"] in ("left", "right") else "x"
            ln = r.ny if w["side"] in ("left", "right") else r.nx
            k0, k1 = int(np.floor(w["span"][0] * r.n + 1e-9)), int(np.ceil(w["span"][1] * r.n - 1e-9))
            k0, k1 = max(k0, p["bound"]), min(k1, ln - p["bound"])
            moved = int((wdepth[s, k0:k1] > p["bound"]).sum()) if k1 > k0 else 0
            kind = {"inlet": "entrée", "outlet": "sortie", "wall": "mur"}[w["type"]]
            vel = f" v = ({w['velocity'][0]:g}, {w['velocity'][1]:g})," if w["type"] == "inlet" else ""
            if w["type"] == "outlet":
                vel = f" pression imposée p = {w['pressure']:g}," if w.get("pressure") else " libre (p = 0),"
            elif w["type"] == "wall":
                vel = f" frottement β = {w.get('friction', 0.0):g},"
            note = f", portée par la face d'un obstacle sur {moved} cellules" if moved else ""
            row(self.item_bc, f"{k}. paroi {SIDE_LABELS[w['side']]}",
                f"{kind},{vel} {axis} ∈ [{w['span'][0]:.3f}, {w['span'][1]:.3f}]{note}")

    def _build_scene_dock(self) -> None:
        w = QWidget()
        grid = QGridLayout(w)
        # ---- cellules (intérieur)
        self.lbl_sel = QLabel("Aucune sélection\n(bouton gauche : boîte de cellules)")
        grid.addWidget(self.lbl_sel, 0, 0, 1, 2)
        grid.addWidget(QLabel("vx"), 1, 0)
        self.spin_vx = QDoubleSpinBox()
        self.spin_vx.setRange(-100, 100)
        grid.addWidget(self.spin_vx, 1, 1)
        grid.addWidget(QLabel("vy"), 2, 0)
        self.spin_vy = QDoubleSpinBox()
        self.spin_vy.setRange(-100, 100)
        grid.addWidget(self.spin_vy, 2, 1)
        actions = [("Eau", lambda m: self.runner.set_fluid(m, self._vel())),
                   ("Solide", lambda m: self.runner.set_solid(m, self._vel())),
                   ("Obstacle", self.runner.set_obstacle),
                   ("Vitesse initiale (vx, vy)", lambda m: self.runner.set_velocity(m, self._vel())),
                   ("Effacer", self.runner.clear)]
        self._cell_buttons = []
        for row, (text, fn) in enumerate(actions, start=3):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: self._apply_cells(f))
            b.setEnabled(False)
            grid.addWidget(b, row, 0, 1, 2)
            self._cell_buttons.append(b)
        base = 3 + len(actions)
        # ---- parois (conditions aux limites)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        grid.addWidget(sep, base, 0, 1, 2)
        self.lbl_wall = QLabel("Aucune paroi sélectionnée\n(clic sur un bord : tout le mur ; glisser : un segment)")
        self.lbl_wall.setWordWrap(True)
        grid.addWidget(self.lbl_wall, base + 1, 0, 1, 2)
        grid.addWidget(QLabel("vx"), base + 2, 0)
        self.spin_wvx = QDoubleSpinBox()
        self.spin_wvx.setRange(-100, 100)
        self.spin_wvx.setValue(1.0)
        grid.addWidget(self.spin_wvx, base + 2, 1)
        grid.addWidget(QLabel("vy"), base + 3, 0)
        self.spin_wvy = QDoubleSpinBox()
        self.spin_wvy.setRange(-100, 100)
        grid.addWidget(self.spin_wvy, base + 3, 1)
        grid.addWidget(QLabel("Sortie : pression p"), base + 4, 0)
        self.spin_wp = QDoubleSpinBox()                 # pression imposée sur une sortie, 0 = libre
        self.spin_wp.setRange(0.0, 1e9)
        self.spin_wp.setDecimals(3)
        self.spin_wp.setSpecialValueText("libre (p = 0)")
        self.spin_wp.setToolTip("Pression imposée sur la sortie (Dirichlet, mode incompressible) ; 0 = sortie libre")
        grid.addWidget(self.spin_wp, base + 4, 1)
        grid.addWidget(QLabel("Mur : frottement β"), base + 5, 0)
        self.spin_wf = QDoubleSpinBox()                 # 0 = glissant (défaut), 1 = adhérent
        self.spin_wf.setRange(0.0, 1.0)
        self.spin_wf.setDecimals(2)
        self.spin_wf.setSingleStep(0.1)
        self.spin_wf.setToolTip("Frottement du mur : 0 = glissant (défaut, efface le segment), 1 = adhérent (no-slip)")
        grid.addWidget(self.spin_wf, base + 5, 1)
        self._wall_buttons = []
        for row, (text, kind) in enumerate([("Entrée (vx, vy)", "inlet"), ("Sortie", "outlet"),
                                            ("Mur (frottement β)", "wall")], start=base + 6):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=kind: self._apply_wall(k))
            b.setEnabled(False)
            grid.addWidget(b, row, 0, 1, 2)
            self._wall_buttons.append(b)
        base = base + 9
        # ---- affichage
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        grid.addWidget(sep2, base, 0, 1, 2)
        self.chk_grid = QCheckBox("Maillage (si assez zoomé)")
        self.chk_grid.setChecked(True)
        self.chk_tint = QCheckBox("Teintes des cellules initiales")
        self.chk_tint.setChecked(True)
        grid.addWidget(self.chk_grid, base + 1, 0, 1, 2)
        grid.addWidget(self.chk_tint, base + 2, 0, 1, 2)
        grid.addWidget(QLabel("Couleur du fluide"), base + 3, 0)
        self.combo_color = QComboBox()
        self.combo_color.addItems(FLUID_MODES)
        self.combo_color.currentIndexChanged.connect(self._on_color_mode)
        grid.addWidget(self.combo_color, base + 3, 1)
        self.lbl_scale = QLabel("")
        grid.addWidget(self.lbl_scale, base + 4, 0, 1, 2)
        grid.setRowStretch(base + 5, 1)
        dock = QDockWidget("Scène", self)
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Simulation")
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, checkable=False):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
                a.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            a.setCheckable(checkable)
            (a.toggled if checkable else a.triggered).connect(slot)
            tb.addAction(a)
            return a

        self.act_play = act("▶ Play", self._set_playing, "Space", checkable=True)
        act("Step", self._step, "S")
        act("Reset", self._reset, "R")
        act("Vue entière", self.viewport.reset_view, "F")
        tb.addSeparator()
        act("Ouvrir…", self._open, "Ctrl+O")
        act("Enregistrer", self._save, "Ctrl+S")
        act("Enregistrer sous…", self._save_as, "Ctrl+Shift+S")

    def _build_statusbar(self) -> None:
        self.lbl_hover = QLabel()
        self.lbl_time = QLabel()
        self.lbl_particles = QLabel()
        self.lbl_perf = QLabel()
        for lbl in (self.lbl_hover, self.lbl_time, self.lbl_particles, self.lbl_perf):
            lbl.setMargin(4)
            self.statusBar().addPermanentWidget(lbl)

    # ------------------------------------------------------------ boucle
    def _tick(self) -> None:
        if self.solver is None:
            return
        if self.playing:
            self.solver.step()
        vp = self.viewport
        self.solver.fluid_mode = self.combo_color.currentIndex()
        img = self.solver.render(vp.x0, vp.y0, vp.scale, self.chk_grid.isChecked(), self.chk_tint.isChecked())
        vp.set_image(img)
        self._frames += 1
        if self._frames % 6 == 0:
            st = self.solver.stats()
            mode = self.solver.fluid_mode
            if mode == 0:
                self.lbl_scale.setText("")
            else:
                rng = f"0 … {st['scalar_max']:.3g}" if mode == 1 else f"−{st['scalar_max']:.3g} … +{st['scalar_max']:.3g}"
                self.lbl_scale.setText(f"échelle : {rng} (auto)")
            self.lbl_time.setText(f"t = {st['t']:.3f} s   dt = {st['dt']:.2e}")
            cap = f" / {st['capacity']}" if st["capacity"] != st["n_fluid"] else ""
            self.lbl_particles.setText(f"fluide {st['n_fluid']}{cap}   solide {st['n_solid']}   "
                                       f"D max {st['D_max']:.2f}   rompues {st['n_broken']}")
            self.lbl_perf.setText(f"{self.solver.arch} · {st['ms']:.1f} ms/image" if self.playing else f"{self.solver.arch} · pause")

    def _rebuild(self) -> None:
        self.statusBar().showMessage("Semis et compilation des kernels…")
        self.statusBar().repaint()
        QApplication.processEvents()                      # laisse la fenêtre se peindre avant le calcul bloquant
        if self.solver is not None:
            self.solver.release()
            self.solver = None
        solver = self.runner.solver()
        solver.step(1)                                    # compile les kernels de simulation maintenant,
        solver.reset()                                    # pas au premier clic sur Play
        self.solver = solver
        self.dirty = False
        self.banner.hide()
        self.statusBar().clearMessage()

    def _mark_dirty(self) -> None:
        self.dirty = True
        self.banner.show()
        self._refresh_summaries()
        self._update_title(modified=True)
        self.viewport.update()

    # ------------------------------------------------------------ actions
    def _set_playing(self, on: bool) -> None:
        self.playing = bool(on) and self.solver is not None
        if self.act_play.isChecked() != self.playing:
            self.act_play.blockSignals(True)
            self.act_play.setChecked(self.playing)
            self.act_play.blockSignals(False)
        self.act_play.setText("❚❚ Pause" if self.playing else "▶ Play")
        if self.playing and self.dirty:
            self._rebuild()

    def _step(self) -> None:
        self._set_playing(False)
        if self.dirty:
            self._rebuild()
        if self.solver is not None:
            self.solver.step()

    def _reset(self) -> None:
        if self.dirty or self.solver is None:
            self._rebuild()
        else:
            self.solver.reset()

    def _on_param(self, key: str, value) -> None:
        if key in SimulationRunner.STRUCTURAL:
            if key in ("nx", "ny"):
                nx, ny = (int(value), self.runner.ny) if key == "nx" else (self.runner.nx, int(value))
                self.runner.resize(nx, ny)
            else:
                self.runner.p[key] = value
            self._mark_dirty()
        else:
            self.runner.p[key] = value
            if self.solver is not None:
                self.solver.set_params({key: value})
            self._update_title(modified=True)

    def _on_color_mode(self, _index: int) -> None:
        if self.solver is not None:
            self.solver.scalar_max = 0.0             # l'échelle se recalcule sur la nouvelle quantité

    def _vel(self) -> tuple[float, float]:
        return self.spin_vx.value(), self.spin_vy.value()

    def _on_selection(self, sel) -> None:
        for b in self._cell_buttons:
            b.setEnabled(sel is not None)
        if sel is None:
            self.lbl_sel.setText("Aucune sélection\n(bouton gauche : boîte de cellules)")
        else:
            i0, j0, i1, j1 = sel
            self.lbl_sel.setText(f"Sélection : i {i0}..{i1}, j {j0}..{j1}\n{(i1 - i0 + 1) * (j1 - j0 + 1)} cellules")

    def _on_wall_selection(self, sel) -> None:
        for b in self._wall_buttons:
            b.setEnabled(sel is not None)
        if sel is None:
            self.lbl_wall.setText("Aucune paroi sélectionnée\n(clic sur un bord : tout le mur ; glisser : un segment)")
        else:
            side, k0, k1 = sel
            n = self.runner.n
            axis = "y" if side in ("left", "right") else "x"
            ln = self.viewport.side_len(side)
            whole = " (mur entier)" if (k0, k1) == (self.runner.p["bound"], ln - self.runner.p["bound"] - 1) else ""
            self.lbl_wall.setText(f"Paroi {SIDE_LABELS[side]}{whole}\n{axis} de {k0 / n:.3f} à {(k1 + 1) / n:.3f}, "
                                  f"cellules {k0}..{k1}")

    def _apply_cells(self, fn) -> None:
        if self.viewport.sel is None:
            return
        i0, j0, i1, j1 = self.viewport.sel
        mask = np.zeros((self.runner.nx, self.runner.ny), bool)
        mask[i0:i1 + 1, j0:j1 + 1] = True
        fn(mask)
        self._mark_dirty()

    def _apply_wall(self, kind: str) -> None:
        if self.viewport.wsel is None:
            return
        side, k0, k1 = self.viewport.wsel
        n = self.runner.n
        pressure = self.spin_wp.value() if kind == "outlet" and self.spin_wp.value() > 0 else None
        friction = self.spin_wf.value() if kind == "wall" else None
        self.runner.set_wall(side, kind, velocity=(self.spin_wvx.value(), self.spin_wvy.value()),
                             span=(k0 / n, (k1 + 1) / n), pressure=pressure, friction=friction)
        self._mark_dirty()

    # ------------------------------------------------------------ fichiers
    def _load(self, runner: SimulationRunner, path: str | None) -> None:
        self._set_playing(False)
        self.runner = runner
        self.viewport.runner = runner
        self.path = path
        for key, w in self._editors.items():
            w.blockSignals(True)
            v = runner.p[key]
            w.setChecked(bool(v)) if isinstance(w, QCheckBox) else w.setValue(v)
            w.blockSignals(False)
        self.viewport.sel = None
        self.viewport.wsel = None
        self.viewport.reset_view()
        self._on_selection(None)
        self._on_wall_selection(None)
        self._refresh_summaries()
        self._rebuild()
        self._update_title()

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une simulation", "", "Simulation (*.json)")
        if not path:
            return
        try:
            self._load(SimulationRunner.load(path), path)
        except Exception as exc:
            QMessageBox.critical(self, "Ouverture impossible", str(exc))

    def _save(self) -> None:
        if self.path is None:
            self._save_as()
            return
        self.runner.save(self.path)
        self._update_title()
        self.statusBar().showMessage(f"Enregistré : {self.path}", 3000)

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la simulation", self.path or "simulation.json",
                                              "Simulation (*.json)")
        if path:
            self.path = path
            self._save()

    def _update_title(self, modified: bool = False) -> None:
        name = os.path.basename(self.path) if self.path else "simulation"
        self.setWindowTitle(f"{name}{'*' if modified else ''} — APIC / MPM")

    # ------------------------------------------------------------ fin
    def _autoplay(self) -> None:
        if self.solver is None:
            QTimer.singleShot(200, self._autoplay)
        else:
            self._set_playing(True)

    def _autoquit(self) -> None:
        st = self.solver.stats() if self.solver is not None else {}
        print(f"[diag] frames={self._frames} stats={st} view={self.viewport.scale}", flush=True)
        self.close()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self.solver is not None:
            self.solver.release()
        super().closeEvent(event)

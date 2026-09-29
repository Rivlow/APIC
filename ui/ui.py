"""Interface Qt : Viewport (image rendue sur le GPU, zoom, pan, sélection de cellules et de parois) et UI.

Souris dans le viewport : molette = zoom autour du curseur, bouton droit ou milieu = déplacement de la
vue, F = vue entière, Échap = désélection. Bouton gauche :
  - à l'intérieur du domaine : boîte de sélection de cellules (eau, solide, obstacle, vitesse, effacer) ;
  - près d'un bord ou dans la bande de paroi, le mur s'éclaire : clic = tout le mur, glisser le long du
    mur = un segment, aimanté aux frontières de cellules ; les segments existants (vert entrée, rouge
    sortie) ont deux poignées à étirer. Les conditions aux limites n'existent que sur les quatre parois.
Toute modification de la scène ou d'un paramètre structurel demande un Reset (bandeau orange).
En 3D (dim = 3), la vue est ui/view3d.Viewport3D (caméra orbitale, zones de paroi rectangulaires, primitives) et
un dock « Primitives » remplace la boîte de cellules. En pause, l'image n'est recalculée que si quelque chose
change (vue, scène, couleur, pas de simulation) : aucune copie GPU -> CPU inutile.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
                               QFrame, QGridLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPushButton,
                               QSpinBox, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ui.kernels import MODE_DENSITY, fluid_modes, mode_signed
from ui.runner import SIDES, SimulationRunner
from ui.view3d import PrimPanel, Viewport3D

INT_RANGES = {"nx": (8, 8192), "bound": (1, 16), "ppc": (1, 4), "substeps": (1, 1000), "seed": (0, 10**9),
              "capacity": (0, 50_000_000), "res": (200, 2000), "color_mode": (0, 1), "dim": (2, 3)}
SIDE_LABELS = {"left": "gauche", "right": "droite", "bottom": "bas", "top": "haut", "back": "arrière",
               "front": "avant"}
FLOAT_RANGES = {"obstacle_friction": (0.0, 1.0), "volume_correction": (0.0, 2.0), "Lx": (1e-4, 1e6),
                "Ly": (1e-4, 1e6), "Lz": (1e-4, 1e6), "fluid_nu": (0.0, 1e3), "smagorinsky": (0.0, 20.0)}
WALL_COLORS = {"inlet": QColor(60, 200, 90), "outlet": QColor(230, 70, 70), "wall": QColor(230, 190, 60)}
SNAP_PX = 12                                   # distance au bord (px) qui fait passer en mode paroi
HANDLE_PX = 7


def _parse(text: str):
    """Boundary value typed in a field: number if it parses, else expression string.

    **Inputs**

    - `text` : str

    **Outputs**

    - float | str
    """
    text = text.strip() or "0"
    try:
        return float(text)
    except ValueError:
        return text


def _fmt(value) -> str:
    """Display a boundary value (number or expression).

    **Inputs**

    - `value` : float | str

    **Outputs**

    - str
    """
    return f"{value:g}" if isinstance(value, (int, float)) else f"« {value} »"


class Viewport(QWidget):
    selectionChanged = Signal(object)          # (i0, j0, i1, j1) cellules incluses, ou None
    wallSelectionChanged = Signal(object)      # (side, k0, k1) cellules le long du mur incluses, ou None
    wallEdited = Signal()                      # une poignée de segment a été déplacée (runner.walls modifié)
    hoverChanged = Signal(str)
    viewChanged = Signal()                     # zoom, déplacement ou redimensionnement : l'image doit être refaite

    def __init__(self, runner: SimulationRunner, parent=None):
        """Create the viewport on a runner (full view, no selection).

        **Inputs**

        - `runner` : SimulationRunner
        - `parent` : QWidget | None
        """
        super().__init__(parent)
        self.runner = runner
        # vue = toute la fenêtre : coin bas-gauche (x0, y0) en m, mpp = mètres par pixel écran ; _fit : la vue suit
        # la fenêtre (boîte entière) tant que l'on n'a ni zoomé ni déplacé
        self.x0, self.y0, self.mpp, self._fit = 0.0, 0.0, 0.0, True
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
    def n(self) -> float:
        """Cells per metre, 1 / dx (float) : k / n = cell -> m, x * n = m -> cell."""
        return 1.0 / self.runner.dx

    @property
    def Lx(self) -> float:
        """Box width (m)."""
        return self.runner.Lx

    @property
    def Ly(self) -> float:
        """Box height (m)."""
        return self.runner.Ly

    @property
    def nx(self) -> int:
        """Number of cells in x (int)."""
        return self.runner.nx

    @property
    def ny(self) -> int:
        """Number of cells in y (int)."""
        return self.runner.ny

    def side_len(self, side: str) -> int:
        """Number of cells along a wall.

        **Inputs**

        - `side` : str   left / right / bottom / top

        **Outputs**

        - `int` : ny for left/right, nx for bottom/top
        """
        return self.ny if side in ("left", "right") else self.nx

    def side_extent(self, side: str) -> float:
        """Wall length (m).

        **Inputs**

        - `side` : str   left / right / bottom / top

        **Outputs**

        - float
        """
        return self.side_len(side) / self.n

    @property
    def bound(self) -> int:
        """Wall band thickness in cells (int)."""
        return self.runner.p["bound"]

    # ------------------------------------------------------------ image
    def set_image(self, buf: np.ndarray) -> None:
        """Display a rendered image.

        **Inputs**

        - `buf` : np.ndarray u8 (res_y, res_x, 3), box aspect ratio

        **Outputs**

        - self._buf, self._qimg set, repaint scheduled

        **Note** : keeps a reference to the buffer (QImage does not own the memory).
        """
        self._buf = np.ascontiguousarray(buf)             # QImage ne possède pas la mémoire : garder une référence
        h, w, _ = self._buf.shape
        self._qimg = QImage(self._buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.update()

    def sizeHint(self) -> QSize:
        """Preferred widget size.

        **Outputs**

        - QSize
        """
        return QSize(700, 700)

    # ------------------------------------------------------------ transformations
    def _fit_mpp(self) -> float:
        """Metres per screen pixel that fit the whole box in the widget (small margin).

        **Outputs**

        - float
        """
        W, H = max(self.width(), 1), max(self.height(), 1)
        return 1.03 * max(self.Lx / W, self.Ly / H)

    def _fit_view(self) -> None:
        """Center the whole box in the widget.

        **Outputs**

        - self.x0, self.y0, self.mpp set
        """
        self.mpp = self._fit_mpp()
        self.x0 = 0.5 * self.Lx - 0.5 * self.width() * self.mpp
        self.y0 = 0.5 * self.Ly - 0.5 * self.height() * self.mpp

    def _view(self) -> float:
        """Current metres per pixel (fits the box first if the view follows the window).

        **Outputs**

        - float
        """
        if self._fit or self.mpp <= 0.0:
            self._fit_view()
        return self.mpp

    def px_to_dom(self, pos) -> tuple[float, float]:
        """Convert a widget pixel position to box coordinates (m).

        **Inputs**

        - `pos` : QPointF

        **Outputs**

        - tuple[float, float]   (x, y)
        """
        mpp = self._view()
        return self.x0 + pos.x() * mpp, self.y0 + (self.height() - pos.y()) * mpp

    def dom_to_px(self, x: float, y: float) -> tuple[float, float]:
        """Convert box coordinates (m) to widget pixels.

        **Inputs**

        - `x`, `y` : float

        **Outputs**

        - tuple[float, float]   (px, py)
        """
        mpp = self._view()
        return (x - self.x0) / mpp, self.height() - (y - self.y0) / mpp

    def px_per_unit(self) -> float:
        """Screen pixels per metre at the current zoom.

        **Outputs**

        - float
        """
        return 1.0 / self._view()

    def render_view(self) -> tuple[float, float, float, tuple[int, int]]:
        """Image the solver must render for the current view (window aspect, at most res pixels on the long side).

        **Outputs**

        - tuple : x0, y0 (m, bottom-left), k (m per image pixel), (width, height) px
        """
        mpp = self._view()
        W, H = max(self.width(), 1), max(self.height(), 1)
        s = min(1.0, self.runner.p["res"] / max(W, H))
        return self.x0, self.y0, mpp / s, (max(1, int(W * s)), max(1, int(H * s)))

    def cell_at(self, pos) -> tuple[int, int]:
        """Cell under a pixel position (clamped to the grid).

        **Inputs**

        - `pos` : QPointF

        **Outputs**

        - tuple[int, int]   (i, j)
        """
        x, y = self.px_to_dom(pos)
        return int(np.clip(int(x * self.n), 0, self.nx - 1)), int(np.clip(int(y * self.n), 0, self.ny - 1))

    def wall_at(self, pos):
        """Wall under the cursor, if within SNAP_PX of a domain edge or in the wall band.

        **Inputs**

        - `pos` : QPointF

        **Outputs**

        - tuple[str, int] | None   (side, k), k = cell along the wall

        **Note** : k clamped to the usable zone [bound, len - bound - 1].
        """
        x, y = self.px_to_dom(pos)
        ppu = self.px_per_unit()
        n, b = self.n, self.bound
        Lx, Ly = self.Lx, self.Ly
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
        """Keep the box in view : centered along an axis where the view is larger than the box, else no
        scrolling past its edges.

        **Outputs**

        - self.x0, self.y0 clamped
        """
        for attr, L, npx in (("x0", self.Lx, self.width()), ("y0", self.Ly, self.height())):
            span = npx * self.mpp
            v = 0.5 * (L - span) if span >= L else float(np.clip(getattr(self, attr), 0.0, L - span))
            setattr(self, attr, v)

    def reset_view(self) -> None:
        """Show the whole box, fitted to the window.

        **Outputs**

        - view fitted and following the window, repaint scheduled
        """
        self._fit = True
        self._fit_view()
        self.viewChanged.emit()
        self.update()

    def resizeEvent(self, event) -> None:
        """Keep the view consistent when the widget is resized (refit if not zoomed).

        **Inputs**

        - `event` : QResizeEvent
        """
        if self._fit:
            self._fit_view()
        elif self.mpp > 0.0:
            self._clamp_view()
        self.viewChanged.emit()
        super().resizeEvent(event)

    # ------------------------------------------------------------ segments de paroi (géométrie écran)
    def _edge_point(self, side: str, along: float, depth_px: float = 0.0) -> QPointF:
        """Screen point on a wall edge.

        **Inputs**

        - `side` : str     left / right / bottom / top
        - `along` : float   coordinate along the wall, domain units
        - `depth_px` : float   inward offset (px)

        **Outputs**

        - QPointF
        """
        if side == "left":
            px, py = self.dom_to_px(0.0, along)
            return QPointF(px + depth_px, py)
        if side == "right":
            px, py = self.dom_to_px(self.Lx, along)
            return QPointF(px - depth_px, py)
        if side == "bottom":
            px, py = self.dom_to_px(along, 0.0)
            return QPointF(px, py - depth_px)
        px, py = self.dom_to_px(along, self.Ly)
        return QPointF(px, py + depth_px)

    def _segment_rect(self, side: str, a: float, b: float, thick: float, inset: float = 0.0) -> QRectF:
        """Screen rectangle of a wall segment.

        **Inputs**

        - `side` : str     left / right / bottom / top
        - `a`, `b` : float   span along the wall, domain units
        - `thick` : float   thickness (px)
        - `inset` : float   inward offset (px)

        **Outputs**

        - QRectF
        """
        p0, p1 = self._edge_point(side, a, inset), self._edge_point(side, b, inset + thick)
        return QRectF(p0, p1).normalized()

    def _walls2d(self):
        """Wall segments drawable in 2D (flat span on one of the 4 sides ; 3D zones of a 3D scene are skipped).

        **Outputs**

        - list of (index in runner.walls, segment dict)
        """
        return [(i, w) for i, w in enumerate(self.runner.walls)
                if w["side"] in SIDES[:4] and np.ndim(w["span"]) == 1]

    def _handle_at(self, pos):
        """Segment handle under the cursor.

        **Inputs**

        - `pos` : QPointF

        **Outputs**

        - tuple[int, int] | None   (segment index, end 0/1)
        """
        for idx, w in self._walls2d():
            for end in (0, 1):
                c = self._edge_point(w["side"], w["span"][end], 5.0)
                if abs(c.x() - pos.x()) <= HANDLE_PX and abs(c.y() - pos.y()) <= HANDLE_PX:
                    return idx, end
        return None

    def _snap_along(self, side: str, pos) -> float:
        """Cursor coordinate along a wall, snapped to cell boundaries.

        **Inputs**

        - `side` : str       left / right / bottom / top
        - `pos` : QPointF

        **Outputs**

        - `float` : domain units, clamped to [bound, len - bound] cells
        """
        x, y = self.px_to_dom(pos)
        along = y if side in ("left", "right") else x
        n, b = self.n, self.bound
        k = int(np.clip(round(along * n), b, self.side_len(side) - b))
        return k / n

    # ------------------------------------------------------------ souris / clavier
    def wheelEvent(self, event) -> None:
        """Zoom around the cursor.

        **Inputs**

        - `event` : QWheelEvent

        **Outputs**

        - self.x0, self.y0, self.mpp updated
        """
        pos = event.position()
        xc, yc = self.px_to_dom(pos)
        fit = self._fit_mpp()
        mpp = self._view() / 1.25 ** (event.angleDelta().y() / 120.0)
        mpp = float(np.clip(mpp, self.runner.dx / 40.0, fit))      # au plus : une cellule = 40 px
        if mpp >= fit * 0.999:                          # dézoom complet : la vue suit de nouveau la fenêtre
            self.reset_view()
            return
        self._fit, self.mpp = False, mpp
        self.x0 = xc - pos.x() * mpp                    # le point sous le curseur reste fixe
        self.y0 = yc - (self.height() - pos.y()) * mpp
        self._clamp_view()
        self.viewChanged.emit()
        self.update()

    def mousePressEvent(self, event) -> None:
        """Start a handle drag, wall selection, cell box selection (left) or pan (right / middle).

        **Inputs**

        - `event` : QMouseEvent

        **Outputs**

        - drag state and selections updated, selection signals emitted
        """
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
        """Update the active drag / pan and the hover state.

        **Inputs**

        - `event` : QMouseEvent

        **Outputs**

        - selections, wall spans or view updated; hoverChanged emitted
        """
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
            d = pos - self._pan
            self._pan = pos
            mpp = self._view()
            self._fit = False
            self.x0 -= d.x() * mpp
            self.y0 += d.y() * mpp
            self._clamp_view()
            self.viewChanged.emit()
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
            self.hoverChanged.emit(f"paroi {SIDE_LABELS[side]}  {axis} = {(k + 0.5) / self.n:.4g} m  (cellule {k})")
        elif 0.0 <= x < self.Lx and 0.0 <= y < self.Ly:
            self.hoverChanged.emit(f"cellule ({int(x * self.n)}, {int(y * self.n)})  x = {x:.4g} m  y = {y:.4g} m")

    def mouseReleaseEvent(self, event) -> None:
        """End drags; a click without motion on a wall selects the whole wall.

        **Inputs**

        - `event` : QMouseEvent

        **Outputs**

        - drag state cleared; wallSelectionChanged / wallEdited emitted
        """
        if self._wdrag is not None and not self._moved:          # clic simple : tout le mur
            side, _ = self._wdrag
            self.wsel = (side, self.bound, self.side_len(side) - self.bound - 1)
            self.wallSelectionChanged.emit(self.wsel)
            self.update()
        if self._hdrag is not None and self._moved:
            self.wallEdited.emit()
        self._drag = self._wdrag = self._hdrag = self._pan = None

    def leaveEvent(self, event) -> None:
        """Clear wall hover highlight when the cursor leaves.

        **Inputs**

        - `event` : QEvent
        """
        if self.hover_wall is not None:
            self.hover_wall = None
            self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        """F: whole view; Escape: clear selections.

        **Inputs**

        - `event` : QKeyEvent
        """
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
        """Draw the rendered image, wall segments and handles, hover and selections.

        **Inputs**

        - `event` : QPaintEvent
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(16, 16, 24))
        clip = QRectF(self.rect())
        if self._qimg is not None:
            painter.drawImage(self.rect(), self._qimg)  # image rendue aux proportions de la fenêtre
        n, b = self.n, self.bound
        thick = max(4.0, min(10.0, b / n * self.px_per_unit()))

        # segments de paroi de la scène (état courant du runner, même avant Reset)
        wtype = self.runner.wall_table()[0][:, :, 0]         # tables (4, nm, 1) en 2D
        for s, sname in enumerate(SIDES[:4]):
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
        for _i, w in self._walls2d():                        # contour + poignées du segment tel que défini
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
    def __init__(self, runner: SimulationRunner, path: str | None = None, vtk_dir: str | None = None,
                 vtk_every: int | None = None):
        """Build the main window (viewport, docks, toolbar, status bar) and start the frame timer.

        **Inputs**

        - `runner` : SimulationRunner
        - `path` : str | None         file path for save
        - `vtk_dir` : str | None      VTK export folder: the export starts with the solver, no folder dialog
        - `vtk_every` : int | None    VTK export period (steps)

        **Note** : the solver is built after the window is shown; APIC_UI_AUTOQUIT=<s> autoplays then quits.
        """
        super().__init__()
        self.runner = runner
        self.path = path
        self.playing = False
        self.dirty = False
        self._frames = 0
        self._editors: dict[str, QWidget] = {}

        self._render_needed = True                        # en pause : image refaite seulement si quelque chose change
        self.viewport = self._make_viewport(runner)
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
        self._central = lay
        self.setCentralWidget(central)

        self._build_params_dock()
        self._build_scene_dock()
        self._build_prim_dock()
        self._build_toolbar()
        self._build_statusbar()
        self._connect_viewport()
        self._setup_mode()
        self.vtk, self.vtk_dir = None, vtk_dir
        if vtk_every:
            self.spin_vtk.setValue(int(vtk_every))

        self.solver = None                                # construit après l'affichage de la fenêtre
        self.statusBar().showMessage("Semis et compilation des kernels…")
        self._vtk_autostart = bool(vtk_dir)               # export lancé à la fin du premier _rebuild
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
    def _make_viewport(self, runner: SimulationRunner):
        """2D or 3D viewport depending on runner.dim."""
        return Viewport3D(runner) if runner.dim == 3 else Viewport(runner)

    def _connect_viewport(self) -> None:
        """Connect the current viewport signals."""
        vp = self.viewport
        vp.wallSelectionChanged.connect(self._on_wall_selection)
        vp.hoverChanged.connect(self.lbl_hover.setText)
        vp.viewChanged.connect(self._need_render)
        if isinstance(vp, Viewport3D):
            vp.primSelected.connect(self._on_prim_pick)
        else:
            vp.selectionChanged.connect(self._on_selection)
            vp.wallEdited.connect(self._mark_dirty)

    def _setup_mode(self) -> None:
        """Adapt the window to the runner dimension (viewport, docks, color modes, fields)."""
        three = self.runner.dim == 3
        want = Viewport3D if three else Viewport
        if not isinstance(self.viewport, want):
            old = self.viewport
            self.viewport = self._make_viewport(self.runner)
            self._central.replaceWidget(old, self.viewport)
            old.hide()
            old.deleteLater()
            self._connect_viewport()
        self.viewport.runner = self.runner
        self.prim_dock.setVisible(three)
        self.prim_panel.set_runner(self.runner)
        for wdg in self._cells_widgets:
            wdg.setVisible(not three)
        for wdg in self._3d_widgets:
            wdg.setVisible(three)
        for wdg in (self.chk_grid, self.chk_tint):
            wdg.setVisible(not three)
        self.combo_color.blockSignals(True)
        cur = self.combo_color.currentData()
        self.combo_color.clear()
        for mode, label in fluid_modes(self.runner.dim):
            self.combo_color.addItem(label, mode)
        idx = self.combo_color.findData(cur if cur is not None else MODE_DENSITY)
        self.combo_color.setCurrentIndex(max(idx, 0))
        self.combo_color.blockSignals(False)
        for key in ("Lz",):
            self._param_items[key].setHidden(not three)
        self._need_render()

    def _need_render(self) -> None:
        """Ask for a new image at the next tick (view, scene or colors changed)."""
        self._render_needed = True

    def _make_editor(self, key: str):
        """Create the editor widget of a parameter (check box, int or float spin box).

        **Inputs**

        - `key` : str   parameter name

        **Outputs**

        - `QWidget` : (also stored in self._editors)
        """
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
        """Build the parameter tree dock (folders + initial-state and boundary-condition summaries).

        **Outputs**

        - self.tree, self.item_ic, self.item_bc created
        """
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

        self._param_items = {}

        def add_params(parent, keys):
            for key in keys:
                label = SimulationRunner.LABELS.get(key, key) + (" *" if key in SimulationRunner.STRUCTURAL else "")
                item = QTreeWidgetItem(parent, [label])
                self.tree.setItemWidget(item, 1, self._make_editor(key))
                self._param_items[key] = item

        add_params(folder(None, "Domaine"), ["dim", "Lx", "Ly", "Lz", "nx", "bound", "ppc", "res", "seed",
                                             "capacity"])
        add_params(folder(None, "Solveur"), ["cfl", "substeps", "gravity", "incompressible", "free_surface",
                                             "cg_iters", "multigrid", "volume_correction", "density_iters", "visc_iters",
                                             "obstacle_friction",
                                             "use_damage", "use_rupture", "color_mode"])
        mats = folder(None, "Matériaux")
        add_params(folder(mats, "Fluide"), ["fluid_rho", "fluid_E", "fluid_nu", "smagorinsky"])
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
        """Refill the initial-state and boundary-condition folders from the runner.

        **Outputs**

        - self.item_ic, self.item_bc children replaced
        """
        p, r = self.runner.p, self.runner
        try:
            m = r.masks()                               # matrices + primitives
        except Exception as exc:                        # maillage illisible : on résume les matrices seules
            m = r.m
            self.statusBar().showMessage(f"Primitive ignorée : {exc}", 5000)
        dim = r.dim
        vk = ("vx0", "vy0", "vz0")[:dim]
        for item in (self.item_ic, self.item_bc):
            item.takeChildren()

        def row(parent, name, text):
            QTreeWidgetItem(parent, [name, text])

        def vel_groups(mask):
            """Group mask cells by identical initial velocity: list of ((v...), cell count)."""
            if not mask.any():
                return []
            vals = np.stack([m[k][mask] for k in vk], axis=1)
            uniq, counts = np.unique(vals, axis=0, return_counts=True)
            return [(tuple(float(a) for a in u), int(c)) for u, c in zip(uniq, counts)]

        for name, key in (("Fluide", "fluid"), ("Solide", "solid")):
            n_c = int(m[key].sum())
            row(self.item_ic, name, f"{n_c} cellules" if n_c else "aucune")
            for vel, c in vel_groups(m[key]):
                if any(vel):
                    row(self.item_ic, "   vitesse", f"({', '.join(f'{v:g}' for v in vel)}) sur {c} cellules")
        n_o = int(m["obstacle"].sum())
        row(self.item_ic, "Obstacles", f"{n_o} cellules" if n_o else "aucun")
        if r.prims:
            row(self.item_ic, "Primitives", f"{len(r.prims)} (dock « Primitives »)")

        row(self.item_bc, "Par défaut", f"mur glissant, bande de {p['bound']} cellules ; "
                                        f"obstacles : frottement β = {p['obstacle_friction']:g}")
        try:
            wdepth = r.wall_table().depth
        except Exception:
            wdepth = None
        if not r.walls:
            row(self.item_bc, "Zones", "aucune (clic sur une face du domaine)")
        from Solver.walls import frame, span_box
        for k, w in enumerate(r.walls, start=1):
            s = SIDES.index(w["side"])
            if s >= 2 * dim:
                continue
            ts = frame(s, dim)[2]
            box = span_box(w["span"])
            if len(box) != len(ts):
                continue
            ext = " × ".join(f"{'xyz'[t]} ∈ [{iv[0]:.4g}, {iv[1]:.4g}]" for t, iv in zip(ts, box))
            moved = 0
            if wdepth is not None:
                ks = [(max(int(np.floor(iv[0] / r.dx + 1e-9)), p["bound"]),
                       min(int(np.ceil(iv[1] / r.dx - 1e-9)), r.shape[t] - p["bound"])) for iv, t in zip(box, ts)]
                if all(b1 > b0 for b0, b1 in ks):
                    kb = slice(*ks[1]) if dim == 3 else slice(0, 1)
                    moved = int((wdepth[s, slice(*ks[0]), kb] > p["bound"]).sum())
            kind = {"inlet": "entrée", "outlet": "sortie", "wall": "mur"}[w["type"]]
            vel = f" v = ({', '.join(_fmt(c) for c in w['velocity'])})," if w["type"] == "inlet" else ""
            if w["type"] == "outlet":
                vel = f" pression imposée p = {_fmt(w['pressure'])}," if w.get("pressure") else " libre (p = 0),"
            elif w["type"] == "wall":
                vel = f" frottement β = {w.get('friction', 0.0):g},"
            note = f", portée par la face d'un obstacle sur {moved} cellules" if moved else ""
            row(self.item_bc, f"{k}. paroi {SIDE_LABELS[w['side']]}", f"{kind},{vel} {ext} m{note}")

    def _build_scene_dock(self) -> None:
        """Build the scene dock: cell actions, wall actions, display options."""
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
                   ("Obstacle", lambda m: self.runner.set_obstacle(m)),
                   ("Vitesse initiale (vx, vy)", lambda m: self.runner.set_velocity(m, self._vel())),
                   ("Effacer", lambda m: self.runner.clear(m))]
        self._cell_buttons = []
        for row, (text, fn) in enumerate(actions, start=3):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: self._apply_cells(f))
            b.setEnabled(False)
            grid.addWidget(b, row, 0, 1, 2)
            self._cell_buttons.append(b)
        base = 3 + len(actions)
        self._cells_widgets = [self.lbl_sel, self.spin_vx, self.spin_vy] + self._cell_buttons + [
            grid.itemAtPosition(1, 0).widget(), grid.itemAtPosition(2, 0).widget()]
        # ---- parois (conditions aux limites)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        grid.addWidget(sep, base, 0, 1, 2)
        self.lbl_wall = QLabel("Aucune paroi sélectionnée\n(clic sur un bord : tout le mur ; glisser : un segment ; "
                               "3D : clic sur une face, Maj + glisser : un rectangle)")
        self.lbl_wall.setWordWrap(True)
        grid.addWidget(self.lbl_wall, base + 1, 0, 1, 2)
        grid.addWidget(QLabel("vx"), base + 2, 0)
        tip = ("Nombre ou expression de x, y, t (unités domaine, s) : + - * / **, min, max, abs, sqrt, exp, log, "
               "sin, cos, tan, tanh ; constantes rho, g, pi, Lx, Ly, dx et runner.consts")
        self.spin_wvx = QLineEdit("1.0")               # nombre ou expression, ex. "U*min(t/T, 1)"
        self.spin_wvx.setToolTip(tip)
        grid.addWidget(self.spin_wvx, base + 2, 1)
        grid.addWidget(QLabel("vy"), base + 3, 0)
        self.spin_wvy = QLineEdit("0.0")
        self.spin_wvy.setToolTip(tip)
        grid.addWidget(self.spin_wvy, base + 3, 1)
        self.lbl_wvz = QLabel("vz")
        self.spin_wvz = QLineEdit("0.0")
        self.spin_wvz.setToolTip(tip)
        grid.addWidget(self.lbl_wvz, base + 10, 0)
        grid.addWidget(self.spin_wvz, base + 10, 1)
        grid.addWidget(QLabel("Sortie : pression p"), base + 4, 0)
        self.spin_wp = QLineEdit("0")                  # pression imposée sur une sortie, 0 = libre
        self.spin_wp.setPlaceholderText("0 = libre, ou rho*g*(H - y)")
        self.spin_wp.setToolTip("Pression imposée sur la sortie (Dirichlet, mode incompressible), 0 = sortie libre. "
                                + tip)
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
        base = base + 11
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
        for mode, label in fluid_modes(self.runner.dim):
            self.combo_color.addItem(label, mode)
        self.combo_color.setCurrentIndex(max(self.combo_color.findData(MODE_DENSITY), 0))
        self.combo_color.currentIndexChanged.connect(self._on_color_mode)
        grid.addWidget(self.combo_color, base + 3, 1)
        self.lbl_scale = QLabel("")
        grid.addWidget(self.lbl_scale, base + 4, 0, 1, 2)
        # ---- 3D : plan de coupe des particules (voir l'intérieur)
        self.chk_clip = QCheckBox("Coupe (masque les particules au-delà)")
        self.chk_clip.toggled.connect(self._need_render)
        self.combo_clip = QComboBox()
        self.combo_clip.addItems(["x", "y", "z"])
        self.combo_clip.setCurrentIndex(2)
        self.combo_clip.currentIndexChanged.connect(self._need_render)
        self.spin_clip = QDoubleSpinBox()
        self.spin_clip.setRange(-1e6, 1e6)
        self.spin_clip.setDecimals(3)
        self.spin_clip.setSingleStep(0.02)
        self.spin_clip.setValue(0.5 * float(self.runner.p.get("Lz", 1.0)))
        self.spin_clip.valueChanged.connect(self._need_render)
        self.chk_translucent = QCheckBox("Obstacles translucides (voir l'eau dedans)")
        self.chk_translucent.setChecked(True)
        self.chk_translucent.toggled.connect(self._need_render)
        grid.addWidget(self.chk_translucent, base + 7, 0, 1, 2)
        grid.addWidget(self.chk_clip, base + 5, 0, 1, 2)
        grid.addWidget(self.combo_clip, base + 6, 0)
        grid.addWidget(self.spin_clip, base + 6, 1)
        grid.setRowStretch(base + 8, 1)
        self._3d_widgets = [self.chk_clip, self.combo_clip, self.spin_clip, self.lbl_wvz, self.spin_wvz,
                            self.chk_translucent]
        dock = QDockWidget("Scène", self)
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_prim_dock(self) -> None:
        """Build the primitives dock (3D scenes)."""
        self.prim_panel = PrimPanel(self.runner)
        self.prim_panel.changed.connect(self._on_prims_changed)
        self.prim_panel.selected.connect(self._on_prim_list)
        self.prim_dock = QDockWidget("Primitives (3D)", self)
        self.prim_dock.setWidget(self.prim_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.prim_dock)

    def _on_prims_changed(self) -> None:
        """Primitive edited in the panel: scene dirty, overlay refreshed."""
        self._mark_dirty()
        self.viewport.update()

    def _on_prim_list(self, idx) -> None:
        """Primitive selected in the list: highlight it in the view."""
        if isinstance(self.viewport, Viewport3D):
            self.viewport.prim_sel = idx
            self.viewport.update()

    def _on_prim_pick(self, idx) -> None:
        """Primitive clicked in the view: select it in the list."""
        self.prim_panel.select(idx)

    def _build_toolbar(self) -> None:
        """Build the toolbar actions (play, step, reset, view, open, save) with shortcuts."""
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
        act("Vue entière", lambda: self.viewport.reset_view(), "F")
        tb.addSeparator()
        act("Ouvrir…", self._open, "Ctrl+O")
        act("Enregistrer", self._save, "Ctrl+S")
        act("Enregistrer sous…", self._save_as, "Ctrl+Shift+S")
        tb.addSeparator()
        self.act_vtk = act("Export VTK", self._set_vtk, checkable=True)
        self.act_vtk.setToolTip("Écrit particules, grille et maillages (roue à son angle) dans un dossier, "
                                "séries .pvd à ouvrir dans ParaView")
        self.spin_vtk = QSpinBox()
        self.spin_vtk.setRange(1, 10000)
        self.spin_vtk.setValue(5)
        self.spin_vtk.setPrefix("tous les ")
        self.spin_vtk.setSuffix(" pas")
        tb.addWidget(self.spin_vtk)

    def _build_statusbar(self) -> None:
        """Build the status bar labels (hover, time, particles, perf)."""
        self.lbl_hover = QLabel()
        self.lbl_time = QLabel()
        self.lbl_particles = QLabel()
        self.lbl_perf = QLabel()
        for lbl in (self.lbl_hover, self.lbl_time, self.lbl_particles, self.lbl_perf):
            lbl.setMargin(4)
            self.statusBar().addPermanentWidget(lbl)

    # ------------------------------------------------------------ boucle
    def _tick(self) -> None:
        """Timer slot: step if playing, render the view, refresh stats every 6 frames.

        **Outputs**

        - viewport image and status labels updated
        """
        if self.solver is None or self.solver.dim != self.runner.dim:   # dimension changée : attendre le Reset
            return
        if self.playing:
            self.solver.step()
            self._vtk_after_step()
        elif not self._render_needed:                     # pause, rien n'a changé : pas de rendu ni de copie
            return
        self._render_needed = False
        vp = self.viewport
        self.solver.fluid_mode = self.combo_color.currentData() or 0
        if self.solver.dim == 3:
            cam, size = vp.render_view()
            clip = (self.combo_clip.currentIndex(), self.spin_clip.value(), True) if self.chk_clip.isChecked() else None
            alpha = 0.25 if self.chk_translucent.isChecked() else 1.0
            img = self.solver.render3d(cam, size=size, clip=clip, obs_alpha=alpha)
        else:
            x0, y0, k, size = vp.render_view()
            img = self.solver.render(x0, y0, grid=self.chk_grid.isChecked(), tint=self.chk_tint.isChecked(), k=k,
                                     size=size)
        vp.set_image(img)
        self._frames += 1
        if self._frames % 6 == 0 or not self.playing:
            st = self.solver.stats()
            if not self.playing and self._frames < 3:
                self._render_needed = True                # échelle de couleur initiale : un second rendu
            mode = self.solver.fluid_mode
            if mode == 0:
                self.lbl_scale.setText("")
            else:
                m = st["scalar_max"]
                if not mode_signed(mode, self.solver.dim):
                    rng = f"0 … {m:.3g}"
                elif mode == MODE_DENSITY:
                    rng = f"ρ0 − {m:.3g} … ρ0 + {m:.3g} kg/m³"
                else:
                    rng = f"−{m:.3g} … +{m:.3g}"
                self.lbl_scale.setText(f"échelle : {rng} (auto)")
            rotor = ""
            if "rotor_rpm" in st:                         # roue : vitesse imposée, couple et puissance de l'eau
                rotor = (f"   roue {st['rotor_rpm']:.0f} tr/min, couple {st['rotor_torque']:.3g} N·m, "
                         f"puissance {st['rotor_power']:.3g} W")
            self.lbl_time.setText(f"t = {st['t']:.3f} s   dt = {st['dt']:.2e}{rotor}")
            cap = f" / {st['capacity']}" if st["capacity"] != st["n_fluid"] else ""
            self.lbl_particles.setText(f"fluide {st['n_fluid']}{cap}   solide {st['n_solid']}   "
                                       f"D max {st['D_max']:.2f}   rompues {st['n_broken']}")
            self.lbl_perf.setText(f"{self.solver.arch} · {st['ms']:.1f} ms/image" if self.playing else f"{self.solver.arch} · pause")

    def _rebuild(self) -> None:
        """Rebuild the solver from the runner (seed, compile kernels, reset).

        **Outputs**

        - self.solver replaced, dirty flag cleared
        """
        self.statusBar().showMessage("Semis et compilation des kernels…")
        self.statusBar().repaint()
        QApplication.processEvents()                      # laisse la fenêtre se peindre avant le calcul bloquant
        if self.solver is not None:
            self.solver.release()
            self.solver = None
        try:
            solver = self.runner.solver()
        except Exception as exc:                          # maillage illisible, mémoire GPU insuffisante...
            self.statusBar().clearMessage()
            QMessageBox.critical(self, "Construction du solveur impossible", str(exc))
            return
        solver.step(1)                                    # compile les kernels de simulation maintenant,
        solver.reset()                                    # pas au premier clic sur Play
        self.solver = solver
        self.dirty = False
        self._need_render()
        self.banner.hide()
        self.statusBar().clearMessage()
        if self._vtk_autostart:                           # dossier imposé (show(vtk_dir=...)) : pas de dialogue
            self._vtk_autostart = False
            self.act_vtk.setChecked(True)
        else:
            self._vtk_restart()

    def _mark_dirty(self) -> None:
        """Flag the scene as modified (Reset required).

        **Outputs**

        - self.dirty = True, banner shown, summaries and title refreshed
        """
        self.dirty = True
        self.banner.show()
        self._refresh_summaries()
        self._update_title(modified=True)
        self.viewport.update()
        self._need_render()

    # ------------------------------------------------------------ actions
    def _set_playing(self, on: bool) -> None:
        """Play / pause the simulation (rebuilds first if dirty).

        **Inputs**

        - `on` : bool

        **Outputs**

        - self.playing and play action updated
        """
        self.playing = bool(on) and self.solver is not None
        if self.act_play.isChecked() != self.playing:
            self.act_play.blockSignals(True)
            self.act_play.setChecked(self.playing)
            self.act_play.blockSignals(False)
        self.act_play.setText("❚❚ Pause" if self.playing else "▶ Play")
        if self.playing and self.dirty:
            self._rebuild()

    def _step(self) -> None:
        """Pause and advance one frame (rebuilds first if dirty)."""
        self._set_playing(False)
        if self.dirty:
            self._rebuild()
        if self.solver is not None:
            self.solver.step()
            self._vtk_after_step()
        self._need_render()

    def _set_vtk(self, on: bool) -> None:
        """Start (choose a folder, write the current state) or stop the VTK export.

        **Inputs**

        - `on` : bool

        **Outputs**

        - self.vtk exporter | None ; files written by ui.export_vtk (GPU -> CPU copy at each export only)
        """
        if not on:
            if getattr(self, "vtk", None) is not None:
                self.statusBar().showMessage(f"Export VTK arrêté : {self.vtk.count} instants dans {self.vtk.folder}",
                                             5000)
            self.vtk = None
            return
        if self.dirty or self.solver is None:
            self._rebuild()
        folder = self.vtk_dir
        if not folder:
            start = os.path.join(os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd(), "vtk")
            folder = QFileDialog.getExistingDirectory(self, "Dossier d'export VTK (ParaView)", start)
        if not folder or self.solver is None:
            self.act_vtk.blockSignals(True)
            self.act_vtk.setChecked(False)
            self.act_vtk.blockSignals(False)
            return
        from ui.export_vtk import VTKExporter
        self.vtk = VTKExporter(folder, self.runner, self.solver)
        self._vtk_steps = 0
        self.vtk.write()
        self.statusBar().showMessage(f"Export VTK vers {folder} : ouvrir les .pvd dans ParaView", 5000)

    def _vtk_after_step(self) -> None:
        """Export one VTK instant every spin_vtk steps while the export is on."""
        ex = getattr(self, "vtk", None)
        if ex is None:
            return
        if ex.s is not self.solver:                       # solveur remplacé sans passer par _rebuild / _reset
            self._vtk_restart()
            return
        self._vtk_steps += 1
        if self._vtk_steps % self.spin_vtk.value() == 0:
            ex.write()

    def _vtk_restart(self) -> None:
        """Keep an active export going after a reset / rebuild: new series (t = 0) in the same folder.

        **Outputs**

        - self.vtk bound to the current solver, initial state written (instants renumbered from 00000)
        """
        ex = getattr(self, "vtk", None)
        if ex is None or self.solver is None:
            return
        from ui.export_vtk import VTKExporter
        self.vtk = VTKExporter(ex.folder, self.runner, self.solver)
        self._vtk_steps = 0
        self.vtk.write()
        self.statusBar().showMessage(f"Export VTK : nouvelle série (t = 0) dans {ex.folder}", 5000)

    def _reset(self) -> None:
        """Reset the solver, rebuilding it if the scene is dirty (an active VTK export starts a new series)."""
        if self.dirty or self.solver is None:
            self._rebuild()
        else:
            self.solver.reset()
            self._vtk_restart()
        self._need_render()

    def _on_param(self, key: str, value) -> None:
        """Apply a parameter edit.

        **Inputs**

        - `key` : str
        - `value` : bool | int | float

        **Outputs**

        - runner.p updated; structural keys mark dirty, others are applied to the solver live
        """
        if key in SimulationRunner.STRUCTURAL:
            if key in ("Lx", "Ly", "Lz", "nx", "dim"):  # boîte, résolution, dimension : matrices rééchantillonnées
                self.runner.p[key] = value
                self.runner.regrid()
                if key == "dim":
                    self._setup_mode()
                self.viewport.reset_view()
            else:
                self.runner.p[key] = value
            self._mark_dirty()
        else:
            self.runner.p[key] = value
            if self.solver is not None:
                self.solver.set_params({key: value})
            self._update_title(modified=True)

    def _on_color_mode(self, _index: int) -> None:
        """Fluid color mode changed: reset the color scale.

        **Inputs**

        - `_index` : int   combo box index (unused)
        """
        if self.solver is not None:
            self.solver.scalar_max = 0.0             # l'échelle se recalcule sur la nouvelle quantité
        self._need_render()

    def _vel(self) -> tuple[float, float]:
        """Initial velocity from the cell spin boxes.

        **Outputs**

        - tuple[float, float]   (vx, vy)
        """
        return self.spin_vx.value(), self.spin_vy.value()

    def _on_selection(self, sel) -> None:
        """Update cell buttons and label for a new cell selection.

        **Inputs**

        - `sel` : tuple[int, int, int, int] | None   (i0, j0, i1, j1) inclusive
        """
        for b in self._cell_buttons:
            b.setEnabled(sel is not None)
        if sel is None:
            self.lbl_sel.setText("Aucune sélection\n(bouton gauche : boîte de cellules)")
        else:
            i0, j0, i1, j1 = sel
            self.lbl_sel.setText(f"Sélection : i {i0}..{i1}, j {j0}..{j1}\n{(i1 - i0 + 1) * (j1 - j0 + 1)} cellules")

    def _on_wall_selection(self, sel) -> None:
        """Update wall buttons and label for a new wall selection.

        **Inputs**

        - `sel` : tuple[str, int, int] | None   (side, k0, k1) inclusive
        """
        for b in self._wall_buttons:
            b.setEnabled(sel is not None)
        if sel is None:
            self.lbl_wall.setText("Aucune paroi sélectionnée\n(clic sur un bord : tout le mur ; glisser : un segment ; "
                                  "3D : clic sur une face, Maj + glisser : un rectangle)")
        elif self.runner.dim == 3:
            from Solver.walls import frame
            side, (a0, a1), (b0, b1) = sel
            ts = frame(SIDES.index(side), 3)[2]
            dx = self.runner.dx
            self.lbl_wall.setText(f"Paroi {SIDE_LABELS[side]}\n{'xyz'[ts[0]]} de {a0 * dx:.4g} à {(a1 + 1) * dx:.4g} m, "
                                  f"{'xyz'[ts[1]]} de {b0 * dx:.4g} à {(b1 + 1) * dx:.4g} m")
        else:
            side, k0, k1 = sel
            n = 1.0 / self.runner.dx                   # cellules par mètre
            axis = "y" if side in ("left", "right") else "x"
            ln = self.viewport.side_len(side)
            whole = " (mur entier)" if (k0, k1) == (self.runner.p["bound"], ln - self.runner.p["bound"] - 1) else ""
            self.lbl_wall.setText(f"Paroi {SIDE_LABELS[side]}{whole}\n{axis} de {k0 / n:.4g} à {(k1 + 1) / n:.4g} m, "
                                  f"cellules {k0}..{k1}")

    def _apply_cells(self, fn) -> None:
        """Apply a runner action to the selected cell box.

        **Inputs**

        - `fn` : callable(np.ndarray bool (nx, ny))

        **Outputs**

        - runner.m updated, marks scene dirty
        """
        if self.viewport.sel is None:
            return
        i0, j0, i1, j1 = self.viewport.sel
        mask = np.zeros((self.runner.nx, self.runner.ny), bool)
        mask[i0:i1 + 1, j0:j1 + 1] = True
        fn(mask)
        self._mark_dirty()

    def _apply_wall(self, kind: str) -> None:
        """Set a wall segment of the given kind on the selected wall span.

        **Inputs**

        - `kind` : str   inlet / outlet / wall

        **Outputs**

        - runner walls updated, marks scene dirty
        """
        if self.viewport.wsel is None:
            return
        dx = self.runner.dx
        vx, vy, vz, p = (_parse(w.text()) for w in (self.spin_wvx, self.spin_wvy, self.spin_wvz, self.spin_wp))
        if self.runner.dim == 3:
            side, (a0, a1), (b0, b1) = self.viewport.wsel
            span, vel = ((a0 * dx, (a1 + 1) * dx), (b0 * dx, (b1 + 1) * dx)), (vx, vy, vz)
        else:
            side, k0, k1 = self.viewport.wsel
            span, vel = (k0 * dx, (k1 + 1) * dx), (vx, vy)
        pressure = p if kind == "outlet" and p != 0.0 else None
        friction = self.spin_wf.value() if kind == "wall" else None
        backup = [dict(w) for w in self.runner.walls]
        try:
            self.runner.set_wall(side, kind, velocity=vel, span=span, pressure=pressure, friction=friction)
            self.runner.wall_table()                   # vérifie les noms des expressions (constantes connues)
        except ValueError as exc:
            self.runner.walls = backup                 # segment refusé : parois inchangées
            QMessageBox.warning(self, "Condition limite", str(exc))
            return
        self._mark_dirty()

    # ------------------------------------------------------------ fichiers
    def _load(self, runner: SimulationRunner, path: str | None) -> None:
        """Switch to another runner and rebuild everything.

        **Inputs**

        - `runner` : SimulationRunner
        - `path` : str | None
        """
        self._set_playing(False)
        self.runner = runner
        self.viewport.runner = runner
        self.path = path
        for key, w in self._editors.items():
            w.blockSignals(True)
            v = runner.p[key]
            w.setChecked(bool(v)) if isinstance(w, QCheckBox) else w.setValue(v)
            w.blockSignals(False)
        self._setup_mode()
        self.viewport.sel = None
        self.viewport.wsel = None
        self.viewport.reset_view()
        self._on_selection(None)
        self._on_wall_selection(None)
        self._refresh_summaries()
        self._rebuild()
        self._update_title()

    def _open(self) -> None:
        """Ask for a JSON file and load it."""
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une simulation", "", "Simulation (*.json)")
        if not path:
            return
        try:
            self._load(SimulationRunner.load(path), path)
        except Exception as exc:
            QMessageBox.critical(self, "Ouverture impossible", str(exc))

    def _save(self) -> None:
        """Save to self.path (asks for a path if none)."""
        if self.path is None:
            self._save_as()
            return
        self.runner.save(self.path)
        self._update_title()
        self.statusBar().showMessage(f"Enregistré : {self.path}", 3000)

    def _save_as(self) -> None:
        """Ask for a path, then save.

        **Outputs**

        - self.path updated
        """
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la simulation", self.path or "simulation.json",
                                              "Simulation (*.json)")
        if path:
            self.path = path
            self._save()

    def _update_title(self, modified: bool = False) -> None:
        """Set the window title from the file name.

        **Inputs**

        - `modified` : bool   append '*'
        """
        name = os.path.basename(self.path) if self.path else "simulation"
        self.setWindowTitle(f"{name}{'*' if modified else ''} — APIC / MPM")

    # ------------------------------------------------------------ fin
    def _autoplay(self) -> None:
        """Start playing once the solver exists (diagnostic mode)."""
        if self.solver is None:
            QTimer.singleShot(200, self._autoplay)
        else:
            self._set_playing(True)

    def _autoquit(self) -> None:
        """Print diagnostics and close the window (diagnostic mode)."""
        st = self.solver.stats() if self.solver is not None else {}
        print(f"[diag] dim={self.runner.dim} frames={self._frames} stats={st}", flush=True)
        self.close()

    def closeEvent(self, event) -> None:
        """Stop the timer and release the solver on close.

        **Inputs**

        - `event` : QCloseEvent
        """
        self.timer.stop()
        if self.solver is not None:
            self.solver.release()
        super().closeEvent(event)

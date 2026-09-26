"""Viewport : la fenêtre GGUI de Taichi embarquée dans Qt.

- La fenêtre GLFW créée par ti.ui.Window est retrouvée par son titre (HWND) puis insérée dans la
  fenêtre Qt via QWindow.fromWinId + QWidget.createWindowContainer. Si cela échoue (autre OS,
  APIC_UI_NO_EMBED=1, HWND introuvable), la fenêtre GGUI reste autonome à côté de la fenêtre Qt.
- Rendu : sur l'arch Vulkan, toute la scène (fond, particules, overlays) est dessinée par un kernel
  dans une ti.Texture que GGUI présente directement : zéro passage par le CPU. Sur les autres archs
  (CUDA, CPU) on retombe sur canvas.set_image / circles / lines, qui transitent par des tampons numpy.
- Les overlays (contours des formes, sélection, état « modifié ») sont des segments dans de petits
  champs Taichi, mis à jour par from_numpy uniquement quand la géométrie ou la sélection change.
- La souris et le clavier du viewport sont lus via l'API d'événements GGUI et relayés en signaux Qt.
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
import uuid

import numpy as np
import taichi as ti
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QWindow
from PySide6.QtWidgets import QLabel, QWidget

from ui.model.project import Circle, Project, Rect
from ui.sim.kernels import clear_texture

N_MAX_SEG = 1024            # largement assez pour quelques dizaines de formes
CIRCLE_SEGMENTS = 32
HANDLE = 0.008              # demi-côté des poignées de sélection (unités domaine)
BACKGROUND = (0.02, 0.02, 0.08)

ROLE_COLORS = {
    "fluid": (0.40, 0.70, 1.00),
    "solid": (0.95, 0.85, 0.30),
    "obstacle": (0.65, 0.65, 0.65),
}
SELECTED_COLOR = (1.0, 1.0, 1.0)
DIRTY_COLOR = (1.0, 0.55, 0.10)
RELAYED_KEYS = {ti.ui.SPACE, ti.ui.ESCAPE, "r", "s"}


def is_vulkan() -> bool:
    return ti.lang.impl.current_cfg().arch == ti.vulkan


class SquareContainer(QWidget):
    """Garde le widget enfant carré et centré (le domaine est [0,1]², la fenêtre GGUI aussi)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._child: QWidget | None = None
        self.setMinimumSize(200, 200)
        self.setStyleSheet("background-color: #101018;")
        self.setAutoFillBackground(True)

    def set_child(self, child: QWidget) -> None:
        self._child = child
        child.setParent(self)
        child.show()
        self._layout_child()

    def sizeHint(self) -> QSize:
        return QSize(700, 700)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_child()

    def _layout_child(self) -> None:
        if self._child is None:
            return
        s = min(self.width(), self.height())
        self._child.setGeometry((self.width() - s) // 2, (self.height() - s) // 2, s, s)


class GguiViewport(QObject):
    shapeClicked = Signal(str)                 # id de forme, ou "" si clic dans le vide
    shapeDragged = Signal(str, float, float)   # id, dx, dy (unités domaine)
    dragFinished = Signal(str)
    keyPressed = Signal(str)                   # touches GGUI relayées (ti.ui.SPACE, 'r', 's', ti.ui.ESCAPE)
    closed = Signal()                          # la fenêtre GGUI a été fermée

    def __init__(self, res: int, parent_widget: QWidget | None = None):
        super().__init__(parent_widget)
        self.res = res
        self.title = f"apic-viewport-{uuid.uuid4().hex[:6]}"
        self.window = ti.ui.Window(self.title, (res, res), vsync=False, show_window=True)
        self.canvas = self.window.get_canvas()

        # Chemin zéro copie : texture présentée directement par GGUI (Vulkan uniquement).
        self.zero_copy = is_vulkan() and not os.environ.get("APIC_UI_NO_TEXTURE")
        self.texture = ti.Texture(ti.Format.rgba8, (res, res)) if self.zero_copy else None

        # Overlays : segments (a, b, couleur) + version entrelacée pour canvas.lines (repli).
        fb = ti.FieldsBuilder()
        self.seg_a = ti.Vector.field(2, ti.f32)
        self.seg_b = ti.Vector.field(2, ti.f32)
        self.seg_c = ti.Vector.field(3, ti.f32)
        self.lines_vf = ti.Vector.field(2, ti.f32)
        self.lines_col = ti.Vector.field(3, ti.f32)
        fb.dense(ti.i, N_MAX_SEG).place(self.seg_a, self.seg_b, self.seg_c)
        fb.dense(ti.i, 2 * N_MAX_SEG).place(self.lines_vf, self.lines_col)
        self._tree = fb.finalize()
        self.n_seg = 0
        self._a = np.full((N_MAX_SEG, 2), -1.0, np.float32)
        self._b = np.full((N_MAX_SEG, 2), -1.0, np.float32)
        self._c = np.zeros((N_MAX_SEG, 3), np.float32)
        self._overlays_pending = False
        self._upload_overlays()
        if self.zero_copy:
            # Première écriture dans la texture dès maintenant (avant toute allocation de la session)
            clear_texture(self.texture, res, *BACKGROUND, self.seg_a, self.seg_b, self.seg_c, 0)
            ti.sync()

        self.project: Project | None = None
        self.selected: str | None = None
        self.dirty = False
        self._drag_id: str | None = None
        self._drag_last = (0.0, 0.0)
        self._drag_moved = False
        self._closed_emitted = False

        self.embedded = False
        self.widget: QWidget = self._embed(parent_widget)

    # ------------------------------------------------------------ embarquement
    def _embed(self, parent: QWidget | None) -> QWidget:
        square = SquareContainer(parent)
        if os.environ.get("APIC_UI_NO_EMBED") or sys.platform != "win32":
            return self._fallback(square, "embarquement désactivé")
        try:
            user32 = ctypes.windll.user32
            user32.FindWindowW.restype = ctypes.c_void_p
            user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
            hwnd = user32.FindWindowW(None, self.title)
            if not hwnd:
                return self._fallback(square, "fenêtre GGUI introuvable")
            qwin = QWindow.fromWinId(int(hwnd))
            container = QWidget.createWindowContainer(qwin, square)
            container.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            square.set_child(container)
            self.embedded = True
            return square
        except Exception as exc:  # embarquement « best effort » : on retombe sur deux fenêtres
            return self._fallback(square, f"{type(exc).__name__}: {exc}")

    def _fallback(self, square: SquareContainer, reason: str) -> QWidget:
        label = QLabel(f"Viewport dans la fenêtre GGUI séparée\n({reason})")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #c0c0c0;")
        square.set_child(label)
        self.embedded = False
        return square

    # ------------------------------------------------------------ API
    def set_project(self, project: Project) -> None:
        self.project = project
        self._rebuild_overlays()

    def set_selected(self, shape_id: str | None) -> None:
        if shape_id != self.selected:
            self.selected = shape_id
            self._rebuild_overlays()

    def set_dirty(self, dirty: bool) -> None:
        if dirty != self.dirty:
            self.dirty = dirty
            self._rebuild_overlays()

    def hit_test(self, x: float, y: float) -> str | None:
        if self.project is None:
            return None
        for s in reversed(self.project.shapes):        # la dernière dessinée gagne
            if s.contains(x, y):
                return s.id
        return None

    def frame(self, session) -> bool:
        """Un tick : événements, dessin de la scène + overlays, présentation. False si fermé."""
        if not self.window.running:
            if not self._closed_emitted:
                self._closed_emitted = True
                self.closed.emit()
            return False
        self._poll_events()
        if self._overlays_pending:               # téléversement ici, jamais depuis un slot Qt
            self._upload_overlays()
        built = session is not None and session.built
        if self.zero_copy:
            if built:
                session.draw_texture(self.texture, self.res, self.seg_a, self.seg_b, self.seg_c, self.n_seg)
            else:
                clear_texture(self.texture, self.res, *BACKGROUND, self.seg_a, self.seg_b, self.seg_c, self.n_seg)
            self.canvas.set_image(self.texture)
        else:
            if built:
                session.draw_canvas(self.canvas)
            else:
                self.canvas.set_background_color(BACKGROUND)
            self.canvas.lines(self.lines_vf, width=0.003, per_vertex_color=self.lines_col)
        self.window.show()
        return True

    def destroy(self) -> None:
        try:
            self.window.running = False
            self.window.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------ événements GGUI
    def _poll_events(self) -> None:
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                x, y = self.window.get_cursor_pos()
                sid = self.hit_test(x, y)
                self._drag_id = sid
                self._drag_last = (x, y)
                self._drag_moved = False
                self.shapeClicked.emit(sid or "")
            elif e.key in RELAYED_KEYS:
                self.keyPressed.emit(e.key)
        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB and self._drag_id is not None:
                if self._drag_moved:
                    self.dragFinished.emit(self._drag_id)
                self._drag_id = None
        if self._drag_id is not None and self.window.is_pressed(ti.ui.LMB):
            x, y = self.window.get_cursor_pos()
            dx, dy = x - self._drag_last[0], y - self._drag_last[1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                self._drag_last = (x, y)
                self._drag_moved = True
                self.shapeDragged.emit(self._drag_id, dx, dy)

    # ------------------------------------------------------------ overlays
    def _rebuild_overlays(self) -> None:
        segs: list[tuple[tuple, tuple, tuple]] = []

        def seg(a, b, c):
            segs.append((a, b, c))

        def square(cx, cy, h, c):
            seg((cx - h, cy - h), (cx + h, cy - h), c)
            seg((cx + h, cy - h), (cx + h, cy + h), c)
            seg((cx + h, cy + h), (cx - h, cy + h), c)
            seg((cx - h, cy + h), (cx - h, cy - h), c)

        if self.project is not None:
            for s in self.project.shapes:
                is_sel = s.id == self.selected
                c = SELECTED_COLOR if is_sel else (DIRTY_COLOR if self.dirty else ROLE_COLORS.get(s.role, (1, 1, 1)))
                if isinstance(s, Rect):
                    x0, x1 = sorted((s.x0, s.x1))
                    y0, y1 = sorted((s.y0, s.y1))
                    seg((x0, y0), (x1, y0), c)
                    seg((x1, y0), (x1, y1), c)
                    seg((x1, y1), (x0, y1), c)
                    seg((x0, y1), (x0, y0), c)
                    handles = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                elif isinstance(s, Circle):
                    pts = [(s.cx + s.r * math.cos(2 * math.pi * k / CIRCLE_SEGMENTS),
                            s.cy + s.r * math.sin(2 * math.pi * k / CIRCLE_SEGMENTS)) for k in range(CIRCLE_SEGMENTS)]
                    for k in range(CIRCLE_SEGMENTS):
                        seg(pts[k], pts[(k + 1) % CIRCLE_SEGMENTS], c)
                    handles = [(s.cx + s.r, s.cy), (s.cx, s.cy + s.r), (s.cx - s.r, s.cy), (s.cx, s.cy - s.r)]
                else:
                    handles = []
                if is_sel:
                    for hx, hy in handles:
                        square(hx, hy, HANDLE, c)
            if self.dirty:                                   # cadre orange = « modifié, Reset pour appliquer »
                m = 0.004
                seg((m, m), (1 - m, m), DIRTY_COLOR)
                seg((1 - m, m), (1 - m, 1 - m), DIRTY_COLOR)
                seg((1 - m, 1 - m), (m, 1 - m), DIRTY_COLOR)
                seg((m, 1 - m), (m, m), DIRTY_COLOR)

        n = min(len(segs), N_MAX_SEG)
        self._a[:] = -1.0                                    # segments inutilisés : hors écran
        self._b[:] = -1.0
        self._c[:] = 0.0
        if n:
            self._a[:n] = np.asarray([s[0] for s in segs[:n]], np.float32)
            self._b[:n] = np.asarray([s[1] for s in segs[:n]], np.float32)
            self._c[:n] = np.asarray([s[2] for s in segs[:n]], np.float32)
        self.n_seg = n
        self._overlays_pending = True            # from_numpy différé au prochain frame() (voir mainwindow)

    def _upload_overlays(self) -> None:
        self._overlays_pending = False
        self.seg_a.from_numpy(self._a)
        self.seg_b.from_numpy(self._b)
        self.seg_c.from_numpy(self._c)
        verts = np.empty((2 * N_MAX_SEG, 2), np.float32)
        verts[0::2], verts[1::2] = self._a, self._b
        cols = np.repeat(self._c, 2, axis=0)
        self.lines_vf.from_numpy(verts)
        self.lines_col.from_numpy(cols)

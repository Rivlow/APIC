"""Interface Qt : Viewport (image rendue sur le GPU, zoom, pan, sélection de cellules) et UI (fenêtre).

Souris dans le viewport : molette = zoom autour du curseur, bouton gauche = boîte de sélection de
cellules, bouton droit ou milieu = déplacement de la vue, F = vue entière, Échap = désélection.
Le panneau « Cellules » applique eau / solide / obstacle / entrée / sortie / vitesse / effacer sur la
sélection ; toute modification des matrices ou d'un paramètre structurel demande un Reset (bandeau).
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton, QSpinBox,
                               QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ui.kernels import FLUID_MODES
from ui.runner import SimulationRunner

INT_RANGES = {"n": (16, 1024), "bound": (1, 16), "ppc": (1, 4), "substeps": (1, 1000), "seed": (0, 10**9),
              "capacity": (0, 5_000_000), "res": (200, 2000), "color_mode": (0, 1)}


class Viewport(QWidget):
    selectionChanged = Signal(object)          # (i0, j0, i1, j1) cellules incluses, ou None
    hoverChanged = Signal(int, int, float, float)

    def __init__(self, n: int, parent=None):
        super().__init__(parent)
        self.n = n
        self.x0, self.y0, self.scale = 0.0, 0.0, 1.0     # vue = [x0, x0 + 1/scale] × [y0, y0 + 1/scale]
        self.sel = None
        self._buf = None
        self._qimg = None
        self._drag = None                                 # cellule de départ de la boîte de sélection
        self._pan = None                                  # dernière position souris pendant le pan
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(300, 300)
        self.setAutoFillBackground(True)

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

    def cell_at(self, pos) -> tuple[int, int]:
        x, y = self.px_to_dom(pos)
        return int(np.clip(int(x * self.n), 0, self.n - 1)), int(np.clip(int(y * self.n), 0, self.n - 1))

    def _clamp_view(self) -> None:
        span = 1.0 / self.scale
        self.x0 = float(np.clip(self.x0, 0.0, 1.0 - span))
        self.y0 = float(np.clip(self.y0, 0.0, 1.0 - span))

    def reset_view(self) -> None:
        self.x0, self.y0, self.scale = 0.0, 0.0, 1.0
        self.update()

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
            i, j = self.cell_at(event.position())
            self._drag = (i, j)
            self.sel = (i, j, i, j)
            self.selectionChanged.emit(self.sel)
            self.update()
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None:
            i, j = self.cell_at(event.position())
            i0, j0 = self._drag
            self.sel = (min(i0, i), min(j0, j), max(i0, i), max(j0, j))
            self.selectionChanged.emit(self.sel)
            self.update()
        elif self._pan is not None:
            _, _, side = self._square()
            d = event.position() - self._pan
            self._pan = event.position()
            self.x0 -= d.x() / (side * self.scale)
            self.y0 += d.y() / (side * self.scale)
            self._clamp_view()
            self.update()
        x, y = self.px_to_dom(event.position())
        if 0.0 <= x < 1.0 and 0.0 <= y < 1.0:
            self.hoverChanged.emit(int(x * self.n), int(y * self.n), x, y)

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        self._pan = None

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F:
            self.reset_view()
        elif event.key() == Qt.Key.Key_Escape:
            self.sel = None
            self.selectionChanged.emit(None)
            self.update()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------ dessin
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(16, 16, 24))
        ox, oy, side = self._square()
        if self._qimg is not None:
            painter.drawImage(QRect(ox, oy, side, side), self._qimg)
        if self.sel is not None:
            i0, j0, i1, j1 = self.sel
            px0, py1 = self.dom_to_px(i0 / self.n, j0 / self.n)
            px1, py0 = self.dom_to_px((i1 + 1) / self.n, (j1 + 1) / self.n)
            r = QRectF(px0, py0, px1 - px0, py1 - py0).intersected(QRectF(ox, oy, side, side))
            painter.fillRect(r, QColor(255, 255, 255, 40))
            painter.setPen(QPen(QColor(255, 255, 255), 1.5, Qt.PenStyle.DashLine))
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

        self.viewport = Viewport(runner.n)
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
        self._build_cells_dock()
        self._build_toolbar()
        self._build_statusbar()

        self.viewport.selectionChanged.connect(self._on_selection)
        self.viewport.hoverChanged.connect(lambda i, j, x, y: self.lbl_hover.setText(f"cellule ({i}, {j})  x={x:.3f} y={y:.3f}"))

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
            w.setRange(-1e6, 1e9)
            w.setSingleStep(abs(default) / 10 if default else 0.1)
            w.setValue(float(value))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, k=key: self._on_param(k, float(v)))
        self._editors[key] = w
        return w

    def _build_params_dock(self) -> None:
        """Arbre de paramètres rangé par dossiers ; les dossiers « État initial » et « Conditions aux
        limites » résument les matrices courantes (mis à jour à chaque modification)."""
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

        add_params(folder(None, "Domaine"), ["n", "bound", "ppc", "res", "seed", "capacity"])
        add_params(folder(None, "Solveur"), ["cfl", "substeps", "gravity", "incompressible", "free_surface",
                                             "cg_iters", "use_damage", "use_rupture", "color_mode"])
        mats = folder(None, "Matériaux")
        add_params(folder(mats, "Fluide"), ["fluid_rho", "fluid_E"])
        add_params(folder(mats, "Solide"), ["solid_rho", "solid_E", "solid_nu", "eps0", "epsf", "tau_D", "k_res"])
        self.item_ic = folder(None, "État initial")
        self.item_bc = folder(None, "Conditions aux limites")
        note = QTreeWidgetItem(self.tree, ["* : appliqué au Reset"])
        note.setFirstColumnSpanned(True)
        self._refresh_summaries()

        dock = QDockWidget("Paramètres", self)
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [360], Qt.Orientation.Horizontal)

    def _refresh_summaries(self) -> None:
        """Dossiers « État initial » et « Conditions aux limites » d'après les matrices du runner."""
        m, p = self.runner.m, self.runner.p
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

        row(self.item_bc, "Parois", f"glissantes, bande de {p['bound']} cellules")
        n_o = int(m["obstacle"].sum())
        row(self.item_bc, "Obstacles", f"{n_o} cellules" if n_o else "aucun")
        groups = vel_groups(m["inlet"], m["inlet_vx"], m["inlet_vy"])
        if not groups:
            row(self.item_bc, "Entrées", "aucune")
        for k, ((vx, vy), c) in enumerate(groups, start=1):
            row(self.item_bc, f"Entrée {k}", f"v = ({vx:g}, {vy:g}), {c} cellules")
        n_out = int(m["outlet"].sum())
        row(self.item_bc, "Sorties", f"{n_out} cellules (p = 0, particules détruites)" if n_out else "aucune")

    def _build_cells_dock(self) -> None:
        w = QWidget()
        grid = QGridLayout(w)
        self.lbl_sel = QLabel("Aucune sélection\n(bouton gauche : boîte de cellules)")
        grid.addWidget(self.lbl_sel, 0, 0, 1, 2)
        grid.addWidget(QLabel("vx"), 1, 0)
        self.spin_vx = QDoubleSpinBox()
        self.spin_vx.setRange(-100, 100)
        self.spin_vx.setValue(1.0)
        grid.addWidget(self.spin_vx, 1, 1)
        grid.addWidget(QLabel("vy"), 2, 0)
        self.spin_vy = QDoubleSpinBox()
        self.spin_vy.setRange(-100, 100)
        grid.addWidget(self.spin_vy, 2, 1)
        actions = [("Eau", lambda m: self.runner.set_fluid(m, self._vel())),
                   ("Solide", lambda m: self.runner.set_solid(m, self._vel())),
                   ("Obstacle", self.runner.set_obstacle),
                   ("Entrée (vx, vy)", lambda m: self.runner.set_inlet(m, self._vel())),
                   ("Sortie", self.runner.set_outlet),
                   ("Vitesse initiale (vx, vy)", lambda m: self.runner.set_velocity(m, self._vel())),
                   ("Effacer", self.runner.clear)]
        self._cell_buttons = []
        for row, (text, fn) in enumerate(actions, start=3):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: self._apply_cells(f))
            b.setEnabled(False)
            grid.addWidget(b, row, 0, 1, 2)
            self._cell_buttons.append(b)
        self.chk_grid = QCheckBox("Maillage (si assez zoomé)")
        self.chk_grid.setChecked(True)
        self.chk_tint = QCheckBox("Teintes des cellules initiales")
        self.chk_tint.setChecked(True)
        base = 3 + len(actions)
        grid.addWidget(self.chk_grid, base, 0, 1, 2)
        grid.addWidget(self.chk_tint, base + 1, 0, 1, 2)
        grid.addWidget(QLabel("Couleur du fluide"), base + 2, 0)
        self.combo_color = QComboBox()
        self.combo_color.addItems(FLUID_MODES)
        self.combo_color.currentIndexChanged.connect(self._on_color_mode)
        grid.addWidget(self.combo_color, base + 2, 1)
        self.lbl_scale = QLabel("")
        grid.addWidget(self.lbl_scale, base + 3, 0, 1, 2)
        grid.setRowStretch(base + 4, 1)
        dock = QDockWidget("Cellules", self)
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
        self.viewport.n = self.runner.n
        self.dirty = False
        self.banner.hide()
        self.statusBar().clearMessage()

    def _mark_dirty(self) -> None:
        self.dirty = True
        self.banner.show()
        self._refresh_summaries()
        self._update_title(modified=True)

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
            if key == "n":
                self.runner.resize(int(value))
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

    def _apply_cells(self, fn) -> None:
        if self.viewport.sel is None:
            return
        i0, j0, i1, j1 = self.viewport.sel
        mask = np.zeros((self.runner.n, self.runner.n), bool)
        mask[i0:i1 + 1, j0:j1 + 1] = True
        fn(mask)
        self._mark_dirty()

    # ------------------------------------------------------------ fichiers
    def _load(self, runner: SimulationRunner, path: str | None) -> None:
        self._set_playing(False)
        self.runner = runner
        self.path = path
        for key, w in self._editors.items():
            w.blockSignals(True)
            v = runner.p[key]
            w.setChecked(bool(v)) if isinstance(w, QCheckBox) else w.setValue(v)
            w.blockSignals(False)
        self.viewport.sel = None
        self.viewport.reset_view()
        self._on_selection(None)
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

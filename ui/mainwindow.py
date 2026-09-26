"""Fenêtre principale : docks (arbre, propriétés), viewport GGUI embarqué, toolbar, menus, timer."""
from __future__ import annotations

import os

import taichi as ti
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QToolBar)

from ui.model.project import (INITIAL, STRUCTURAL, Circle, FluidMaterial, Project, Rect, SolidMaterial,
                              default_project, scope_of)
from ui.sim.session import SimSession
from ui.widgets.ggui_viewport import GguiViewport
from ui.widgets.project_tree import ProjectTree
from ui.widgets.properties import DataclassForm

TITLES = {"project": "Projet", "domain": "Domaine", "material": "Matériau", "shape": "Forme",
          "ic": "Condition initiale", "bc": "Conditions aux limites", "solver": "Solveur"}


def new_project() -> Project:
    p = Project(name="Nouveau projet")
    p.materials = [FluidMaterial(name="water"), SolidMaterial(name="beam")]
    p.shapes = [Rect(name="Water", role="fluid", material="water", x0=0.3, y0=0.5, x1=0.7, y1=0.9)]
    p.sync_ic()
    return p


class MainWindow(QMainWindow):
    def __init__(self, project: Project, arch: str = "?", path: str | None = None):
        super().__init__()
        self.project = project
        self.path = path
        self.arch = arch
        self.session = SimSession()
        self.session_dirty = True          # champs à (re)construire
        self.needs_reset = False
        self.build_error = False
        self.playing = False
        self.modified = False
        self.selected_shape = None
        # glfwPollEvents (dans get_events / show de GGUI) dispatche aussi les messages Windows de Qt :
        # un slot Qt peut donc s'exécuter AU MILIEU d'un appel Taichi. Règle : aucun slot ne touche
        # Taichi ; les actions sont mises en file (_pending) et exécutées au début du tick.
        self._in_tick = False
        self._pending: list = []
        self._close_requested = False
        self._shutdown_done = False

        # ---- viewport (crée la fenêtre GGUI ; QApplication existe déjà)
        self.viewport = GguiViewport(project.solver.viewport_res, self)
        self.setCentralWidget(self.viewport.widget)

        # ---- docks
        self.tree = ProjectTree()
        dock_tree = QDockWidget("Project", self)
        dock_tree.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_tree)

        self.form = DataclassForm(choices_provider=self._choices_for)
        dock_props = QDockWidget("Properties", self)
        dock_props.setWidget(self.form)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_props)
        self.resizeDocks([dock_tree, dock_props], [260, 320], Qt.Orientation.Horizontal)

        self._build_actions()
        self._build_statusbar()

        # ---- câblage
        self.tree.nodeSelected.connect(self._on_node_selected)
        self.tree.addShapeRequested.connect(self._add_shape)
        self.tree.deleteShapeRequested.connect(self._delete_shape)
        self.form.valueChanged.connect(self._on_project_changed)
        self.viewport.shapeClicked.connect(self._on_shape_clicked)
        self.viewport.shapeDragged.connect(self._on_shape_dragged)
        self.viewport.dragFinished.connect(lambda sid: self._on_project_changed(self.project.shape_by_id(sid), "x0"))
        self.viewport.keyPressed.connect(self._on_viewport_key)
        self.viewport.closed.connect(self.close)

        self.tree.rebuild(self.project)
        self.viewport.set_project(self.project)
        self._update_title()

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._frames = 0
        self._setup_diagnostics()

    def _setup_diagnostics(self) -> None:
        """Mode non interactif : APIC_UI_AUTOQUIT=<s> ferme après s secondes ; APIC_UI_SCREENSHOT=<prefix>
        enregistre <prefix>_viewport.png (rendu GGUI) et <prefix>_desktop.png (écran entier) juste avant."""
        secs = os.environ.get("APIC_UI_AUTOQUIT")
        if not secs:
            return
        self._set_playing(True)

        def finish():
            self._defer(finish_impl)

        def finish_impl():
            prefix = os.environ.get("APIC_UI_SCREENSHOT")
            if prefix:
                try:
                    self.viewport.window.save_image(prefix + "_viewport.png")
                    screen = QApplication.primaryScreen()
                    screen.grabWindow(0).save(prefix + "_desktop.png")
                except Exception as exc:
                    print(f"[diag] capture impossible : {exc}")
            st = self.session.stats() if self.session.built else {}
            print(f"[diag] embedded={self.viewport.embedded} frames={self._frames} built={self.session.built} "
                  f"geometry={self.geometry().width()}x{self.geometry().height()} stats={st}")
            self.close()

        QTimer.singleShot(int(float(secs) * 1000), finish)

    # ------------------------------------------------------------ construction UI
    def _build_actions(self) -> None:
        tb = QToolBar("Simulation")
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, checkable=False):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
                a.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            a.setCheckable(checkable)
            if checkable:
                a.toggled.connect(slot)
            else:
                a.triggered.connect(slot)
            return a

        self.act_play = act("▶ Play", self._set_playing, "Space", checkable=True)
        self.act_step = act("Step", self._step, "S")
        self.act_reset = act("Reset", self._reset, "R")
        for a in (self.act_play, self.act_step, self.act_reset):
            tb.addAction(a)
        tb.addSeparator()
        self.act_add_rect = act("+ Rect", lambda: self._add_shape("rect"))
        self.act_add_circle = act("+ Circle", lambda: self._add_shape("circle"))
        self.act_delete = act("Delete", self._delete_selected, "Del")
        for a in (self.act_add_rect, self.act_add_circle, self.act_delete):
            tb.addAction(a)

        m_file = self.menuBar().addMenu("&Fichier")
        m_file.addAction(act("Nouveau", self._new, "Ctrl+N"))
        m_file.addAction(act("Ouvrir…", self._open, "Ctrl+O"))
        m_file.addAction(act("Enregistrer", self._save, "Ctrl+S"))
        m_file.addAction(act("Enregistrer sous…", self._save_as, "Ctrl+Shift+S"))
        m_file.addSeparator()
        m_file.addAction(act("Quitter", self.close, "Ctrl+Q"))

        m_edit = self.menuBar().addMenu("&Édition")
        m_edit.addAction(self.act_add_rect)
        m_edit.addAction(self.act_add_circle)
        m_edit.addAction(self.act_delete)

        m_sim = self.menuBar().addMenu("&Simulation")
        for a in (self.act_play, self.act_step, self.act_reset):
            m_sim.addAction(a)

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.lbl_backend = QLabel()
        self.lbl_time = QLabel()
        self.lbl_particles = QLabel()
        self.lbl_damage = QLabel()
        self.lbl_perf = QLabel()
        for lbl in (self.lbl_backend, self.lbl_time, self.lbl_particles, self.lbl_damage, self.lbl_perf):
            lbl.setMargin(4)
            sb.addPermanentWidget(lbl)
        mode = "zéro copie" if self.viewport.zero_copy else "staging numpy"
        self.lbl_backend.setText(f"{self.arch} · {mode} · viewport {'embarqué' if self.viewport.embedded else 'séparé'}")

    # ------------------------------------------------------------ boucle
    def _defer(self, fn) -> None:
        """Exécute fn au début du prochain tick, hors de tout appel Taichi en cours."""
        self._pending.append(fn)

    def _tick(self) -> None:
        if self._in_tick or self._shutdown_done:  # jamais réentrant (dialogues, repaint synchrone...)
            return
        self._in_tick = True
        try:
            pending, self._pending = self._pending, []
            for fn in pending:
                fn()
            if not self._close_requested:
                self._tick_impl()
        finally:
            self._in_tick = False
        if self._close_requested:
            self.close()

    def _tick_impl(self) -> None:
        trace = os.environ.get("APIC_UI_TRACE")
        if trace:
            print(f"[trace] tick {self._frames} built={self.session.built} dirty={self.session_dirty} playing={self.playing}", flush=True)
        if (not self.session.built and not self.build_error) or (self.playing and self.session_dirty):
            self._ensure_built()
        if self.playing and self.session.built:
            if trace:
                print("[trace]   step", flush=True)
            self.session.step(self.project.solver.substeps_per_frame)
        if trace:
            print("[trace]   frame", flush=True)
        if not self.viewport.frame(self.session):
            return
        if trace:
            print("[trace]   frame done", flush=True)
        self._frames += 1
        if self._frames % 6 == 0:                # ~10 Hz suffit pour la status bar (8 octets GPU -> CPU)
            self._update_status()

    def _ensure_built(self) -> None:
        if self.session_dirty or not self.session.built:
            self.statusBar().showMessage("Construction des champs et compilation des kernels…")
            self.statusBar().repaint()           # peinture synchrone, sans relancer la boucle d'événements
            self.project.sync_ic()
            try:
                self.session.build(self.project)
            except ValueError as exc:
                self.build_error = True
                self.statusBar().showMessage(f"Projet invalide : {exc}")
                self._set_playing(False)
                return
            self.session_dirty = False
            self.needs_reset = False
            self.build_error = False
            self.viewport.set_dirty(False)
            self.statusBar().clearMessage()
        elif self.needs_reset:
            self.session.reset()
            self.needs_reset = False
        self.session.apply_runtime(self.project)

    def _update_status(self) -> None:
        if not self.session.built:
            self.lbl_time.setText("—")
            return
        st = self.session.stats()
        self.lbl_time.setText(f"t = {st['t']:.3f} s   dt = {st['dt']:.2e}")
        self.lbl_particles.setText(f"fluide {st['n_fluid']}   solide {st['n_solid']}")
        self.lbl_damage.setText(f"D max {st['D_max']:.2f}   rompues {st['n_broken']}")
        self.lbl_perf.setText(f"{st['last_step_ms']:.1f} ms/image" if self.playing else "pause")

    # ------------------------------------------------------------ actions simulation
    def _set_playing(self, on: bool) -> None:
        self.playing = bool(on)              # le tick construit la session si besoin avant de simuler
        if self.act_play.isChecked() != self.playing:
            self.act_play.blockSignals(True)
            self.act_play.setChecked(self.playing)
            self.act_play.blockSignals(False)
        self.act_play.setText("❚❚ Pause" if self.playing else "▶ Play")

    def _step(self) -> None:
        self._set_playing(False)

        def do_step():
            self._ensure_built()
            if self.session.built:
                self.session.step(self.project.solver.substeps_per_frame)
        self._defer(do_step)

    def _reset(self) -> None:
        def do_reset():
            if self.session_dirty or not self.session.built:
                self._ensure_built()
            elif self.session.built:
                self.session.reset()
                self.needs_reset = False
        self._defer(do_reset)

    def _on_viewport_key(self, key: str) -> None:
        if key == ti.ui.SPACE:
            self._set_playing(not self.playing)
        elif key == "r":
            self._reset()
        elif key == "s":
            self._step()
        elif key == ti.ui.ESCAPE:
            self._set_playing(False)

    # ------------------------------------------------------------ modèle -> session
    def _on_project_changed(self, obj, field: str) -> None:
        if obj is None:
            return
        self.modified = True
        scope = scope_of(obj, field)
        if field == "name":
            self.tree.refresh_labels()
        if scope == STRUCTURAL:
            self.session_dirty = True
            self.build_error = False
            self.viewport.set_dirty(True)
            if field == "role":                      # les CI et les matériaux possibles changent
                self.project.sync_ic()
                self.tree.rebuild(self.project)
                self.form.set_object(obj, TITLES["shape"])
            self.tree.refresh_labels()
            self.viewport.set_project(self.project)
        elif scope == INITIAL:
            self.needs_reset = True
        elif self.session.built:
            self.session.apply_runtime(self.project)
        self._update_title()

    def _choices_for(self, obj, name: str):
        if name == "material" and hasattr(obj, "role"):
            if obj.role == "obstacle":
                return [""]
            return [m.name for m in self.project.materials_of(obj.role)] or [""]
        return None

    # ------------------------------------------------------------ sélection
    def _on_node_selected(self, obj, kind: str) -> None:
        if kind == "shape":
            self.selected_shape = obj
            self.form.set_object(obj, f"{TITLES['shape']} : {obj.name}")
            self.viewport.set_selected(obj.id)
        else:
            self.selected_shape = None
            self.viewport.set_selected(None)
            if kind == "ic":
                shape = self.project.shape_by_id(obj.shape_id)
                self.form.set_object(obj, f"{TITLES['ic']} : {shape.name if shape else '?'}")
            elif kind in TITLES:
                self.form.set_object(obj, TITLES[kind])
            else:
                self.form.set_object(None)

    def _on_shape_clicked(self, sid: str) -> None:
        if sid:
            self.tree.select_shape(sid)
        else:
            self.tree.setCurrentItem(None)

    def _on_shape_dragged(self, sid: str, dx: float, dy: float) -> None:
        shape = self.project.shape_by_id(sid)
        if shape is None:
            return
        shape.translate(dx, dy)
        if self.selected_shape is shape:
            self.form.refresh()
        self.viewport.set_project(self.project)

    # ------------------------------------------------------------ formes
    def _add_shape(self, kind: str) -> None:
        fluids = self.project.materials_of("fluid")
        if kind == "circle":
            shape = Circle(name=f"Circle {len(self.project.shapes) + 1}", role="obstacle")
        else:
            shape = Rect(name=f"Rect {len(self.project.shapes) + 1}", role="fluid",
                         material=fluids[0].name if fluids else "")
        self.project.shapes.append(shape)
        self.project.sync_ic()
        self.tree.rebuild(self.project)
        self.tree.select_shape(shape.id)
        self._on_project_changed(shape, "role")

    def _delete_selected(self) -> None:
        if self.selected_shape is not None:
            self._delete_shape(self.selected_shape.id)

    def _delete_shape(self, sid: str) -> None:
        shape = self.project.shape_by_id(sid)
        if shape is None:
            return
        self.project.shapes.remove(shape)
        self.project.sync_ic()
        self.selected_shape = None
        self.form.set_object(None)
        self.viewport.set_selected(None)
        self.tree.rebuild(self.project)
        self._on_project_changed(self.project.domain, "n_grid")   # changement structural

    # ------------------------------------------------------------ fichiers
    def _load_project(self, project: Project, path: str | None) -> None:
        self._set_playing(False)
        self.project = project
        self.path = path
        self.modified = False
        self.selected_shape = None
        self.session_dirty = True
        self.build_error = False
        self.form.set_object(None)
        self.tree.rebuild(self.project)
        self.viewport.set_selected(None)
        self.viewport.set_project(self.project)
        self.viewport.set_dirty(True)
        self._update_title()

    def _new(self) -> None:
        self._load_project(new_project(), None)

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un projet", "", "Projet APIC (*.json)")
        if not path:
            return
        try:
            self._load_project(Project.load(path), path)
        except Exception as exc:
            QMessageBox.critical(self, "Ouverture impossible", str(exc))

    def _save(self) -> None:
        if self.path is None:
            self._save_as()
            return
        self.project.save(self.path)
        self.modified = False
        self._update_title()
        self.statusBar().showMessage(f"Enregistré : {self.path}", 3000)

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le projet", self.path or "projet.json",
                                              "Projet APIC (*.json)")
        if not path:
            return
        self.path = path
        self._save()

    def _update_title(self) -> None:
        name = os.path.basename(self.path) if self.path else self.project.name
        self.setWindowTitle(f"{name}{'*' if self.modified else ''} — APIC / MPM editor")

    # ------------------------------------------------------------ fermeture
    def closeEvent(self, event) -> None:
        if self._in_tick and not self._shutdown_done:
            self._close_requested = True     # on est peut-être au milieu d'un appel Taichi : plus tard
            event.ignore()
            return
        self._shutdown()
        super().closeEvent(event)

    def _shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.timer.stop()
        self.session.release()
        self.viewport.destroy()

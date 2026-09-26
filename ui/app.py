"""Application : ti.init, puis QApplication, puis la fenêtre principale (qui crée la fenêtre GGUI)."""
from __future__ import annotations

import os
import sys


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from ui.sim.backend import ensure_taichi
    arch = ensure_taichi()

    from PySide6.QtWidgets import QApplication
    app = QApplication(argv)
    app.setApplicationName("APIC / MPM editor")

    from ui.mainwindow import MainWindow
    from ui.model.project import Project, default_project

    path = next((a for a in argv[1:] if a.lower().endswith(".json")), None)
    project = Project.load(path) if path else default_project()
    win = MainWindow(project, arch=arch, path=path)
    win.show()
    return app.exec()

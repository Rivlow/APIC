"""Arbre du projet : Domain / Materials / Geometry / Initial Conditions / Boundary Conditions / Solver."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMenu, QTreeWidget, QTreeWidgetItem

from ui.model.project import Project

ROLE_OBJ = Qt.ItemDataRole.UserRole
ROLE_KIND = Qt.ItemDataRole.UserRole + 1


class ProjectTree(QTreeWidget):
    nodeSelected = Signal(object, str)        # (objet, kind) ; kind : project, domain, material, shape, ic, bc, solver, group
    addShapeRequested = Signal(str)           # "rect" | "circle"
    deleteShapeRequested = Signal(str)        # id de forme

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.project: Project | None = None
        self._shape_items: dict[str, QTreeWidgetItem] = {}
        self._geometry_item: QTreeWidgetItem | None = None
        self.currentItemChanged.connect(self._on_current_changed)

    # ------------------------------------------------------------ construction
    def rebuild(self, project: Project) -> None:
        self.project = project
        selected = self.current_shape_id()
        self.blockSignals(True)
        self.clear()
        self._shape_items.clear()

        root = self._item(None, project.name, project, "project")

        self._item(root, "Domain", project.domain, "domain")

        mats = self._item(root, "Materials", None, "group")
        for m in project.materials:
            self._item(mats, f"{m.name}  ({m.TAG})", m, "material")

        self._geometry_item = self._item(root, "Geometry", None, "group")
        for s in project.shapes:
            self._shape_items[s.id] = self._item(self._geometry_item, self._shape_label(s), s, "shape")

        ic = self._item(root, "Initial Conditions", None, "group")
        for e in project.initial_conditions.entries:
            shape = project.shape_by_id(e.shape_id)
            self._item(ic, shape.name if shape else e.shape_id, e, "ic")

        bc = self._item(root, "Boundary Conditions", None, "group")
        self._item(bc, "Parois", project.boundary_conditions, "bc")
        for s in project.shapes_of("obstacle"):
            self._item(bc, f"Obstacle : {s.name}", s, "shape")

        self._item(root, "Solver", project.solver, "solver")

        self.expandAll()
        self.blockSignals(False)
        if selected and selected in self._shape_items:
            self.setCurrentItem(self._shape_items[selected])

    def refresh_labels(self) -> None:
        if self.project is None:
            return
        self.topLevelItem(0).setText(0, self.project.name)
        for s in self.project.shapes:
            item = self._shape_items.get(s.id)
            if item is not None:
                item.setText(0, self._shape_label(s))

    def select_shape(self, sid: str) -> None:
        item = self._shape_items.get(sid)
        if item is not None:
            self.setCurrentItem(item)

    def current_shape_id(self) -> str | None:
        item = self.currentItem()
        if item is not None and item.data(0, ROLE_KIND) == "shape":
            return item.data(0, ROLE_OBJ).id
        return None

    # ------------------------------------------------------------ interne
    @staticmethod
    def _shape_label(s) -> str:
        return f"{s.name}  [{s.role}]"

    def _item(self, parent, text: str, obj, kind: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([text]) if parent is None else QTreeWidgetItem(parent, [text])
        if parent is None:
            self.addTopLevelItem(item)
        item.setData(0, ROLE_OBJ, obj)
        item.setData(0, ROLE_KIND, kind)
        return item

    def _on_current_changed(self, current, _previous) -> None:
        if current is None:
            self.nodeSelected.emit(None, "none")
            return
        self.nodeSelected.emit(current.data(0, ROLE_OBJ), current.data(0, ROLE_KIND))

    def contextMenuEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        menu = QMenu(self)
        act_rect = menu.addAction("Ajouter un rectangle")
        act_circle = menu.addAction("Ajouter un cercle")
        act_del = None
        if item is not None and item.data(0, ROLE_KIND) == "shape":
            menu.addSeparator()
            act_del = menu.addAction(f"Supprimer « {item.data(0, ROLE_OBJ).name} »")
        chosen = menu.exec(event.globalPos())
        if chosen is act_rect:
            self.addShapeRequested.emit("rect")
        elif chosen is act_circle:
            self.addShapeRequested.emit("circle")
        elif act_del is not None and chosen is act_del:
            self.deleteShapeRequested.emit(item.data(0, ROLE_OBJ).id)

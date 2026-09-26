"""Panneau de propriétés : formulaire généré depuis les métadonnées d'une dataclass."""
from __future__ import annotations

from dataclasses import fields, is_dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget, QColorDialog)


def _is_color(value) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value)


class DataclassForm(QWidget):
    """Affiche et édite les champs simples (int, float, bool, str, couleur) d'un objet dataclass.

    `choices_provider(obj, field_name) -> list[str] | None` permet des listes de choix dynamiques
    (ex. : les matériaux disponibles pour une forme).
    """
    valueChanged = Signal(object, str, object)   # (objet, nom du champ, nouvelle valeur)

    def __init__(self, choices_provider=None, parent=None):
        super().__init__(parent)
        self._provider = choices_provider
        self._obj = None
        self._editors: dict[str, tuple[QWidget, str]] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        self._title = QLabel("Aucune sélection")
        self._title.setStyleSheet("font-weight: bold;")
        outer.addWidget(self._title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._inner = QWidget()
        self._form = QFormLayout(self._inner)
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        scroll.setWidget(self._inner)
        outer.addWidget(scroll, 1)

    # ------------------------------------------------------------ API
    def set_object(self, obj, title: str = "") -> None:
        self._clear()
        self._obj = obj
        if obj is None or not is_dataclass(obj):
            self._title.setText("Aucune sélection")
            return
        self._title.setText(title or type(obj).__name__)
        for f in fields(obj):
            meta = f.metadata
            if meta.get("hidden"):
                continue
            value = getattr(obj, f.name)
            if is_dataclass(value) or (isinstance(value, list) and not _is_color(value)):
                continue
            editor, kind = self._make_editor(obj, f.name, value, dict(meta))
            self._editors[f.name] = (editor, kind)
            self._form.addRow(meta.get("label") or f.name, editor)

    def refresh(self) -> None:
        """Relit les valeurs de l'objet sans réémettre valueChanged."""
        if self._obj is None:
            return
        for name, (editor, kind) in self._editors.items():
            value = getattr(self._obj, name)
            editor.blockSignals(True)
            if kind == "float" or kind == "int":
                editor.setValue(value)
            elif kind == "bool":
                editor.setChecked(bool(value))
            elif kind == "choice":
                editor.setCurrentText(str(value))
            elif kind == "str":
                editor.setText(str(value))
            elif kind == "color":
                self._paint_color_button(editor, value)
            elif kind == "label":
                editor.setText(str(value))
            editor.blockSignals(False)

    # ------------------------------------------------------------ interne
    def _clear(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._editors.clear()

    def _emit(self, name: str, value) -> None:
        if self._obj is None:
            return
        setattr(self._obj, name, value)
        self.valueChanged.emit(self._obj, name, value)

    def _make_editor(self, obj, name: str, value, meta: dict):
        if meta.get("readonly"):
            return QLabel(str(value)), "label"

        choices = meta.get("choices")
        if choices is None and self._provider is not None:
            choices = self._provider(obj, name)
        if choices is not None:
            box = QComboBox()
            box.addItems([str(c) for c in choices])
            if str(value) not in [str(c) for c in choices]:
                box.insertItem(0, str(value))
            box.setCurrentText(str(value))
            box.currentTextChanged.connect(lambda t, n=name: self._emit(n, t))
            return box, "choice"

        if isinstance(value, bool):
            cb = QCheckBox()
            cb.setChecked(value)
            cb.toggled.connect(lambda v, n=name: self._emit(n, bool(v)))
            return cb, "bool"

        if isinstance(value, int):
            sb = QSpinBox()
            sb.setRange(int(meta.get("lo") if meta.get("lo") is not None else -10**9),
                        int(meta.get("hi") if meta.get("hi") is not None else 10**9))
            sb.setValue(value)
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda v, n=name: self._emit(n, int(v)))
            return sb, "int"

        if isinstance(value, float):
            sb = QDoubleSpinBox()
            sb.setDecimals(meta.get("decimals") if meta.get("decimals") is not None else 4)
            sb.setRange(float(meta.get("lo") if meta.get("lo") is not None else -1e12),
                        float(meta.get("hi") if meta.get("hi") is not None else 1e12))
            sb.setSingleStep(float(meta.get("step") or 0.1))
            sb.setValue(value)
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda v, n=name: self._emit(n, float(v)))
            return sb, "float"

        if _is_color(value):
            btn = QPushButton()
            self._paint_color_button(btn, value)
            btn.clicked.connect(lambda _=False, n=name, b=btn: self._pick_color(n, b))
            return btn, "color"

        le = QLineEdit(str(value))
        le.editingFinished.connect(lambda n=name, w=le: self._emit(n, w.text()) if w.text() != getattr(self._obj, n) else None)
        return le, "str"

    def _pick_color(self, name: str, btn: QPushButton) -> None:
        cur = getattr(self._obj, name)
        qc = QColorDialog.getColor(QColor.fromRgbF(*cur), self, "Couleur")
        if qc.isValid():
            rgb = [round(qc.redF(), 3), round(qc.greenF(), 3), round(qc.blueF(), 3)]
            self._paint_color_button(btn, rgb)
            self._emit(name, rgb)

    @staticmethod
    def _paint_color_button(btn: QPushButton, rgb) -> None:
        r, g, b = (int(255 * c) for c in rgb)
        btn.setText(f"({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})")
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {'black' if (r + g + b) > 380 else 'white'};")

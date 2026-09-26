"""Semis des particules et masque d'obstacles, en numpy (CPU), une fois par build/reset.

Reproduit la logique de main_fsi.py : fluide = points uniformes (ppc² par cellule en moyenne),
solide = réseau régulier ppc × ppc par cellule, obstacles = nœuds de grille à l'intérieur des formes.
"""
from __future__ import annotations

import numpy as np

from ui.model.project import Circle, Project, Rect


def particle_spacing(project: Project) -> float:
    return 1.0 / project.domain.n_grid / project.solver.ppc_side


def _bbox(shape):
    if isinstance(shape, Rect):
        return min(shape.x0, shape.x1), min(shape.y0, shape.y1), max(shape.x0, shape.x1), max(shape.y0, shape.y1)
    return shape.cx - shape.r, shape.cy - shape.r, shape.cx + shape.r, shape.cy + shape.r


def _inside(shape, pts: np.ndarray) -> np.ndarray:
    if isinstance(shape, Rect):
        return np.ones(len(pts), dtype=bool)          # les points sont déjà tirés dans le rectangle
    d2 = (pts[:, 0] - shape.cx) ** 2 + (pts[:, 1] - shape.cy) ** 2
    return d2 <= shape.r ** 2


def seed_fluid(project: Project, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Positions et vitesses initiales des particules fluides, (N, 2) float32 chacune."""
    n_grid, ppc = project.domain.n_grid, project.solver.ppc_side
    xs, vs = [], []
    for s in project.shapes_of("fluid"):
        x0, y0, x1, y1 = _bbox(s)
        n_bbox = int(ppc * ppc * (x1 - x0) * (y1 - y0) * n_grid * n_grid)
        if n_bbox <= 0:
            continue
        pts = np.column_stack([rng.uniform(x0, x1, n_bbox), rng.uniform(y0, y1, n_bbox)])
        pts = pts[_inside(s, pts)]
        vx, vy = project.velocity_of(s.id)
        xs.append(pts)
        vs.append(np.tile([vx, vy], (len(pts), 1)))
    if not xs:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.vstack(xs).astype(np.float32), np.vstack(vs).astype(np.float32)


def seed_solid(project: Project) -> tuple[np.ndarray, np.ndarray]:
    """Réseau régulier de particules solides (comme init_beam de Code_tuto/mpm_solid.py)."""
    spacing = particle_spacing(project)
    xs, vs = [], []
    for s in project.shapes_of("solid"):
        x0, y0, x1, y1 = _bbox(s)
        n_px = int(round((x1 - x0) / spacing))
        n_py = int(round((y1 - y0) / spacing))
        if n_px <= 0 or n_py <= 0:
            continue
        i, j = np.meshgrid(np.arange(n_px), np.arange(n_py), indexing="xy")
        pts = np.column_stack([x0 + (i.ravel() + 0.5) * spacing, y0 + (j.ravel() + 0.5) * spacing])
        pts = pts[_inside(s, pts)]
        vx, vy = project.velocity_of(s.id)
        xs.append(pts)
        vs.append(np.tile([vx, vy], (len(pts), 1)))
    if not xs:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.vstack(xs).astype(np.float32), np.vstack(vs).astype(np.float32)


def build_mask(project: Project, n_grid: int) -> np.ndarray:
    """Masque d'obstacle sur les nœuds de grille (1.0 = obstacle), position du nœud = (i, j) * dx."""
    dx = 1.0 / n_grid
    i, j = np.meshgrid(np.arange(n_grid), np.arange(n_grid), indexing="ij")
    px, py = i * dx, j * dx
    mask = np.zeros((n_grid, n_grid), dtype=bool)
    for s in project.shapes_of("obstacle"):
        if isinstance(s, Rect):
            x0, y0, x1, y1 = _bbox(s)
            mask |= (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        elif isinstance(s, Circle):
            mask |= (px - s.cx) ** 2 + (py - s.cy) ** 2 <= s.r ** 2
    return mask.astype(np.float32)

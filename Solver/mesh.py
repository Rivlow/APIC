# Solver/mesh.py -- Import de maillages (STL binaire / ASCII, OBJ) et voxelisation sur la grille (numpy + SciPy).
#
#     tri = load_mesh("piece.stl")                          # (T, 3, 3) sommets des triangles
#     tri = transform(tri, scale=0.001, rotate=(0, 90, 0), translate=(0.5, 0.2, 0.3))   # mm -> m, degrés
#     mask = voxelize(tri, (nx, ny, nz), dx)                # cellules intérieures (bool)
#
# Voxelisation : la surface est échantillonnée (points sur chaque triangle, pas <= dx / 2) et marque les cellules
# qu'elle traverse ; l'intérieur est ensuite rempli (scipy.ndimage.binary_fill_holes), ce qui tolère des maillages
# réels pas parfaitement fermés tant que la coque voxelisée l'est. fill=False garde seulement la coque (épaissie de
# `shell` cellules) : pour une pièce ouverte (tôle, aube).

import numpy as np


def load_mesh(path: str) -> np.ndarray:
    """Read a triangle mesh.

    **Inputs**

    - `path` : str, .stl (binary or ASCII) or .obj (polygons triangulated as fans)

    **Outputs**

    - np.ndarray f64 (T, 3, 3) triangle vertices (file units)
    """
    low = path.lower()
    if low.endswith(".stl"):
        return _load_stl(path)
    if low.endswith(".obj"):
        return _load_obj(path)
    raise ValueError(f"format de maillage non pris en charge : {path} (attendu .stl ou .obj)")


def _load_stl(path: str) -> np.ndarray:
    """Read an STL file (binary, or ASCII 'solid ... facet ... vertex').

    **Inputs**

    - `path` : str

    **Outputs**

    - np.ndarray f64 (T, 3, 3)
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        count = int(np.frombuffer(data, np.uint32, 1, 80)[0])
        if 84 + 50 * count == len(data):              # binaire : en-tête 80 o + nombre + 50 o par triangle
            rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            return np.frombuffer(data, rec, count, 84)["v"].astype(np.float64)
    text = data.decode("ascii", errors="ignore")
    verts = [line.split()[1:4] for line in text.splitlines() if line.strip().startswith("vertex")]
    if not verts or len(verts) % 3:
        raise ValueError(f"STL illisible : {path}")
    return np.asarray(verts, np.float64).reshape(-1, 3, 3)


def _load_obj(path: str) -> np.ndarray:
    """Read the vertices and faces of an OBJ file (fan triangulation).

    **Inputs**

    - `path` : str

    **Outputs**

    - np.ndarray f64 (T, 3, 3)
    """
    vs, tris = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vs.append([float(c) for c in parts[1:4]])
            elif parts[0] == "f":
                idx = [int(tok.split("/")[0]) for tok in parts[1:]]
                idx = [i - 1 if i > 0 else len(vs) + i for i in idx]      # indices OBJ : 1.., négatifs relatifs
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    if not tris:
        raise ValueError(f"OBJ sans face : {path}")
    return np.asarray(vs, np.float64)[np.asarray(tris)]


def rotation(deg) -> np.ndarray:
    """Rotation matrix from Euler angles (degrees, applied about x, then y, then z).

    **Inputs**

    - `deg` : (3,) float

    **Outputs**

    - np.ndarray (3, 3)
    """
    ax, ay, az = np.radians(np.asarray(deg, np.float64))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(tri: np.ndarray, scale=1.0, rotate=(0.0, 0.0, 0.0), translate=(0.0, 0.0, 0.0),
              pivot=None) -> np.ndarray:
    """Scale, rotate and move a mesh about a pivot.

    **Inputs**

    - `tri` : np.ndarray (T, 3, 3)
    - `scale` : float | (3,) float (e.g. 0.001 for mm -> m)
    - `rotate` : (3,) float Euler angles in degrees (x, then y, then z)
    - `translate` : (3,) float, where the pivot lands, m
    - `pivot` : (3,) float in file units | None (bounding-box center of this mesh) ; parts of one assembly
      share a pivot to stay aligned

    **Outputs**

    - np.ndarray (T, 3, 3)
    """
    v = tri.reshape(-1, 3)
    pv = 0.5 * (v.min(axis=0) + v.max(axis=0)) if pivot is None else np.asarray(pivot, np.float64)
    v = (v - pv) * np.asarray(scale, np.float64)
    v = v @ rotation(rotate).T + np.asarray(translate, np.float64)
    return v.reshape(-1, 3, 3)


def bounds(tri: np.ndarray) -> tuple:
    """Axis-aligned bounding box.

    **Inputs**

    - `tri` : np.ndarray (T, 3, 3)

    **Outputs**

    - (lo (3,), hi (3,))
    """
    v = tri.reshape(-1, 3)
    return v.min(axis=0), v.max(axis=0)


def voxelize(tri: np.ndarray, shape, dx: float, fill: bool = True, shell: int = 1) -> np.ndarray:
    """Cells occupied by a mesh.

    **Inputs**

    - `tri` : np.ndarray (T, 3, 3) triangle vertices (m)
    - `shape` : (nx, ny, nz) cells ; `dx` : float cell size (m)
    - `fill` : bool, fill the enclosed interior ; False : surface shell only
    - `shell` : int, shell thickness in cells when fill is False (dilation)

    **Outputs**

    - np.ndarray bool (shape)

    **Note** : surface sampled with a step <= dx / 2 on every triangle (vectorized by subdivision count).
    """
    from scipy import ndimage
    shape = tuple(int(v) for v in shape)
    surf = np.zeros(shape, bool)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    edge = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                            np.linalg.norm(a - c, axis=1)]), axis=0)
    k_all = np.maximum(1, np.ceil(edge / (0.5 * dx)).astype(int))
    lim = np.array(shape) - 1
    for k in np.unique(k_all):                        # triangles regroupés par nombre de subdivisions
        sel = k_all == k
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        u = (i[keep] / k)[None, :, None]
        w = (j[keep] / k)[None, :, None]
        ta, tb, tc = a[sel][:, None, :], b[sel][:, None, :], c[sel][:, None, :]
        pts = (ta + u * (tb - ta) + w * (tc - ta)).reshape(-1, 3)
        idx = np.floor(pts / dx).astype(np.int64)
        ok = np.all((idx >= 0) & (idx <= lim), axis=1)
        idx = idx[ok]
        surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if not fill:
        if shell > 1:
            surf = ndimage.binary_dilation(surf, iterations=shell - 1)
        return surf
    # intérieur : centre de cellule dedans (parité des croisements le long de z, exacte pour un maillage fermé),
    # restreint à la coque remplie (un maillage troué ne laisse pas de traînée de parité jusqu'au bord)
    return _parity_z(tri, shape, dx) & ndimage.binary_fill_holes(surf)


def _parity_z(tri: np.ndarray, shape, dx: float) -> np.ndarray:
    """Cells whose center is inside a closed mesh (ray parity along +z through every column center).

    **Inputs**

    - `tri` : np.ndarray (T, 3, 3) ; `shape` : (nx, ny, nz) ; `dx` : float

    **Outputs**

    - np.ndarray bool (shape)
    """
    nx, ny, nz = shape
    jit = np.array([1.37e-4, 2.91e-4]) * dx           # décalage irrationnel : pas de rayon exactement sur une arête
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    lo = np.floor(np.minimum(np.minimum(p0, p1), p2)[:, :2] / dx - 0.5).astype(np.int64) + 1
    hi = np.floor(np.maximum(np.maximum(p0, p1), p2)[:, :2] / dx - 0.5).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [nx - 1, ny - 1])
    cnt = np.maximum(hi - lo + 1, 0)
    tot = cnt[:, 0] * cnt[:, 1]
    keep = tot > 0
    crossings = np.zeros((nx, ny, nz + 1), np.int32)
    idx_t = np.repeat(np.nonzero(keep)[0], tot[keep])  # une ligne par (triangle, colonne candidate)
    if idx_t.size:
        start = np.repeat(np.cumsum(tot[keep]) - tot[keep], tot[keep])
        k = np.arange(idx_t.size) - start
        ci = lo[idx_t, 0] + k // cnt[idx_t, 1]
        cj = lo[idx_t, 1] + k % cnt[idx_t, 1]
        qx, qy = (ci + 0.5) * dx + jit[0], (cj + 0.5) * dx + jit[1]
        a, b, c = p0[idx_t], p1[idx_t], p2[idx_t]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        ok = np.abs(det) > 1e-30
        det = np.where(ok, det, 1.0)
        u = ((qx - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (qy - a[:, 1])) / det
        v = ((b[:, 0] - a[:, 0]) * (qy - a[:, 1]) - (qx - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
        z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
        k0 = np.clip(np.ceil(z / dx - 0.5).astype(np.int64), 0, nz)   # première cellule dont le centre est au-dessus
        np.add.at(crossings, (ci[hit], cj[hit], k0[hit]), 1)
    return (np.cumsum(crossings, axis=2)[:, :, :nz] % 2) == 1

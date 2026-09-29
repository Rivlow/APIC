"""Export VTK (ParaView) d'une simulation : particules, grille, maillages STL (roue tournante comprise).

    from ui.export_vtk import VTKExporter
    ex = VTKExporter("sortie_vtk", runner, solver)       # maillages fixes écrits une fois
    for k in range(200):
        solver.step()
        if k % 5 == 0:
            ex.write()                                   # un instant : fichiers + séries .pvd mises à jour

Fichiers (VTK XML binaire, lus directement par ParaView, sans dépendance) :
  fluid.pvd   -> fluid_NNNNN.vtp   particules fluides : velocity, speed, pressure, density (incompressible)
  solid.pvd   -> solid_NNNNN.vtp   particules solides : velocity, damage, broken
  grid.pvd    -> grid_NNNNN.vti    cellules : pressure, cell_type (0 air, 1 fluide, 2 solide, 3 solide MPM,
                                   4 rotor), obstacle
  rotor.pvd   -> rotor_NNNNN.vtp   maillage de la roue à son angle courant (primitive rotor de type maillage)
  mesh_K.vtp                        maillages STL fixes (obstacles), une fois
Ouvrir les .pvd dans ParaView (lecture animée) ; unités SI, mêmes axes que la simulation (y vertical).
Chaque écriture copie les champs du GPU vers le CPU : à faire toutes les quelques images, pas à chaque sous-pas.
"""
import base64
import os

import numpy as np

from ui import kernels as K


def _b64(a: np.ndarray) -> str:
    """VTK XML inline binary block: UInt32 byte count + raw little-endian data, base64."""
    raw = np.ascontiguousarray(a).tobytes()
    return base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode("ascii")


def _vtype(a: np.ndarray) -> str:
    return {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64", np.dtype(np.int32): "Int32",
            np.dtype(np.uint8): "UInt8", np.dtype(np.int64): "Int64"}[a.dtype]


def _array(name: str, a: np.ndarray) -> str:
    a = np.asarray(a)
    ncomp = 1 if a.ndim == 1 else a.shape[1]
    return (f'<DataArray type="{_vtype(a)}" Name="{name}" NumberOfComponents="{ncomp}" format="binary">'
            f'{_b64(a)}</DataArray>\n')


def write_points(path: str, pos: np.ndarray, data: dict) -> None:
    """Point cloud as VTK PolyData (.vtp), one vertex per point.

    **Inputs**

    - `path` : str ; `pos` : np.ndarray (N, 2 or 3) ; `data` : dict name -> (N,) or (N, k) array
    """
    n = len(pos)
    p3 = np.zeros((n, 3), np.float32)
    p3[:, :pos.shape[1]] = pos
    idx = np.arange(n, dtype=np.int32)
    with open(path, "w", encoding="ascii") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" '
                'header_type="UInt32">\n<PolyData>\n')
        f.write(f'<Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" '
                'NumberOfPolys="0">\n<PointData>\n')
        for k, v in data.items():
            f.write(_array(k, np.asarray(v, np.float32)))
        f.write('</PointData>\n<Points>\n' + _array("Points", p3) + '</Points>\n<Verts>\n')
        f.write(_array("connectivity", idx) + _array("offsets", idx + 1))
        f.write('</Verts>\n</Piece>\n</PolyData>\n</VTKFile>\n')


def index_mesh(tri: np.ndarray) -> tuple:
    """Merge the shared vertices of a triangle soup.

    **Inputs**

    - `tri` : np.ndarray (T, 3, 3)

    **Outputs**

    - (points np.ndarray (V, 3) f64, connectivity np.ndarray (3T,) i32)
    """
    pts, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    return pts, inv.reshape(-1).astype(np.int32)


def write_triangles(path: str, tri: np.ndarray, conn: np.ndarray | None = None) -> None:
    """Triangle mesh as VTK PolyData (.vtp).

    **Inputs**

    - `path` : str
    - `tri` : np.ndarray (T, 3, 3) triangle soup, or (V, 3) points when `conn` is given
    - `conn` : np.ndarray (3T,) i32 | None   vertex indices (see index_mesh)
    """
    if conn is None:
        tri, conn = index_mesh(tri)
    pts = np.asarray(tri, np.float32)
    t = len(conn) // 3
    off = np.arange(1, t + 1, dtype=np.int32) * 3
    with open(path, "w", encoding="ascii") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" '
                'header_type="UInt32">\n<PolyData>\n')
        f.write(f'<Piece NumberOfPoints="{len(pts)}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" '
                f'NumberOfPolys="{t}">\n<Points>\n' + _array("Points", pts) + '</Points>\n<Polys>\n')
        f.write(_array("connectivity", conn) + _array("offsets", off))
        f.write('</Polys>\n</Piece>\n</PolyData>\n</VTKFile>\n')


def write_grid(path: str, dx: float, cell_data: dict) -> None:
    """Cell fields on the regular grid as VTK ImageData (.vti).

    **Inputs**

    - `path` : str ; `dx` : float cell size (m) ; `cell_data` : dict name -> np.ndarray (nx, ny[, nz]) scalars
    """
    nx, ny, nz = (tuple(next(iter(cell_data.values())).shape) + (1,))[:3]
    ext = f"0 {nx} 0 {ny} 0 {nz}"
    with open(path, "w", encoding="ascii") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" '
                'header_type="UInt32">\n')
        f.write(f'<ImageData WholeExtent="{ext}" Origin="0 0 0" Spacing="{dx} {dx} {dx}">\n'
                f'<Piece Extent="{ext}">\n<CellData>\n')
        for k, v in cell_data.items():                  # ordre VTK : x le plus rapide
            f.write(_array(k, np.asarray(v).reshape(nx, ny, nz).transpose(2, 1, 0).ravel()))
        f.write('</CellData>\n</Piece>\n</ImageData>\n</VTKFile>\n')


def write_pvd(path: str, entries: list) -> None:
    """Time series collection (.pvd).

    **Inputs**

    - `path` : str ; `entries` : list of (time float, file name relative to the .pvd)
    """
    with open(path, "w", encoding="ascii") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for t, name in entries:
            f.write(f'<DataSet timestep="{t:.6g}" group="" part="0" file="{name}"/>\n')
        f.write('</Collection>\n</VTKFile>\n')


class VTKExporter:
    def __init__(self, folder: str, runner, solver):
        """Prepare an export folder; write the fixed STL obstacles once.

        **Inputs**

        - `folder` : str (created) ; `runner` : SimulationRunner ; `solver` : Solver built from it
        """
        self.folder, self.r, self.s = folder, runner, solver
        os.makedirs(folder, exist_ok=True)
        self.series = {"fluid": [], "solid": [], "grid": [], "rotor": []}
        self.count = 0
        k = 0
        for prim in runner.prims:                       # maillages fixes : une fois
            if prim["kind"] == "mesh" and prim["material"] != "rotor":
                write_triangles(os.path.join(folder, f"mesh_{k}.vtp"), runner.mesh_triangles(prim))
                k += 1
        rot = [q for q in runner.prims if q["material"] == "rotor"]
        self.rotor_mesh = None                          # (points, connectivité) à l'angle 0, indexés une fois
        if rot and rot[0]["kind"] == "mesh" and solver.rotor is not None:
            self.rotor_mesh = index_mesh(runner.mesh_triangles(rot[0]))

    def write(self) -> str:
        """Export the current state (one time step) and refresh the .pvd series.

        **Outputs**

        - str name of the step (e.g. "00012")
        """
        s = self.s
        tag = f"{self.count:05d}"
        t = float(s.t)
        if s.has_fluid:
            alive = s.alive.to_numpy() == 1
            x, v = s.x_f.to_numpy()[alive], s.v_f.to_numpy()[alive]
            data = {"velocity": np.pad(v, ((0, 0), (0, 3 - v.shape[1]))), "speed": np.linalg.norm(v, axis=1)}
            if s.incompressible:                        # pression et masse volumique à la cellule de la particule
                keep = s.fluid_mode
                for mode, name in ((K.MODE_P, "pressure"), (K.MODE_DENSITY, "density")):
                    s.fluid_mode = mode
                    s._colors()
                    data[name] = s.sc_f.to_numpy()[alive] + (float(s.p["fluid_rho"]) if name == "density" else 0.0)
                s.fluid_mode = keep
            write_points(os.path.join(self.folder, f"fluid_{tag}.vtp"), x, data)
            self.series["fluid"].append((t, f"fluid_{tag}.vtp"))
        if s.has_solid:
            xs, vs = s.x_s.to_numpy(), s.v_s.to_numpy()
            write_points(os.path.join(self.folder, f"solid_{tag}.vtp"), xs,
                         {"velocity": np.pad(vs, ((0, 0), (0, 3 - vs.shape[1]))), "damage": s.D_s.to_numpy(),
                          "broken": s.broken_s.to_numpy().astype(np.float32)})
            self.series["solid"].append((t, f"solid_{tag}.vtp"))
        cells = {"obstacle": (s.cells.to_numpy() & K.OBSTACLE > 0).astype(np.uint8)}
        if s.incompressible:
            cells["pressure"] = (s.mac.q.to_numpy() * s.p["fluid_rho"] / s.dt).astype(np.float32)
            cells["cell_type"] = s.mac.ctype.to_numpy().astype(np.int32)
        write_grid(os.path.join(self.folder, f"grid_{tag}.vti"), s.dx, cells)
        self.series["grid"].append((t, f"grid_{tag}.vti"))
        if self.rotor_mesh is not None:                 # roue à son angle omega t
            pts, conn = self.rotor_mesh
            c = np.asarray(s.rotor["center"], np.float64)
            ax = np.asarray(s.rotor["axis"], np.float64)
            ang = s.rot_omega * s.t
            kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * kx @ kx       # Rodrigues
            write_triangles(os.path.join(self.folder, f"rotor_{tag}.vtp"), (pts - c) @ R.T + c, conn)
            self.series["rotor"].append((t, f"rotor_{tag}.vtp"))
        for name, entries in self.series.items():
            if entries:
                write_pvd(os.path.join(self.folder, f"{name}.pvd"), entries)
        self.count += 1
        return tag


__all__ = ["VTKExporter", "write_points", "write_triangles", "write_grid", "write_pvd"]

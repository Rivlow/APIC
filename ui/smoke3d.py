"""Test de fumée 3D sans fenêtre : python -m ui.smoke3d (depuis la racine du dépôt, ou n'importe où)."""
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _write_cube_stl(path: str, lo, hi) -> None:
    """Binary STL of an axis-aligned box (12 triangles)."""
    import numpy as np
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float32)
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
         (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    rec = np.zeros(len(f), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]))
    rec["v"] = v[np.array(f)]
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80 + np.uint32(len(f)).tobytes() + rec.tobytes())


def _write_sphere_obj(path: str, center, radius: float, n: int = 48) -> None:
    """OBJ UV sphere (quads and triangles, 1-based indices)."""
    import numpy as np
    lines, idx = [], {}
    for i in range(n + 1):
        th = np.pi * i / n
        for j in range(n):
            ph = 2 * np.pi * j / n
            idx[i, j] = len(idx) + 1
            p = np.asarray(center) + radius * np.array([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)])
            lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    for i in range(n):
        for j in range(n):
            a, b = idx[i, j], idx[i, (j + 1) % n]
            c, d = idx[i + 1, (j + 1) % n], idx[i + 1, j]
            lines.append(f"f {a} {b} {c} {d}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    """Run headless 3D smoke tests.

    **Outputs**

    - `int` : 0 on success (AssertionError otherwise)
    """
    import numpy as np

    from ui.render3d import Camera
    from ui.runner import SimulationRunner

    # ---- rupture de barrage 3D + obstacle sphérique : projection convergée, le front avance
    r = SimulationRunner(dim=3, nx=24, Lx=1.0, Ly=1.0, Lz=1.0, incompressible=True, substeps=2, cg_iters=6,
                         density_iters=3)
    r.set_fluid(r.box((0.1, 0.1, 0.1), (0.4, 0.6, 0.9)))
    r.set_obstacle(r.sphere((0.7, 0.2, 0.5), 0.12))
    s = r.solver()
    x0 = s.positions()[:, 0].mean()
    for _ in range(10):
        s.step()
    st = s.stats()
    x = s.positions()
    print(f"[smoke3d] barrage : t = {st['t']:.3f}  n = {st['n_fluid']}  div max = {st['div_max']:.1e}  "
          f"x moyen {x0:.3f} -> {x[:, 0].mean():.3f}  {st['ms']:.0f} ms/image")
    assert x.shape[1] == 3 and np.isfinite(x).all(), "positions 3D"
    assert st["div_max"] < 1e-2, "projection 3D non convergée"
    assert x[:, 0].mean() > x0 + 0.05, "le barrage 3D ne s'effondre pas"

    # ---- rendu 3D : image non uniforme, la caméra et le plan de coupe changent l'image
    cam = Camera.fit(r.extent)
    img = s.render3d(cam, size=(160, 120))
    assert img.shape == (120, 160, 3) and img.dtype == np.uint8 and (img != img[0, 0]).any(), "rendu 3D vide"
    cam.orbit(90, 0)
    img2 = s.render3d(cam, size=(160, 120))
    assert (img2 != img).any(), "la caméra ne change rien"
    img3 = s.render3d(cam, size=(160, 120), clip=(0, 0.0, True))          # toutes les particules masquées
    assert (img3 != img2).any(), "le plan de coupe ne change rien"
    zb = s.view3d.zbuf.to_numpy()[:120, :160]
    assert np.isfinite(zb).any() and (zb < 1e29).any(), "profondeur"
    s.release()

    # ---- profondeur : une sphère proche cache une sphère lointaine (même pixel central)
    r = SimulationRunner(dim=3, nx=24, Lx=1.0, Ly=1.0, Lz=1.0, incompressible=True, gravity=0.0, substeps=1,
                         cg_iters=2, density_iters=0)
    r.set_fluid(r.sphere((0.5, 0.5, 0.25), 0.1), velocity=(0, 0, 0))
    r.set_solid(r.sphere((0.5, 0.5, 0.75), 0.1))
    s = r.solver()
    s.fluid_mode = 0
    cam = Camera(target=(0.5, 0.5, 0.5), yaw=180.0, pitch=0.0, dist=2.0)      # œil côté z < 0 : le fluide devant
    img = s.render3d(cam, size=(64, 64))
    c = img[32, 32].astype(int)
    assert c[2] > c[0] + 40, f"la sphère de fluide (bleue) doit cacher le solide : {c}"
    cam = Camera(target=(0.5, 0.5, 0.5), yaw=0.0, pitch=0.0, dist=2.0)        # œil côté z > 0 : le solide devant
    img = s.render3d(cam, size=(64, 64))
    c2 = img[32, 32].astype(int)
    assert abs(c2[0] - c2[2]) < 60 and (c2 != c).any(), f"le solide (gris) doit cacher le fluide : {c2}"
    s.release()

    # ---- entrée sur un rectangle de la face gauche, sortie libre à droite : le fluide traverse
    r = SimulationRunner(dim=3, nx=32, Lx=1.0, Ly=0.5, Lz=0.5, incompressible=True, gravity=0.0,
                         free_surface=False, substeps=2, cg_iters=6, density_iters=0, capacity=200_000)
    b = r.band                                                  # intérieur seulement (hors bande de paroi)
    r.set_fluid(r.box((b, b, b), (1.0 - b, 0.5 - b, 0.5 - b)))
    r.set_wall("left", "inlet", velocity=(1.0, 0.0, 0.0), span=((0.15, 0.35), (0.15, 0.35)))
    r.set_wall("right", "outlet")
    tab = r.wall_table()
    n, b = r.shape, r.p["bound"]
    assert tab.type.shape == (6, max(n[1], n[0]), max(n[2], n[1])), f"table {tab.type.shape}"
    ks = slice(int(0.15 / r.dx) + 1, int(0.35 / r.dx) - 1)
    assert (tab.type[0, ks, ks] == 1).all() and (tab.v[0, 8, 8] == [1.0, 0.0, 0.0]).all(), "entrée rectangle"
    assert (tab.type[1, b:n[1] - b, b:n[2] - b] == 2).all() and (tab.type[1, :b] == 0).all(), "sortie + coins"
    s = r.solver()
    n0 = s.n_fluid_init
    for _ in range(20):
        s.step()
    st = s.stats()
    v = s.velocities()
    print(f"[smoke3d] entrée/sortie : n = {n0} -> {st['n_fluid']}  vx moyen = {v[:, 0].mean():.3f}  "
          f"|v| max = {np.abs(v).max():.2f}  div max = {st['div_max']:.1e}")
    assert 0.2 < v[:, 0].mean() < 1.0 and np.abs(v).max() < 1.5, "écoulement de l'entrée vers la sortie"
    assert abs(st["n_fluid"] - n0) < 0.05 * n0, "débit entrant = débit sortant (conduit plein)"
    s.release()

    # ---- obstacle collé au mur droit sur une sortie : la face de l'obstacle porte la sortie
    r = SimulationRunner(dim=3, nx=40, Lx=1.0, Ly=1.0, Lz=1.0, incompressible=True)
    r.set_obstacle(r.box((0.85, 0.4, 0.4), (1.0, 0.6, 0.6)))
    r.set_wall("right", "outlet")
    tab = r.wall_table()
    d = tab.depth[1, 18:22, 18:22]
    assert (d == 6).all() and tab.depth[1, 10, 10] == r.p["bound"], f"profondeur de la sortie collée : {d}"

    # ---- JSON v5 (3D, primitives, zones rectangulaires) + lecture d'un fichier 2D v4
    r = SimulationRunner(dim=3, nx=20, Lz=0.5, incompressible=True)
    r.set_fluid(r.box((0.1, 0.1, 0.1), (0.5, 0.5, 0.4)), velocity=(0.1, 0.0, 0.2))
    r.add_prim("sphere", "obstacle", center=(0.7, 0.3, 0.25), radius=0.1)
    r.add_prim("cylinder", "solid", velocity=(0, 0.5, 0), p0=(0.2, 0.7, 0.1), p1=(0.2, 0.7, 0.4), radius=0.05)
    r.set_wall("bottom", "wall", friction=0.5, span=((0.0, 0.5), (0.0, 0.25)))
    r.set_wall("front", "inlet", velocity=(0, 0, -0.3), span=((0.3, 0.6), (0.2, 0.4)))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s3.json")
        r.save(path)
        r2 = SimulationRunner.load(path)
    assert r.equals(r2) and r2.dim == 3 and r2.shape == r.shape, "aller-retour JSON 3D"
    m = r2.masks()
    assert m["obstacle"].any() and m["solid"].any() and m["vz0"].max() > 0.19, "primitives appliquées"
    old = SimulationRunner.load(os.path.join(_ROOT, "test_cases", "von_karman.json"))
    assert old.dim == 2 and len(old.shape) == 2 and old.m["fluid"].any(), "lecture d'un fichier 2D v4"

    # ---- maillages : STL binaire (cube) et OBJ (sphère) voxelisés, volume à quelques % près
    from Solver import mesh
    with tempfile.TemporaryDirectory() as tmp:
        stl, obj = os.path.join(tmp, "cube.stl"), os.path.join(tmp, "sphere.obj")
        _write_cube_stl(stl, (0.2, 0.2, 0.2), (0.5, 0.5, 0.5))
        _write_sphere_obj(obj, (0.0, 0.0, 0.0), 0.2)
        dx = 1.0 / 60
        vc = mesh.voxelize(mesh.load_mesh(stl), (60, 60, 60), dx).sum() * dx ** 3
        tri = mesh.transform(mesh.load_mesh(obj), translate=(0.5, 0.5, 0.5))
        vs = mesh.voxelize(tri, (60, 60, 60), dx).sum() * dx ** 3
        print(f"[smoke3d] maillages : cube {vc:.4f} m³ (0.0270)  sphère {vs:.4f} m³ ({4 / 3 * np.pi * 0.008:.4f})")
        assert abs(vc - 0.027) / 0.027 < 0.15 and abs(vs - 4 / 3 * np.pi * 0.008) / (4 / 3 * np.pi * 0.008) < 0.15
        r = SimulationRunner(dim=3, nx=30)
        r.add_prim("mesh", "obstacle", path=stl, translate=(0.5, 0.5, 0.5))
        assert r.masks()["obstacle"].sum() > 0, "primitive maillage"

    # ---- solide 3D dans le fluide : un bloc lourd coule, un bloc léger remonte (CG simple, voir ui/smoke.py)
    for rho_s, sign in ((3.0, -1), (0.5, 1)):
        r = SimulationRunner(dim=3, nx=20, Lx=1.0, Ly=1.0, Lz=0.5, incompressible=True, substeps=2, cg_iters=60,
                             solid_rho=rho_s, solid_E=2000.0, multigrid=False)
        r.set_fluid(r.box((0.05, 0.05, 0.05), (0.95, 0.7, 0.45)))
        r.set_solid(r.box((0.35, 0.3, 0.12), (0.65, 0.38, 0.38)))
        s = r.solver()
        y0 = s.solid_positions()[:, 1].mean()
        for _ in range(15):
            s.step()
        y1 = s.solid_positions()[:, 1].mean()
        st = s.stats()
        print(f"[smoke3d] solide rho = {rho_s} : y {y0:.3f} -> {y1:.3f}  D max {st['D_max']:.2f}")
        assert np.isfinite(s.solid_positions()).all() and sign * (y1 - y0) > 0.005, "flottabilité 3D"
        s.release()
    print("[smoke3d] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

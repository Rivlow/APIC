"""Écoulement 3D en conduite pleine à travers une roue de turbine Francis importée d'un STL.

    python ui/examples/turbine_3d.py                     # interface 3D : Espace = lecture continue
    python ui/examples/turbine_3d.py --headless [--png vue.png] [--save]

La roue (francis-turbine-stl1/19.stl, ~650 000 triangles, en mm) est voxelisée comme obstacle (primitive
« maillage », éditable dans le dock Primitives de l'UI). Conduite pleine (pas de surface libre, pas de
gravité), entrée uniforme à gauche, sortie libre à droite, parois glissantes.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from ui.runner import SimulationRunner  # noqa: E402

STL = os.path.join(_ROOT, "francis-turbine-stl1", "19.stl")


def build(nx: int = 80) -> SimulationRunner:
    """Scene: pipe 1.0 x 0.5 x 0.5 m, runner (diameter ~0.35 m, axis along x) in the middle.

    **Inputs**

    - `nx` : int cells along x

    **Outputs**

    - SimulationRunner
    """
    r = SimulationRunner(dim=3, Lx=1.0, Ly=0.5, Lz=0.5, nx=nx, incompressible=True, free_surface=False,
                         gravity=0.0, substeps=2, cg_iters=6, density_iters=3, cfl=0.8, capacity=3_000_000)
    b = r.band
    r.set_fluid(r.box((b, b, b), (1.0 - b, 0.5 - b, 0.5 - b)), velocity=(0.5, 0.0, 0.0))
    r.add_prim("mesh", "obstacle", path=STL, scale=0.0032, rotate=[0.0, 90.0, 0.0], translate=[0.5, 0.25, 0.25])
    r.set_wall("left", "inlet", velocity=(0.5, 0.0, 0.0))
    r.set_wall("right", "outlet")
    return r


if __name__ == "__main__":
    if not os.path.exists(STL):
        raise SystemExit(f"maillage introuvable : {STL}")
    r = build()
    if "--save" in sys.argv:
        r.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "turbine_3d.json"))
    if "--headless" not in sys.argv:
        r.show()                                                 # Play (Espace) : la simulation tourne en continu
    else:
        import numpy as np

        from ui.render3d import Camera
        s = r.solver()
        print(f"grille {s.shape}, {s.n_fluid_init} particules, {int(r.masks()['obstacle'].sum())} cellules d'obstacle")
        for k in range(20):
            s.step()
        st = s.stats()
        print(f"t = {st['t']:.3f} s  div max = {st['div_max']:.1e}  |v| max = "
              f"{np.linalg.norm(s.velocities(), axis=1).max():.2f} m/s  {st['ms']:.0f} ms/image")
        if "--png" in sys.argv:
            from PIL import Image
            s.fluid_mode = 1                                     # |v|
            cam = Camera.fit(r.extent, yaw=35, pitch=25)
            s.render3d(cam, size=(900, 600))                     # calcule |v| par particule...
            s.stats()                                            # ... puis l'échelle de couleur
            img = s.render3d(cam, size=(900, 600), clip=(2, 0.25, True))   # moitié avant masquée : la roue
            Image.fromarray(img).save(sys.argv[sys.argv.index("--png") + 1])

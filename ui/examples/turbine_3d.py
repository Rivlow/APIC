"""Turbine Francis simplifiée en 3D : conduite d'entrée, chambre, roue tournante (STL), tube d'aspiration.

    python ui/examples/turbine_3d.py                     # interface 3D : Espace = lecture continue
    python ui/examples/turbine_3d.py --headless [--frames 60] [--png vue.png] [--save]
Export VTK (ParaView) par défaut dans VTK_DIR, un instant tous les VTK_EVERY pas ; changer avec
[--vtk dossier] [--every N], couper avec --no-vtk.

Dans l'UI : cocher « Coupe » (axe y, ~0.52 m) pour ouvrir le bloc et voir la chambre, la roue et le tourbillon ;
couleur « vitesse |v| ».

Géométrie (primitives, éditables dans le dock « Primitives » de l'UI) : un bloc plein dans lequel on creuse, en eau,
  - une conduite d'entrée à section carrée, horizontale (x), tangente à la chambre : l'eau y entre en tourbillon ;
  - une chambre cylindrique d'axe vertical (y) autour de la roue (bâche spirale simplifiée) ;
  - le tube d'aspiration : cylindre vers le bas sous la roue, puis section carrée jusqu'à la face inférieure.
La roue (francis-turbine-stl1/19.stl, ~650 000 triangles, en mm) est un rotor : voxelisée une fois, elle tourne à
vitesse imposée (paramètres rpm et axe de la primitive) ; ses cellules imposent la vitesse omega × r au fluide. La
barre d'état affiche le couple et la puissance transmis par l'eau. Conduite pleine (pas de surface libre, pas de
gravité), entrée à vitesse uniforme à gauche, sortie libre (p = 0) en bas. Les tronçons qui touchent les parois
sont carrés et calés sur les cellules : l'entrée / la sortie (rectangles) couvrent exactement leur section.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from ui.runner import SimulationRunner  # noqa: E402

STL = os.path.join(_ROOT, "francis-turbine-stl1", "19.stl")
VTK_DIR = r"C:\Users\lucas\Dev\Fun\outputs\pompe_3D"      # export ParaView (UI et --headless)
T_STOP = None                                             # entrée coupée à t = T_STOP s (None : débit continu)
T_RAMP = 0.5                                             # décroissance de la vitesse d'entrée sur T_RAMP s avant
DRY = 0                                               # démarrage à sec : conduites vides, l'entrée les remplit
VTK_EVERY = 10                                         # un instant exporté tous les 10 pas


def build(nx: int = 100, rpm: float = 60.0, u_in: float = 1.0, t_stop: float | None = T_STOP,
          t_ramp: float = T_RAMP, dry: bool = DRY) -> SimulationRunner:
    """Francis-like scene in a 1.0 x 0.9 x 1.0 m box.

    **Inputs**

    - `nx` : int cells along x (dx = 1 / nx m)
    - `rpm` : float runner speed (tr/min, sign = direction about +y)
    - `u_in` : float inlet velocity (m/s)
    - `t_stop` : float | None   time (s) at which the inflow is fully stopped ; None : continuous inflow
    - `t_ramp` : float          ramp-down duration (s) before t_stop
    - `dry` : bool   True : empty ducts at t = 0 (free surface, gravity 9.81 m/s² down y) ; the inlet fills them
      and they drain through the bottom outlet once the inflow stops. False : ducts full of water, no gravity

    **Outputs**

    - SimulationRunner
    """
    # à sec : les cellules vides sont de l'air (p = 0) grâce à la surface libre, et la gravité fait tomber l'eau ;
    # en charge : conduites pleines, toute cellule non solide est de l'eau, pas de gravité
    r = SimulationRunner(dim=3, Lx=1.0, Ly=0.9, Lz=1.0, nx=nx, incompressible=True, free_surface=dry,
                         gravity=9.81 if dry else 0.0, fluid_rho=1000.0, substeps=2, cg_iters=10, density_iters=3,
                         cfl=0.8, capacity=40_000_000)
    dx = r.dx
    snap = lambda v: round(v / dx) * dx                  # bornes sur les frontières de cellules
    c = (0.5, 0.5, 0.5)                                  # centre de la roue
    # Carter calé sur la roue (mesurée dans 19.stl une fois mise à l'échelle, en m) : ceinture (anneau du bas)
    # y 0.40-0.48, r ext 0.158 ; entrée des aubes = surface extérieure entre le haut de la ceinture (y 0.48) et la
    # couronne (y 0.585, r 0.124) ; couronne pleine y 0.585-0.60 ; alésage d'axe r 0.03 ouvert. L'eau ne peut aller
    # de la chambre à l'aspirateur QU'À TRAVERS les aubes : chambre limitée à la hauteur d'entrée, paroi fixe autour
    # de la ceinture (pas de passage dessous), couvercle sur la couronne (pas de passage dessus), arbre dans l'alésage.
    Y_BAND, Y_IN0, Y_IN1, R_BAND = 0.40, 0.48, 0.585, 0.158
    # bloc plein, puis conduites creusées : « clear » les laisse vides, « fluid » les remplit d'eau
    hole = "clear" if dry else "fluid"
    r.add_prim("box", "obstacle", lo=(0.0, 0.0, 0.0), hi=r.extent)
    r.add_prim("cylinder", hole, p0=(c[0], Y_IN0, c[2]), p1=(c[0], Y_IN1, c[2]), radius=0.30)   # chambre (distributeur)
    r.add_prim("cylinder", hole, p0=(c[0], Y_BAND, c[2]), p1=(c[0], Y_IN0, c[2]), radius=R_BAND)  # logement ceinture
    duct_lo = (0.0, snap(Y_IN0), snap(0.66))                                                   # entrée tangente
    duct_hi = (snap(0.52), snap(Y_IN1), snap(0.80))
    r.add_prim("box", hole, lo=duct_lo, hi=duct_hi)
    r.add_prim("cylinder", hole, p0=(c[0], 0.10, c[2]), p1=(c[0], Y_BAND + 0.01, c[2]), radius=0.13)  # aspirateur
    r.add_prim("cylinder", "obstacle", p0=(c[0], 0.43, c[2]), p1=(c[0], r.extent[1], c[2]), radius=0.03)  # arbre
    out_lo = (snap(0.37), 0.0, snap(0.37))
    out_hi = (snap(0.63), snap(0.13), snap(0.63))
    r.add_prim("box", hole, lo=out_lo, hi=out_hi)                                             # sortie carrée
    # roue : axe du fichier (z) remis à la verticale (y), diamètre ~0.32 m, tourne autour de +y
    r.add_prim("mesh", "rotor", path=STL, scale=0.0029, rotate=[-90.0, 0.0, 0.0], translate=list(c),
               rpm=rpm, axis=[0.0, 1.0, 0.0])
    # conditions aux limites : exactement la section des conduites carrées
    # entrée coupée à t_stop : la vitesse décroît linéairement jusqu'à 0 sur t_ramp (pas de coup de bélier dans la
    # conduite pleine) ; l'émission de particules, proportionnelle à la vitesse, s'arrête avec elle
    vx = u_in if t_stop is None else f"{u_in}*min(max(({t_stop} - t)/{t_ramp}, 0), 1)"
    r.set_wall("left", "inlet", velocity=(vx, 0.0, 0.0), span=((duct_lo[1], duct_hi[1]), (duct_lo[2], duct_hi[2])))
    r.set_wall("bottom", "outlet", span=((out_lo[0], out_hi[0]), (out_lo[2], out_hi[2])))
    return r


if __name__ == "__main__":
    if not os.path.exists(STL):
        raise SystemExit(f"maillage introuvable : {STL}")
    r = build()
    if "--save" in sys.argv:
        r.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "turbine_3d.json"))
    vtk_dir = sys.argv[sys.argv.index("--vtk") + 1] if "--vtk" in sys.argv else VTK_DIR
    vtk_dir = None if "--no-vtk" in sys.argv else vtk_dir
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else VTK_EVERY
    if "--headless" not in sys.argv:
        r.show(vtk_dir=vtk_dir, vtk_every=every)                 # Play (Espace) : export pendant la lecture
    else:
        import numpy as np

        from ui.render3d import Camera
        s = r.solver()
        m = r.masks()
        print(f"grille {s.shape}, {s.n_fluid_init} particules, {int(m['fluid'].sum())} cellules d'eau, "
              f"roue {int(r.rotor()['mask'].sum())} cellules")
        ex = None
        if vtk_dir:                                              # ParaView : ouvrir fluid.pvd, rotor.pvd, mesh_*.vtp
            from ui.export_vtk import VTKExporter
            ex = VTKExporter(vtk_dir, r, s)
            ex.write()
        for k in range(int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 20):
            s.step()
            if ex is not None and (k + 1) % every == 0:
                ex.write()
        st = s.stats()
        print(f"t = {st['t']:.3f} s  div max = {st['div_max']:.1e}  |v| max = "
              f"{np.linalg.norm(s.velocities(), axis=1).max():.2f} m/s  {st['ms']:.0f} ms/image  roue "
              f"{st['rotor_rpm']:.0f} tr/min, angle {st['rotor_angle']:.0f}°, couple {st['rotor_torque']:.3g} N·m, "
              f"puissance {st['rotor_power']:.3g} W")
        if "--png" in sys.argv:
            from PIL import Image
            s.fluid_mode = 1                                     # |v|
            cam = Camera.fit(r.extent, yaw=20, pitch=55)
            clip = (1, 0.52, True)                               # coupe horizontale à mi-hauteur de la roue
            s.render3d(cam, size=(900, 700), clip=clip)          # calcule |v| par particule...
            s.stats()                                            # ... puis l'échelle de couleur
            img = s.render3d(cam, size=(900, 700), clip=clip)
            Image.fromarray(img).save(sys.argv[sys.argv.index("--png") + 1])

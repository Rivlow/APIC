import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ui.runner import SimulationRunner

test_case = "hydraulic_jump_3d"

# Ressaut hydraulique en tranche 3D : le cas 2D (test_cases/hydraulic_jump.py) extrudé sur Lz, parois avant /
# arrière glissantes. y vertical, z profondeur. Mêmes résultats moyens que le 2D (ressaut ~11.55 m), plus la
# turbulence 3D du rouleau si la tranche est assez épaisse.
# Params (SI : m, s, kg, Pa)
g = 9.81
rho = 1000.0
x0, x1 = 7.0, 15.25                                                  # tronçon simulé (m, coordonnées de la thèse)
Lx = x1 - x0
Lz = 0.1                                                             # épaisseur de la tranche (dont 2 x 3 cellules de bande)
q = 0.18                                                             # débit linéique (m²/s)
h1 = 0.4224                                                          # hauteur amont sous-critique
h2 = 0.33                                                            # hauteur aval (Fr = 0.3)

nx = 825                                                             # dx = 1 cm ; ny, nz déduits
h_in = round(h1 * nx / Lx) * Lx / nx                                 # entrée calée sur les cellules
U1 = q / h_in
U2 = q / h2                                                          # sortie : vitesse imposée

sim = SimulationRunner(dim=3, Lx=Lx, Ly=1.0, Lz=Lz, nx=nx, gravity=g, fluid_rho=rho, incompressible=True,
                       free_surface=True, substeps=4, cg_iters=8, multigrid=True, obstacle_friction=0.0,
                       capacity=4_000_000, cfl=0.9, density_iters=5)

bathy = lambda x: np.maximum(0.2 - 0.05 * (x - 10.0) ** 2, 0.0)
X, Z, _ = sim.centers()                                             # centres des cellules (m) ; Z = hauteur ici
b = sim.band
x_m, z_m = X - b + x0, Z - b                                         # coordonnées de la thèse ; z_m = 0 sur le fond
zb = bathy(x_m)

sim.set_obstacle(z_m < zb)                                           # rect / masques : extrudés sur toute la profondeur
sim.set_fluid((x_m < 10) & (z_m >= zb) & (z_m < h1), velocity=(U1, 0.0, 0.0))
sim.set_fluid((x_m >= 10) & (z_m >= zb) & (z_m < h2), velocity=(U2, 0.0, 0.0))

sim.set_wall("left", "inlet", velocity=(U1, 0.0, 0.0), span=((b, b + h_in), (0.0, Lz)))   # rectangle (y, z)
sim.set_wall("bottom", "wall", friction=0.0)
sim.set_wall("right", "inlet", velocity=(U2, 0.0, 0.0))             # vitesse sortante : sortie à vitesse imposée

if __name__ == "__main__":
    sim.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{test_case}.json"))
    sim.show(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{test_case}.json"))

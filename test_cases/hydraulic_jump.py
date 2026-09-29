import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ui.runner import SimulationRunner

test_case = "hydraulic_jump"

# Ressaut hydraulique (Archambeau ; configuration SPlisHSPlasH) : x in [7, 15.25] m, obstacle zb = 0.2 - 0.05 (x - 10)^2
# Params (SI : m, s, kg, Pa)
g = 9.81
rho = 1000.0
x0, x1 = 7.0, 15.25                                                  # tronçon simulé (m, coordonnées de la thèse)
Lx = x1 - x0
Lz = 1.0
q = 0.18                                                             # débit linéique (m²/s)
h1 = 0.4224                                                          # hauteur amont sous-critique (H = 0.4234 m, lit plat)
h2 = 0.33                                                            # hauteur aval (thèse : Fr = 0.3, H = 0.3452 m)


nx = 825                                                             # dx = 1 cm ; nz = Lz / dx déduit
h_in = round(h1 * nx / Lx) * Lx / nx                                 # entrée calée sur les cellules (sinon débit faux)
U1 = q / h_in                                                        # vitesse uniforme sur la hauteur d'eau amont : débit q
U2 = q / h2                                                          # sortie : vitesse imposée (champ d'animation SPH)

sim = SimulationRunner(Lx=Lx, Ly=Lz, nx=nx, gravity=g, fluid_rho=rho, incompressible=True, free_surface=True,
                       substeps=4, cg_iters=8, multigrid=True, obstacle_friction=0.0,
                       capacity=1_000_000,
                       cfl=0.9,
                       density_iters=5)

bathy = lambda x: np.maximum(0.2 - 0.05 * (x - 10.0) ** 2, 0.0)
X, Z = sim.centers()                                                # centres des cellules (m)
b = sim.band                                                        # épaisseur de la bande de paroi (m) : le fond
x_m, z_m = X - b + x0, Z - b                                        # coordonnées de la thèse ; z = 0 sur le fond
zb = bathy(x_m)

# Initial conditions : état proche du stationnaire (niveaux amont / aval, vitesse q / h), pas d'eau au repos
sim.set_obstacle(z_m < zb)
sim.set_fluid((x_m < 10) & (z_m >= zb) & (z_m < h1), velocity=(U1, 0.0))
sim.set_fluid((x_m >= 10) & (z_m >= zb) & (z_m < h2), velocity=(U2, 0.0))

# Boundary conditions
sim.set_wall("left", "inlet", velocity=(U1, 0.0), span=(b, b + h_in))
sim.set_wall("bottom", "wall", friction=0.0)
# Sortie à vitesse imposée (comme SPlisHSPlasH) : débit sortant U2 h, la hauteur aval se règle seule sur q / U2 = h2.
# (Alternative : sortie à pression hydrostatique, set_constants(H=b + h2) + set_wall("right", "outlet",
# pressure="rho*g*max(H - y, 0)") ; elle impose le niveau et réfléchit les ondes : ressaut plus lent à se fixer.)
sim.set_wall("right", "inlet", velocity=(U2, 0.0))

if __name__ == "__main__":
    sim.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{test_case}.json"))
    sim.show(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{test_case}.json"))

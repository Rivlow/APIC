# test_cases/hydraulic_jump.py -- Ressaut hydraulique 2D dans le plan XZ (Z vertical), fluide incompressible.
#
# Débit imposé en entrée (mur gauche, sur la seule hauteur d'eau amont h1), pression imposée cellule par cellule en
# sortie (mur droit : profil hydrostatique de la hauteur aval h2 ; le gradient de pression pilote l'écoulement),
# lit défini par une fonction z_b(x) (obstacle sous le lit).
#
# Grille rectangulaire nx × ny (cellules carrées) : le canal fait Lx × Lz mètres. Mise à l'échelle : 1 unité domaine
# = L = Lx mètres (dx = 1 / nx). Similitude de Froude : longueurs et vitesses divisées par L, gravité divisée par L,
# temps inchangé (Fr = U / sqrt(g h) identique). La bande de paroi du bas (sim.band) est le fond : z = 0 m y
# correspond. Frottement : fond (mur du bas) et bosse (obstacle), beta = 0 glissant ... 1 adhérent.
#
# Lancer :  python test_cases/hydraulic_jump.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from ui.runner import SimulationRunner  # noqa: E402

test_case = "hydraulic_jump"

# ---------------------------------------------------------------- données physiques (SI)
g = 9.81
rho = 1000.0
Lx = 20.0                              # longueur du canal (m)
Lz = 1.25                              # hauteur du domaine (m) : eau (< 0,6 m) + air au-dessus
q = 0.3                                # débit par unité de largeur (m²/s)
h2 = 0.333                             # hauteur d'eau imposée en sortie (m)

Fr2 = q / (h2 * np.sqrt(g * h2))
h1 = 0.5 * h2 * (np.sqrt(1.0 + 8.0 * Fr2**2) - 1.0)
U1 = q / h1
print(f"h1 = {h1:.3f} m, U1 = {U1:.2f} m/s, Fr1 = {U1 / np.sqrt(g * h1):.2f}   |   h2 = {h2:.3f} m, Fr2 = {Fr2:.2f}")


def bed(x):
    return np.maximum(0.2 - 0.05 * (x - 10.0) ** 2, 0.0)


x_jump0 = 3.0                          # position initiale du ressaut (m) : amont h1, aval h2
beta_bed = 0.0                         # frottement du fond plat (mur du bas) : 0 glissant, 1 adhérent
beta_bump = 0.0                        # frottement de la bosse (obstacle)

# ---------------------------------------------------------------- simulation (unités domaine)
N = 512                               # nx : dx = Lx / N ~ 2 cm, h1 ~ 6 cellules
NZ = int(round(Lz / Lx * N))           # ny : 64 cellules sur la hauteur
L = Lx
sim = SimulationRunner(nx=N, ny=NZ, gravity=g / L, fluid_rho=rho, incompressible=True, free_surface=True,
                       substeps=4, cg_iters=100, obstacle_friction=beta_bump,
                       capacity=400_000,
                       cfl=0.9,             # APIC incompressible : pas d'advection ~1 cellule (2x moins de sous-pas qu'à 0,4)
                       density_iters=5)     # correction de densité : l'erreur se rattrape d'un pas à l'autre
X, Z = sim.centers()
b = sim.band
x_m, z_m = X * L, (Z - b) * L
zb = bed(x_m)

sim.set_obstacle(z_m < zb)
sim.set_fluid((x_m < x_jump0) & (z_m >= zb) & (z_m < zb + h1), velocity=(U1 / L, 0.0))
sim.set_fluid((x_m >= x_jump0) & (z_m >= zb) & (z_m < zb + h2), velocity=(q / h2 / L, 0.0))

zb_in, zb_out = float(bed(np.array(0.0))), float(bed(np.array(Lx)))
sim.set_wall("left", "inlet", velocity=(U1 / L, 0.0), span=(b + zb_in / L, b + (zb_in + h1) / L))
sim.set_wall("bottom", "wall", friction=beta_bed)


P_out = lambda z: rho * g * max(zb_out + h2 - z, 0.0)
sim.set_wall("right", "outlet")                                    # p = 0 par défaut (au-dessus de l'eau)
for k in range(NZ):                                                # cellules le long du mur droit
    p_k = P_out(((k + 0.5) / N - b) * L)
    if p_k > 0.0:
        sim.set_wall("right", "outlet", pressure=p_k / L**2, span=(k / N, (k + 1) / N))

if __name__ == "__main__":
    s = sim.run(100, callback=lambda s, k: k % 20 == 0 and print(k, s.stats()))
    s.release()
    sim.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{test_case}.json"))
    sim.show(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{test_case}.json"))

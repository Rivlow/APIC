"""Allée de von Kármán : conduit horizontal rempli d'eau, cylindre au milieu, entrée à gauche et sortie
à droite à vitesse purement horizontale, sans gravité. Fluide incompressible (grille MAC + projection de
pression par gradient conjugué sur le GPU).

    python test.py              # ouvre l'interface (Espace pour lancer, molette pour zoomer)
    python test.py --no-show    # calcule seulement 100 images et affiche des stats
"""
import sys

sys.path.insert(0, r"C:\Users\lucas\Dev\Fun\APIC")
from ui.runner import SimulationRunner  # noqa: E402

U = 0.5                                   # vitesse d'entrée (m/s) ; c = sqrt(E/rho) = 20 -> Mach 0.1
N = 256
r = SimulationRunner(n=N, gravity=0.0, fluid_rho=1.0, capacity=int(1.3 * 4 * N * N),
                     incompressible=True, free_surface=False, substeps=2,   # conduit plein : pas d'air, pression résolue partout
                     cg_iters=150)
X, Y = r.centers()

r.set_fluid(r.rect(r.band, r.band, 1 - r.band, 1 - r.band), velocity=(U, 0.0))   # conduit plein (hors bande de paroi)
r.set_obstacle(r.circle(0.30, 0.50, 0.05))                          # cylindre au milieu du conduit
r.set_wall("left", "inlet", velocity=(U, 0.0))                       # tout le mur gauche : entrée
r.set_wall("right", "outlet")                                        # tout le mur droit : sortie

if "--no-show" in sys.argv:
    s = r.run(100, callback=lambda s, k: k % 20 == 0 and print(k, s.stats()))
    s.release()
else:
    r.save("von_karman.json")
    r.show("von_karman.json")

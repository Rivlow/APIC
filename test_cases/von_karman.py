import sys

sys.path.insert(0, r"C:\Users\lucas\Dev\Fun\APIC")
from ui.runner import SimulationRunner  # noqa: E402

U = 0.5                                   # vitesse d'entrée (m/s) ; c = sqrt(E/rho) = 20 -> Mach 0.1
N = 256
sim = SimulationRunner(n=N, gravity=0.0, fluid_rho=1.0, capacity=int(1.3 * 4 * N * N),
                       incompressible=True, free_surface=False, substeps=2,   # conduit plein : pas d'air, pression résolue partout
                       cg_iters=20)
X, Y = sim.centers()

sim.set_fluid(sim.rect(sim.band, sim.band, 1 - sim.band, 1 - sim.band), velocity=(U, 0.0))   # conduit plein (hors bande de paroi)
sim.set_obstacle(sim.circle(0.30, 0.50, 0.05))
sim.set_wall("left", "inlet", velocity=(U, 0.0))
sim.set_wall("right", "outlet")

s = sim.run(100, callback=lambda s, k: k % 20 == 0 and print(k, s.stats()))
s.release()
sim.save("test_cases/von_karman.json")
sim.show("test_cases/von_karman.json")



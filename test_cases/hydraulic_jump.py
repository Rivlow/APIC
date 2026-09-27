import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ui.runner import SimulationRunner

test_case = "hydraulic_jump"

# Params
g = 9.81
rho = 1000.0
Lx = 20.0
Lz = 1.25
q = 0.18
h2 = 0.333


U1 = q / 1.


nx = 512
ny = int(round(Lz / Lx * nx))

sim = SimulationRunner(nx=nx, ny=ny, gravity=g, fluid_rho=rho, incompressible=True, free_surface=True,
                       substeps=4, cg_iters=100, obstacle_friction=0.0,
                       capacity=400_000,
                       cfl=0.9,
                       density_iters=5)

bathy = lambda x: np.maximum(0.2 - 0.05 * (x - 10.0) ** 2, 0.0)
X, Z = sim.centers()
b = sim.band
x_m, z_m = X * Lx, (Z - b) * Lx
zb = bathy(x_m)

# Initial conditions
sim.set_obstacle(z_m < zb)
sim.set_fluid((x_m < 8) & (z_m >= zb), velocity=(0.0, 0.0))
sim.set_fluid((x_m >= 12) & (z_m >= zb) , velocity=(0.0, 0.0))

# Boundary conditions
zb_in, zb_out = float(bathy(np.array(0.0))), float(bathy(np.array(Lx)))
sim.set_wall("left", "inlet", velocity=(U1 / Lx, 0.0), span=(b + zb_in / Lx, b + (2*zb_in) / Lx))
sim.set_wall("bottom", "wall", friction=0.0)


P_out = lambda z: rho * g * max(zb_out + h2 - z, 0.0)
sim.set_wall("right", "outlet")                                    # p = 0 par défaut (au-dessus de l'eau)
for k in range(ny):                                                # cellules le long du mur droit
    p_k = P_out(((k + 0.5) / nx - b) * Lx)
    if p_k > 0.0:
        sim.set_wall("right", "outlet", pressure=p_k / Lx**2, span=(k / nx, (k + 1) / nx))

if __name__ == "__main__":
    s = sim.run(100, callback=lambda s, k: k % 20 == 0 and print(k, s.stats()))
    s.release()
    sim.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{test_case}.json"))
    sim.show(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{test_case}.json"))

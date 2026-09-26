import sys; sys.path.insert(0, r"C:\Users\lucas\Dev\Fun\APIC")
from ui.runner import SimulationRunner

r = SimulationRunner(n=128, gravity=9.81, solid_E=20000.0)
X, Y = r.centers()
r.set_fluid(Y < 0.20)
r.set_solid(r.rect(0.55, 0.20, 0.58, 0.55))
r.set_obstacle(r.circle(0.35, 0.28, 0.05))
r.set_inlet((X < 0.06) & (Y > 0.25) & (Y < 0.45), velocity=(3.0, 0.0))
r.set_outlet(X > 0.95)
s = r.run(150, callback=lambda s, k: print(k, s.stats()["n_fluid"]))
r.save("cas.json")
r.show()

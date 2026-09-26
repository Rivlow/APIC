"""Test de fumée sans fenêtre : python -m ui.smoke (depuis la racine du dépôt, ou n'importe où)."""
import math
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    import numpy as np

    from ui.runner import SimulationRunner

    # ---- scène de référence (main_fsi) : calcul, stats, rendu
    r = SimulationRunner.demo()
    s = r.solver()
    s.step()                                   # première image : compilation JIT
    s.step(10 * r.p["substeps"])
    st = s.stats()
    print(f"[smoke] arch={s.arch} 10 images : {st['ms']:.0f} ms  stats={st}")
    assert st["n_fluid"] > 0 and st["n_solid"] > 0, "particules absentes"
    assert st["t"] > 0 and math.isfinite(st["D_max"]), "état incohérent"
    img = s.render(0.0, 0.0, 1.0, grid=True, tint=True)
    assert img.shape == (r.p["res"], r.p["res"], 3) and img.dtype == np.uint8 and (img != img[0, 0]).any(), "rendu vide"
    img2 = s.render(0.2, 0.3, 4.0, grid=True, tint=False)
    assert (img2 != img).any(), "le zoom ne change rien"
    s.release()

    # ---- paramètres à chaud + JSON aller-retour
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "demo.json")
        r.save(path)
        r2 = SimulationRunner.load(path)
        assert r.equals(r2), "aller-retour JSON"
    r2.resize(64)
    r2.p["gravity"] = 2.0
    s = r2.solver()
    s.step()
    s.set_params({"gravity": 20.0, "fluid_E": 800.0})
    s.step()
    assert s.stats()["t"] > 0
    s.release()

    # ---- entrée / sortie sur masques numpy
    r = SimulationRunner(n=96, substeps=10, capacity=60000)
    X, Y = r.centers()
    r.set_fluid(Y < 0.15)
    r.set_solid(r.rect(0.60, 0.15, 0.63, 0.40))
    r.set_obstacle(r.rect(0.55, 0.0, 0.68, 0.15))
    r.set_inlet((X < 0.06) & (Y > 0.3) & (Y < 0.4), velocity=(3.0, 0.0))
    r.set_outlet(X > 0.95)
    s = r.solver()
    s.step()
    n0 = s.stats()["n_fluid"]
    for _ in range(40):
        s.step()
    st = s.stats()
    print(f"[smoke] entrée/sortie : fluide {n0} -> {st['n_fluid']} / {st['capacity']}, t = {st['t']:.3f}")
    assert n0 < st["n_fluid"] <= st["capacity"], "l'entrée n'émet pas"
    assert st["n_fluid"] - n0 < 40 * 10 * 4 * 30, "émission non bornée"
    x = s.positions()
    assert len(x) == st["n_fluid"] and x.min() >= 0.0, "positions incohérentes"
    assert (x[:, 0] < 0.97).all(), "la sortie ne détruit pas"
    s.release()
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

    # ---- entrée / sortie sur les parois ; un obstacle collé au mur droit porte la sortie sur sa face gauche
    r = SimulationRunner(n=96, substeps=10, capacity=60000)
    X, Y = r.centers()
    r.set_fluid(Y < 0.15)
    r.set_solid(r.rect(0.60, 0.15, 0.63, 0.40))
    r.set_obstacle(r.rect(0.55, 0.0, 0.68, 0.15))
    r.set_obstacle(r.rect(0.90, 0.5, 1.0, 0.6))                 # bloc collé au mur droit
    r.set_wall("left", "inlet", velocity=(3.0, 0.0), span=(0.3, 0.4))
    r.set_wall("right", "outlet")
    wt, wv, wd, wl = r.wall_table()
    n, b = r.n, r.p["bound"]
    assert (wt[0, int(0.3 * n):int(0.4 * n)] == 1).all() and (wv[0, int(0.35 * n)] == [3.0, 0.0]).all(), "entrée"
    assert (wt[1, b:n - b] == 2).all(), "sortie"
    assert (wd[1, int(0.5 * n) + 1:int(0.6 * n) - 1] == 10).all() and (wd[1, b:int(0.5 * n) - 1] == b).all(), \
        "sortie portée par la face de l'obstacle (10 cellules depuis le bord)"
    assert (wt[:, :b] == 0).all() and (wt[:, n - b:] == 0).all(), "coins"
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
    assert (x[:, 0] <= 1.0 - r.band + 1e-6).all(), "positions hors bande"
    assert (x[:, 0] > 0.9).sum() < len(x), "tout le fluide au mur droit"
    s.release()

    # ---- fichier v2 (bandes de cellules d'entrée / sortie) migré en segments de paroi
    import base64
    n = 64
    inlet = np.zeros((n, n), bool)
    inlet[3:6, 20:30] = True
    outlet = np.zeros((n, n), bool)
    outlet[n - 6:, :] = True
    ivx = np.where(inlet, 2.5, 0.0).astype(np.float32)

    def enc(a):
        raw = np.packbits(a.ravel()) if a.dtype == bool else a.ravel()
        return {"dtype": str(a.dtype), "shape": [n, n], "data": base64.b64encode(raw.tobytes()).decode()}

    r2 = SimulationRunner.from_dict({"version": 2, "params": {"n": n},
                                     "matrices": {"inlet": enc(inlet), "outlet": enc(outlet), "inlet_vx": enc(ivx)}})
    kinds = sorted((w["side"], w["type"]) for w in r2.walls)
    assert kinds == [("left", "inlet"), ("right", "outlet")], f"migration v2 : {r2.walls}"
    w_in = next(w for w in r2.walls if w["type"] == "inlet")
    assert abs(w_in["span"][0] - 20 / n) < 1e-9 and abs(w_in["span"][1] - 30 / n) < 1e-9 and w_in["velocity"] == [2.5, 0.0]

    # ---- incompressible : rupture de barrage, la divergence doit être ~0 après projection
    r = SimulationRunner(n=64, incompressible=True, substeps=2, cg_iters=200)
    r.set_fluid(r.rect(0.05, 0.05, 0.4, 0.6))
    r.set_obstacle(r.circle(0.7, 0.2, 0.08))
    s = r.solver()
    for _ in range(15):
        s.step()
    st = s.stats()
    x = s.positions()
    print(f"[smoke] incompressible : t = {st['t']:.3f}  dt = {st['dt']:.2e}  cg = {st['cg_iters']} it  "
          f"div max = {st['div_max']:.2e}  x moyen = {x[:, 0].mean():.3f}")
    assert st["div_max"] < 1e-2, "projection non convergée"
    assert x[:, 0].mean() > 0.23, "le barrage ne s'effondre pas"
    assert np.isfinite(x).all() and x.max() <= 1.0
    s.release()

    # ---- incompressible + solide : poutre légère immergée, elle doit remonter (Archimède)
    r = SimulationRunner(n=64, incompressible=True, substeps=2, cg_iters=150, solid_rho=0.5, solid_E=2000.0)
    r.set_fluid(r.rect(0.05, 0.05, 0.95, 0.7))
    r.set_solid(r.rect(0.35, 0.25, 0.65, 0.32))
    s = r.solver()
    y0 = s.solid_positions()[:, 1].mean()
    for _ in range(25):
        s.step()
    st = s.stats()
    y1 = s.solid_positions()[:, 1].mean()
    print(f"[smoke] fluide-structure : t = {st['t']:.3f}  poutre y {y0:.3f} -> {y1:.3f}  D max {st['D_max']:.2f}  "
          f"div max = {st['div_max']:.2e}")
    assert np.isfinite(s.solid_positions()).all() and y1 > y0 + 0.01, "la poutre légère ne remonte pas"
    s.release()
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

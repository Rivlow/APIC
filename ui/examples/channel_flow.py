"""Exemple : canal avec entrée, sortie, obstacle et lame solide, piloté depuis un script externe.

    python ui/examples/channel_flow.py            # calcule 150 images et affiche des stats
    python ui/examples/channel_flow.py --show     # puis ouvre l'interface Qt
    python ui/examples/channel_flow.py --save     # écrit channel_flow.json à côté du script
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.runner import SimulationRunner  # noqa: E402


def main() -> None:
    r = SimulationRunner(n=128, gravity=9.81, substeps=20, solid_E=20000.0, epsf=0.5)
    X, Y = r.centers()

    r.set_fluid(Y < 0.20)                                            # bassin initial
    r.set_solid(r.rect(0.55, 0.20, 0.58, 0.55))                      # lame encastrée dans le socle
    r.set_obstacle(r.circle(0.35, 0.28, 0.05))                       # rocher
    r.set_obstacle(r.rect(0.50, 0.0, 0.62, 0.20))                    # socle
    r.set_inlet((X < 0.06) & (Y > 0.25) & (Y < 0.45), velocity=(3.0, 0.0))   # jet entrant à gauche
    r.set_outlet(X > 0.95)                                           # tout ce qui touche le bord droit disparaît

    def report(s, k):
        if k % 25 == 0:
            st = s.stats()
            print(f"image {k:4d}  t = {st['t']:.3f} s  fluide {st['n_fluid']:6d} / {st['capacity']}  "
                  f"D max {st['D_max']:.2f}  {st['ms']:.1f} ms")

    s = r.run(150, callback=report)
    x = s.positions()
    print(f"particules vivantes : {len(x)}   x moyen = {x[:, 0].mean():.3f}   y moyen = {x[:, 1].mean():.3f}")
    s.release()

    if "--save" in sys.argv:
        r.save(os.path.join(os.path.dirname(__file__), "channel_flow.json"))
    if "--show" in sys.argv:
        r.show()


if __name__ == "__main__":
    main()

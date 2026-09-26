# MpM/beton_section.py -- Référence "résistance des matériaux" à comparer au calcul MPM
#
# Analyse de section à la manière de l'article (2.1) : pour une courbure phi donnée, on cherche la
# déformation au centre de gravité qui annule l'effort normal N, puis on calcule le moment M(phi).
# On intègre ensuite les courbures le long de la poutre (flexion 4 points) pour obtenir la flèche.
#
# Lancer :  python MpM/beton_section.py [beton_courbe.csv]   (trace la comparaison si le csv existe)

import sys
import numpy as np

# mêmes données que MpM/beton.py
b, h = 0.2, 0.2
E_c, fc, fcr, eps_cu = 30.4e9, 47.1e6, 3.5e6, 0.004
E_s, fy = 230e9, 309e6
rebars = [(0.019, 4 * 113.1e-6), (0.072, 2 * 56.5e-6), (0.128, 2 * 56.5e-6), (0.181, 4 * 113.1e-6)]
L, a = 2.0, 0.6

ny = 400
y = (np.arange(ny) + 0.5) * h / ny                # fibres de béton, y mesuré depuis la base
dy = h / ny
yc = h / 2


def sigma_concrete(eps):
    """Loi du béton de l'article : Desayi en compression (nul après eps_cu), raidissement en traction."""
    eps = np.asarray(eps, dtype=float)
    s = np.zeros_like(eps)
    ecr = fcr / E_c
    t = eps > 0
    lin = t & (eps <= ecr)
    s[lin] = E_c * eps[lin]
    soft = t & (eps > ecr)
    s[soft] = fcr / (1.0 + np.sqrt(500.0 * eps[soft]))
    eu = 1.8 * fc / E_c
    c = (eps < 0) & (-eps <= eps_cu)
    x = -eps[c]
    s[c] = -2.0 * fc * x / (eu * (1.0 + (x / eu) ** 2))
    return s


def sigma_steel(eps):
    return np.clip(E_s * eps, -fy, fy)


def forces(eps0, phi):
    """Effort normal N et moment M (autour du centre de gravité) pour la déformation eps0 et la courbure phi."""
    e = eps0 + phi * (y - yc)                       # phi > 0 : fibre inférieure tendue
    sc = sigma_concrete(e)
    N = np.sum(sc) * b * dy
    M = np.sum(sc * (y - yc)) * b * dy
    for d, A in rebars:
        es = eps0 + phi * (d - yc)
        ss = sigma_steel(es)
        sc_lost = sigma_concrete(np.array([es]))[0]  # le béton déplacé par la barre
        N += (ss - sc_lost) * A
        M += (ss - sc_lost) * A * (d - yc)
    return N, M


def moment_curvature(phis):
    Ms = []
    e0 = 0.0
    for phi in phis:
        lo, hi = -0.02, 0.02
        for _ in range(60):                          # dichotomie sur eps0 : N(eps0) croissant
            mid = 0.5 * (lo + hi)
            if forces(mid, phi)[0] > 0:
                hi = mid
            else:
                lo = mid
        e0 = 0.5 * (lo + hi)
        Ms.append(forces(e0, phi)[1])
    return np.array(Ms)


def load_deflection():
    phis = np.linspace(1e-6, 0.06, 600)
    Ms = moment_curvature(phis)
    imax = int(np.argmax(Ms))                        # on reste avant le pic du moment
    phis, Ms = phis[:imax + 1], Ms[:imax + 1]
    # courbure le long de la demi-poutre (symétrie), moment de flexion 4 points M(x) = P/2 * min(x, a)
    xs = np.linspace(0.0, L / 2, 201)
    shape = np.minimum(xs, a) / 2.0                  # M(x) / P
    m_unit = xs / 2.0                                # moment dû à une charge unitaire à mi-portée
    Ps = np.linspace(0.0, Ms[-1] / (a / 2.0), 300)
    out = []
    for P in Ps:
        kappa = np.interp(P * shape, Ms, phis)       # courbure en chaque x
        delta = 2.0 * np.trapezoid(kappa * m_unit, xs)   # théorème de la charge unitaire
        out.append((delta, P))
    out = np.array(out)
    return out[:, 0] * 1000.0, out[:, 1] / 1000.0, phis, Ms


if __name__ == "__main__":
    d, P, phis, Ms = load_deflection()
    # rigidité initiale, moment de fissuration et moment/charge maximaux
    k0 = (P[5] / d[5])
    icr = np.argmax(np.abs(np.gradient(P, d)) < 0.6 * np.gradient(P, d)[3])
    print(f"raideur initiale (section)      : {k0:.1f} kN/mm")
    print(f"charge à la rupture de pente   : {P[icr]:.1f} kN a {d[icr]:.2f} mm")
    print(f"moment max de la section       : {Ms.max() / 1000:.1f} kN.m  ->  P = {Ms.max() / (a / 2) / 1000:.1f} kN")
    np.savetxt("beton_section.csv", np.column_stack([d, P]), delimiter=",", header="fleche_mm,P_kN", comments="")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(d, P, "k--", lw=1.4, label="analyse de section (RDM)")
        if len(sys.argv) > 1:
            r = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
            ax.plot(r["fleche_mm"], r["P_kN"], color="tab:blue", lw=1.4, label="MPM")
        ax.set_xlabel("flèche à mi-portée (mm)")
        ax.set_ylabel("charge P (kN)")
        ax.set_xlim(0, 26)
        ax.set_ylim(0, 160)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("tuto_beton_courbe.png", dpi=170)
        print("figure : tuto_beton_courbe.png")
    except ImportError:
        print("matplotlib absent : pas de figure")

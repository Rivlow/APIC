# `ui/` — solveur APIC / MPM 2D en trois classes

```
SimulationRunner  (runner.py)  définit la simulation : paramètres plats + matrices n × n ; JSON ; lance
Solver            (solver.py)  reçoit paramètres + matrices, alloue les champs Taichi, calcule, rend l'image
UI                (ui.py)      fenêtre Qt : viewport zoomable, sélection de cellules, paramètres, fichiers
kernels.py                     les kernels Taichi (fluide avec réservoir, grille, émission, rendu)
kernels_inc.py                 mode incompressible : grille MAC + gradient conjugué sur GPU
```

Les kernels solides (MLS-MPM corotationnel, endommagement, rupture) viennent de `Code_tuto/mpm_solid.py`,
inchangé. Rien d'autre dans le dépôt n'est touché.

## Lancer

```bash
python -m ui                     # scène de main_fsi.py (eau + pont)
python -m ui canal.json          # une simulation enregistrée
python -m ui.smoke               # test sans fenêtre (Vulkan puis APIC_UI_ARCH=cuda)
python ui/examples/channel_flow.py [--show] [--save]
```

Dépendances : `pip install -r ui/requirements.txt` (PySide6 ≥ 6.8, taichi 1.7.4, numpy).

## Définir une simulation depuis un script

```python
import sys; sys.path.insert(0, r"C:\Users\lucas\Dev\Fun\APIC")
from ui.runner import SimulationRunner

r = SimulationRunner(n=128, gravity=9.81, solid_E=20000.0)
X, Y = r.centers()                                   # (n, n) : X[i, j] = (i + 0.5) dx
r.set_fluid(Y < 0.20)                                # masque numpy booléen = cellules pleines d'eau
r.set_solid(r.rect(0.55, 0.20, 0.58, 0.55))          # rect / circle renvoient un masque
r.set_obstacle(r.circle(0.35, 0.28, 0.05))
r.set_inlet((X < 0.06) & (Y > 0.25) & (Y < 0.45), velocity=(3.0, 0.0))
r.set_outlet(X > 0.95)
r.set_velocity(r.rect(0.3, 0.1, 0.5, 0.2), (0.0, 2.0))   # vitesse initiale d'une zone

s = r.run(150, callback=lambda s, k: print(k, s.stats()["n_fluid"]))   # sans fenêtre
x = s.positions()                                    # (N, 2) numpy, copié du GPU à la demande
r.save("canal.json")                                 # paramètres + matrices dans un seul fichier
r.show()                                             # interface Qt
```

Tout est matrices `(n, n)` indexées `[i, j]` = (x, y) comme les champs Taichi (`m[:, 0]` = rangée du bas) :

| Matrice | Type | Rôle |
|---|---|---|
| `fluid`, `solid`, `obstacle` | bool | cellules initialement eau / solide / obstacle (exclusifs, le dernier gagne) |
| `inlet` + `inlet_vx`, `inlet_vy` | bool + float | entrée : vitesse imposée (grille et particules) et émission jusqu'à `ppc²` particules par cellule |
| `outlet` | bool | sortie : les particules qui y entrent sont détruites |
| `vx0`, `vy0` | float | vitesse initiale des particules semées dans la cellule |

Paramètres (`SimulationRunner.PARAMS`) : `n, bound, ppc, cfl, gravity, substeps, seed, capacity, res,
fluid_rho, fluid_E, solid_rho, solid_E, solid_nu, eps0, epsf, tau_D, k_res, use_damage, use_rupture,
color_mode`. Les clés `n, bound, ppc, capacity, seed, res` sont structurelles (nouveau solveur au Reset) ;
les autres s'appliquent à chaud (`Solver.set_params`).

**Bande de paroi** : les particules restent dans `[bound·dx, 1 − bound·dx]` (`r.band`). Le semis y est
borné et les cellules d'entrée / sortie situées dans la bande sont ignorées : mettre l'entrée un peu
au-delà (`X < r.band + 2 * r.dx`). Sans cela le stencil 3 × 3 de P2G sortirait de la grille (corruption
silencieuse sur Vulkan, « illegal address » sur CUDA).

## Interface

- **Viewport** : molette = zoom autour du curseur (×1,25, jusqu'à 64×), bouton droit ou milieu = déplacer
  la vue, `F` = vue entière, bouton gauche = boîte de sélection de cellules, `Échap` = désélection. Le
  maillage s'affiche dès qu'une cellule fait 6 px ; les cellules initiales sont teintées (eau bleu,
  solide jaune, obstacle gris, entrée vert, sortie rouge).
- **Cellules** (dock droit) : sur la sélection, Eau / Solide / Obstacle / Entrée (vx, vy) / Sortie /
  Vitesse initiale / Effacer. Toute modification allume le bandeau orange : `Reset` (R) reconstruit.
  Liste « Couleur du fluide » : uniforme, |v| (échelle 0…max, bleu → rouge), vx, vy, pression, vorticité
  (échelle ±max, bleu / blanc / rouge). L'échelle est le max courant lissé, recalculé toutes les 6 images
  (`Solver.fluid_mode`, `Solver.scalar_max`). Pression : E (1 − J) en compressible, q ρ / dt en
  incompressible ; vorticité : rotationnel de la vitesse de grille à la cellule de la particule.
- **Paramètres** (dock gauche) : formulaire généré depuis `PARAMS` ; les clés marquées `*` demandent un
  Reset, les autres agissent immédiatement (gravité, matériaux, CFL…).
- Toolbar : Play/Pause (Espace), Step (S), Reset (R), Vue entière (F), Ouvrir / Enregistrer (JSON).
- `APIC_UI_AUTOQUIT=<s>` : lance la lecture et ferme après s secondes en imprimant les stats (tests).

## Trafic GPU ↔ CPU

Semis : numpy → GPU une fois au build / reset. Simulation : 7 lancements de kernels par sous-pas, rien ne
redescend. Affichage : le kernel `render` dessine la vue zoomée dans une image u8 `(res, res)` sur le
GPU ; une seule copie GPU → CPU par image affichée (≈ 0,6 ms à 700²). Stats : quelques octets toutes les
6 images. `positions()` / `velocities()` / `damage()` copient à la demande.

Arch : Vulkan, puis CUDA, puis CPU ; forcer avec `APIC_UI_ARCH=cuda`. Pour déboguer un accès mémoire
douteux, CUDA lève une erreur là où Vulkan corrompt silencieusement.

## Mode incompressible (`incompressible=True`)

Grille décalée MAC (`u` sur les faces verticales, `v` sur les
horizontales), transfert APIC particules ↔ faces, projection de pression par gradient conjugué
(`kernels_inc.py`) :

- cellules : **air** (p = 0 : surface libre et sorties), **fluide** (inconnue), **solide** (bande de paroi,
  obstacles, entrées : vitesse imposée sur les faces, Neumann) ;
- système A q = −dx (u_e − u_w + v_n − v_s) sur les cellules fluides, A = laplacien à 5 points (un voisin
  solide est exclu, un voisin air compte avec q = 0), puis u −= ∇q ;
- `cg_iters` itérations fixes par sous-pas, entièrement sur le GPU (α, β et les produits scalaires
  vivent dans un champ de 3 flottants ; aucune lecture dans la boucle). Point de départ : la pression du
  sous-pas précédent. `stats()` donne `cg_rr` (résidu) et `div_max` (divergence) comme diagnostics ;
- pas de temps : CFL d'advection sur la vitesse max (plus de limite acoustique), donc `substeps=2` à 4
  suffisent au lieu de 20.

**Couplage fluide-structure** (solide MPM + fluide incompressible), partitionné et explicite :
- le solide garde son pas de temps élastique `dt_solid = cfl·dx/c_s` et est sous-cyclé à l'intérieur de
  chaque pas d'advection du fluide (`_solid_substep`) ;
- les cellules occupées par des particules solides sont classées `MOVING` : leurs faces reçoivent la
  vitesse du solide (moyenne pondérée par la masse des deux nœuds de sa grille collocalisée), donc
  non-pénétration et entraînement du fluide ; elles sont exclues du Laplacien comme un solide fixe ;
- en retour, `pressure_force` calcule −∇p / ρ_s sur les nœuds massiques du solide (p = q ρ_f / dt, q pris
  dans les cellules fluides voisines, 0 ailleurs) et cette accélération est appliquée à chaque sous-pas
  du solide. Elle donne la poussée d'Archimède (vérifiée dans le smoke test) et le chargement
  hydrodynamique (flexion du pont sous la chute d'eau : `pont_incompressible.json`).
- Limites : couplage explicite, donc instable si le solide est beaucoup plus léger que le fluide avec un
  grand pas fluide (masse ajoutée) ; quelques particules fluides peuvent se retrouver dans les cellules
  solides (elles suivent alors la vitesse du solide) ; la surface du solide est résolue à la cellule près.

Coût : ~0,5 ms par itération à 256² (limité par le lancement des kernels) ; 2 sous-pas × 150 itérations
≈ 120 ms par image. Une lecture GPU → CPU coûte ~4 ms sur Vulkan : c'est pour ça que la boucle n'en fait
aucune. Réduire `n` ou `cg_iters` pour l'interactivité ; 150 itérations donnent un résidu ~1e-9 à 256²
grâce au démarrage à chaud.

## JSON

```json
{"version": 2,
 "params": {"n": 128, "gravity": 9.81, ...},
 "matrices": {"fluid": {"dtype": "bool", "shape": [128, 128], "data": "<base64 packbits>"},
              "inlet_vx": {"dtype": "float32", "shape": [128, 128], "data": "<base64>"}}}
```
Un seul fichier auto-suffisant ; les matrices entièrement nulles sont omises.

## Limites

- 2D, domaine unitaire `[0, 1]²`, un fluide et un solide.
- Entrées / sorties : fluide uniquement.
- Changer `n` dans l'interface rééchantillonne les matrices au plus proche voisin.

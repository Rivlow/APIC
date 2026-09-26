# `ui/` — éditeur de projet APIC / MPM (PySide6 + Taichi GGUI)

Éditeur de type CAE pour les simulations 2D du dépôt : arbre de projet, panneau de propriétés,
viewport GGUI embarqué, toolbar Play / Step / Reset, sauvegarde JSON. Le dossier est additif :
aucun script existant n'est modifié, seules les bibliothèques de kernels `APIC/APIC.py` et
`Code_tuto/mpm_solid.py` sont importées.

## Lancer

```bash
# depuis la racine du dépôt (ou n'importe où : ui/__main__.py ajoute la racine au sys.path)
python -m ui                 # scène par défaut = main_fsi.py (eau + pont)
python -m ui projet.json     # ouvre un projet
python -m ui.smoke           # test de fumée sans fenêtre (build, 10 images, JSON, rebuild, release)
```

Dépendances : `pip install -r ui/requirements.txt` (PySide6 ≥ 6.8, taichi 1.7.4, numpy).

## Répartition Qt / Taichi

| Qt (PySide6)                                   | Taichi                                              |
|------------------------------------------------|-----------------------------------------------------|
| fenêtre, docks, arbre, propriétés, toolbar     | kernels APIC / MPM, particules, grille              |
| menus, dialogues fichiers, raccourcis          | rendu (fond, particules, contours) dans une texture |
| sélection dans l'arbre ↔ viewport              | fenêtre GGUI, événements souris / clavier du viewport |

GGUI seule ne peut pas faire l'éditeur (pas d'arbre, de combo, de saisie texte, de docks ni de
dialogue fichier dans Taichi 1.7), Qt seule ne peut pas afficher sans copier l'image sur le CPU.

## Trafic GPU ↔ CPU

Constat vérifié dans `taichi/ui/staging_buffer.py` (1.7.4) : `canvas.circles`, `canvas.lines` et
`canvas.set_image(champ)` passent par des **tampons numpy** (un `np.ndarray` rempli par un kernel
puis renvoyé au C++). La GGUI « classique » fait donc un aller-retour GPU → CPU → GPU des sommets à
chaque image (≈ 4 ms pour 88 k particules ici), y compris dans `main_fsi.py`.

La seule voie zéro copie est `canvas.set_image(ti.Texture)` sur l'**arch Vulkan**. C'est le mode par
défaut de l'éditeur : un kernel (`render_scene`, `ui/sim/kernels.py`) dessine fond, particules et
contours dans une texture RGBA8 que GGUI présente telle quelle (≈ 0,3 ms). La simulation est aussi
rapide sur Vulkan que sur CUDA sur cette machine (≈ 12–14 ms pour 20 sous-pas).

Par image, le seul retour vers Python est le `vec2` de `solid_stats` (8 octets, toutes les 6 images).
Semis des particules : numpy → GPU une seule fois au build / reset. Overlays : petits champs
téléversés uniquement quand la géométrie ou la sélection change.

Sur CUDA / CPU (`APIC_UI_ARCH=cuda`), repli automatique sur les primitives GGUI (avec staging).

## Règle de réentrance (importante)

`glfwPollEvents`, appelé par GGUI dans `get_events` et `show`, dispatche aussi les messages Windows
des fenêtres Qt. Un slot Qt (timer, bouton, spinbox) peut donc s'exécuter **au milieu d'un appel
Taichi**. Règle appliquée dans `MainWindow` : aucun slot ne touche Taichi. Les actions (step, reset,
build, fermeture, captures) sont mises en file (`_defer`) et exécutées au début du tick ; les
overlays sont téléversés dans `GguiViewport.frame()`. Ne pas contourner cette règle : c'est la cause
des access violations rencontrées pendant le développement.

## Structure

```
ui/app.py                  ensure_taichi() -> QApplication -> MainWindow
ui/mainwindow.py           docks, toolbar, menus, QTimer(16 ms), file d'actions, routage par scope
ui/model/project.py        dataclasses (Domain, Materials, Rect/Circle, IC, BC, Solver), JSON,
                           métadonnées `scope` : structural (rebuild) / initial (reset) / runtime (immédiat)
ui/sim/backend.py          ti.init une seule fois : Vulkan > CUDA > CPU, ou APIC_UI_ARCH
ui/sim/seeding.py          semis numpy (fluide uniforme, solide en réseau, masque d'obstacles)
ui/sim/kernels.py          kernels UI sans globales + render_scene (texture)
ui/sim/session.py          SimSession : FieldsBuilder / destroy, substep (ordre de main_fsi), draw_*
ui/widgets/ggui_viewport.py fenêtre GGUI embarquée (HWND -> QWindow.fromWinId), overlays, hit-test, drag
ui/widgets/project_tree.py arbre du projet (menu contextuel : ajouter / supprimer une forme)
ui/widgets/properties.py   formulaire généré depuis dataclasses.fields() + métadonnées
```

## Variables d'environnement

| Variable                | Effet                                                         |
|-------------------------|---------------------------------------------------------------|
| `APIC_UI_ARCH`          | `vulkan` / `cuda` / `cpu` (défaut : vulkan, puis cuda, puis cpu) |
| `APIC_UI_NO_EMBED=1`    | fenêtre GGUI séparée (repli, tous OS)                         |
| `APIC_UI_NO_TEXTURE=1`  | force les primitives GGUI même sur Vulkan                     |
| `APIC_UI_AUTOQUIT=<s>`  | lance la lecture et ferme après s secondes (tests)            |
| `APIC_UI_SCREENSHOT=<p>`| avec AUTOQUIT : `<p>_viewport.png` et `<p>_desktop.png`       |
| `APIC_UI_TRACE=1`       | trace du tick sur stdout                                      |

## Limites de la v1

- 2D, domaine unitaire `[0,1]²` (imposé par les kernels existants), viewport carré.
- Un seul matériau fluide et un seul solide *utilisés* à la fois (les kernels prennent un jeu de
  scalaires par phase) ; plusieurs formes par phase possibles.
- Embarquement de la fenêtre GGUI : Windows uniquement (HWND) ; ailleurs, deux fenêtres.
- Une fenêtre GGUI plein écran ou un dialogue modal fige le viewport pendant sa durée (normal).
- `imgui.ini` continue d'être écrit par GGUI dans le répertoire courant.

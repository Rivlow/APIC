# -*- coding: utf-8 -*-
"""Génère docs/cartographie_APIC.pdf : cartographie du code, optimisation, ergonomie."""
import json
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, Flowable)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Group

OUT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\lucas\Dev\Fun\APIC\docs\cartographie_APIC.pdf"
RES = json.load(open(os.path.join(os.path.dirname(__file__), "results.json"), encoding="utf-8"))

# ---------------------------------------------------------------- polices
F = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("UI", os.path.join(F, "segoeui.ttf")))
pdfmetrics.registerFont(TTFont("UI-B", os.path.join(F, "segoeuib.ttf")))
pdfmetrics.registerFont(TTFont("UI-I", os.path.join(F, "segoeuii.ttf")))
pdfmetrics.registerFont(TTFont("UI-BI", os.path.join(F, "segoeuiz.ttf")))
pdfmetrics.registerFont(TTFont("Mono", os.path.join(F, "consola.ttf")))
pdfmetrics.registerFont(TTFont("Mono-B", os.path.join(F, "consolab.ttf")))
pdfmetrics.registerFontFamily("UI", normal="UI", bold="UI-B", italic="UI-I", boldItalic="UI-BI")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-B", italic="Mono", boldItalic="Mono-B")

# ---------------------------------------------------------------- couleurs (palette catégorielle validée)
INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
GRID = colors.HexColor("#e1e0d9")
BLUE, ORANGE, AQUA, YELLOW = (colors.HexColor(h) for h in ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"))
VIOLET, RED = colors.HexColor("#4a3aa7"), colors.HexColor("#e34948")
LIGHT = {BLUE: "#dce9f9", ORANGE: "#fbe3d9", AQUA: "#d6f2e7", YELLOW: "#fdf0cc", VIOLET: "#e2dff4", RED: "#f9dada"}

# ---------------------------------------------------------------- styles
S = {
    "title": ParagraphStyle("title", fontName="UI-B", fontSize=24, leading=30, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="UI", fontSize=12, leading=16, textColor=INK2, spaceAfter=18),
    "h1": ParagraphStyle("h1", fontName="UI-B", fontSize=17, leading=22, textColor=INK, spaceBefore=14, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="UI-B", fontSize=13, leading=17, textColor=INK, spaceBefore=12, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="UI-B", fontSize=10.5, leading=14, textColor=INK2, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="UI", fontSize=9.6, leading=13.4, textColor=INK, alignment=TA_JUSTIFY,
                           spaceAfter=5),
    "small": ParagraphStyle("small", fontName="UI", fontSize=8.2, leading=11, textColor=INK2, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName="UI", fontSize=9.6, leading=13.4, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="UI", fontSize=8.3, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="UI-B", fontSize=8.3, leading=10.6, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=10.5, textColor=INK, leftIndent=8,
                           backColor=colors.HexColor("#f4f4f1"), borderPadding=(4, 6, 4, 6), spaceBefore=3,
                           spaceAfter=8),
    "quote": ParagraphStyle("quote", fontName="UI-I", fontSize=9.4, leading=13, textColor=INK2, leftIndent=14,
                            spaceAfter=6),
}


import re


def fr(text):
    """Décimales à la française (1.3 → 1,3 ; 3.3e-04 → 3,3e-4) sans toucher aux noms de fichiers."""
    text = re.sub(r"(\d)\.(\d)", r"\1,\2", str(text))
    return re.sub(r"e-0(\d)", r"e-\1", text)


def num(v):
    return f"{v:,}".replace(",", " ")


def P(text, style="body"):
    return Paragraph(text if style in ("h1", "h2", "h3") else fr(text), S[style])


def B(items):
    return [Paragraph(fr(t), S["bullet"], bulletText="•") for t in items]


def C(text):
    return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"), S["code"])


def T(rows, widths, header=True, zebra=True, align_right=()):
    data = []
    for r_i, row in enumerate(rows):
        cells = []
        for c_i, c in enumerate(row):
            st = "cellb" if (header and r_i == 0) else "cell"
            cells.append(c if isinstance(c, Flowable) else Paragraph(fr(c), S[st]))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2) if header else ("LINEBELOW", (0, 0), (-1, 0), 0.3, GRID),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRID),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f9f9f7")))
    for c in align_right:
        style.append(("ALIGN", (c, 1), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------- dessin : boîtes et flèches
def box(d, x, y, w, h, title, lines=(), color=BLUE, fs=8.2):
    d.add(Rect(x, y, w, h, rx=3, ry=3, fillColor=colors.HexColor(LIGHT[color]), strokeColor=color, strokeWidth=0.9))
    d.add(String(x + w / 2, y + h - 11, title, fontName="UI-B", fontSize=fs, fillColor=INK, textAnchor="middle"))
    for k, ln in enumerate(lines):
        d.add(String(x + w / 2, y + h - 11 - (k + 1) * (fs + 2.2), ln, fontName="UI", fontSize=fs - 0.8, fillColor=INK2,
                     textAnchor="middle"))


def arrow(d, x1, y1, x2, y2, label=None, color=INK2, fs=7, lx=0, ly=3):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.9))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    L = 5
    p1 = (x2 - L * math.cos(a - 0.45), y2 - L * math.sin(a - 0.45))
    p2 = (x2 - L * math.cos(a + 0.45), y2 - L * math.sin(a + 0.45))
    d.add(Polygon([x2, y2, p1[0], p1[1], p2[0], p2[1]], fillColor=color, strokeColor=color))
    if label:
        d.add(String((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontName="UI-I", fontSize=fs, fillColor=INK2,
                     textAnchor="middle"))


def diagram_architecture():
    d = Drawing(500, 300)
    # gros blocs
    box(d, 10, 205, 130, 85, "A · Définition de scène", ["ui/runner.py", "SimulationRunner", "params + 9 matrices n×n",
                                                        "JSON v2 (base64)"], BLUE)
    box(d, 185, 205, 130, 85, "B · Solveur", ["ui/solver.py", "Solver", "semis, champs Taichi,", "orchestration des kernels"],
        ORANGE)
    box(d, 360, 205, 130, 85, "D · Interface Qt", ["ui/ui.py", "Viewport + UI", "docks, toolbar, timer 16 ms"], AQUA)
    # kernels
    d.add(Rect(10, 60, 480, 125, rx=4, ry=4, fillColor=colors.HexColor("#f7f7f4"), strokeColor=MUTED, strokeWidth=0.6,
               strokeDashArray=[3, 2]))
    d.add(String(18, 172, "C · Kernels GPU (Taichi)", fontName="UI-B", fontSize=8.6, fillColor=INK2))
    box(d, 20, 75, 105, 88, "C1 · Fluide WC", ["ui/kernels.py", "P2G / grille / G2P", "advection, émission,",
                                              "réservoir de particules"], BLUE)
    box(d, 138, 75, 105, 88, "C2 · Fluide inc.", ["ui/kernels_inc.py", "grille MAC u, v", "classification, CL",
                                                  "CG sur GPU, projection"], VIOLET)
    box(d, 256, 75, 105, 88, "C3 · Solide MPM", ["Code_tuto/mpm_solid.py", "corotationnel, F",
                                                 "endommagement non local", "rupture"], YELLOW)
    box(d, 374, 75, 105, 88, "C4 · Rendu", ["kernels.render", "image u8 (res×res)", "cellules, maillage,",
                                            "particules colorées"], AQUA)
    # points d'entrée
    box(d, 10, 8, 200, 40, "E · Points d'entrée et tests", ["ui/__main__.py, ui/smoke.py, test.py,", "ui/examples/channel_flow.py"], RED)
    box(d, 290, 8, 200, 40, "F · Scripts historiques (hors ui/)", ["main.py, main_fsi.py, APIC/, Time_integration/,",
                                                                    "MpM/beton*.py, MpM/main.py, Code_tuto/demo_solid.py"],
        RED)
    # flèches
    arrow(d, 140, 250, 185, 250, "params, matrices", ly=6)
    arrow(d, 360, 262, 315, 262, "step(), render(), stats()", ly=6)
    arrow(d, 315, 236, 360, 236, "image u8, stats", ly=-10)
    arrow(d, 250, 205, 250, 165, "lancements", lx=32, ly=8)
    arrow(d, 425, 205, 425, 165)
    arrow(d, 75, 205, 75, 48 + 0, "", lx=0)
    d.add(String(78, 190, "run() / show()", fontName="UI-I", fontSize=7, fillColor=INK2))
    arrow(d, 390, 165, 425, 205)
    return d


def diagram_dataflow():
    d = Drawing(500, 130)
    d.add(Rect(10, 15, 200, 108, rx=4, ry=4, fillColor=colors.HexColor("#f7f7f4"), strokeColor=MUTED, strokeWidth=0.6))
    d.add(String(110, 110, "CPU (Python, numpy, Qt)", fontName="UI-B", fontSize=8.6, fillColor=INK2, textAnchor="middle"))
    d.add(Rect(290, 15, 200, 108, rx=4, ry=4, fillColor=colors.HexColor("#f7f7f4"), strokeColor=MUTED, strokeWidth=0.6))
    d.add(String(390, 110, "GPU (champs Taichi)", fontName="UI-B", fontSize=8.6, fillColor=INK2, textAnchor="middle"))
    box(d, 20, 55, 85, 32, "semis numpy", ["x0, v0, cells"], BLUE, fs=7.6)
    box(d, 115, 55, 85, 32, "QImage", ["(res, res, 3) u8"], AQUA, fs=7.6)
    box(d, 300, 55, 85, 32, "particules", ["x, v, C, J, alive"], BLUE, fs=7.6)
    box(d, 395, 55, 85, 32, "grille + image", ["grid_*, u, v, q, img"], ORANGE, fs=7.6)
    arrow(d, 105, 80, 300, 80)
    d.add(String(250, 93, "1× au Reset : from_numpy (positions, vitesses, cellules)", fontName="UI-I", fontSize=6.8,
                 fillColor=INK2, textAnchor="middle"))
    arrow(d, 395, 62, 200, 62)
    d.add(String(250, 50, "1 copie par image affichée : img.to_numpy() = 4,3 ms sur Vulkan, 1,0 ms sur CUDA", fontName="UI-I",
                 fontSize=6.8, fillColor=INK2, textAnchor="middle"))
    d.add(String(250, 34, "stats() : quelques scalaires toutes les 6 images (≈ 2 ms par lecture)", fontName="UI-I",
                 fontSize=6.8, fillColor=INK2, textAnchor="middle"))
    d.add(String(250, 20, "boucle de simulation : 0 lecture GPU → CPU", fontName="UI-B", fontSize=7.4,
                 fillColor=INK2, textAnchor="middle"))
    return d


def pipeline(steps, width=500, h=44, color=BLUE, gap=8, fs=7.6):
    """Chaîne horizontale de boîtes : steps = [(titre, [lignes]), ...]."""
    n = len(steps)
    w = (width - gap * (n - 1)) / n
    d = Drawing(width, h + 6)
    for k, (title, lines) in enumerate(steps):
        x = k * (w + gap)
        box(d, x, 3, w, h, title, lines, color, fs=fs)
        if k < n - 1:
            arrow(d, x + w, 3 + h / 2, x + w + gap, 3 + h / 2)
    return d


def diagram_incompressible():
    d = Drawing(500, 205)
    # boucle solide
    d.add(Rect(5, 130, 490, 70, rx=4, ry=4, fillColor=colors.HexColor(LIGHT[YELLOW]), strokeColor=YELLOW, strokeWidth=0.8))
    d.add(String(12, 188, "si solide : n_in = ceil(dt / dt_solid) sous-pas MPM (grille collocalisée)  ·  puis P2G_solid + grid_update pour les faces MOVING",
                 fontName="UI-B", fontSize=7.4, fillColor=INK2))
    steps = [("fill ×2", ["grid_m, grid_v"]), ("P2G_solid", ["masse, qdm,", "force élastique"]),
             ("grid_update", ["v = p/m, g, parois"]), ("add_accel", ["+ dt · fp"]),
             ("G2P_solid", ["v, C, F, advection"]), ("clear/scatter/", ["update_damage"])]
    n = len(steps)
    gap, w = 6, (470 - 6 * (n - 1)) / n
    for k, (t, ln) in enumerate(steps):
        x = 15 + k * (w + gap)
        box(d, x, 138, w, 42, t, ln, YELLOW, fs=7.2)
        if k < n - 1:
            arrow(d, x + w, 159, x + w + gap, 159)
    # fluide
    steps = [("mac_p2g", ["particules → faces"]), ("mac_classify", ["AIR/FLUID/SOLID/", "MOVING"]),
             ("mac_bc", ["gravité + vitesses", "imposées"]), ("projection", ["cg_init +", "cg_iters × 2 kernels"]),
             ("mac_bc", ["vitesses imposées"]), ("pressure_force", ["−grad p / ρ_s → fp"]),
             ("mac_g2p", ["faces → particules"]), ("advect + emit", ["sorties, entrées"])]
    n = len(steps)
    gap, w = 6, (490 - 6 * (n - 1)) / n
    for k, (t, ln) in enumerate(steps):
        x = 5 + k * (w + gap)
        box(d, x, 60, w, 46, t, ln, VIOLET, fs=6.9)
        if k < n - 1:
            arrow(d, x + w, 83, x + w + gap, 83)
    arrow(d, 250, 130, 250, 106)
    d.add(Rect(120, 8, 260, 36, rx=3, ry=3, fillColor=colors.HexColor("#f7f7f4"), strokeColor=MUTED, strokeWidth=0.6))
    d.add(String(250, 31, "projection : cg_init → [cg_apply, cg_update] × cg_iters → mac_project",
                 fontName="Mono", fontSize=7.2, fillColor=INK, textAnchor="middle"))
    d.add(String(250, 17, "α, β, r·r vivent dans le champ cg[3] : aucune lecture CPU dans la boucle",
                 fontName="UI-I", fontSize=7.2, fillColor=INK2, textAnchor="middle"))
    return d


def chart_breakdown(rows):
    """Barres horizontales empilées : rows = [(label, total_ms, [(segment, ms, couleur)])]."""
    W, H = 500, 34 * len(rows) + 40
    d = Drawing(W, H)
    x0, bar_w = 120, 300
    vmax = max(r[1] for r in rows)
    for k, (label, total, segs) in enumerate(rows):
        y = H - 30 - k * 34
        d.add(String(x0 - 8, y + 4, label, fontName="UI", fontSize=8, fillColor=INK, textAnchor="end"))
        x = x0
        for name, ms, col in segs:
            w = bar_w * ms / vmax
            d.add(Rect(x, y - 4, max(w - 1.5, 0), 16, fillColor=col, strokeColor=None))
            if w > 34:
                d.add(String(x + w / 2, y + 1, f"{ms:.0f}", fontName="UI", fontSize=7.2, fillColor=colors.white,
                             textAnchor="middle"))
            x += w
        d.add(String(x + 5, y + 1, f"{total:.0f} ms / image", fontName="UI-B", fontSize=7.6, fillColor=INK2))
    # légende
    lx = x0
    for name, col in (("sous-cyclage solide", YELLOW), ("projection CG", VIOLET), ("reste (P2G, G2P, advection…)", MUTED)):
        d.add(Rect(lx, 8, 9, 9, fillColor=col, strokeColor=None))
        d.add(String(lx + 13, 10, name, fontName="UI", fontSize=7.4, fillColor=INK2))
        lx += 20 + 4.2 * len(name)
    return d


def diagram_ui_mockup():
    d = Drawing(500, 235)
    d.add(Rect(5, 5, 490, 225, fillColor=colors.HexColor("#f7f7f4"), strokeColor=MUTED, strokeWidth=0.7))
    # menu + toolbar
    d.add(Rect(5, 214, 490, 16, fillColor=colors.HexColor("#e6e5df"), strokeColor=None))
    d.add(String(12, 219, "Fichier  Édition (Ctrl+Z / Ctrl+Y)  Scène  Simulation  Vue  Mesures  Aide", fontName="UI",
                 fontSize=7.2, fillColor=INK))
    d.add(Rect(5, 198, 490, 16, fillColor=colors.HexColor("#ecebe6"), strokeColor=None))
    d.add(String(12, 203, "Play · Step · Reset  |  Outils : rectangle, cercle, pinceau, polygone, sonde  |  Champ : particules / pression / |v|  ·  échelle : auto / fixe / gelée  |  Capture · Exporter",
                 fontName="UI", fontSize=6.6, fillColor=INK))
    # outliner (gauche)
    box(d, 8, 60, 105, 135, "Outliner (scène)", ["› Fluide : bassin (rect)", "› Fluide : bloc (rect)",
                                                "› Solide : tablier (rect)", "› Obstacle : pilier G", "› Obstacle : pilier D",
                                                "› Entrée jet (v = 3,0)", "› Sortie droite", "› Sonde P1 (0,5 ; 0,3)",
                                                "› Ligne L1 (profil)"], BLUE, fs=7)
    # propriétés (gauche bas)
    box(d, 8, 8, 105, 48, "Propriétés", ["objet sélectionné :", "x0 y0 x1 y1, vitesse,", "matériau, unités, doc"], BLUE, fs=7)
    # viewport
    d.add(Rect(118, 8, 262, 187, fillColor=colors.HexColor("#101018"), strokeColor=MUTED, strokeWidth=0.6))
    d.add(String(249, 130, "Viewport (taille = pixels réels du widget)", fontName="UI-B", fontSize=8, fillColor=colors.white,
                 textAnchor="middle"))
    d.add(String(249, 116, "champ de grille en fond (pression, |v|) + particules", fontName="UI", fontSize=7,
                 fillColor=colors.HexColor("#c3c2b7"), textAnchor="middle"))
    d.add(String(249, 104, "légende des cellules, barre d'échelle, curseur : (i, j) x y p |v|", fontName="UI", fontSize=7,
                 fillColor=colors.HexColor("#c3c2b7"), textAnchor="middle"))
    # colorbar
    for k in range(20):
        d.add(Rect(366, 40 + k * 5, 8, 5, fillColor=colors.Color(0.2 + 0.04 * k, 0.3, 1.0 - 0.045 * k), strokeColor=None))
    d.add(String(364, 142, "max", fontName="UI", fontSize=6, fillColor=colors.white, textAnchor="end"))
    d.add(String(364, 40, "0", fontName="UI", fontSize=6, fillColor=colors.white, textAnchor="end"))
    # timeline
    d.add(Rect(118, 8, 262, 14, fillColor=colors.HexColor("#2c2c2a"), strokeColor=None))
    d.add(Rect(124, 13, 250, 4, fillColor=colors.HexColor("#383835"), strokeColor=None))
    d.add(Rect(124, 13, 150, 4, fillColor=AQUA, strokeColor=None))
    d.add(Rect(272, 10, 3, 10, fillColor=colors.white, strokeColor=None))
    d.add(String(249, 24, "timeline : images en cache (vert) · scrub · t = 0,842 s · 60 fps", fontName="UI", fontSize=6.6,
                 fillColor=colors.HexColor("#c3c2b7"), textAnchor="middle"))
    # moniteurs (droite)
    box(d, 386, 108, 106, 87, "Moniteurs", ["résidu CG (échelle log)", "div max, dt, CFL réel",
                                            "énergie cinétique", "N particules / capacité", "ms/image, sim/réel"], ORANGE, fs=7)
    box(d, 386, 60, 106, 44, "Sondes", ["P1 : p(t), |v|(t)", "L1 : profil u(y)", "Obstacle : Fx, Fy"], ORANGE, fs=7)
    box(d, 386, 8, 106, 48, "Journal", ["16:02 NaN détecté → pause", "16:01 capacité 98 %",
                                        "16:00 rebuild 1,3 s"], RED, fs=7)
    return d


# ================================================================ contenu
def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="APIC / MPM 2D — cartographie, optimisation, ergonomie",
                            author="Claude (revue du dépôt APIC)")
    W = A4[0] - 36 * mm
    st = []
    R = RES

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("UI", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 9 * mm, "APIC / MPM 2D — cartographie du code, optimisation et ergonomie · septembre 2026")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{doc_.page}")
        canvas.restoreState()

    # ---------------------------------------------------------------- page de titre / résumé
    st += [P("APIC / MPM 2D", "title"),
           P("Cartographie du code, opportunités d'optimisation du temps de calcul, critique ergonomique de l'interface",
             "sub")]
    st += [P("Résumé", "h1")]
    st += [P("Le dépôt contient un solveur 2D de dynamique des fluides et des solides par méthode des points matériels "
             "(APIC pour le fluide, MLS-MPM corotationnel pour le solide) écrit en Taichi, exécuté sur GPU, et une "
             "interface Qt pour définir des scènes, les lancer et les observer. Le cœur actif tient dans le paquet "
             "<b>ui/</b> (≈ 1 650 lignes) et le module solide <b>Code_tuto/mpm_solid.py</b> ; le reste du dépôt "
             "(≈ 1 300 lignes) est constitué de scripts historiques autonomes qui ont servi de tutoriel et de "
             "prototypes.")]
    st += [P("Trois constats chiffrés sur la RTX 5070 Ti (mesures de cette revue, détail en section 3) :")]
    st += B([
        f"<b>Le sous-cyclage du solide domine le couplage fluide-structure.</b> Sur la scène "
        f"<i>pont_incompressible.json</i>, chaque pas fluide entraîne {R['pont_n_in']} sous-pas solides de "
        f"{R['pont_solid_substep_ms']:.2f} ms (8 à 9 lancements de kernels chacun) : environ deux tiers des "
        f"{R['pont_ms_frame']:.0f} ms par image. Un seul kernel fusionné fait le même travail en "
        f"{R['pont_solid_substep_fused_ms']:.2f} ms (prototype mesuré).",
        f"<b>Le gradient conjugué converge trop lentement sans préconditionnement.</b> "
        f"Sur <i>von_karman.json</i> (256², domaine plein), la projection coûte {R['vk_project_ms']:.0f} ms par "
        f"sous-pas, soit plus de 85 % de l'image ; après 150 itérations le résidu relatif n'est descendu qu'à "
        f"{R['vk_cg_rel_at_150']:.0e} et la divergence maximale résiduelle vaut {R['vk_div_max']:.2f} (elle devrait "
        f"tendre vers 0). Un préconditionneur multigrille ramènerait le compte d'itérations sous 20 quel que soit n.",
        f"<b>Chaque Reset recompile les kernels</b> ({R['rebuild_ms']/1000:.1f} s avec le cache disque chaud, "
        f"{R['rebuild_cold_ms']/1000:.1f} s à froid) parce que les kernels sont instanciés par <i>template</i> sur des "
        f"champs recréés à chaque reconstruction. C'est le premier frein à l'itération interactive sur une scène.",
    ])
    st += [P("Côté interface, l'outil est un bon éditeur de scène mais pas encore un poste de travail de simulation : "
             "il manque la <b>timeline avec cache</b> (revenir en arrière, scruber), les <b>moniteurs</b> (résidu, "
             "divergence, énergie, dt), les <b>sondes</b> et mesures (forces sur un obstacle, profils), l'<b>undo</b>, "
             "une description de scène par <b>objets paramétriques</b> plutôt que par matrices de pixels, et la "
             "séparation du calcul et de l'affichage dans des fils distincts. La section 4 détaille ces points en "
             "s'appuyant sur les pratiques de Houdini, ParaView, Fluent, Blender et PreonLab.")]
    st += [PageBreak()]

    # ---------------------------------------------------------------- 1. gros blocs
    st += [P("1. Vue d'ensemble : les gros blocs", "h1")]
    st += [P("Le code se lit comme trois couches et deux annexes. La couche <b>A</b> décrit une scène sans rien "
             "calculer ; la couche <b>B</b> transforme cette description en champs GPU et enchaîne les kernels de la "
             "couche <b>C</b> ; la couche <b>D</b> pilote B depuis une fenêtre Qt. Les blocs E et F sont des points "
             "d'entrée et des scripts autonomes qui n'entrent pas dans la boucle interactive.")]
    st += [diagram_architecture(), Spacer(1, 6)]
    st += [T([["Bloc", "Fichiers", "Lignes", "Rôle", "Dépend de"],
              ["A · Scène", "ui/runner.py", "214", "Paramètres plats + 9 matrices n×n (fluid, solid, obstacle, inlet, outlet, vx0, vy0, inlet_vx, inlet_vy) ; primitives rect/circle ; JSON v2 ; resize ; run/show", "numpy"],
              ["B · Solveur", "ui/solver.py", "367", "Semis des particules, allocation d'un FieldsBuilder, dt, deux boucles de sous-pas (WC et incompressible), stats, rendu", "A, C"],
              ["C1 · Fluide WC", "ui/kernels.py", "332", "APIC faiblement compressible avec réservoir de particules (alive + pile de slots libres), émission, quantité colorée, rendu", "Taichi"],
              ["C2 · Fluide inc.", "ui/kernels_inc.py", "313", "Grille MAC, transfert APIC sur faces, classification, CL, gradient conjugué sans lecture, projection, force de pression sur le solide", "C1 (constantes)"],
              ["C3 · Solide", "Code_tuto/mpm_solid.py", "260", "MLS-MPM corotationnel, endommagement non local à vitesse limitée, rupture, couleurs, stats", "Taichi"],
              ["D · Interface", "ui/ui.py", "550", "Viewport (zoom, pan, sélection), docks Paramètres et Cellules, toolbar, barre d'état, boucle QTimer, rebuild, fichiers", "A, B"],
              ["E · Entrées", "ui/__main__.py, ui/smoke.py, ui/examples/channel_flow.py, test.py", "≈ 180", "Lancement, test de fumée à 5 scènes, exemple scripté, allée de von Kármán", "A"],
              ["F · Historique", "main.py, main_fsi.py, APIC/APIC.py, Time_integration/, MpM/main.py, MpM/beton.py, MpM/beton_section.py, Code_tuto/demo_solid.py", "≈ 1 300", "Prototypes GGUI autonomes ; béton armé (lois Desayi, acier bilinéaire) ; référence RDM", "Taichi, GGUI"]],
             [22 * mm, 40 * mm, 12 * mm, W - 110 * mm, 20 * mm])]
    st += [Spacer(1, 8), P("1.1 Chemin des données GPU ↔ CPU", "h2")]
    st += [P("Le principe de conception, respecté par le code, est qu'aucune donnée ne redescend du GPU pendant la "
             "simulation : les positions semées montent une fois, la boucle de sous-pas ne fait que des lancements, "
             "et l'image rendue est la seule copie par image affichée. Les statistiques sont lues toutes les six images "
             "et coûtent une synchronisation complète du pipeline à chaque fois.")]
    st += [diagram_dataflow(), Spacer(1, 6)]
    st += [P("1.2 Modes de calcul", "h2")]
    st += [T([["Mode", "Fluide", "Pas de temps", "Sous-pas / image", "Couplage solide"],
              ["Faiblement compressible (défaut)", "APIC sur grille collocalisée, pression p = E (1 − J) déposée comme contrainte dans P2G", "CFL acoustique : dt = cfl · dx / max(c_fluide, c_solide, v_entrée)", "20 (paramètre)", "Grille partagée : P2G fluide puis P2G solide additif, une seule mise à jour de grille"],
              ["Incompressible", "Grille MAC (u sur faces verticales, v sur faces horizontales), projection de pression par CG", "CFL d'advection : dt = cfl · dx / v_max, borné par 0,5 √(dx/g)", "2 à 4", "Partitionné explicite : solide sous-cyclé à dt_solid, cellules MOVING imposent la vitesse aux faces, retour −grad p / ρ_s sur les nœuds"]],
             [30 * mm, 48 * mm, 40 * mm, 20 * mm, W - 138 * mm])]
    st += [PageBreak()]

    # ---------------------------------------------------------------- 2. détail
    st += [P("2. Détail bloc par bloc", "h1")]
    st += [P("2.1 Bloc A — SimulationRunner (ui/runner.py)", "h2")]
    st += [P("Un objet <b>sans Taichi</b>, importable partout, qui porte la définition d'une scène : un dictionnaire de "
             "paramètres <i>p</i> (copie de PARAMS surchargée) et un dictionnaire de matrices <i>m</i> indexées "
             "[i, j] = (x, y) comme les champs Taichi. Ses sous-blocs :")]
    st += B(["<b>Géométrie</b> : n, dx, band (épaisseur de la bande de paroi = bound · dx), centers() renvoie les grilles X, Y "
             "des centres de cellules, rect() et circle() renvoient des masques booléens.",
             "<b>Définition</b> : set_fluid / set_solid / set_obstacle sont exclusifs (le dernier appel gagne), set_inlet / "
             "set_outlet exclusifs entre eux, set_velocity écrit vx0/vy0, clear efface tout sur le masque. resize() "
             "rééchantillonne au plus proche voisin quand n change dans l'interface.",
             "<b>Fichiers</b> : to_dict / from_dict sérialisent les matrices non nulles en base64 (packbits pour les booléens, "
             "float32 brut sinon) ; save / load ; equals sert au test aller-retour.",
             "<b>Exécution</b> : solver() instancie un Solver (import paresseux de Taichi), run(frames, callback) calcule "
             "sans fenêtre, show() ouvre l'interface Qt ; demo() reconstruit la scène de main_fsi.py (bassin, bloc d'eau, "
             "tablier sur deux piliers) en n = 250."])
    st += [P("Paramètres : 24 clés. Sept sont <b>structurelles</b> (n, bound, ppc, capacity, seed, res, incompressible) "
             "et exigent un nouveau solveur ; les autres (gravité, CFL, sous-pas, matériaux, endommagement, cg_iters, "
             "free_surface, color_mode) s'appliquent à chaud via Solver.set_params.", "small")]

    st += [P("2.2 Bloc B — Solver (ui/solver.py)", "h2")]
    st += [P("Le constructeur fait tout le travail CPU en une fois, puis la boucle ne touche plus au CPU.")]
    st += [T([["Sous-bloc", "Ce qu'il fait", "Points d'attention"],
              ["ensure_taichi()", "ti.init une seule fois par processus : APIC_UI_ARCH sinon Vulkan → CUDA → CPU", "Une seule arch par processus ; impossible de changer sans redémarrer"],
              ["Semis", "_fluid_points : ppc² points aléatoires par cellule (graine) ; _solid_points : réseau régulier ppc×ppc ; vitesse initiale lue dans vx0/vy0 ; positions bornées à [band, 1 − band]", "Le solide est semé en réseau, le fluide au hasard ; pas de jitter possible sur le solide"],
              ["Grille de bits", "cells = FLUID0 | SOLID0 | OBSTACLE | INLET | OUTLET, entrées/sorties tronquées à la zone utilisable (hors bande) ; bc_v porte (vx, vy) imposées", "Une entrée dans la bande est silencieusement ignorée"],
              ["Capacités", "capacity = max(n_fluid_init, p.capacity ou ppc²·n²) si entrée ; sinon n_fluid_init ; ≥ 1", "Sans capacity explicite, 4·n² slots : toutes les boucles particules parcourent la capacité, pas les vivantes"],
              ["FieldsBuilder", "Tous les champs (particules fluide et solide, grille, image, MAC + CG si incompressible) dans un seul arbre libéré par release()", "Les champs d'un même place() sont entrelacés (AoS) : sc_f et free_stack partagent la ligne de cache de x_f, v_f, C_f"],
              ["set_params()", "Recalcule dx, masses, Lamé, vitesses d'ondes, dt, dt_solid, tau_D (borné à 2·dt)", "Lit deux fois bc_v.to_numpy() alors que _inlet_speed est déjà calculé"],
              ["reset()", "from_numpy des positions/vitesses semées, init_pool (pile de slots libres), init_solid_state", "Ne remet pas cell_count, grid_* : ils sont réécrits au premier sous-pas"],
              ["step()", "Choisit la boucle selon incompressible, ti.sync() pour mesurer last_step_ms, avance t", "Le ti.sync() force l'attente du GPU dans le fil Qt"],
              ["stats()", "count_alive, solid_stats, cg_rr, div_max, scalar_absmax (échelle de couleur lissée 0,7/0,3)", "4 à 5 lectures GPU→CPU par appel (≈ 2 ms chacune)"],
              ["render()", "solid_colors + fluid_scalar_* + kernel render dans img (res×res u8) puis img.to_numpy()", f"Mesuré {R['render_total_ms']:.1f} ms au total à 700² dont {R['img_to_numpy_ms']:.1f} ms de copie"]],
             [26 * mm, 78 * mm, W - 104 * mm])]
    st += [Spacer(1, 6), P("2.2.1 Sous-pas faiblement compressible (_substep)", "h3")]
    st += [pipeline([("P2G_fluid", ["clear + masse,", "qdm, pression"]), ("P2G_solid", ["additif : masse,", "qdm, τ"]),
                     ("grid_update", ["v = p/m, g,", "parois, entrée"]), ("G2P_fluid", ["v, C, J", "entrée : v fixée"]),
                     ("G2P_solid", ["v, C, F,", "advection"]), ("damage ×3", ["clear, scatter,", "update"]),
                     ("advect_fluid", ["clamp bande,", "sortie → pile"]), ("emit", ["1 thread par", "cellule d'entrée"])],
                    color=BLUE, fs=6.8)]
    st += [P(f"Huit à neuf lancements par sous-pas, vingt sous-pas par image : ≈ 170 lancements par image. Mesuré sur la "
             f"scène demo (n = 250, {num(R['demo_n_fluid'])} particules fluides, {num(R['demo_n_solid'])} solides) : "
             f"{R['demo_ms_frame']:.0f} ms par image soit {R['demo_ms_substep']:.2f} ms par sous-pas.")]
    st += [P("2.2.2 Sous-pas incompressible (_substep_incompressible)", "h3")]
    st += [diagram_incompressible()]
    st += [P("Chaque pas fluide contient donc : n_in × (8 à 9) lancements pour le solide, 1 P2G_solid + grid_update "
             "supplémentaires pour les faces MOVING, 7 kernels fluides, et 1 + 2 × cg_iters lancements pour la "
             "projection. Avec cg_iters = 150 et n_in = 51 (pont), c'est ≈ 770 lancements par pas fluide, "
             "≈ 1 540 par image.")]

    st += [P("2.3 Bloc C1 — kernels du fluide faiblement compressible (ui/kernels.py)", "h2")]
    st += [T([["Kernel", "Boucle", "Lit", "Écrit", "Rôle"],
              ["init_pool", "particules", "—", "alive, x, C, J, free_stack, free_top", "n_init vivantes, les autres garées en (−1, −1) et empilées"],
              ["P2G_fluid", "grille puis particules", "x, v, C, J, alive", "grid_m, grid_v (atomiques)", "Remise à zéro + dépôt B-spline quadratique 3×3, stress = −dt E V (J − 1) 4/dx²"],
              ["grid_update", "grille", "grid_m, cells, bc_v", "grid_v", "qdm → vitesse, gravité, parois glissantes sur bound cellules, obstacle v = 0, entrée v = bc_v"],
              ["G2P_fluid", "particules", "grid_v, cells, bc_v", "v, C, J", "APIC : C = Σ w · outer(v, d) · 4/dx² ; J *= 1 + dt tr C ; entrée : v imposée, C = 0"],
              ["advect_fluid", "grille puis particules", "v, cells", "x, alive, cell_count, free_stack, free_top", "x += dt v, clamp bande ; sortie : mort + push ; entrée : comptage"],
              ["emit", "grille", "cells, cell_count, bc_v", "x, v, C, J, alive, free_top", "Complète chaque cellule d'entrée à target particules en dépilant des slots"],
              ["fluid_scalar_wc / _inc", "particules", "v, J ou u, v, q, grid_v", "sc_f", "Quantité colorée : |v|, vx, vy, pression, vorticité"],
              ["scalar_absmax, count_alive", "particules", "sc_f, alive", "retour scalaire", "Réductions par atomiques (lecture CPU)"],
              ["render", "image, particules fluides, particules solides", "cells, x_f, sc_f, x_s, col_s", "img u8", "Fond (cellules, teintes, maillage) puis splat carré de rayon r_px par particule"]],
             [26 * mm, 24 * mm, 30 * mm, 36 * mm, W - 116 * mm])]
    st += [P("Le réservoir de particules est l'idée clé du bloc : la capacité est fixe, la vie d'une particule est un "
             "drapeau, et une pile de slots libres sur GPU (free_stack, free_top avec atomic_add / atomic_sub) relie les "
             "sorties aux entrées sans jamais compacter ni lire le CPU.", "small")]

    st += [P("2.4 Bloc C2 — kernels incompressibles (ui/kernels_inc.py)", "h2")]
    st += [T([["Kernel", "Boucle", "Rôle"],
              ["mac_p2g", "faces u, faces v, particules, faces u, faces v", "Remise à zéro, dépôt APIC séparé sur les deux grilles décalées (stencil décalé de ½ cellule), normalisation par les poids mu, mv"],
              ["mac_classify", "cellules, particules solides, particules fluides", "SOLID (paroi, obstacle, entrée) ; MOVING si une particule solide s'y trouve ; FLUID si une particule fluide (ou domaine plein) ; AIR sinon et sorties"],
              ["mac_bc", "faces v, faces u, faces v", "Gravité sur v (optionnelle), puis vitesse imposée sur toute face touchant SOLID/MOVING : 0, bc_v, ou moyenne massique des deux nœuds solides"],
              ["cg_init", "cellules ×2", "rhs = −dx · div ; r = b − A q (démarrage à chaud sur la q précédente) ; p = r ; cg[0] = r·r"],
              ["cg_apply", "cellules", "Ap = A p sur FLUID (laplacien 5 points, voisin AIR : q = 0, voisin SOLID exclu) ; cg[1] = p·Ap"],
              ["cg_update", "cellules ×2", "α = cg[0]/cg[1] ; q += α p ; r −= α Ap ; cg[2] = r·r ; β ; p = r + β p ; cg[0] = cg[2]"],
              ["mac_project", "faces u, faces v", "u −= (q_e − q_w)/dx entre deux cellules non solides dont l'une est fluide"],
              ["pressure_force", "grille collocalisée", "fp = coef · grad q (moyenne des 4 cellules autour du nœud, q = 0 hors FLUID) sur les nœuds massiques du solide"],
              ["add_accel", "grille", "grid_v += dt · fp (appliqué à chaque sous-pas solide)"],
              ["mac_g2p", "particules", "v et C reconstruits face par face en ignorant les faces sans poids (mu = 0) ; sinon la particule garde sa vitesse"],
              ["max_speed, divergence_max, cg_residual", "particules / cellules", "Diagnostics et CFL (lectures CPU)"]],
             [28 * mm, 42 * mm, W - 70 * mm])]
    st += [P("Le système résolu est A q = −dx · div(u) avec q = dt p / ρ et A le laplacien positif ; les réductions "
             "r·r et p·Ap sont des accumulations sur un scalaire de kernel, que Taichi implémente par atomiques sur les "
             "backends GPU. Le point de départ à chaud (q précédente) est ce qui rend 150 itérations suffisantes sur "
             "les scènes à surface libre.", "small")]

    st += [P("2.5 Bloc C3 — kernels du solide (Code_tuto/mpm_solid.py)", "h2")]
    st += B(["<b>kirchhoff_stress</b> : τ = 2μ (F − R) Fᵀ + λ J (J − 1) I, R = U Vᵀ par SVD 2×2 (modèle corotationnel de Stomakhin 2012).",
             "<b>P2G_solid</b> (additif, ne remet pas la grille à zéro) : raideur effective k = max(1 − D, k_res), 1 si rompue ; affine = (−dt V 4/dx²) τ + m C.",
             "<b>G2P_solid</b> : v, C ; F ← (I + dt C) F ; si rompue, valeurs singulières écrêtées dans [0,1 ; 1] (pas de traction) ; advection et clamp bande.",
             "<b>clear_eps / scatter_eps / update_damage</b> : l'allongement principal max(σ) − 1 des particules saines est lissé par la grille (non local, ≈ 2 cellules), D suit clamp((ε − ε0)/(εf − ε0)) à vitesse limitée dt/τ_D, rupture à D = 1.",
             "<b>solid_colors, solid_stats, init_beam</b> : affichage (endommagement ou déformation), (n rompues, D max), semis en réseau."])
    st += [P("Quatre SVD par particule et par sous-pas (P2G, G2P si rompue, scatter_eps, update_damage) ; la fonction "
             "grid_update de ce module n'est pas utilisée par ui/ (remplacée par kernels.grid_update, sans amortissement).",
             "small")]

    st += [P("2.6 Bloc D — interface Qt (ui/ui.py)", "h2")]
    st += [T([["Sous-bloc", "Contenu", "Comportement"],
              ["Viewport (QWidget)", "x0, y0, scale ; _buf / _qimg ; sel ; _drag ; _pan", "Molette : zoom ×1,25^k autour du curseur (1 à 64) ; gauche : boîte de cellules (signal selectionChanged) ; droit/milieu : pan ; F : vue entière ; Échap ; paintEvent dessine l'image dans le carré inscrit puis le rectangle de sélection"],
              ["Dock Paramètres", "QTreeWidget : Domaine, Solveur, Matériaux (Fluide, Solide), État initial, Conditions aux limites", "Éditeurs générés depuis PARAMS (QCheckBox / QSpinBox / QDoubleSpinBox, keyboardTracking off) ; structurel → resize/dirty, sinon set_params à chaud ; résumés recalculés à chaque modification"],
              ["Dock Cellules", "vx, vy ; 7 boutons (Eau, Solide, Obstacle, Entrée, Sortie, Vitesse, Effacer) ; maillage, teintes ; couleur du fluide ; échelle", "Applique la fonction du runner sur le masque rectangulaire de la sélection, puis _mark_dirty (bandeau orange)"],
              ["Toolbar / raccourcis", "Play (Espace), Step (S), Reset (R), Vue entière (F), Ouvrir (Ctrl+O), Enregistrer (Ctrl+S / Ctrl+Maj+S)", "Play ou Step sur une scène modifiée déclenche _rebuild"],
              ["Boucle _tick", "QTimer 16 ms", "step() si playing ; render() et set_image à chaque tick ; stats et barre d'état toutes les 6 images"],
              ["_rebuild", "release() de l'ancien solveur, nouveau Solver, step(1) pour compiler, reset()", "Bloquant dans le fil Qt (message dans la barre d'état + processEvents)"],
              ["Fichiers / titre", "_open, _save, _save_as, _update_title", "Le JSON du runner ; l'état des particules n'est pas sauvegardé"],
              ["APIC_UI_AUTOQUIT", "autoplay + autoquit", "Diagnostic sans intervention (tests)"]],
             [28 * mm, 58 * mm, W - 86 * mm])]

    st += [P("2.7 Blocs E et F — entrées, tests, scripts historiques", "h2")]
    st += [T([["Fichier", "Statut", "Contenu"],
              ["ui/__main__.py", "actif", "python -m ui [scène.json] : charge ou demo(), show()"],
              ["ui/smoke.py", "actif", "5 scènes sans fenêtre : demo (stats, rendu, zoom), paramètres à chaud + JSON, entrée/sortie, barrage incompressible (div < 1e-2), poutre légère qui remonte (Archimède)"],
              ["ui/examples/channel_flow.py", "actif", "Canal avec jet, rocher, socle, lame solide ; 150 images + stats ; --show, --save"],
              ["test.py", "actif", "Allée de von Kármán : conduit plein, cylindre, entrée/sortie à U = 0,5, incompressible sans surface libre, n = 256"],
              ["main.py + APIC/APIC.py + Time_integration/", "historique", "Premier solveur APIC GGUI : bloc d'eau sur un cylindre, dt CFL calculé sur GPU à chaque pas ; P2G/G2P identiques à kernels.py sans réservoir"],
              ["main_fsi.py", "historique", "Eau + pont sur grille commune, GGUI ; c'est la scène demo() de ui/"],
              ["Code_tuto/demo_solid.py, MpM/main.py", "historique (doublon)", "Poutre encastrée seule, charge réglable, modèles 1/2/3 ; MpM/main.py est une copie plus ancienne de mpm_solid + demo dans un seul fichier, avec E = 30 000"],
              ["MpM/beton.py, MpM/beton_section.py", "historique, spécifique", "Poutre en béton armé (flexion 4 points, lois Desayi/Neville, acier bilinéaire, résistances lognormales, vérins à vitesse imposée, courbe P-flèche CSV) et référence RDM par analyse de section"],
              ["cas.json, von_karman.json, pont_incompressible.json", "données", "Scènes sauvegardées (v2) ; cas.json fait 730 ko car les matrices float32 ne sont pas compressées"]],
             [46 * mm, 26 * mm, W - 72 * mm])]
    st += [PageBreak()]

    # ---------------------------------------------------------------- 3. optimisation
    st += [P("3. Opportunités d'optimisation du temps de calcul", "h1")]
    st += [P("3.1 Où va le temps : mesures", "h2")]
    st += [P(f"Mesures réalisées avec les scripts docs/bench/bench.py, bench2.py (prototypes de kernels fusionnés) et "
             f"bench3.py (copies d'image, réductions) sur les trois scènes de référence, backend Vulkan sauf mention, "
             f"GPU RTX 5070 Ti, Taichi 1.7.4. Les kernels sont chronométrés après échauffement avec ti.sync() ; les "
             f"valeurs sont dans docs/bench/results.json et ce document se régénère avec docs/bench/make_pdf.py.")]
    st += [chart_breakdown([
        (f"demo (WC, n = 250)", R["demo_ms_frame"], [("reste", R["demo_ms_frame"], MUTED)]),
        (f"pont (inc., n = 160)", R["pont_ms_frame"],
         [("solide", R["pont_solid_ms_frame"], YELLOW), ("CG", R["pont_project_ms"] * 2, VIOLET),
          ("reste", max(R["pont_ms_frame"] - R["pont_solid_ms_frame"] - 2 * R["pont_project_ms"], 0), MUTED)]),
        (f"von Kármán (inc., n = 256)", R["vk_ms_frame"],
         [("CG", R["vk_project_ms"] * 2, VIOLET), ("reste", max(R["vk_ms_frame"] - 2 * R["vk_project_ms"], 0), MUTED)]),
    ])]
    st += [P("Répartition du temps par image (ms), somme des composantes mesurées séparément ; la scène demo n'a ni "
             "projection ni sous-cyclage.", "small")]
    st += [T([["Mesure", "Vulkan", "CUDA", "Commentaire"],
              ["Lancement d'un kernel minuscule", f"{R['tiny_launch_ms']:.2f} ms", f"{R['tiny_launch_ms_cuda']:.2f} ms", "Surcoût pur Python + runtime ; plancher de tout kernel"],
              ["Lecture d'un scalaire GPU → CPU", f"{R['readback_scalar_ms']:.1f} ms", f"{R['readback_scalar_ms_cuda']:.1f} ms", "Synchronisation complète : à bannir des boucles"],
              ["Une itération CG, pont (7 442 cellules fluides / 25 600)", f"{R['pont_cg_pair_ms']:.2f} ms", f"{R['pont_cg_pair_ms_cuda']:.2f} ms", "2 lancements ; limité par le lancement, pas par le calcul"],
              ["Une itération CG, von Kármán (≈ 65 000 cellules)", f"{R['vk_cg_pair_ms']:.2f} ms", f"{R['vk_cg_pair_ms_cuda']:.2f} ms", "Le calcul commence à compter"],
              ["Itération CG fusionnée en 1 kernel (prototype)", f"{R['pont_cg_fused_ms']:.2f} / {R['vk_cg_fused_ms']:.2f} ms", f"{R['pont_cg_fused_ms_cuda']:.2f} / {R['vk_cg_fused_ms_cuda']:.2f} ms", "pont / von Kármán : gain seulement quand le lancement domine"],
              ["Réduction r·r sur 65 536 cellules : atomique global / par ligne / stencil seul", f"{R['reduce_atomic_ms']:.2f} / {R['reduce_rows_ms']:.2f} / {R['stencil_only_ms']:.2f} ms", f"{R['reduce_atomic_ms_cuda']:.2f} / {R['reduce_rows_ms_cuda']:.2f} / {R['stencil_only_ms_cuda']:.2f} ms", "L'accumulation atomique double le coût d'un passage de stencil"],
              ["Copie de l'image 700²×3 u8 (variantes : champ vectoriel, plat, i32, ndarray)", R["copy_variants_vulkan"], R["copy_variants_cuda"], "Plancher de synchronisation, indépendant du format"],
              ["Sous-pas solide (9 lancements)", f"{R['pont_solid_substep_ms']:.2f} ms", f"{R['pont_solid_substep_ms_cuda']:.2f} ms", f"× {R['pont_n_in']} par pas fluide × 2 pas par image"],
              ["Sous-pas solide fusionné en 1 kernel (prototype)", f"{R['pont_solid_substep_fused_ms']:.2f} ms", f"{R['pont_solid_substep_fused_ms_cuda']:.2f} ms", "Mêmes calculs, un seul lancement"],
              ["Image demo : ms par image (20 sous-pas)", f"{R['demo_ms_frame']:.0f} ms", f"{R['demo_ms_frame_cuda']:.0f} ms", "≈ 170 lancements"],
              ["Image pont", f"{R['pont_ms_frame']:.0f} ms", f"{R['pont_ms_frame_cuda']:.0f} ms", "≈ 1 540 lancements"],
              ["Image von Kármán", f"{R['vk_ms_frame']:.0f} ms", f"{R['vk_ms_frame_cuda']:.0f} ms", "≈ 320 lancements, dominés par le CG"],
              ["render() complet à 700², vue entière", f"{R['render_total_ms']:.1f} ms", f"{R['render_total_ms_cuda']:.1f} ms", f"kernel {R['render_kernel_only_ms']:.1f} ms + copie {R['img_to_numpy_ms']:.1f} ms + solid_colors"],
              ["render() au zoom 64×", f"{R['render_scale64_ms']:.1f} ms", f"{R['render_scale64_ms_cuda']:.1f} ms", "Le splat carré tourne pour les particules hors champ"],
              ["Reconstruction du solveur (Reset structurel)", f"{R['rebuild_ms']/1000:.1f} s", f"{R['rebuild_ms_cuda']/1000:.1f} s", f"Cache disque chaud ; {R['rebuild_cold_ms']/1000:.1f} s à froid"]],
             [58 * mm, 24 * mm, 24 * mm, W - 106 * mm])]
    st += [Spacer(1, 4)]
    st += [P("Convergence du gradient conjugué (résidu relatif r·r / r0·r0, démarrage à chaud) :", "body")]
    st += [T([["Scène", "Cellules fluides", "Itérations pour 1e-2", "pour 1e-4", "pour 1e-6", "Résidu relatif à 150 it."],
              ["pont (surface libre)", num(R['pont_fluid_cells']), "—", str(R["cg_it_1e-4"]), "> 150", f"{R['cg_rr_final_rel']:.1e}"],
              ["von Kármán (domaine plein)", num(R['vk_fluid_cells']), str(R["vk_cg_it_1e-2"]), str(R["vk_cg_it_1e-4"] or "> 400"), str(R["vk_cg_it_1e-6"] or "> 400"), f"{R['vk_cg_rel_at_150']:.1e}"]],
             [40 * mm, 26 * mm, 30 * mm, 22 * mm, 22 * mm, W - 140 * mm])]
    st += [P("Lecture : le CG non préconditionné a besoin d'un nombre d'itérations proportionnel à la taille du domaine "
             "(≈ n pour un laplacien 2D). Sur le domaine plein 256², 150 itérations laissent un champ de pression loin de "
             "convergé ; la divergence résiduelle est absorbée image après image par le démarrage à chaud, ce qui "
             "fonctionne visuellement mais fausse les pressions et les forces mesurées.", "small")]

    st += [P("3.2 Pistes classées par gain / effort", "h2")]
    opt = [
        ["1", "Fusionner le sous-pas solide en un kernel",
         f"Un kernel à 5 boucles de premier niveau (clear, P2G, grille, G2P + scatter, damage) : mesuré {R['pont_solid_substep_ms']:.2f} → {R['pont_solid_substep_fused_ms']:.2f} ms. Sur le pont, ≈ {R['pont_solid_ms_frame']:.0f} ms par image deviennent ≈ {R['pont_solid_fused_ms_frame']:.0f} ms.",
         f"× {R['pont_solid_gain']:.1f} sur le solide, image pont ≈ {R['pont_ms_frame_est1']:.0f} ms", "faible", "Aucun changement de résultat ; garder les kernels séparés pour les tests"],
        ["2", "Préconditionner la projection (MGPCG)",
         "Multigrille géométrique en préconditionneur du CG, comme mgpcg_advanced.py des exemples Taichi : 3 à 4 niveaux, lisseur Jacobi amorti ou rouge-noir, un V-cycle par itération. Le compte d'itérations tombe à 10–20 quel que soit n. Chaque V-cycle peut tenir dans un seul kernel (boucles par niveau déroulées avec ti.static).",
         "× 5 à 8 sur la projection ; von Kármán ≈ 100 → 20 ms", "moyen–élevé", "Le cas surface libre + cellules MOVING demande de restreindre correctement le masque ; valider avec div_max"],
        ["3", "Fusionner l'itération CG en un kernel",
         f"cg_apply + cg_update en un kernel à 3 boucles : {R['pont_cg_pair_ms']:.2f} → {R['pont_cg_fused_ms']:.2f} ms sur le pont (7 442 inconnues, lancement dominant) mais {R['vk_cg_pair_ms']:.2f} → {R['vk_cg_fused_ms']:.2f} ms sur von Kármán (60 480 inconnues, calcul dominant). Compatible avec la piste 2.",
         f"× {R['pont_cg_gain']:.1f} sur la projection en surface libre ; rien sur un domaine plein", "faible", "—"],
        ["4", "Réductions sans atomique global",
         f"r·r et p·Ap accumulent 60 000 atomiques flottants sur une seule adresse par kernel : sur 65 536 cellules, un passage de stencil coûte {R['stencil_only_ms']:.2f} ms seul et {R['reduce_atomic_ms']:.2f} ms avec l'accumulation. Une réduction en deux étages (≈ 1 024 sommes partielles par autant de threads, puis une somme finale) rend l'accumulation presque gratuite ; la variante « une ligne par thread » testée gagne sur Vulkan ({R['reduce_rows_ms']:.2f} ms) mais perd sur CUDA (trop peu de threads). Même idée pour scalar_absmax, count_alive, max_speed, divergence_max.",
         "× 1,5 à 2 sur chaque kernel CG à 256²", "faible", "Mesurer les deux backends ; garder la version atomique en référence"],
        ["5", "Réduire le sous-cyclage solide",
         "n_in = 51 vient de dt_solid = cfl·dx/c_s avec c_s ≈ 45 m/s. Options : mettre à jour l'endommagement une fois par pas fluide au lieu de 51 (3 kernels sur 9), fusionner les deux fill dans P2G_solid, et surtout accepter un dt fluide plus petit quand le solide domine (substeps = 4 réduit n_in à 13 pour le même coût total mais un couplage plus précis).",
         "× 1,3 à 1,5 en plus de la piste 1", "faible", "L'endommagement à vitesse limitée dépend de dt : garder dt/τ_D cohérent"],
        ["6", "Éviter la recompilation au Reset",
         f"Les kernels prennent les champs en ti.template() : chaque FieldsBuilder est une nouvelle instanciation ({R['rebuild_ms']/1000:.1f} s). Passer aux arguments ti.types.ndarray() (compilés une fois par type et rang) ou allouer une seule fois à la capacité max et ne réinitialiser que les contenus.",
         "Reset de 1,3 s → ≈ 50 ms", "moyen", "Les ndarrays supportent les atomiques et ti.grouped ; l'arbre FieldsBuilder disparaît"],
        ["7", "Rendu : rejeter les particules hors champ, ne rendre que si nécessaire, accepter le plancher de copie",
         f"Dans render, tester le rectangle [col − r, col + r] contre l'image avant la boucle de splat (zoom 64× : {R['render_scale64_ms']:.0f} ms → ≈ {R['render_total_ms']:.0f} ms). Ne pas appeler render() quand la simulation est en pause et que ni la vue ni les options n'ont changé. La copie de l'image coûte {R['img_to_numpy_ms']:.1f} ms sur Vulkan quel que soit le format (champ vectoriel, plat, i32, ndarray : toutes les variantes mesurées entre 4 et 5 ms) : c'est un plancher de synchronisation, pas un débit. Le seul moyen de l'éviter est l'affichage zéro copie (texture Vulkan partagée avec un QRhiWidget), un chantier à part ; la piste 12 la masque en la sortant du fil Qt.",
         f"Zoom fort × 3 ; pause : {R['render_total_ms']:.0f} ms par tick économisés", "très faible", "—"],
        ["8", "Choisir le backend par scène",
         f"APIC_UI_ARCH=cuda ne change pas le code. Mesuré ici : CUDA est plus lent sur les kernels (pont {R['pont_ms_frame_cuda']:.0f} contre {R['pont_ms_frame']:.0f} ms, von Kármán {R['vk_ms_frame_cuda']:.0f} contre {R['vk_ms_frame']:.0f} ms) et plus rapide sur les lectures et la copie d'image ({R['img_to_numpy_ms_cuda']:.1f} contre {R['img_to_numpy_ms']:.1f} ms). Vulkan reste le bon défaut pour la simulation ; CUDA sert au débogage des accès mémoire.",
         "0 à 15 % selon la scène", "nul", "Une seule arch par processus"],
        ["9", "Compacter les particules vivantes",
         f"Avec une entrée, capacity = 4 n² ({num(R['vk_capacity'])} slots pour {num(R['vk_n_fluid'])} vivantes sur von Kármán) : toutes les boucles particules parcourent la capacité. Un compactage périodique (préfixe-somme Taichi, ti.algorithms.PrefixSumExecutor) ou une capacité dimensionnée réduit les boucles au nécessaire.",
         "10 à 25 % sur P2G/G2P", "moyen", "L'ordre des particules change : sans effet physique"],
        ["10", "Liste compacte des cellules fluides pour le CG",
         "Sur une surface libre (pont : 29 % de cellules fluides), les boucles CG parcourent n² cellules et testent ctype. Construire une liste d'indices FLUID dans mac_classify et itérer dessus divise le travail par 3 ; sans effet sur un domaine plein.",
         "× 2 à 3 sur le CG en surface libre", "faible", "—"],
        ["11", "Regrouper les lectures de stats()",
         "count_alive, solid_stats, cg_residual, divergence_max, scalar_absmax font 4 à 5 synchronisations (≈ 2 ms chacune). Un kernel unique qui écrit un champ de 8 flottants, lu en une fois, ramène cela à une lecture.",
         "≈ 8 ms toutes les 6 images", "très faible", "—"],
        ["12", "Séparer calcul et affichage (fil de calcul)",
         "Le ti.sync() de step() et le render() tournent dans le fil Qt : une image de 180 ms gèle l'interface. Un QThread dédié à Taichi (tous les appels Taichi depuis ce fil) avec une file d'ordres, et le fil Qt qui ne fait que recevoir l'image u8, garde l'interface fluide et permet de rendre à 60 Hz pendant qu'un pas de 200 ms se calcule.",
         "Réactivité, pas de temps CPU", "moyen", "Aucun appel Taichi depuis un slot Qt"],
    ]
    st += [T([["#", "Piste", "Constat et proposition", "Gain estimé", "Effort", "Risque / remarque"]] + opt,
             [6 * mm, 30 * mm, W - 116 * mm, 30 * mm, 14 * mm, 36 * mm])]
    st += [Spacer(1, 6)]
    st += [P("3.3 Ordre recommandé", "h2")]
    st += B(["<b>Semaine 1</b> : pistes 1, 3, 4, 7, 11 (fusions, réductions, rendu) — quelques heures chacune, gains mesurables immédiatement, aucun changement de physique. Le pont passe d'environ "
             f"{R['pont_ms_frame']:.0f} ms à ≈ {R['pont_ms_frame_est2']:.0f} ms par image avec les seules pistes 1 et 3.",
             "<b>Semaine 2</b> : piste 6 (ndarray) — c'est la plus rentable pour le confort d'itération, et elle simplifie la piste 12.",
             "<b>Ensuite</b> : piste 2 (MGPCG), qui change la classe de complexité de la projection et rend les résolutions 512² raisonnables ; puis 5, 9, 10 selon les scènes réellement utilisées.",
             "<b>À ne pas faire</b> : réintroduire une lecture CPU dans la boucle CG pour un arrêt anticipé — la lecture coûte l'équivalent de 5 à 8 itérations, et le compte fixe reste prévisible. Une alternative sans lecture : adapter cg_iters au résidu lu dans stats() toutes les 6 images."])
    st += [P("Ce qui est déjà bien fait et à conserver : zéro copie dans la boucle, réservoir de particules sans "
             "compactage, α et β sur GPU, rendu de la vue zoomée sur GPU en u8, fills et kernels sans globales (arguments "
             "explicites), bande de paroi qui protège le stencil.", "quote")]
    st += [Spacer(1, 8)]

    # ---------------------------------------------------------------- 4. UI
    st += [P("4. Critique de l'interface et propositions", "h1")]
    st += [P("4.1 Ce qui fonctionne", "h2")]
    st += B(["Le viewport est correct : zoom autour du curseur, pan, sélection de cellules, maillage à partir de 6 px, coordonnées au survol.",
             "La séparation runner / solver / ui est saine : la scène est un objet sérialisable, l'interface n'a pas d'état caché, les scripts et l'interface partagent le même chemin.",
             "Le mécanisme dirty + bandeau orange + Reset est explicite ; les paramètres à chaud contre structurels sont distingués.",
             "Les raccourcis essentiels existent (Espace, S, R, F, Ctrl+O/S) et les tests de fumée couvrent le pipeline."])
    st += [P("4.2 Points faibles, par ordre d'impact sur un travail de simulation", "h2")]
    weak = [
        ["Pas de retour en arrière", "Play, Pause, Step, Reset : impossible de revoir l'image précédente, de scruber, ni de comparer deux instants. Un phénomène vu trop tard (rupture, instabilité) oblige à tout relancer.", "Houdini, Blender : cache de frames + timeline scrubable ; PreonLab : lecture indépendante du calcul"],
        ["Aucun moniteur", "cg_rr et div_max sont calculés mais jamais affichés ; pas de courbe de résidu, d'énergie, de dt, de nombre de particules dans le temps. On ne sait pas si la projection converge ni si le pas est stable.", "Fluent, OpenFOAM (foamMonitor), COMSOL : fenêtre de résidus par défaut"],
        ["Aucune mesure", "Pas de sonde ponctuelle, de profil sur une ligne, de force sur un obstacle, d'intégrale sur une zone. Sans cela on ne peut valider (Strouhal, Cd, flèche d'une poutre).", "ParaView : Probe, Plot Over Line ; Fluent : report definitions"],
        ["Interface gelée pendant le calcul", "step() bloque le fil Qt (ti.sync) : à 180 ms par image, les spinbox et la molette répondent par à-coups, et un rebuild de 1,3 s fige tout.", "Toutes les applications pro calculent dans un processus ou un fil séparé"],
        ["Scène décrite en pixels", "Les objets sont des matrices booléennes : impossible de déplacer un rectangle, changer un rayon, éditer une vitesse d'entrée après coup sans tout redessiner ; changer n rééchantillonne au plus proche voisin.", "Outliner + objets paramétriques (Blender, Houdini, tout CAO)"],
        ["Pas d'undo", "Une erreur de sélection sur « Effacer » est irréversible.", "QUndoStack est standard dans Qt"],
        ["Champs visibles seulement via les particules", "Pas de rendu de la pression ou de la vitesse sur la grille en fond, pas de vecteurs, de lignes de courant, ni de barre de couleur dans le viewport ; l'échelle auto (lissée) change en permanence.", "ParaView : colorbar, range auto/fixe/gelé ; glyphs, streamlines"],
        ["Outils de dessin pauvres", "Seule la boîte rectangulaire ; pas de cercle, polygone, pinceau, ligne, ni import d'un masque depuis une image PNG.", "Éditeurs 2D ; import d'images pour la géométrie"],
        ["Silences dangereux", "Une entrée dans la bande de paroi est ignorée sans message ; la capacité pleine arrête l'émission sans avertir ; un NaN se propage sans détection ; un CFL non respecté n'est pas signalé.", "Journal + avertissements bloquants"],
        ["Rendu à résolution fixe 700²", "L'image est étirée dans le carré inscrit : floue sur un grand écran ou en HiDPI, et l'espace latéral d'une fenêtre large est perdu.", "Rendre à la taille réelle du widget × devicePixelRatio"],
        ["Paramètres sans contexte", "Pas d'unité, de plage, de tooltip, ni de valeurs dérivées (c_s, dt, Mach, nombre de particules, mémoire) ; le suffixe « * » est discret.", "COMSOL/Fluent affichent les dérivées ; tooltips partout"],
        ["Pas d'export", "Ni capture, ni séquence d'images/vidéo, ni CSV, ni checkpoint de l'état des particules pour reprendre.", "Export image/vidéo, CSV, VTK pour ParaView"],
    ]
    st += [T([["Point", "Constat", "Ce que font les pros"]] + weak, [30 * mm, W - 92 * mm, 62 * mm])]
    st += [Spacer(1, 6), P("4.3 Proposition d'organisation", "h2")]
    st += [P("Maquette d'un poste de travail de simulation, en gardant l'architecture actuelle (runner / solver / ui) : "
             "un outliner d'objets à gauche, le viewport avec timeline au centre, moniteurs, sondes et journal à droite.")]
    st += [diagram_ui_mockup(), Spacer(1, 6)]
    st += [P("4.4 Les changements, du plus structurant au plus simple", "h2")]
    st += [P("Fil de calcul et boucle d'affichage découplés", "h3")]
    st += [P("Un QThread « SimWorker » possède le solveur et exécute tous les appels Taichi ; il reçoit des ordres "
             "(step, reset, set_params, render(vue)) par une file et émet des signaux (image prête, stats). Le fil Qt "
             "ne fait que peindre la dernière image reçue. Conséquences : l'interface reste fluide pendant un rebuild, "
             "le rendu peut tourner à 60 Hz sur une vue qui bouge même si le pas fait 200 ms, et l'on peut afficher "
             "un vrai rapport temps simulé / temps réel avec une cible (« temps réel », « au plus vite », « N pas par image »).")]
    st += [P("Timeline avec cache de frames", "h3")]
    st += [P("Conserver en mémoire GPU (ou CPU si l'on accepte une copie au moment de l'enregistrement, hors boucle) "
             "les N dernières images rendues, ou mieux l'état complet (x, v, C, J, alive, F, D) toutes les k images "
             "sous forme de checkpoints. Une barre de temps montre les images en cache, un glisser scrube, "
             "Maj+S revient d'une image, et « reprendre ici » relance depuis un checkpoint. Le coût mémoire d'un "
             "checkpoint est celui des champs (≈ 12 flottants par particule fluide : 250 000 particules ≈ 12 Mo) ; "
             "100 checkpoints tiennent dans la carte.")]
    st += [P("Moniteurs et sondes", "h3")]
    st += B(["Un dock « Moniteurs » avec pyqtgraph (léger, rapide) : résidu CG en échelle log, div_max, dt, CFL réel (v_max dt / dx), énergie cinétique, nombre de particules / capacité, ms par image. Les données proviennent d'un seul kernel de stats (piste 11) lu toutes les k images.",
             "Sondes : un clic avec l'outil sonde crée un objet « Sonde » dans l'outliner (cellule i, j) ; sa série p(t), |v|(t) s'affiche. Une « Ligne » donne un profil (u(y) dans le canal). Un « Obstacle » ou un « Solide » expose Fx, Fy intégrés à partir de q sur ses faces (le kernel pressure_force fait déjà l'essentiel).",
             "Export CSV de toutes les séries en un clic, pour comparer à une référence (comme MpM/beton_section.py le fait déjà pour le béton)."])
    st += [P("Scène par objets", "h3")]
    st += [P("Remplacer la description par matrices par une liste d'objets (rect, circle, polygon, image importée) "
             "portant leur type (fluide, solide, obstacle, entrée, sortie, vitesse) et leurs paramètres. Les matrices "
             "actuelles deviennent le résultat d'une rastérisation faite par le runner au moment du build : le format "
             "JSON reste compatible (on peut stocker les deux), les scripts existants continuent de marcher, et "
             "l'interface gagne : déplacer, redimensionner, dupliquer, désactiver un objet, changer n sans perte. "
             "Un calque « peinture » (matrice libre) garde les cas particuliers.")]
    st += [P("Édition", "h3")]
    st += B(["QUndoStack sur toutes les opérations de scène et de paramètres ; Ctrl+Z / Ctrl+Y.",
             "Outils : rectangle, cercle, polygone, pinceau avec rayon, ligne (pour une lame fine), pipette ; import PNG (blanc = fluide, etc.) à la résolution de la grille.",
             "Presets : « Nouveau depuis un modèle » (barrage, canal, von Kármán, pont, poutre) ; la scène demo devient un fichier."])
    st += [P("Visualisation", "h3")]
    st += B(["Champ de grille en fond (pression, |v|, vorticité, type de cellule) avec une barre de couleur dans le viewport, plage auto / fixe / gelée, et un mode « ±max symétrique » pour les quantités signées.",
             "Vecteurs de vitesse sur la grille (un sur k), lignes de courant optionnelles, particules coloriées comme aujourd'hui.",
             "Légende des teintes de cellules dans le viewport, barre d'échelle en unités du domaine, position du curseur avec les valeurs de champ sous le curseur.",
             "Rendu à la taille réelle du widget (res = pixels du widget × devicePixelRatio) ; res cesse d'être un paramètre de scène."])
    st += [P("Robustesse et confort", "h3")]
    st += B(["Journal (dock) avec horodatage : avertissements pour entrée dans la bande, capacité atteinte, dt tronqué, NaN détecté (pause automatique), temps de rebuild.",
             "Tooltips sur chaque paramètre : signification, unité, plage, effet ; panneau « Dérivées » : c_fluide, c_solide, dt, sous-pas solides, particules, mémoire.",
             "Fichier récent, autosave de la scène, checkpoint de l'état (npz) séparé de la scène (json).",
             "Export : capture PNG, séquence d'images / MP4 (ffmpeg), CSV des moniteurs, VTK des particules pour ParaView.",
             "Paramétrique : file d'exécution sans fenêtre (balayage d'un paramètre, N graines) avec un tableau de résultats — le runner sait déjà tout faire, il manque l'interface."])
    st += [P("4.5 Feuille de route indicative", "h2")]
    st += [T([["Phase", "Contenu", "Dépend de", "Effet"],
              ["1", "Fil de calcul, kernel de stats unique, moniteurs pyqtgraph, journal + détection de NaN, colorbar et champ de grille en fond, rendu à la taille du widget", "pistes 11 et 12 de la section 3", "Interface fluide et instrumentée"],
              ["2", "Timeline + cache de checkpoints, Maj+S, capture et export image/vidéo/CSV", "phase 1", "Observation et communication des résultats"],
              ["3", "Scène par objets + outliner + undo + outils de dessin + presets", "—", "Itération rapide sur les cas"],
              ["4", "Sondes, lignes, forces sur obstacle / solide ; balayage paramétrique sans fenêtre", "phases 1 et 3", "Validation quantitative"]],
             [14 * mm, W - 96 * mm, 40 * mm, 42 * mm])]
    st += [Spacer(1, 8)]
    st += [P("Annexe — inventaire complet des fichiers", "h2")]
    st += [T([["Fichier", "Lignes", "Statut"],
              ["ui/ui.py", "550", "actif"], ["ui/solver.py", "367", "actif"], ["ui/kernels.py", "332", "actif"],
              ["ui/kernels_inc.py", "313", "actif"], ["ui/runner.py", "214", "actif"], ["ui/smoke.py", "105", "actif (test)"],
              ["ui/examples/channel_flow.py", "45", "actif (exemple)"], ["ui/__main__.py", "14", "actif"], ["ui/README.md", "142", "documentation"],
              ["Code_tuto/mpm_solid.py", "260", "actif (solide)"], ["Code_tuto/demo_solid.py", "169", "historique"],
              ["MpM/beton.py", "538", "spécifique, autonome"], ["MpM/beton_section.py", "125", "spécifique, autonome"], ["MpM/main.py", "348", "historique (doublon de mpm_solid + demo)"],
              ["main_fsi.py", "189", "historique"], ["main.py", "138", "historique"], ["APIC/APIC.py", "108", "historique"], ["Time_integration/time_integration.py", "26", "historique"],
              ["test.py", "31", "actif (von Kármán)"], ["cas.json / von_karman.json / pont_incompressible.json", "—", "scènes"]],
             [70 * mm, 20 * mm, W - 90 * mm])]
    doc.build(st, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    build()
    print("écrit :", OUT)

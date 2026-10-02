#!/usr/bin/env python3
"""Graphical abstract for EPJ B submission.

"Strain couplings and Akhiezer shear viscosity in cubic SrTiO3"

EPJ B spec: max width 480 px, aspect ratio 11:6, .png/.jpg, color encouraged.
Renders a 600-dpi master (2880x1571) and the exact 480x262 submission file.

Three blocks: (1) mechanism; (2) the coupling test: eta from the transferred (A), projected harmonic (B) and projected
SCPH (C) couplings on the two input models; (3) the number with its status, next to the inner-mesh sequence.
Numbers are hard-coded deliberately (a graphical abstract is a schematic of the paper's claims and this script must run
standalone); they are the values of data/processed/coupling_constructions_SrTiO3.csv (300 K, 11^3 outer mesh).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------- palette
# Okabe-Ito (colorblind-safe by construction)
BLUE = "#0072B2"       # computed / this work
VERMIL = "#D55E00"     # BaTiO3 critical enhancement
GRAY_BAND = "#C9C9C9"  # measured band
GRAY_TXT = "#444444"
LIGHT = "#F2F2F2"

# ---------------------------------------------------------------- data
# eta(300 K), 10^-3 Pa s: constructions A, B, C at inner mesh 12^3 on both input models
CONS = {"hybrid model": (0.920, 0.694, 0.475), "diagnostic surface": (1.229, 0.640, 0.349)}
MESH_N = [2, 4, 8, 12]
ETA_C_MESH = [0.475, 0.449, 0.450, 0.475]         # hybrid model, consistent linewidths
ETA_C_FIXED = [0.475, 0.546, 0.609, 0.644]        # linewidths held at inner mesh 2^3

FW, FH = 4.80, 2.618
fig = plt.figure(figsize=(FW, FH), facecolor="white")
ax_m = fig.add_axes([0.005, 0.02, 0.28, 0.96]); ax_m.axis("off")
ax_a = fig.add_axes([0.355, 0.20, 0.27, 0.66])
ax_b = fig.add_axes([0.725, 0.20, 0.255, 0.66])
FS_TITLE, FS_LAB, FS_TICK, FS_BIG = 6.2, 5.4, 4.8, 7.4

# ------------------------------------------------ (1) mechanism
ax_m.set_xlim(0, 1); ax_m.set_ylim(0, 1)
rows_y = [0.82, 0.70, 0.58]
shift = [0.055, 0.0, -0.055]
for y, dx in zip(rows_y, shift):
    xs = np.linspace(0.16, 0.66, 5) + dx
    ax_m.scatter(xs, [y] * 5, s=13, c=BLUE, zorder=3, edgecolors="white", linewidths=0.4, clip_on=False)
ax_m.add_patch(FancyArrowPatch((0.70, 0.84), (0.88, 0.84), arrowstyle="-|>", mutation_scale=7, color=GRAY_TXT, lw=1.0))
ax_m.add_patch(FancyArrowPatch((0.14, 0.56), (-0.04, 0.56), arrowstyle="-|>", mutation_scale=7, color=GRAY_TXT, lw=1.0))
ax_m.text(0.41, 0.935, "shear strain", ha="center", fontsize=FS_LAB, color=GRAY_TXT)
ax_m.text(0.41, 0.45, "phonon occupations lag;\nrelaxation dissipates", ha="center", va="center",
          fontsize=FS_LAB, color=GRAY_TXT, linespacing=1.25)
ax_m.text(0.41, 0.265, r"$\eta \;\propto\; \sum_{\mathbf{q}s}\gamma^{2}\,\tau$", ha="center", va="center",
          fontsize=FS_BIG + 1.4, color="black")
ax_m.text(0.45, 0.085, "self-consistent phonons;\ncouplings in the\nrenormalised basis", ha="center", va="center",
          fontsize=FS_LAB - 0.3, color=GRAY_TXT, linespacing=1.25)

# ------------------------------------------------ (2) the A -> B -> C test
x = np.arange(3)
for k, (label, vals) in enumerate(CONS.items()):
    ax_a.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=[VERMIL, "#7FB3D5", BLUE] if k == 0 else "none",
             edgecolor=[VERMIL, "#7FB3D5", BLUE], hatch=None if k == 0 else "////", lw=0.8, label=label)
ax_a.set_xticks(x, ["A\ntransferred", "B\nprojected\nharmonic", "C\nprojected\nSCPH"], fontsize=FS_TICK - 0.3)
ax_a.set_ylabel(r"$\eta_{xyxy}$(300 K) ($10^{-3}$ Pa s)", fontsize=FS_LAB, labelpad=1.5)
ax_a.set_title("strain coupling: transfer vs projection", fontsize=FS_TITLE, pad=9.0)
from matplotlib.patches import Patch  # noqa: E402
ax_a.legend(handles=[Patch(facecolor="0.45", label="hybrid model"), Patch(facecolor="white", edgecolor="0.45", hatch="////", label="diagnostic surface")],
           fontsize=FS_TICK - 0.6, frameon=False, loc="upper right", handlelength=1.2)
ax_a.text(0.5, 1.012, "both models: correction mesh $2^3$, inner mesh $12^3$", transform=ax_a.transAxes, ha="center", va="bottom",
          fontsize=FS_TICK - 0.6, color=GRAY_TXT)
ax_a.text(0.62, 0.95, "A: $\\times$1.9–3.5 vs C\n($2^3$ and $12^3$, both models)", fontsize=FS_TICK - 0.3, color=VERMIL)
ax_a.set_ylim(0, 1.50)
ax_a.tick_params(labelsize=FS_TICK, length=2, pad=1.5)
for s_ in ("top", "right"):
    ax_a.spines[s_].set_visible(False)

# ------------------------------------------------ (3) the number with its status
ax_b.plot(MESH_N, ETA_C_MESH, "-o", color=BLUE, ms=2.6, lw=1.1, mec="white", mew=0.3, label="consistent linewidths")
ax_b.plot(MESH_N, ETA_C_FIXED, "--o", color="#7FB3D5", ms=2.4, lw=0.9, mfc="white", label="linewidths fixed")
ax_b.set_xticks(MESH_N)
ax_b.set_ylim(0.3, 0.8)
ax_b.set_xlabel("inner mesh $n$ ($n^3$)", fontsize=FS_LAB, labelpad=1.0)
ax_b.set_ylabel(r"$\eta_C$ ($10^{-3}$ Pa s)", fontsize=FS_LAB, labelpad=1.5)
ax_b.set_title("hybrid model, 300 K", fontsize=FS_TITLE, pad=2.5)
ax_b.text(2.2, 0.36, "$0.475\\times10^{-3}$ Pa s at $12^3$\nconvergence not established", fontsize=FS_TICK - 0.3, color=BLUE)
ax_b.legend(fontsize=FS_TICK - 0.8, frameon=False, loc="upper left", handlelength=1.6)
ax_b.tick_params(labelsize=FS_TICK, length=2, pad=1.5)
for s_ in ("top", "right"):
    ax_b.spines[s_].set_visible(False)

for x0, x1 in ():
    fig.add_artist(FancyArrowPatch((x0, 0.52), (x1, 0.52), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=8, color=GRAY_TXT, lw=1.1))

# ---------------------------------------------------------------- output
fig.savefig("graphical_abstract_master.png", dpi=600, facecolor="white")
from PIL import Image  # noqa: E402
im = Image.open("graphical_abstract_master.png")
im = im.resize((480, 262), Image.LANCZOS)
im.save("graphical_abstract_480x262.png")
print("master:", Image.open("graphical_abstract_master.png").size)

#!/usr/bin/env python3
"""Fig. 1 (data-side realization of skeleton slot 4.1) — mode-resolved
eta_xyxy contribution spectrum vs bare omega0 at 300 K, with the
175 cm-1 Route S/H partition boundary.

The skeleton's Fig. 1 is "phonon spectra and linewidths"; this figure
realizes its viscosity-facing content (where the eta weight sits in the
spectrum). A dispersion+linewidth overlay panel remains [TODO] for the
final composite.

Data provenance (committed CSVs only, no hand-edited data):
  data/processed/eta_spectral_density_SrTiO3.csv
    <- scripts/audit_eta_assembly.py::export_spectral_density
       (generation-3 assembly; see eta_SrTiO3_stageC.md)

Writes figures/fig1_eta_spectral_density.{png,pdf}.
Usage: uv run python scripts/fig1_eta_spectral_density.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "processed" / "eta_spectral_density_SrTiO3.csv"
OUT = REPO / "figures" / "fig1_eta_spectral_density"
CUTOFF = 175.0

rows = [line.split(",") for line in SRC.read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("omega0")]
lo = np.array([float(r[0]) for r in rows])
hi = np.array([float(r[1]) for r in rows])
centers = 0.5 * (lo + hi)
width = hi - lo
routeS = np.array([float(r[2]) for r in rows])
h_stab = np.array([float(r[3]) for r in rows])
h_unst = np.array([float(r[4]) for r in rows])
g_sec = np.array([float(r[5]) for r in rows])

fig, ax = plt.subplots(figsize=(5.6, 3.8))
bottom = np.zeros_like(centers)
for series, label, color in [
        (h_stab, r"Route H, stable", "#4878a8"),
        (h_unst + g_sec, r"Route H, formerly unstable + $\Gamma$ sector", "#a84848"),
        (routeS, r"Route S", "#6aa86a")]:
    ax.bar(centers, series * 1e3, width=width * 0.92, bottom=bottom * 1e3,
           label=label, color=color, alpha=0.88)
    bottom = bottom + series

peak = float((h_stab + h_unst + g_sec + routeS).max())
ax.set_ylim(0, peak * 1e3 * 1.12)
ax.axvline(CUTOFF, color="k", ls="--", lw=1.0)
ax.annotate(r"Route S / Route H partition, $\omega_0$ = 175 cm$^{-1}$"
            "\n(5-point Richardson validated)",
            xy=(CUTOFF, ax.get_ylim()[1] * 0.03), xytext=(240, 0.55),
            textcoords="data", fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.axvspan(lo[0], 0.0, color="0.9", zorder=0)
ax.text(-95, 0.02, "bare-imaginary\nmanifold\n($\\omega_0^2 < 0$)",
        fontsize=6.5, color="0.35", va="bottom")

total = (routeS + h_stab + h_unst + g_sec).sum()
ax.set_xlabel(r"bare harmonic frequency $\omega_0$ (cm$^{-1}$; negative = imaginary)")
ax.set_ylabel(r"$\eta$ contribution per 25 cm$^{-1}$ bin ($10^{-3}$ Pa s)")
ax.set_title(rf"SrTiO$_3$ $\eta_{{xyxy}}$ spectral decomposition, 300 K "
             rf"(total {total * 1e3:.2f}$\times 10^{{-3}}$ Pa s)", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.set_xlim(lo[0], 900)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf  (bin sum {total:.4e} Pa s)")

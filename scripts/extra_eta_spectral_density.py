#!/usr/bin/env python3
"""Auxiliary figure (not in the paper; data of Fig. 1c) — mode-resolved eta_xyxy contribution spectrum
at 300 K on the production model (hybrid model tut_z_od1, construction C), binned by the bare omega_0 of
each renormalised mode's max-overlap bare partner, with constructions B and A of the same surface
as reference lines.

Data provenance (committed CSVs only): data/processed/eta_spectral_density_SrTiO3.csv
  <- scripts/audit_eta_assembly.py::export_spectral_density
Writes figures/extra_eta_spectral_density.{png,pdf}.
Usage: uv run python scripts/extra_eta_spectral_density.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "processed" / "eta_spectral_density_SrTiO3.csv"
OUT = REPO / "figures" / "extra_eta_spectral_density"

rows = [line.split(",") for line in SRC.read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("omega0")]
lo = np.array([float(r[0]) for r in rows])
hi = np.array([float(r[1]) for r in rows])
centers = 0.5 * (lo + hi)
width = hi - lo
eta_c = np.array([float(r[2]) for r in rows])
eta_g = np.array([float(r[3]) for r in rows])
eta_b = np.array([float(r[4]) for r in rows])
eta_a = np.array([float(r[5]) for r in rows])

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.bar(centers, eta_c * 1e3, width=width * 0.92, label=r"construction C (production)", color="#4878a8", alpha=0.88)
ax.bar(centers, eta_g * 1e3, width=width * 0.92, bottom=eta_c * 1e3, label=r"$\Gamma$ TO1 triplet (Vogt-anchored)",
       color="#c8a848", alpha=0.88)
ax.step(np.append(lo, hi[-1]), np.append(eta_b, eta_b[-1]) * 1e3, where="post", color="#a84848", lw=1.1,
        label="construction B (bare coupling, same basis)")
ax.step(np.append(lo, hi[-1]), np.append(eta_a, eta_a[-1]) * 1e3, where="post", color="0.3", lw=1.0, ls="--",
        label="construction A (transferred bare coupling, superseded)")
peak = float(max((eta_c + eta_g).max(), eta_b.max(), eta_a.max()))
ax.set_ylim(0, peak * 1e3 * 1.12)
ax.axvspan(lo[0], 0.0, color="0.9", zorder=0)
ax.text(-50, peak * 1e3 * 0.98, "bare-imaginary\npartner\nmanifold", fontsize=6.5, color="0.35", ha="center", va="top")
total = (eta_c + eta_g).sum()
ax.set_xlabel(r"bare frequency $\omega_0$ of the partner mode (cm$^{-1}$; negative = imaginary)")
ax.set_ylabel(r"$\eta$ contribution per 25 cm$^{-1}$ bin ($10^{-3}$ Pa s)")
ax.set_title(rf"SrTiO$_3$ $\eta_{{xyxy}}$ spectral decomposition, 300 K (C total {total * 1e3:.2f}$\times 10^{{-3}}$ Pa s)",
             fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.set_xlim(lo[0], 900)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf  (bin sum {total:.4e} Pa s)")

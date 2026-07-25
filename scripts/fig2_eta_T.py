#!/usr/bin/env python3
"""Fig. 2 — eta_xyxy(T) for SrTiO3, partition-resolved (skeleton slot 4.2).

Data provenance (committed CSVs only, no hand-edited data):
  data/processed/eta_SrTiO3.csv
    <- scripts/compute_eta_SrTiO3.py (generation-3 assembly, audit-clean;
       see data/processed/reports/eta_SrTiO3_stageC.md)

Provisional flags are rendered IN the figure, not only the caption:
  * T != 300 K: SCPH-coupled linewidth recipe validated at 300 K only ->
    shaded 'method validated at 300 K only' background outside 300 K.
  * T = 350, 400 K: Vogt soft-mode series ends at 298 K; the Gamma-point
    sector falls back to the ALAMODE theory floor -> open markers + label.

Writes figures/fig2_eta_T.{png,pdf}.
Usage: uv run python scripts/fig2_eta_T.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "processed" / "eta_SrTiO3.csv"
OUT = REPO / "figures" / "fig2_eta_T"

rows = [line.split(",") for line in SRC.read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("T_K")]
T = np.array([float(r[0]) for r in rows])
eta = np.array([float(r[1]) for r in rows])
s_route = np.array([float(r[2]) for r in rows])
h_stable = np.array([float(r[3]) for r in rows])
h_unst = np.array([float(r[4]) for r in rows])
g_sec = np.array([float(r[5]) for r in rows])
vogt_ok = np.array([r[7] == "1" for r in rows])

fig, ax = plt.subplots(figsize=(5.2, 3.8))

# provisional-method shading (everything except 300 K)
ax.axvspan(T.min() - 10, 290, color="0.92", zorder=0)
ax.axvspan(310, T.max() + 10, color="0.92", zorder=0)
ax.text(0.985, 0.02, "shaded: linewidth method validated at 300 K only",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        color="0.35")

ax.stackplot(T, h_stable, h_unst, g_sec, s_route,
             labels=[r"Route H, stable ($5 \leq \omega_0 < 175$ cm$^{-1}$)",
                     r"Route H, formerly unstable ($\omega_0^2 < 0$)",
                     r"$\Gamma$ soft-mode sector (Vogt)",
                     r"Route S ($\omega_0 \geq 175$ cm$^{-1}$)"],
             colors=["#4878a8", "#a84848", "#c8a848", "#6aa86a"], alpha=0.85)
ax.plot(T, eta, "k-", lw=1.6, label=r"$\eta_{xyxy}$ total")
filled = vogt_ok
ax.plot(T[filled], eta[filled], "ko", ms=5)
ax.plot(T[~filled], eta[~filled], "ko", ms=6, mfc="white",
        label="open: Vogt series exceeded,\ntheory floor used (350, 400 K)")

ax.set_xlabel("T (K)")
ax.set_ylabel(r"$\eta_{xyxy}$ (Pa s)")
ax.set_xlim(T.min() - 10, T.max() + 10)
ax.set_ylim(0, eta.max() * 1.18)
ax.legend(fontsize=6.5, loc="upper right", framealpha=0.95)
ax.set_title(r"SrTiO$_3$ lattice shear viscosity, partition-resolved", fontsize=9)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf  (eta(300K) = {eta[T == 300][0]:.3e} Pa s)")

#!/usr/bin/env python3
"""BaTiO3 zone-center soft-mode sector viscosity eta_soft(T) — DRAFT figure,
deliberately UNNUMBERED (final manuscript assembly assigns the number).

Data provenance (committed CSVs only, no hand-edited data):
  data/processed/eta_BaTiO3.csv
    <- scripts/compute_eta_BaTiO3.py (zone-center Route-H sector assembly;
       provenance: data/processed/reports/eta_BaTiO3_stageC.md)
  data/processed/softmode_inputs_BaTiO3.csv (omega_s annotation series)

Shows the ~300x critical enhancement approaching T_C = 393 K, with the
sector-scope caveat printed ON the figure (partial quantity; stable
manifold excluded by the Gamma-point-only scope).

Writes figures/fig_eta_BaTiO3_sector.{png,pdf}.
Usage: uv run python scripts/fig_eta_BaTiO3_sector.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "processed" / "eta_BaTiO3.csv"
OUT = REPO / "figures" / "fig_eta_BaTiO3_sector"
T_C = 393.0

rows = [line.split(",") for line in SRC.read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("T_K")]
T = np.array([float(r[0]) for r in rows])
eta = np.array([float(r[1]) for r in rows])
om = np.array([float(r[2]) for r in rows])

fig, ax = plt.subplots(figsize=(5.0, 3.8))
ax.semilogy(T, eta, "o-", color="#a84848", lw=1.5, ms=5,
            label=r"$\eta_{44}^{\rm soft}$ (zone-center sector)")
ax.axvline(T_C, color="k", ls=":", lw=1.0)
ax.text(T_C + 4, eta.min() * 1.5, r"$T_C$ = 393 K", fontsize=7, rotation=90)

ax2 = ax.twinx()
ax2.plot(T, om, "s--", color="#4878a8", lw=1.0, ms=4, alpha=0.7)
ax2.set_ylabel(r"$\omega_s$ (cm$^{-1}$, measured, VSR 1982)", color="#4878a8",
               fontsize=8)
ax2.tick_params(axis="y", labelcolor="#4878a8")

ax.set_xlabel("T (K)")
ax.set_ylabel(r"$\eta_{44}^{\rm soft}$ (Pa s)")
ax.set_title(r"BaTiO$_3$ soft-mode sector viscosity: critical enhancement",
             fontsize=9)
ax.text(0.03, 0.04,
        "SECTOR quantity (Gamma-point-only scope):\n"
        "stable manifold excluded — lower bound on the\n"
        "full-zone $\\eta_{44}$; all points overdamped ($\\tau_{\\rm eff}$)",
        transform=ax.transAxes, fontsize=6.5, color="0.35")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf (enhancement x{eta.max() / eta.min():.0f} from "
      f"{T[np.argmin(eta)]:.0f} K to {T[np.argmax(eta)]:.0f} K)")

#!/usr/bin/env python3
"""Figure 5 — BaTiO3 zone-center-anchored soft-branch sector of eta_44 (exploratory).

Caption (for the manuscript, section 4.5):
  Figure 5. Zone-center-anchored soft-branch sector of the BaTiO3 shear
  coefficient, eta_44^soft (circles, left log axis), in the cubic
  paraelectric phase, computed from the strained-cell coupling Lambda
  (tensor convention), the measured hyper-Raman soft-mode frequency
  omega_s(T) and damping (Vogt, Sanjurjo & Rossbroich; squares, right
  axis show omega_s), and the neutron branch dispersion of Harada, Axe &
  Shirane, with the stress-correlator two-pole kernel. The sector rises
  steeply on approaching T_C as omega_s collapses. It is a partial
  quantity within the diagonal model: the stable manifold lies outside the
  zone-center-anchored scope (every zone-center point is overdamped), and
  no total-viscosity value is inferred from it. Dotted line: T_C = 403 K
  of the melt-grown VSR samples (410 K is T_C + 7 K).

Data provenance (committed CSVs only, no hand-edited data):
  data/processed/eta_BaTiO3.csv
    <- scripts/compute_eta_BaTiO3.py (zone-center Route-H sector assembly;
       provenance: data/processed/reports/eta_BaTiO3_stageC.md)
  data/processed/softmode_inputs_BaTiO3.csv (omega_s annotation series)

The sector-scope caveat is printed ON the figure (partial quantity;
stable manifold excluded by the Gamma-point-only scope).

Writes figures/fig5_eta_BaTiO3_sector.{png,pdf}.
Usage: uv run python scripts/fig5_eta_BaTiO3_sector.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from latvisc.materials import BATIO3_TC_VSR1982  # noqa: E402

SRC = REPO / "data" / "processed" / "eta_BaTiO3.csv"
OUT = REPO / "figures" / "fig5_eta_BaTiO3_sector"
T_C = BATIO3_TC_VSR1982   # melt-grown VSR 1982 samples

rows = [line.split(",") for line in SRC.read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("T_K")]
T = np.array([float(r[0]) for r in rows])
eta = np.array([float(r[1]) for r in rows])
om = np.array([float(r[2]) for r in rows])

fig, ax = plt.subplots(figsize=(5.0, 3.8))
ax.semilogy(T, eta, "o-", color="#a84848", lw=1.5, ms=5,
            label=r"$\eta_{44}^{\rm soft}$ (zone-center-anchored sector, exploratory)")
ax.axvline(T_C, color="k", ls=":", lw=1.0)
ax.text(T_C + 4, eta.min() * 1.5, rf"$T_C$ = {T_C:.0f} K (VSR samples)", fontsize=7, rotation=90)

ax2 = ax.twinx()
ax2.plot(T, om, "s--", color="#4878a8", lw=1.0, ms=4, alpha=0.7)
ax2.set_ylabel(r"$\omega_s$ (cm$^{-1}$, measured, VSR 1982)", color="#4878a8",
               fontsize=8)
ax2.tick_params(axis="y", labelcolor="#4878a8")

ax.set_xlabel("T (K)")
ax.set_ylabel(r"$\eta_{44}^{\rm soft}$ (Pa s)")
ax.set_title(r"BaTiO$_3$ zone-center-anchored soft-branch sector of $\eta_{44}$",
             fontsize=9)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.55, 0.99))
fig.tight_layout(rect=(0, 0.09, 1, 1))
# scope caveat as a figure footnote below the axes — clear of both curves
fig.text(0.5, 0.012,
         "SECTOR quantity ($\\Gamma$-point-only scope): stable manifold excluded; not a total\n"
         "viscosity. All points overdamped; stress-correlator two-pole kernel",
         ha="center", va="bottom", fontsize=6.5, color="0.35")
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf (enhancement x{eta.max() / eta.min():.0f} from "
      f"{T[np.argmin(eta)]:.0f} K to {T[np.argmax(eta)]:.0f} K)")

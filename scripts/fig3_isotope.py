#!/usr/bin/env python3
"""Fig. 3 — eta vs 18O fraction f for SrTiO3 at 300 K (skeleton slot 4.3).

Data provenance (committed CSVs only, no hand-edited data):
  data/processed/eta_isotope_SrTiO3.csv
    <- scripts/compute_eta_isotope_SrTiO3.py (Tamura channel, EXACT g2 sum
       verified against manuscript Eq. (9); generation-3 baseline; see
       data/processed/reports/eta_SrTiO3_stageC.md)

Writes figures/fig3_isotope.{png,pdf}.
Usage: uv run python scripts/fig3_isotope.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "processed" / "eta_isotope_SrTiO3.csv"
OUT = REPO / "figures" / "fig3_isotope"

rows = [line.split(",") for line in SRC.read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("f_18O")]
f = np.array([float(r[0]) for r in rows])
eta = np.array([float(r[2]) for r in rows])
rel = np.array([float(r[3]) for r in rows])

fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.plot(f, eta * 1e3, "o-", color="#4878a8", lw=1.5, ms=6)
ax.set_xlabel(r"$^{18}$O fraction $f$ on the oxygen sublattice")
ax.set_ylabel(r"$\eta_{xyxy}$(300 K) ($10^{-3}$ Pa s)")
ax.set_title(r"Isotope suppression of $\eta$ (Tamura, exact $g_2$)", fontsize=9)

delta_pct = 100.0 * (rel[-1] - 1.0)
ax.annotate(f"{delta_pct:+.1f}% at $f$ = {f[-1]:.2f}",
            xy=(f[-1], eta[-1] * 1e3), xytext=(0.55, 0.55),
            textcoords="axes fraction", fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(0.03, 0.05,
        "site projection approximated by total DOS\n(upper bound on the suppression)",
        transform=ax.transAxes, fontsize=6.5, color="0.35")
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf  (monotone decrease: {np.all(np.diff(eta) < 0)}; "
      f"{delta_pct:+.1f}% at f={f[-1]:.2f})")

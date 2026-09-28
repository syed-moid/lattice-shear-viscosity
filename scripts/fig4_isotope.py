#!/usr/bin/env python3
"""Fig. 4 — eta vs 18O fraction f for SrTiO3 at 300 K (skeleton slot 4.3).

Data provenance (committed CSVs only, no hand-edited data):
  data/processed/eta_isotope_SrTiO3.csv
    <- scripts/compute_eta_isotope_SrTiO3.py (Tamura channel, EXACT g2 sum
       verified against manuscript Eq. (9); eigenvector-resolved O-site
       projection with the renormalised eigenvectors; the bare-eigenvector
       series is carried in the same CSV for comparison)

Writes figures/fig4_isotope.{png,pdf}.
Usage: uv run python scripts/fig4_isotope.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "processed" / "eta_isotope_SrTiO3.csv"
OUT = REPO / "figures" / "fig4_isotope"

lines = [line for line in SRC.read_text().splitlines() if line and not line.startswith("#")]
header = lines[0].split(",")
rows = [line.split(",") for line in lines[1:]]
col = {name: i for i, name in enumerate(header)}
f = np.array([float(r[col["f_18O"]]) for r in rows])
eta = np.array([float(r[col["eta_total_Pas"]]) for r in rows])
rel = np.array([float(r[col["eta_over_eta0"]]) for r in rows])
rel_bare = (np.array([float(r[col["eta_over_eta0_bare_eigenvectors"]]) for r in rows])
            if "eta_over_eta0_bare_eigenvectors" in col else None)

fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.plot(f, 100.0 * (rel - 1.0), "o-", color="#4878a8", lw=1.5, ms=6, label="renormalised eigenvectors")
if rel_bare is not None:
    ax.plot(f, 100.0 * (rel_bare - 1.0), "s--", color="#a84848", lw=1.0, ms=4, label="harmonic eigenvectors")
    ax.legend(fontsize=7, loc="upper right")
ax.set_xlabel(r"$^{18}$O fraction $f$ on the oxygen sublattice")
ax.set_ylabel(r"relative change of $\eta_{xyxy}$(300 K) (%)")
ax.set_title("Mass-variance channel of $^{18}$O substitution\n(Tamura rate, exact $g_2$, O-site projection)", fontsize=8.5)

delta_pct = 100.0 * (rel[-1] - 1.0)
ax.annotate(f"{delta_pct:+.2f} % at $f$ = {f[-1]:.2f}".replace("-", "−"),
            xy=(f[-1], delta_pct), xytext=(0.30, 0.52),
            textcoords="axes fraction", fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(0.03, 0.05,
        "eigenvector-resolved O-site projection (Eq. 10);\nfixed renormalised spectrum and couplings",
        transform=ax.transAxes, fontsize=6.5, color="0.35")
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf  (monotone decrease: {np.all(np.diff(eta) < 0)}; "
      f"{delta_pct:+.1f}% at f={f[-1]:.2f})")

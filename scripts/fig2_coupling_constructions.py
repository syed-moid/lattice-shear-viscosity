#!/usr/bin/env python3
"""Fig. 2 — three ways of evaluating the shear-strain coupling of a renormalised phonon.

(a) per bare-partner frequency bin, the eta ratio of projected harmonic to transferred coupling (B/A) and of
    projected SCPH to projected harmonic coupling (C/B), inner mesh 12^3, both input models;
(b) acoustic limit: Lambda/q^2 of the two transverse acoustic modes along Gamma-X for A, B, C on the hybrid model (a strain
    coupling of an acoustic mode must vanish as q^2, so Lambda/q^2 must stay finite);
(c) eta_A, eta_B, eta_C against the inner mesh on both input models (consistent linewidths), with eta_C at
    linewidths held at the 2^3/2^3 values.
Data (committed CSVs only): data/processed/coupling_ratio_bins_SrTiO3.csv, acoustic_limit_SrTiO3.csv,
coupling_constructions_SrTiO3.csv <- scripts/export_coupling_constructions.py.
Writes figures/fig2_coupling_constructions.{png,pdf}.
Usage: uv run python scripts/fig2_coupling_constructions.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
P = REPO / "data" / "processed"
OUT = REPO / "figures" / "fig2_coupling_constructions"
COL = {"A": "#D55E00", "B": "#0072B2", "C": "#009E73"}
ORDER = ["imaginary", "[0-25)", "[25-50)", "[50-100)", "[100-150)", "[150-175)", "[175-300)", "[300-inf)"]
LABEL = {"imaginary": "imag.", "[0-25)": "0–25", "[25-50)": "25–50", "[50-100)": "50–100", "[100-150)": "100–150",
         "[150-175)": "150–175", "[175-300)": "175–300", "[300-inf)": "≥300"}

bins = pd.read_csv(P / "coupling_ratio_bins_SrTiO3.csv", comment="#")
ac = pd.read_csv(P / "acoustic_limit_SrTiO3.csv", comment="#")
cons = pd.read_csv(P / "coupling_constructions_SrTiO3.csv", comment="#")

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
# (a)
ax = axs[0]
x = np.arange(len(ORDER))
for k, (surf, hatch) in enumerate((("hybrid", None), ("diagnostic", "//"))):
    b = bins[bins.surface == surf].set_index("partner_omega0_bin").reindex(ORDER)
    ax.bar(x - 0.3 + 0.2 * k, b.ratio_B_over_A, 0.2, color=COL["B"], hatch=hatch, edgecolor="white" if hatch is None else "0.3",
           label=f"B/A ({surf})")
    ax.bar(x + 0.1 + 0.2 * k, b.ratio_C_over_B, 0.2, color=COL["C"], hatch=hatch, edgecolor="white" if hatch is None else "0.3",
           label=f"C/B ({surf})")
ax.axhline(1.0, color="0.3", lw=0.8)
ax.set_yscale("log")
ax.set_xticks(x, [LABEL[o] for o in ORDER], rotation=45, fontsize=7)
ax.set_xlabel(r"bare-partner $\omega_0$ bin (cm$^{-1}$)", fontsize=8)
ax.set_ylabel(r"ratio of $\eta$ contributions in the bin", fontsize=8)
ax.set_title("(a) coupling ratios by frequency bin (inner mesh $12^3$)", fontsize=8.5)
ax.legend(fontsize=6, ncol=2, loc="lower left")
# (b)
ax = axs[1]
a = ac[(ac.surface == "hybrid") & (ac.line == "GX")]
for mode, ls in ((1, "-"), (2, "--")):
    m = a[a["mode"] == mode].sort_values("q_reduced")
    for c in ("A", "B", "C"):
        ax.plot(m.q_reduced, m[f"Lambda_{c}_over_q2"] / 1e3, ls, color=COL[c], marker="o", ms=3, label=f"{c}, TA mode {mode}")
ax.axhline(0.0, color="0.6", lw=0.6)
ax.set_xlabel(r"$|q|$ along $\Gamma$–X (units of $2\pi/a$)", fontsize=8)
ax.set_ylabel(r"$\Lambda/q^2$ ($10^3$ cm$^{-2}$)", fontsize=8)
ax.set_title("(b) acoustic limit (TA branches), hybrid model", fontsize=8.5)
ax.legend(fontsize=6, ncol=2, loc="upper left")
# (c)
ax = axs[2]
for surf, mk in (("hybrid", "o"), ("diagnostic", "s")):
    s = cons[(cons.surface == surf) & (cons.linewidths == "consistent")].sort_values("inner_mesh")
    ls = "-" if surf == "hybrid" else "--"
    for c in ("A", "B", "C"):
        ax.plot(s.inner_mesh, s[f"eta_{c}_Pas"] * 1e3, ls, marker=mk, color=COL[c], ms=4, label=f"{c}, {surf}")
    f = cons[(cons.surface == surf) & ((cons.linewidths == "fixed_at_2x2x2") | (cons.inner_mesh == 2))].sort_values("inner_mesh")
    f = f.drop_duplicates("inner_mesh", keep="last")
    ax.plot(f.inner_mesh, f.eta_C_Pas * 1e3, ls, marker=mk, color=COL["C"], mfc="white", ms=4, alpha=0.7,
            label=f"C, {surf}, linewidths fixed")
ax.set_xticks([2, 4, 8, 12])
ax.set_xlabel("inner mesh $n$ ($n^3$ q-points)", fontsize=8)
ax.set_ylabel(r"$\eta_{xyxy}(300\,\mathrm{K})$ ($10^{-3}$ Pa s)", fontsize=8)
ax.set_title("(c) inner-mesh sequences (finite-mesh model results)", fontsize=8.5)
ax.legend(fontsize=5.5, ncol=2, loc="upper right")
for a_ in axs:
    a_.tick_params(labelsize=7)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf")

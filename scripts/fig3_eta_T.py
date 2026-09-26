#!/usr/bin/env python3
"""Fig. 3 — eta_xyxy(T) for SrTiO3 on the production model (hybrid model tut_z_od1, SELF_OFFDIAG = 1, inner mesh 12^3,
construction C), decomposed by the bare-omega_0 class of each renormalised mode's partner.

Data provenance (committed CSVs only): data/processed/eta_SrTiO3.csv <- scripts/compute_eta_SrTiO3.py.
Only the temperatures with converged unstrained and strained SCPH solutions of the hybrid model appear in the CSV;
if that is 300 K alone the figure is not produced (the paper then reports 300 K only). Flags rendered: T != 300 K
(linewidths validated at 300 K only, shaded); T above the Vogt series (298 K) uses the production-surface TO1
frequency and the soft-manifold linewidth (open markers).
Writes figures/fig3_eta_T.{png,pdf}.
Usage: uv run python scripts/fig3_eta_T.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "processed" / "eta_SrTiO3.csv"
OUT = REPO / "figures" / "fig3_eta_T"

rows = [line.split(",") for line in SRC.read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("T_K")]
T = np.array([float(r[0]) for r in rows])
eta = np.array([float(r[1]) for r in rows]) * 1e3
imag = np.array([float(r[2]) for r in rows]) * 1e3
b0 = np.array([float(r[3]) for r in rows]) * 1e3
b50 = np.array([float(r[4]) for r in rows]) * 1e3
b100 = np.array([float(r[5]) for r in rows]) * 1e3
b175 = np.array([float(r[6]) for r in rows]) * 1e3
g_sec = np.array([float(r[7]) for r in rows]) * 1e3
vogt_ok = np.array([r[9] == "1" for r in rows])
if len(T) < 2:
    print(f"only T = {T.tolist()} K available: Fig. 3 is not produced (300 K only is reported)")
    raise SystemExit(0)
order = np.argsort(T)
T, eta, imag, b0, b50, b100, b175, g_sec, vogt_ok = (a[order] for a in (T, eta, imag, b0, b50, b100, b175, g_sec, vogt_ok))

fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.axvspan(T.min() - 10, 290, color="0.92", zorder=0)
ax.axvspan(310, T.max() + 10, color="0.92", zorder=0)
ax.text(0.985, 0.02, "shaded: linewidth method validated at 300 K only",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.35")
ax.stackplot(T, b100 + b175, b50, b0 + imag, g_sec,
             labels=[r"partner $\omega_0 \geq 100$ cm$^{-1}$", r"partner $50 \leq \omega_0 < 100$ cm$^{-1}$",
                     r"partner $\omega_0 < 50$ cm$^{-1}$ (incl. imaginary)", r"$\Gamma$ TO1 triplet (Vogt)"],
             colors=["#4878a8", "#6aa86a", "#a84848", "#c8a848"], alpha=0.85)
ax.plot(T, eta, "k-", lw=1.6, label=r"$\eta_{xyxy}$ total (projected SCPH coupling)")
if vogt_ok.any():
    ax.plot(T[vogt_ok], eta[vogt_ok], "ko", ms=5, label="filled: Vogt soft-mode parameters used")
if (~vogt_ok).any():
    ax.plot(T[~vogt_ok], eta[~vogt_ok], "ko", ms=6, mfc="white", label="open: above the Vogt series;\nmodel TO1 frequency used")
ax.set_xlabel("T (K)")
ax.set_ylabel(r"$\eta_{xyxy}$ ($10^{-3}$ Pa s)")
ax.set_xlim(T.min() - 10, T.max() + 10)
ax.set_ylim(0, eta.max() * 1.12)
ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
ax.set_title(r"SrTiO$_3$, hybrid model (inner mesh $12^3$); converged temperatures only", fontsize=8)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf  (eta(300K) = {eta[T == 300][0]:.4f} x 1e-3 Pa s)")

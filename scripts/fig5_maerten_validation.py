#!/usr/bin/env python3
"""Fig. 5 — experimental adjudication of eta(300 K) against the GHz
Brillouin damping of Maerten et al. (arXiv:1810.00381; no published
journal version exists as of 2026-07-24 — cite as preprint).

This validation figure has no slot in the original 4-figure skeleton plan
(Figs 1-4 = spectra/eta(T)/isotope/near-T_C); it is added as Fig. 5.

Data provenance (committed CSVs only, no hand-edited data):
  data/processed/maerten2018_damping_300K.csv  (transcribed text values +
    conventions, see its header)
  data/processed/eta_SrTiO3.csv  (our eta(300 K), generation-3 assembly)

Physics: amplitude damping rate of an acoustic mode of wavevector q in
the Akhiezer regime (their Fig. 6 CONFIRMS Gamma ~ q^2 at 300 K over
~2 decades in q): Gamma_amp = alpha*v = omega^2 eta/(2 rho v^2)
= eta q^2/(2 rho). Measured rho = 5110 kg/m^3 (Bell & Rupprecht 1963).
Caveat rendered on the figure: their LA phonons probe the longitudinal
component eta_xxxx; ours is the shear eta_xyxy (order-of-magnitude
comparison; same-polarization sharpening = deferred uniaxial pair).

Writes figures/fig5_maerten_validation.{png,pdf}.
Usage: uv run python scripts/fig5_maerten_validation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures" / "fig5_maerten_validation"
RHO = 5110.0  # kg/m^3, measured (Bell & Rupprecht 1963)

# our eta(300 K)
for line in (REPO / "data" / "processed" / "eta_SrTiO3.csv").read_text().splitlines():
    if line.startswith("300,"):
        ETA_OURS = float(line.split(",")[1])

# Maerten reference values
meas = {}
for line in (REPO / "data" / "processed" / "maerten2018_damping_300K.csv").read_text().splitlines():
    if line.startswith("#") or line.startswith("label"):
        continue
    parts = line.split(",")
    meas[parts[0]] = (float(parts[1]), float(parts[2]))


def gamma_ghz(eta, q_um):
    q = q_um * 1e6  # 1/m
    return eta * q * q / (2.0 * RHO) / 1e9


q_axis = np.logspace(-0.5, 2.1, 200)  # um^-1

fig, ax = plt.subplots(figsize=(5.2, 4.0))
ax.loglog(q_axis, gamma_ghz(ETA_OURS, q_axis), "-", color="#4878a8", lw=1.8,
          label=rf"this work: $\eta_{{xyxy}}$ = {ETA_OURS * 1e3:.2f}$\times10^{{-3}}$ Pa s"
                "\n" r"($\Gamma = \eta q^2/2\rho$, Akhiezer)")

# old pre-registered decade, shown as excluded
ax.fill_between(q_axis, gamma_ghz(1e-4, q_axis), gamma_ghz(1e-3, q_axis),
                color="#c86a6a", alpha=0.25,
                label=r"old $10^{-4}$–$10^{-3}$ Pa s decade"
                      "\n(excluded by the measurement)")

# their measured band at q ~ 52-58 um^-1
q_lo, q_hi = 50, 60
ax.fill_between([q_lo, q_hi], [meas["band_min"][1]] * 2, [meas["band_max"][1]] * 2,
                color="#6aa86a", alpha=0.45,
                label="Maerten et al. 300 K: 1–2 GHz\n(measured band, LA [100])")
ax.plot(*[[meas["bulk_STO_BS"][0]], [meas["bulk_STO_BS"][1]]], "s",
        color="#2a682a", ms=7, label="bulk STO, Brillouin (~1 GHz)")
ax.plot(*[[meas["LSMO_TDBS_cubic"][0]], [meas["LSMO_TDBS_cubic"][1]]], "D",
        color="#2a682a", ms=6, mfc="white", label="LSMO sample, TDBS (~2 GHz)")

# our prediction at their wavevector
q_ref = meas["bulk_STO_BS"][0]
g_pred = gamma_ghz(ETA_OURS, q_ref)
ax.plot([q_ref], [g_pred], "*", color="#a84848", ms=14,
        label=f"this work at $q$ = {q_ref:.0f} " r"$\mu$m$^{-1}$: "
              f"{g_pred:.2f} GHz")

ax.set_xlabel(r"acoustic wavevector $q$ ($\mu$m$^{-1}$)")
ax.set_ylabel(r"damping rate $\Gamma$ (GHz)")
ax.set_title(r"$\eta$(300 K) vs GHz Brillouin damping "
             "(Maerten et al., arXiv:1810.00381)", fontsize=8.5)
ax.text(0.03, 0.97,
        "their LA phonons probe $\\eta_{xxxx}$; ours is $\\eta_{xyxy}$\n"
        "(order-of-magnitude comparison; their Fig. 6\n"
        "confirms the $q^2$ law at 300 K)",
        transform=ax.transAxes, fontsize=6.5, va="top", color="0.35")
ax.legend(fontsize=6.3, loc="lower right", framealpha=0.95)
ax.set_xlim(q_axis[0], q_axis[-1])
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=220)
fig.savefig(f"{OUT}.pdf")
print(f"-> {OUT}.png/.pdf  (prediction at q={q_ref:.0f} um^-1: {g_pred:.2f} GHz; "
      f"measured band 1-2 GHz)")

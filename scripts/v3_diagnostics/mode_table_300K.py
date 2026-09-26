#!/usr/bin/env python3
"""flat 300 K mode table for SrTiO3 (the input of every later analysis).

One row per (q, branch) slot that enters the production sum (19,962 = 11^3 x 15
minus the three acoustic translations at Gamma; audit 0.D.1). Columns:

  iq                    q index on the 11^3 Gamma-centred fractional mesh (0-based,
                        iq = 121 i + 11 j + k, q = (i, j, k)/11 in reciprocal-lattice units)
  qa, qb, qc            fractional q label
  weight                mesh weight, 1/1331 (full mesh, no symmetry reduction)
  branch                1-based branch index in the bare (matdyn) ordering
  omega0_cm1            bare PBEsol harmonic frequency (imaginary printed negative)
  omega_r_cm1           renormalised frequency used in the sum (eigenvalue map to the
                        SCPH-coupled ALAMODE 300 K spectrum; Vogt value / branch floor
                        applied in the soft sector exactly as in production)
  Gamma_anh_cm1         anharmonic HWHM, cm-1 (character-aware map; Vogt in the Gamma sector)
  Gamma_iso_dos_f015_cm1   Tamura HWHM at f = 0.15, production total-DOS form
  Gamma_iso_exact_f015_cm1 Tamura HWHM at f = 0.15, exact O-site projection (audit 0.B)
  gamma_xy_used         Grueneisen component as consumed by production:
                        Route S: -D/(2 omega0^2); Route H / Gamma sector: -D/(2 omega_r^2)
  D_domega2_ds_cm2      strained-cell eigenvalue derivative d(omega^2)/ds, central
                        difference over s = +-0.005 (the manuscript's Lambda = d omega^2/d eps_xy)
  sector                routeS / routeH_stable / routeH_unstable / gamma_sector
  n_n_plus_1            Bose factor n(n+1) at omega_r, 300 K
  tau_exact_ps          production kernel (Gamma^2+omega^2)/(2 Gamma omega^2)
  tau_stress_ps         x^2-stress kernel 1/(2 Gamma) + 2 Gamma/omega^2 (audit 0.C)
  eta_contrib_Pas       per-mode contribution with the production kernel
  rank, cum_frac        rank by descending contribution and cumulative fraction of eta
                        up to and including this mode in that ranking
  Gamma_over_omega      Gamma_anh / omega_r
  degenerate            1 if the bare reference frequency sits in a degenerate multiplet
  omega_r_nearest_other_branch_cm1  nearest other branch at the same q (renormalised), for
                        the interbranch-coherence diagnostic 

GRUENEISEN CONVENTION (audit 0.A verdict): the strained cells apply eps_xy =
eps_yx = s, and D is the derivative along that path; gamma_xy_used is
therefore -d ln omega/ds = 2 x the tensor component gamma_xy = -d ln
omega/d eps_xy of the nine-component convention in which the Green-Kubo
eta_xyxy equals the Newtonian eta_44. The table records the quantity
production uses; the tensor-convention value is gamma_xy_used/2 and the
corresponding eta is eta_contrib_Pas/4.

Writes: data/processed/v3_diagnostics/mode_table_SrTiO3_300K.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as K_B
from scipy.constants import hbar as HBAR

from diag_common import CM1, N_Q, REPO, DIAG_DIR, mode_table, tau_two_pole_exact, tau_two_pole_stress
from latvisc.isotope import mass_variance_g2  # noqa: E402

T = 300
F15 = 0.15


def main() -> None:
    eta, sec, flags, details = mode_table(T)
    print(f"eta(300 K) = {eta:.6e} Pa s, {len(details)} modes, flags {flags}")
    iso = pd.read_csv(DIAG_DIR / "isotope_rates_SrTiO3_300K.csv", comment="#")
    iso = iso.set_index(["iq", "branch"])
    g2 = mass_variance_g2([15.999, 17.99916], [1 - F15, F15])

    # nearest other branch at the same q (renormalised frequencies)
    by_q: dict[int, list] = {}
    for d in details:
        by_q.setdefault(d["iq"], []).append((d["branch"], d["omega_r"]))

    recs = []
    for d in details:
        iq, br = d["iq"], d["branch"]
        others = [w for b, w in by_q[iq] if b != br]
        nearest = min(others, key=lambda w: abs(w - d["omega_r"])) if others else np.nan
        w = d["omega_r"] * CM1
        lw = d["gamma_hwhm"] * CM1
        x = HBAR * w / (K_B * T)
        n = 1.0 / np.expm1(x)
        r = iso.loc[(iq, br)]
        recs.append({
            "iq": iq, "qa": (iq // 121) / 11, "qb": ((iq // 11) % 11) / 11, "qc": (iq % 11) / 11,
            "weight": 1.0 / N_Q, "branch": br,
            "omega0_cm1": d["omega0"], "omega_r_cm1": d["omega_r"],
            "Gamma_anh_cm1": d["gamma_hwhm"],
            "Gamma_iso_dos_f015_cm1": g2 * r["rate_dos_per_g2_rad_s"] / 2.0 / CM1,
            "Gamma_iso_exact_f015_cm1": g2 * r["rate_exact_per_g2_rad_s"] / 2.0 / CM1,
            "gamma_xy_used": d["gruneisen"], "D_domega2_ds_cm2": d["D"],
            "sector": d["sector"], "n_n_plus_1": n * (n + 1),
            "tau_exact_ps": float(tau_two_pole_exact(w, lw)) * 1e12,
            "tau_stress_ps": float(tau_two_pole_stress(w, lw)) * 1e12,
            "eta_contrib_Pas": d["eta_contrib"],
            "Gamma_over_omega": d["gamma_hwhm"] / d["omega_r"],
            "degenerate": int(d["degenerate"]),
            "omega_r_nearest_other_branch_cm1": nearest,
        })
    df = pd.DataFrame(recs)
    order = np.argsort(-df["eta_contrib_Pas"].values)
    rank = np.empty(len(df), dtype=int)
    rank[order] = np.arange(1, len(df) + 1)
    cum = np.empty(len(df))
    cum[order] = np.cumsum(df["eta_contrib_Pas"].values[order]) / eta
    df["rank"] = rank
    df["cum_frac"] = cum
    assert abs(df["eta_contrib_Pas"].sum() / eta - 1) < 1e-12

    out = DIAG_DIR / "mode_table_SrTiO3_300K.csv"
    header = [
        "# mode_table_SrTiO3_300K.csv - (mode_table_300K.py); one row per production mode slot",
        f"# eta_xyxy(300 K) = {eta:.6e} Pa s (production kernel); sectors: "
        + ", ".join(f"{k} {v:.4e}" for k, v in sec.items()),
        "# Units: frequencies/linewidths cm-1 (HWHM), D cm-2 per unit path parameter s, tau ps, eta Pa s.",
        "# GRUENEISEN CONVENTION (audit 0.A): eps_xy = eps_yx = s in the strained cells; gamma_xy_used =",
        "#   -d ln omega/ds = 2 x tensor gamma_xy (nine-component convention, eta_xyxy = eta_44).",
        "#   Tensor-convention values: gamma_xy_used/2, eta_contrib_Pas/4.",
        "# Gamma_iso_*: Tamura HWHM at f = 0.15 (g2 = %.6e), sigma = 10 cm-1; dos = production form," % g2,
        "#   exact = O-site eigenvector projection with bare QE eigenvectors (audit 0.B).",
        "# tau_exact = production two-pole kernel; tau_stress = x^2-stress kernel (audit 0.C).",
        "# rank/cum_frac: ranking by descending eta_contrib_Pas.",
    ]
    with open(out, "w") as fh:
        fh.write("\n".join(header) + "\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    print(f"-> {out.relative_to(REPO)}  ({len(df)} rows)")
    print("sector counts:", df["sector"].value_counts().to_dict())
    print("top-20 modes carry %.3f of eta; 50%% of eta reached at rank %d; 90%% at rank %d"
          % (df.sort_values("rank").head(20)["eta_contrib_Pas"].sum() / eta,
             int(df.loc[df.cum_frac >= 0.5, "rank"].min()), int(df.loc[df.cum_frac >= 0.9, "rank"].min())))


if __name__ == "__main__":
    main()

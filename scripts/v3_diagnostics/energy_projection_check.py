#!/usr/bin/env python3
"""numerical check of the conserved-energy projection (R3.1).

The projection of the shear stress onto the conserved energy is
proportional to S1 = sum_qs (hbar w)^2 n(n+1) gamma_xy(qs), which vanishes
by cubic symmetry on a symmetric mesh (the 11^3 Gamma-centred fractional
mesh is invariant under the full cubic group). Reported: S1 relative to
S_abs = sum (hbar w)^2 n(n+1) |gamma_xy|, with the production weights
(renormalised omega_r) and with bare-frequency weights, per sector. The
(gamma_xx - gamma_yy) sum requires the uniaxial strain pair  and
is not available for SrTiO3; noted as a gap.

Within a degenerate multiplet the individual gamma values are gauge
dependent but their sum (the trace of the strain perturbation on the
subspace) is not, so the multiplet-summed S1 is the meaningful residual.

Reads : data/processed/v3_diagnostics/mode_table_SrTiO3_300K.csv
Writes: data/processed/v3_diagnostics/energy_projection_check_300K.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as K_B
from scipy.constants import hbar as HBAR

from diag_common import CM1, REPO, DIAG_DIR

T = 300.0


def main() -> None:
    df = pd.read_csv(DIAG_DIR / "mode_table_SrTiO3_300K.csv", comment="#")
    g = df["gamma_xy_used"].values
    w_r = df["omega_r_cm1"].values * CM1
    n_r = 1 / np.expm1(HBAR * w_r / (K_B * T))
    wt_r = (HBAR * w_r) ** 2 * n_r * (n_r + 1)
    stable = df["omega0_cm1"].values > 5.0
    w_0 = np.where(stable, df["omega0_cm1"].values, np.nan) * CM1
    n_0 = 1 / np.expm1(HBAR * w_0 / (K_B * T))
    wt_0 = np.where(stable, (HBAR * w_0) ** 2 * n_0 * (n_0 + 1), 0.0)
    rows = ["weights,sector,S1,S_abs,S1_over_S_abs,S1_over_sum_wt_gamma2_sqrt"]
    for label, wt in [("renormalised", wt_r), ("bare_stable_only", wt_0)]:
        for sector in ["routeS", "routeH_stable", "routeH_unstable", "gamma_sector", "ALL"]:
            m = np.ones(len(df), bool) if sector == "ALL" else (df["sector"] == sector).values
            s1 = np.sum(wt[m] * g[m])
            sabs = np.sum(wt[m] * np.abs(g[m]))
            s2 = np.sqrt(np.sum(wt[m] * g[m] ** 2) * np.sum(wt[m]))
            print(f"[{label:16s}] {sector:16s} S1/S_abs = {s1 / sabs:+.3e}   S1/sqrt(sum wt gamma^2 * sum wt) = {s1 / s2:+.3e}")
            rows.append(f"{label},{sector},{s1:.6e},{sabs:.6e},{s1 / sabs:.6e},{s1 / s2:.6e}")
    # mesh symmetry check: q and -q both present
    q = df[["qa", "qb", "qc"]].drop_duplicates().values
    qs = {tuple(np.round(x, 6)) for x in q}
    missing = sum(1 for x in q if tuple(np.round((-x) % 1.0, 6)) not in qs)
    print(f"mesh inversion check: {len(qs)} q-points, {missing} without a -q partner")
    out = DIAG_DIR / "energy_projection_check_300K.csv"
    out.write_text("# energy_projection_check_300K.csv - (energy_projection_check.py)\n"
                   "# S1 = sum (hbar w)^2 n(n+1) gamma_xy, S_abs = same with |gamma_xy|; 300 K; gamma in the\n"
                   "# production convention (the ratio is convention independent). gamma_xx - gamma_yy: NOT AVAILABLE\n"
                   "# for SrTiO3 (needs the uniaxial pair, ).\n"
                   + "\n".join(rows) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

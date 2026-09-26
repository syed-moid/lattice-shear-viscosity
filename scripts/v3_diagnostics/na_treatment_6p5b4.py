#!/usr/bin/env python3
"""na_treatment_6p5b4.py - non-analytic treatment, odd-in-strain part, zone-wide.

Lambda_C by the eigenvalue-matched strained route (match_strain_pair_by_overlap on the unstrained
renormalised basis) from
  (i)  the dynmat.py builder (rgd_blk non-analytic term, matdyn-consistent)          -> Lambda_C^rgd
  (ii) anphon's own NONANALYTIC = 3 Ewald frequencies and eigenvectors on the same strained renormalised
       sets (the PRINTEVEC 11^3 sets of )                                    -> Lambda_C^ewald
and their difference per mode; eta-weighted through the per-mode prefactors of the table
(eta_C = pref * Lambda^2), i.e. the effect on eta_C of the ~0.3 cm^-1 sheared-cell discrepancy.
Also the direct comparison of the strained frequencies (max |omega_rgd - omega_ewald|).
Writes data/processed/v3_diagnostics/na_treatment_6p5b4.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from coupling_kernel import H_STEP, Sets, signed_sq  # noqa: E402
from dynmat import diagonalise  # noqa: E402
from latvisc.gruneisen import match_strain_pair_by_overlap  # noqa: E402

OWN = REPO / "data" / "raw" / "alamode_sto" / "own_surface_6p2"


def displacement(E, masses):
    nat = len(masses)
    return (E.T / np.sqrt(np.repeat(masses, 3))[None, :]).reshape(-1, nat, 3)


def main():
    variant = "od1"
    sets = Sets(variant)
    masses = sets.masses
    ref = np.load(OWN / f"mesh11_own_reference_{variant}_300K.npz")
    plus = np.load(OWN / f"mesh11_own_shear_xy_p005_{variant}_300K.npz")
    minus = np.load(OWN / f"mesh11_own_shear_xy_m005_{variant}_300K.npz")
    modes = pd.read_csv(DIAG / "eta_constructions_b5_modes_od1_mesh11.csv", comment="#")
    pref = modes.set_index(["iq", "nu"]).pref_eta_per_Lambda2
    lamC = modes.set_index(["iq", "nu"]).Lambda_C
    qs = ref["q_frac"]
    h = H_STEP["005"]
    rows = []
    dw_max = 0.0
    for iq in range(len(qs)):
        q = qs[iq]
        # (ii) anphon Ewald: frequencies/eigenvectors from the PRINTEVEC sets; eigenvectors as displacement patterns
        wr_a, er_a = ref["omega_cm1"][iq], ref["evec"][iq]          # [mode, comp]
        wp_a, ep_a = plus["omega_cm1"][iq], plus["evec"][iq]
        wm_a, em_a = minus["omega_cm1"][iq], minus["evec"][iq]
        nat = len(masses)
        va = lambda e: (e / np.sqrt(np.repeat(masses, 3))[None, :]).reshape(-1, nat, 3)
        mp, mm = match_strain_pair_by_overlap(wr_a, va(er_a), wp_a, va(ep_a), wm_a, va(em_a), masses, 0.5)
        lam_ew = (signed_sq(mp) - signed_sq(mm)) / (2.0 * h)
        # (i) builder with rgd_blk on the same sets, same matched route, unstrained basis from the builder
        wr, Er = diagonalise(sets.D_SCPH(q))
        wp, Ep = diagonalise(sets.D_SCPH(q, "shear_xy_p005"))
        wm, Em = diagonalise(sets.D_SCPH(q, "shear_xy_m005"))
        mp2, mm2 = match_strain_pair_by_overlap(wr, displacement(Er, masses), wp, displacement(Ep, masses), wm, displacement(Em, masses), masses, 0.5)
        lam_rgd = (signed_sq(mp2) - signed_sq(mm2)) / (2.0 * h)
        dw_max = max(dw_max, float(np.abs(np.sort(wp) - np.sort(wp_a)).max()), float(np.abs(np.sort(wm) - np.sort(wm_a)).max()))
        for nu in range(15):
            key = (iq, nu + 1)
            if key not in pref.index:
                continue
            rows.append({"iq": iq, "nu": nu + 1, "omega_r_cm1": wr[nu], "Lambda_C_proj": lamC[key],
                         "Lambda_C_matched_rgd": lam_rgd[nu], "Lambda_C_matched_ewald": lam_ew[nu],
                         "pref": pref[key]})
    df = pd.DataFrame(rows)
    df["delta_Lambda_odd"] = df.Lambda_C_matched_ewald - df.Lambda_C_matched_rgd
    e_rgd = float((df.pref * df.Lambda_C_matched_rgd ** 2).sum())
    e_ew = float((df.pref * df.Lambda_C_matched_ewald ** 2).sum())
    e_proj = float((df.pref * df.Lambda_C_proj ** 2).sum())
    print(f"max |omega_strained(rgd) - omega_strained(ewald)| over 11^3 = {dw_max:.3f} cm-1")
    print(f"eta_C (11^3, od1): projected {e_proj:.4e}; matched rgd {e_rgd:.4e}; matched ewald {e_ew:.4e} Pa s; "
          f"ewald/rgd - 1 = {e_ew / e_rgd - 1:+.4f}")
    w = df.pref * df.Lambda_C_matched_rgd ** 2
    rel = (df.delta_Lambda_odd.abs() / df.Lambda_C_matched_rgd.abs().clip(lower=1.0))
    print(f"eta-weighted median |dLambda_odd / Lambda| = {np.average(rel, weights=w):.4f} (mean), "
          f"|dLambda_odd| median {df.delta_Lambda_odd.abs().median():.1f} cm-2, p90 {df.delta_Lambda_odd.abs().quantile(0.9):.1f}")
    out = DIAG / "na_treatment_6p5b4.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write(f"# (na_treatment_6p5b4.py), od1, 11^3, h = 0.010: eta_C matched-rgd {e_rgd:.6e}, matched-ewald {e_ew:.6e},\n"
                 f"# projected {e_proj:.6e} Pa s; max strained-frequency difference rgd vs ewald {dw_max:.4f} cm-1.\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)


if __name__ == "__main__":
    main()

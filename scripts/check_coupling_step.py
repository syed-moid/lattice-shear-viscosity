#!/usr/bin/env python3
"""Strain-step statistics of the production coupling Lambda_C (SrTiO3, 300 K, 11^3), alongside the
bare-surface statistics of check_shear_nonlinearity.py / check_5point_richardson.py.

Per renormalised mode: Lambda_C from the +-0.005 pair (h = 0.010, production) and from the +-0.010
pair (h = 0.020); asymptotic when |Lambda(0.020)/Lambda(0.010) - 1| < 0.25 with the same sign; the
Richardson combination (4 Lambda_010 - Lambda_020)/3 formed on the derivative. Reported per bare-
omega_0 bin of the partner mode with the eta weight of the non-asymptotic class.
Writes: data/processed/reports/coupling_step_SrTiO3.csv
Usage: uv run python scripts/check_coupling_step.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T_K = 300
BINS = [("imaginary", -np.inf, 0), ("[0-25)", 0, 25), ("[25-50)", 25, 50), ("[50-100)", 50, 100), ("[100-150)", 100, 150),
        ("[150-175)", 150, 175), ("[175-300)", 175, 300), ("[300-inf)", 300, np.inf)]


def main() -> None:
    vogt, _ = eta_mod.load_vogt()
    surface = eta_mod.load_construction_surface(T_K, with_h020=True)
    modes = eta_mod.construction_modes(T_K, 11, surface)
    if "lam_C_h020" not in modes[0]:
        raise SystemExit("the +-0.010 strained SCPH sets are not available")
    eta, _, _, details = eta_mod.assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt, return_details=True)
    by = {(d["iq"], d["nu"]): d for d in details}
    recs = []
    for m in modes:
        d = by.get((m["iq"], m["nu"]))
        if d is None or m["lam_C"] == 0:
            continue
        r = m["lam_C_h020"] / m["lam_C"]
        asym = abs(r - 1.0) < 0.25 and np.sign(m["lam_C_h020"]) == np.sign(m["lam_C"])
        recs.append((m["omega0"], d["eta_contrib"], r, asym, m["lam_C"], m["lam_C_h020"], m["lam_C_rich"]))
    om = np.array([x[0] for x in recs]); w = np.array([x[1] for x in recs]); r = np.array([x[2] for x in recs])
    asym = np.array([x[3] for x in recs]); l1 = np.array([x[4] for x in recs]); l2 = np.array([x[5] for x in recs]); lr = np.array([x[6] for x in recs])
    pref = w / l1 ** 2
    lines = ["omega0_bin,n_modes,eta_C_h010_Pas,frac_modes_asymptotic,eta_weight_non_asymptotic,"
             "median_ratio_h020_over_h010,eta_C_h020_Pas,eta_C_richardson_asymptotic_only_Pas,eta_C_richardson_all_Pas"]
    print(f"eta_C(300 K) = {eta:.4e} Pa s; asymptotic modes {100 * asym.mean():.1f} %, eta weight of the non-asymptotic class "
          f"{100 * w[~asym].sum() / eta:.1f} %")
    for label, lo, hi in BINS + [("all", -np.inf, np.inf)]:
        m = (om >= lo) & (om < hi)
        if m.sum() == 0:
            continue
        e = w[m].sum()
        e20 = (pref[m] * l2[m] ** 2).sum()
        emix = (pref[m] * np.where(asym[m], lr[m], l1[m]) ** 2).sum()
        eall = (pref[m] * lr[m] ** 2).sum()
        lines.append(f"{label},{m.sum()},{e:.6e},{asym[m].mean():.4f},{(w[m][~asym[m]].sum() / e if e else float('nan')):.4f},"
                     f"{np.median(r[m]):.4f},{e20:.6e},{emix:.6e},{eall:.6e}")
        print(f"  {label:>10}: n={m.sum():5d} eta={e:.3e} asymptotic {100 * asym[m].mean():5.1f} % (eta-weight non-asym "
              f"{100 * (w[m][~asym[m]].sum() / e if e else 0):5.1f} %)  h020/h010 median {np.median(r[m]):.3f}  "
              f"eta: h020 {e20:.3e}, rich-asym {emix:.3e}, rich-all {eall:.3e}")
    out = REPO / "data" / "processed" / "reports" / "coupling_step_SrTiO3.csv"
    header = ["# coupling_step_SrTiO3.csv - produced by scripts/check_coupling_step.py",
              "# strain-step statistics of the production coupling Lambda_C (hybrid model tut_z_od1, SELF_OFFDIAG = 1) at 300 K, 11^3;",
              "# h = 0.010 (production) vs h = 0.020; asymptotic = |ratio - 1| < 0.25 and same sign; Richardson on the derivative."]
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(header) + "\n" + "\n".join(lines) + "\n")
    tmp.rename(out)
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

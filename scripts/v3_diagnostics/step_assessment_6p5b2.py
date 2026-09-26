#!/usr/bin/env python3
"""step_assessment_6p5b2.py - strain-step assessment of construction C, zone-wide (11^3).

Reads the per-mode file of eta_constructions.py (with Lambda_C at h = 0.010 and h = 0.020 and the
Richardson combination formed on the derivative before squaring).  Classifies each renormalised mode as
asymptotic when |Lambda(0.020)/Lambda(0.010) - 1| < 0.25 and the two have the same sign, and reports
eta_C from (i) h = 0.010 only, (ii) Richardson where asymptotic and 0.010 elsewhere, (iii) Richardson
everywhere (sensitivity), plus the eta weight of the non-asymptotic class, per bare-omega_0 bin.
Usage: uv run python step_assessment_6p5b2.py <modes csv> [label]
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    path = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else path.stem
    m = pd.read_csv(path, comment="#")
    r = m.Lambda_C_h020 / m.Lambda_C
    asym = (np.abs(r - 1.0) < 0.25) & (np.sign(m.Lambda_C_h020) == np.sign(m.Lambda_C)) & (m.Lambda_C != 0)
    m["asymptotic"] = asym
    pref = m.pref_eta_per_Lambda2
    eta_010 = float((pref * m.Lambda_C ** 2).sum())
    eta_mixed = float((pref * np.where(asym, m.Lambda_C_rich, m.Lambda_C) ** 2).sum())
    eta_rich = float((pref * m.Lambda_C_rich ** 2).sum())
    eta_020 = float((pref * m.Lambda_C_h020 ** 2).sum())
    w_non = float((pref * m.Lambda_C ** 2)[~asym].sum() / eta_010)
    rows = [{"label": label, "eta_C_h010": eta_010, "eta_C_richardson_asymptotic_only": eta_mixed,
             "eta_C_richardson_all_sensitivity": eta_rich, "eta_C_h020_only": eta_020,
             "eta_weight_non_asymptotic": w_non, "frac_modes_asymptotic": float(asym.mean())}]
    print(pd.DataFrame(rows).to_string())
    binrows = []
    for b, g in m.groupby("omega0_bin"):
        e = (g.pref_eta_per_Lambda2 * g.Lambda_C ** 2)
        binrows.append({"omega0_bin": b, "n": len(g), "eta_C_h010": e.sum(), "frac_asymptotic_modes": g.asymptotic.mean(),
                        "eta_weight_non_asymptotic": e[~g.asymptotic].sum() / e.sum() if e.sum() > 0 else np.nan,
                        "eta_C_rich_asym_only": (g.pref_eta_per_Lambda2 * np.where(g.asymptotic, g.Lambda_C_rich, g.Lambda_C) ** 2).sum(),
                        "eta_C_rich_all": (g.pref_eta_per_Lambda2 * g.Lambda_C_rich ** 2).sum()})
    bd = pd.DataFrame(binrows)
    print(bd.to_string())
    out = path.parent / f"step_assessment_6p5b2_{label}.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# (step_assessment_6p5b2.py): strain-step assessment of Lambda_C on 11^3; asymptotic =\n"
                 "# |Lambda(0.020)/Lambda(0.010)-1| < 0.25 and same sign; Richardson formed on the derivative before squaring.\n")
        pd.DataFrame(rows).to_csv(fh, index=False, float_format="%.6g")
        fh.write("\n")
        bd.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

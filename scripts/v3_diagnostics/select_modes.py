#!/usr/bin/env python3
"""select_modes.py - selection from data/processed/v3_diagnostics/mode_table_SrTiO3_300K_v4.csv (production
model at the 6.0.4 stop point: tutorial surface, direct eigenvector matching, 11^3 mesh).

Groups
  P   : the 20 highest-eta mode slots with bare omega_0 in [0, 50) cm^-1 (reported per bin
        [0,25) / [25,50)), plus every mode at one representative of each of the 5 q shells
        nearest Gamma on the 11^3 mesh ((1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0) / 11);
  C1  : the 10 highest-eta slots with omega_0 in [100, 125);
  C2  : the 5 highest-eta Route-S slots with omega_0 in [225, 275) (around 250 cm^-1);
  C3  : the 5 highest-eta slots at R = (5,5,5)/11 (the mesh point at/nearest R).
Star-equivalent q of every listed slot share the same numbers by cubic symmetry; the
representative q used in is listed in `q_rep` (the first member in the table order).
Writes data/processed/v3_diagnostics/selection_6p3a.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
TABLE = DIAG / "mode_table_SrTiO3_300K_v4.csv"
OUT = DIAG / "selection_6p3a.csv"
SHELLS = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0)]
COLS = ["iq", "qa", "qb", "qc", "branch", "omega0_cm1", "omega_r_cm1", "sector", "eta_contrib_Pas",
        "bare_renorm_overlap", "Gamma_anh_cm1", "gamma_xy_tensor"]


def main():
    df = pd.read_csv(TABLE, comment="#")
    eta_total = df.eta_contrib_Pas.sum()
    parts = []
    p = df[(df.omega0_cm1 >= 0) & (df.omega0_cm1 < 50)].sort_values("eta_contrib_Pas", ascending=False).head(20).copy()
    p["group"] = ["P[0,25)" if w < 25 else "P[25,50)" for w in p.omega0_cm1]
    parts.append(p)
    for shell in SHELLS:
        q = np.array(shell) / 11.0
        m = np.isclose(df.qa, q[0]) & np.isclose(df.qb, q[1]) & np.isclose(df.qc, q[2])
        s = df[m].copy()
        s["group"] = f"P_shell({shell[0]},{shell[1]},{shell[2]})/11"
        parts.append(s)
    c1 = df[(df.omega0_cm1 >= 100) & (df.omega0_cm1 < 125)].sort_values("eta_contrib_Pas", ascending=False).head(10).copy()
    c1["group"] = "C1[100,125)"
    parts.append(c1)
    c2 = df[(df.sector == "routeS") & (df.omega0_cm1 >= 225) & (df.omega0_cm1 < 275)].sort_values(
        "eta_contrib_Pas", ascending=False).head(5).copy()
    c2["group"] = "C2_routeS~250"
    parts.append(c2)
    mr = np.isclose(df.qa, 5 / 11) & np.isclose(df.qb, 5 / 11) & np.isclose(df.qc, 5 / 11)
    c3 = df[mr].sort_values("eta_contrib_Pas", ascending=False).head(5).copy()
    c3["group"] = "C3_R"
    parts.append(c3)
    sel = pd.concat(parts)[COLS + ["group"]]
    sel["eta_share_pct"] = 100.0 * sel.eta_contrib_Pas / eta_total
    # representative q of the cubic star (first member in table order among the selected rows' stars)
    def star_key(row):
        return tuple(sorted(np.round(np.minimum(np.abs([row.qa, row.qb, row.qc]), 1 - np.abs([row.qa, row.qb, row.qc])) * 11).astype(int)))
    sel["star"] = [f"({k[2]},{k[1]},{k[0]})/11" for k in (star_key(r) for r in sel.itertuples())]
    with open(OUT, "w") as fh:
        fh.write("# selection_6p3a.csv - (select_modes.py) from mode_table_SrTiO3_300K_v4.csv;\n"
                 f"# eta_total(300 K) = {eta_total:.6e} Pa s (tutorial surface, direct matching, 11^3 mesh);\n"
                 "# omega_r_cm1 and bare_renorm_overlap are the PRODUCTION (tutorial-surface) values; the \n"
                 "# comparison is done on the own-surface renormalised basis at the same q.  star = cubic star label.\n")
        sel.to_csv(fh, index=False, float_format="%.6g")
    print(sel.groupby("group").agg(n=("branch", "size"), eta=("eta_contrib_Pas", "sum"),
                                   share_pct=("eta_share_pct", "sum")).to_string())
    print("distinct stars:", sorted(set(sel.star)))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

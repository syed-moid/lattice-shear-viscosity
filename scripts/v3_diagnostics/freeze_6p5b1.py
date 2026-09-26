#!/usr/bin/env python3
"""freeze_6p5b1.py - internal SCPH-mesh dependence of construction C (own surface, od1, 300 K).

Surfaces: i2s2 (KMESH_INTERPOLATE 2x2x2 / KMESH_SCPH 2x2x2), i4s4 (4/4), i4s8 (4/8); IFC supercell 2x2x2.
Per surface: SCPH iterations/residual/wall time (runs.csv of the driver); R-point AFD frequency and its
sign; eta_C at 11^3 (eta_constructions_<tag>_zone.csv, same assembly rules); the contribution to eta_C
from the R region (modes whose max-overlap bare partner is bare-imaginary within |q - R| <= 0.15
reduced, i.e. the R4+ branch; and the plain shell |q - R| <= 0.15); kappa(300 K) and the acoustic tau
band from the own 8^3 RTA; TO1/TO2/LO1 and the R manifold vs the anchors.
Writes data/processed/v3_diagnostics/freeze_6p5b1.csv.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from crosscheck_alamode_sto_tau import parse_result  # noqa: E402
from latvisc.coupling import AlamodeSet, diagonalise  # noqa: E402

OWN = REPO / "data" / "raw" / "alamode_sto" / "own_od1"
FC = REPO / "dft" / "qe" / "SrTiO3" / "dispersion_pbesol" / "SrTiO3_pbesol.444.fc"
SURF = {
    "i2s2": {"dir": OWN / "i2s2", "xml": "renorm_reference_i2s2_od1_300K.xml", "rta": "STO_RTA_i2s2_od1_300K",
             "modes": DIAG / "eta_constructions_b5_modes_od1_mesh11.csv", "zone": DIAG / "eta_constructions_b5_zone.csv",
             "kmesh": "2x2x2 / 2x2x2"},
    "i4s4": {"dir": OWN / "i4s4", "xml": "renorm_reference_i4s4_od1_300K.xml", "rta": "STO_RTA_i4s4_od1_300K",
             "modes": DIAG / "eta_constructions_i4s4_modes_od1_mesh11.csv", "zone": DIAG / "eta_constructions_i4s4_zone.csv",
             "kmesh": "4x4x4 / 4x4x4"},
    "i4s8": {"dir": OWN / "i4s8", "xml": "renorm_reference_i4s8_od1_300K.xml", "rta": "STO_RTA_i4s8_od1_300K",
             "modes": DIAG / "eta_constructions_i4s8_modes_od1_mesh11.csv", "zone": DIAG / "eta_constructions_i4s8_zone.csv",
             "kmesh": "4x4x4 / 8x8x8"},
}
R = np.array([0.5, 0.5, 0.5])
ANCH = {"TO1": 88.96, "TO2": 175.0, "LO1": 171.0, "R4+": 52.0, "R5+": 145.0}
BANDS = [(0, 50), (50, 100), (100, 150), (200, 450), (500, 550), (730, 830)]


def qdist(q):
    d = (np.asarray(q) - R + 0.5) % 1.0 - 0.5
    return np.linalg.norm(d, axis=-1)


def main():
    rows = []
    for tag, s in SURF.items():
        rec = {"surface": tag, "KMESH_INTERPOLATE/KMESH_SCPH": s["kmesh"]}
        runs = pd.read_csv(s["dir"] / "runs.csv")
        for _, r in runs[runs.stage == "scph"].iterrows():
            key = {"reference": "ref", "shear_xy_p005": "p005", "shear_xy_m005": "m005"}.get(r["set"], r["set"])
            it = str(r["iterations"])
            it300 = re.search(r"300K:(\d+)", it)
            rec[f"iter_{key}"] = int(it300.group(1)) if it300 else it
            rec[f"resid_{key}"] = r["final_DIFF"]
            rec[f"wall_s_{key}"] = r["wall_s"]
        al = AlamodeSet(s["dir"] / s["xml"], FC)
        wR, _ = diagonalise(al.dynmat(R, asr_onsite=True))
        wG, _ = diagonalise(al.dynmat((0, 0, 0), asr_onsite=True))
        wGx, _ = diagonalise(al.dynmat((0.001, 0, 0), asr_onsite=True))
        rec["R_lowest_cm1"] = float(wR[0]); rec["R_second_cm1"] = float(wR[1]); rec["R_R5+_cm1"] = float(wR[3])
        rec["R_status"] = "imaginary" if wR[0] < 0 else "positive"
        g = np.delete(wG, np.argsort(np.abs(wG))[:3]); gx = np.delete(wGx, np.argsort(np.abs(wGx))[:3])
        rec["TO1_cm1"] = float(g[0]); rec["TO2_cm1"] = float(g[3])
        lo = [v for v in gx if np.abs(g - v).min() > 0.2]
        rec["LO1_cm1"] = float(lo[0]) if lo else np.nan
        for k, v in ANCH.items():
            val = {"TO1": rec["TO1_cm1"], "TO2": rec["TO2_cm1"], "LO1": rec["LO1_cm1"], "R4+": rec["R_lowest_cm1"], "R5+": rec["R_R5+_cm1"]}[k]
            rec[f"{k}_vs_exp_pct"] = 100.0 * (val - v) / v
        z = pd.read_csv(s["zone"], comment="#")
        z = z[(z.variant == "od1") & (z.mesh == 11)].iloc[0]
        rec["eta_C_11_Pas"] = float(z.eta_C_Pas); rec["eta_B_11_Pas"] = float(z.eta_B_Pas); rec["eta_A_11_Pas"] = float(z.eta_A_Pas)
        m = pd.read_csv(s["modes"], comment="#")
        d = qdist(m[["qa", "qb", "qc"]].values)
        near = d <= 0.15
        rec["eta_C_Rregion_R4branch_Pas"] = float(m[near & (m.omega0_partner_cm1 < 0)].eta_C.sum())
        rec["eta_C_Rshell_all_Pas"] = float(m[near].eta_C.sum())
        rec["n_modes_Rshell"] = int(near.sum())
        rec["frac_eta_C_Rregion_R4branch"] = rec["eta_C_Rregion_R4branch_Pas"] / rec["eta_C_11_Pas"]
        rec["frac_eta_C_Rshell"] = rec["eta_C_Rshell_all_Pas"] / rec["eta_C_11_Pas"]
        kl = (s["dir"] / f"{s['rta']}.kl").read_text().splitlines()
        rec["kappa_300K_W_mK"] = float([ln for ln in kl if ln.strip() and not ln.startswith("#")][-1].split()[1])
        rec["kappa_vs_calc_8p7_pct"] = 100.0 * (rec["kappa_300K_W_mK"] / 8.7 - 1.0)
        rec["kappa_vs_meas_11p0_pct"] = 100.0 * (rec["kappa_300K_W_mK"] / 11.0 - 1.0)
        freq, gam = parse_result(s["dir"] / f"{s['rta']}.result", target_temp=300)
        for lo_, hi_ in BANDS:
            taus = [1.0 / (2.0 * gam[k] * 2.0 * np.pi * 2.99792458e10) * 1e12 for k, w in freq.items()
                    if lo_ <= w < hi_ and gam.get(k, 0) > 0]
            rec[f"tau_med_ps_{lo_}_{hi_}"] = float(np.median(taus)) if taus else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(df.T.to_string())
    out = DIAG / "freeze_6p5b1.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# (freeze_6p5b1.py): internal SCPH-mesh dependence of construction C, own surface, SELF_OFFDIAG = 1,\n"
                 "# 300 K, 11^3; R region = modes whose max-overlap bare partner is bare-imaginary within |q-R| <= 0.15 (R4+ branch);\n"
                 "# tau_med = median 1/(2 Gamma) of the own 8^3 RTA per frequency band (ps); kappa from the same RTA.\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)


if __name__ == "__main__":
    main()

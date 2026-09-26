#!/usr/bin/env python3
"""T_stability_9p6.py - (J3): stability range in T of the cubic SCPH model on the internal meshes 4/8 and
4/12 (unstrained, SELF_OFFDIAG = 1): per T (100-300 K) the R-point AFD frequency, Gamma TO1, the lowest frequency on
the solution grid, imaginary counts on the KMESH_SCPH grid and on 11^3 (threshold -0.5 cm^-1, Gamma translations
excluded), SCPH iterations/residual per T (driver log). Sources: data/raw/alamode_sto/own_od1/<tag>_Tstab (100-250 K)
and own_od1/<tag> (300 K). Writes data/processed/v3_diagnostics/T_stability_9p6.csv (temp-then-rename).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from latvisc.coupling import AlamodeSet, diagonalise  # noqa: E402

OWN = REPO / "data" / "raw" / "alamode_sto" / "own_od1"
FC = REPO / "dft" / "qe" / "SrTiO3" / "dispersion_pbesol" / "SrTiO3_pbesol.444.fc"
SRC = {"i4s8": [(f"i4s8T{T}", OWN / "i4s8_Tstab", (T,)) for T in (100, 150, 200, 250)] + [("i4s8", OWN / "i4s8", (300,))],
       "i4s12": [(f"i4s12T{T}", OWN / "i4s12_Tstab", (T,)) for T in (100, 150, 200, 250)] + [("i4s12", OWN / "i4s12", (300,))],
       "i2s2": [("i2s2", OWN / "i2s2", (100, 150, 200, 250, 300, 350, 400))]}
KS = {"i4s8": 8, "i4s12": 12, "i2s2": 2}
THR = -0.5
R = (0.5, 0.5, 0.5)


def grid(n):
    return [(i / n, j / n, k / n) for i in range(n) for j in range(n) for k in range(n)]


def main():
    rows = []
    for mesh, srcs in SRC.items():
        for label, d, temps in srcs:
            log = d / f"scph_reference_{label}_od1.log"
            its = dict(re.findall(r"Temp = ([\d.e+]+) : convergence achieved in\s+(\d+) iterations", log.read_text())) if log.exists() else {}
            for T in temps:
                xml = d / f"renorm_reference_{label}_od1_{T}K.xml"
                if not xml.exists():
                    print(f"[{mesh} {T} K] missing renormalised set (not converged or not run)")
                    rows.append({"mesh": mesh, "T_K": T, "iterations": -1, "stable": False, "note": "SCPH not converged (MAXITER) or not run"}); continue
                al = AlamodeSet(xml, FC)
                rec = {"mesh": mesh, "T_K": T, "iterations": int(its.get(f"{T:.6e}", its.get(f"{float(T):.6e}", -1)))}
                wR, _ = diagonalise(al.dynmat(R, asr_onsite=True)); wG, _ = diagonalise(al.dynmat((0, 0, 0), asr_onsite=True))
                g = np.delete(np.sort(wG), np.argsort(np.abs(np.sort(wG)))[:3])
                rec["R_AFD_cm1"] = float(np.sort(wR)[0]); rec["Gamma_TO1_cm1"] = float(g[0])
                n_im = 0; wmin = np.inf
                for q in grid(KS[mesh]):
                    w = np.sort(diagonalise(al.dynmat(q, asr_onsite=True))[0])
                    if all(abs(x) < 1e-9 for x in q):
                        w = w[3:]
                    n_im += int((w < THR).sum()); wmin = min(wmin, float(w[0]))
                rec["n_imag_solution_grid"] = n_im; rec["min_omega_solution_grid_cm1"] = wmin
                npz = d / f"mesh11_reference_{label}_od1_{T}K.npz"
                if npz.exists():
                    w11 = np.load(npz)["omega_cm1"]
                    rec["n_imag_11cubed"] = int((w11[1:] < THR).sum() + (np.sort(w11[0])[3:] < THR).sum())
                    rec["min_omega_11cubed_cm1"] = float(min(w11[1:].min(), np.sort(w11[0])[3:].min()))
                rec["stable"] = bool(rec["n_imag_solution_grid"] == 0 and rec.get("n_imag_11cubed", 0) == 0)
                rows.append(rec); print(rec, flush=True)
    df = pd.DataFrame(rows).sort_values(["mesh", "T_K"])
    out = DIAG / "T_stability_9p6.csv"; tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# (T_stability_9p6.py): cubic SCPH model stability vs T on the internal meshes (own surface, SELF_OFFDIAG = 1,\n"
                 "# unstrained). Imaginary counts on the KMESH_SCPH solution grid and on 11^3, threshold -0.5 cm^-1, Gamma translations excluded;\n"
                 "# frequencies from the AlamodeSet builder on the dfc2-renormalised XML (asr_onsite). iterations = -1 if not parsed.\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out); print(df.to_string())


if __name__ == "__main__":
    main()

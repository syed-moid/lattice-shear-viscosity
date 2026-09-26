#!/usr/bin/env python3
"""z_check_11p1iii.py - (iii): anphon frequencies at Gamma/X/M/R (+ imaginary count on 8^3) of the SCPH-renormalised Z
reference (STO444_tut re-expression) at 2/2 and 2/8, compared anphon-to-anphon with the STO222 runs (tutorial_kmesh 2/2 of and bench11 2/8 od1/NONANALYTIC 3 of ). Criterion: |delta| < 0.1 cm^-1 for R, TO1, M, X, R5+.
Writes data/processed/v3_diagnostics/z_check_11p1iii.csv."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from anphon_io import gamma_centered_mesh, parse_log_frequencies, write_anphon_input  # noqa: E402

SCRATCH = Path(os.environ["DYNMAT_SCRATCH"])
ANPHON = SCRATCH / "alamode" / "_build" / "anphon" / "anphon"
Z = REPO / "data" / "raw" / "alamode_sto" / "z_tut"
Q = {"Gamma": (0, 0, 0), "Gamma+x": (0.001, 0, 0), "X": (0.5, 0, 0), "M": (0.5, 0.5, 0), "R": (0.5, 0.5, 0.5)}
REF = {"i2s2": SCRATCH / "run" / "tutorial_kmesh" / "pts_tut_i2s2_od1.log", "i2s8": SCRATCH / "run" / "bench11" / "pts_tut_i2s8_od1_na3.log",
       "i2s12": SCRATCH / "run" / "bench11" / "pts_tut_i2s12_od1_na3.log", "i2s4": SCRATCH / "run" / "tutorial_kmesh" / "pts_tut_i2s4_od1.log",
       "i2s16": SCRATCH / "run" / "tutorial_kmesh" / "pts_tut_i2s16_od1.log"}


def extract(log):
    fr = parse_log_frequencies(log)
    f = {lab: np.sort(fr[i][1]) for i, lab in enumerate(Q)}
    mesh8 = np.array([np.sort(fr[i][1]) for i in range(len(Q), len(Q) + 512)])
    g = np.delete(f["Gamma"], np.argsort(np.abs(f["Gamma"]))[:3])
    return {"TO1": g[0], "TO2": g[3], "X_lowest": f["X"][0], "M_lowest": f["M"][0], "R_AFD": f["R"][0], "R5+": f["R"][3],
            "n_imag8": int((mesh8[1:] < -0.5).sum() + (mesh8[0][3:] < -0.5).sum())}


def main():
    rows = []
    for mesh in sys.argv[1:] or ["i2s8", "i2s2"]:
        d = Z / mesh
        xml = d / f"renorm_z_reference_{mesh}_od1_300K.xml"
        if not xml.exists():
            print(f"[{mesh}] renormalised Z set missing"); continue
        p = f"pts_z_{mesh}_od1"
        if not (d / f"{p}.log").exists():
            write_anphon_input(d / f"{p}.in", p, "phonons", "STO_anharm.xml", 7.363, [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               kpoints=list(Q.values()) + gamma_centered_mesh(8), fc2xml=xml.name, born="BORN_z_reference", nonanalytic=3)
            with open(d / f"{p}.log", "w") as fh:
                subprocess.run([str(ANPHON), f"{p}.in"], cwd=d, stdout=fh, stderr=subprocess.STDOUT, check=True,
                               env=dict(os.environ, OMP_NUM_THREADS="4"))
        z = extract(d / f"{p}.log")
        ref = extract(REF[mesh]) if REF.get(mesh) and REF[mesh].exists() else {}
        for k, v in z.items():
            rows.append({"mesh": mesh, "quantity": k, "Z_STO444": v, "STO222": ref.get(k, np.nan), "delta": v - ref.get(k, np.nan)})
    df = pd.DataFrame(rows)
    out = DIAG / "z_check_11p1iii.csv"
    with open(out.with_suffix(".tmp"), "w") as fh:
        fh.write("# (iii): anphon SCPH od1 300 K on the re-expressed STO444_tut vs the original STO222 (both anphon, NONANALYTIC 3, T&T protocol).\n")
        df.to_csv(fh, index=False, float_format="%.4f")
    out.with_suffix(".tmp").rename(out)
    print(df.to_string())
    bad = df[(df.quantity != "n_imag8") & (df.delta.abs() >= 0.1)]
    print("PASS (all |delta| < 0.1 cm-1)" if bad.empty and not df.empty else f"FAIL rows:\n{bad}")


if __name__ == "__main__":
    main()

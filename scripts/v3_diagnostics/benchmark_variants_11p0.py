#!/usr/bin/env python3
"""benchmark_variants_11p0.py - single-variable SCPH runs on the tutorial surface (example STO222.xml + example
anharmonic set, cell 7.363 bohr, unstrained, 300 K, T&T protocol KMESH_INTERPOLATE 2): variants of SELF_OFFDIAG (0/1) and
NONANALYTIC (0/2/3, BORN of the example) at KMESH_SCPH 8^3 and 12^3; then RTA kappa(300 K) at 2/8 for od0 and od1 (NONANALYTIC 3)
on the 8^3 and 12^3 BTE meshes. Frequencies at Gamma/X/M/R from anphon on the dfc2-renormalised set. Writes
data/processed/v3_diagnostics/benchmark_variants_11p0.csv and data/processed/v3_diagnostics/benchmark_kappa_11p0.csv (temp-then-rename). Runs live in DYNMAT_SCRATCH/run/bench11.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from anphon_io import gamma_centered_mesh, parse_log_frequencies, write_anphon_input  # noqa: E402

SCRATCH = Path(os.environ["DYNMAT_SCRATCH"])
BUILD = SCRATCH / "alamode" / "_build"
TUT = REPO / "data" / "raw" / "alamode_sto" / "surface_tutorial"
ALAT = 7.363
ID = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
Q = {"Gamma": (0, 0, 0), "Gamma+x": (0.001, 0, 0), "X": (0.5, 0, 0), "M": (0.5, 0.5, 0), "R": (0.5, 0.5, 0.5)}
VARIANTS = [(8, 1, 3), (8, 0, 3), (8, 1, 2), (8, 0, 2), (8, 1, 0), (12, 1, 3), (12, 0, 3), (12, 1, 2), (12, 0, 2), (12, 1, 0)]


def run(cmd, cwd, log, env):
    t0 = time.time()
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=True)
    return time.time() - t0


def main():
    work = SCRATCH / "run" / "bench11"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("STO_anharm.xml", "bare_fc2.xml", "BORN"):
        if not (work / name).exists():
            (work / name).write_bytes((TUT / name).read_bytes())
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env4 = dict(os.environ, OMP_NUM_THREADS="4")
    anphon = str(BUILD / "anphon" / "anphon")
    rows = []
    out = DIAG / "benchmark_variants_11p0.csv"
    for ks, od, na in VARIANTS:
        tag = f"tut_i2s{ks}_od{od}_na{na}"
        deck = work / f"scph_{tag}.in"
        nproc, nomp = (1, 10) if ks >= 12 else (2, 5)
        if not (work / f"scph_{tag}.scph_dfc2").exists():
            write_anphon_input(deck, f"scph_{tag}", "SCPH", "STO_anharm.xml", ALAT, ID, fc2xml="bare_fc2.xml",
                               born="BORN" if na else None, nonanalytic=na, temps=(300, 300, 50), kmesh=(8, 8, 8),
                               scph=[f"SELF_OFFDIAG = {od}", "MAXITER = 400", "MIXALPHA = 0.2",
                                     "KMESH_INTERPOLATE = 2 2 2", f"KMESH_SCPH = {ks} {ks} {ks}"])
            wall = run(["mpirun", "-np", str(nproc), anphon, deck.name], work, work / f"scph_{tag}.log", dict(env, OMP_NUM_THREADS=str(nomp)))
        else:
            wall = float("nan")
        txt = (work / f"scph_{tag}.log").read_text()
        it = re.findall(r"convergence achieved in\s+(\d+) iterations", txt)
        diffs = re.findall(r"DIFF =\s+([\d.eE+-]+)", txt)
        xml = work / f"renorm_{tag}.xml"
        if not xml.exists():
            run([str(BUILD / "tools" / "dfc2"), "bare_fc2.xml", xml.name, f"scph_{tag}.scph_dfc2", "300"], work, work / f"dfc2_{tag}.log", env)
        p2 = f"pts_{tag}"
        write_anphon_input(work / f"{p2}.in", p2, "phonons", "STO_anharm.xml", ALAT, ID, kpoints=list(Q.values()) + gamma_centered_mesh(8),
                           fc2xml=xml.name, born="BORN" if na else None, nonanalytic=na)
        run([anphon, f"{p2}.in"], work, work / f"{p2}.log", env4)
        fr = parse_log_frequencies(work / f"{p2}.log")
        f = {lab: np.sort(fr[i][1]) for i, lab in enumerate(Q)}
        mesh8 = np.array([np.sort(fr[i][1]) for i in range(len(Q), len(Q) + 512)])
        n_imag8 = int((mesh8[1:] < -0.5).sum() + (mesh8[0][3:] < -0.5).sum())
        g = np.delete(f["Gamma"], np.argsort(np.abs(f["Gamma"]))[:3])
        gx = np.delete(f["Gamma+x"], np.argsort(np.abs(f["Gamma+x"]))[:3])
        lo = [v for v in gx if np.abs(g - v).min() > 0.2]
        rec = {"KMESH_SCPH": ks, "SELF_OFFDIAG": od, "NONANALYTIC": na, "iterations": int(it[0]) if it else -1,
               "final_DIFF": diffs[-1] if diffs else "", "wall_s": wall, "TO1_cm1": g[0], "TO2_cm1": g[3],
               "LO1_cm1": lo[0] if lo else np.nan, "X_lowest_cm1": f["X"][0], "M_lowest_cm1": f["M"][0],
               "R_AFD_cm1": f["R"][0], "R5+_cm1": f["R"][3], "n_imaginary_8x8x8": n_imag8}
        rows.append(rec)
        print(rec, flush=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            fh.write("# (benchmark_variants_11p0.py): tutorial surface (example STO222.xml + example anharmonic set, cell 7.363 bohr),\n"
                     "# unstrained, 300 K, KMESH_INTERPOLATE 2, MIXALPHA 0.2, TOL 1e-10; frequencies from anphon (cm-1); imaginary count on the\n"
                     "# interpolated 8^3 list excluding the three Gamma translations (threshold -0.5 cm-1). T&T Table I (300 K): R 37 (8^3), 35 (12^3);\n"
                     "# TO1 136 / 135; M 85 / 85.\n")
            fh.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                fh.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
        tmp.rename(out)
    # kappa at 2/8 (od0, od1; NA 3) on 8^3 and 12^3 BTE meshes
    krows = []
    kout = DIAG / "benchmark_kappa_11p0.csv"
    for od in (1, 0):
        tag = f"tut_i2s8_od{od}_na3"
        for kb in (8, 12):
            p3 = f"rta_{tag}_bte{kb}"
            if not (work / f"{p3}.kl").exists():
                write_anphon_input(work / f"{p3}.in", p3, "RTA", "STO_anharm.xml", ALAT, ID, fc2xml=f"renorm_{tag}.xml", born="BORN",
                                   nonanalytic=3, temps=(300, 300, 50), kmesh=(kb, kb, kb))
                wall = run(["mpirun", "-np", "4", anphon, f"{p3}.in"], work, work / f"{p3}.log", env)
            else:
                wall = float("nan")
            kl = [ln for ln in (work / f"{p3}.kl").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
            kappa = float(kl[-1].split()[1])
            krows.append({"SELF_OFFDIAG": od, "KMESH_SCPH": 8, "BTE_mesh": kb, "kappa_300K_W_mK": kappa, "vs_TT_calc_8p7_pct": 100 * (kappa / 8.7 - 1),
                          "vs_meas_11p0_pct": 100 * (kappa / 11.0 - 1), "wall_s": wall})
            print(krows[-1], flush=True)
            tmp = kout.with_suffix(".tmp")
            with open(tmp, "w") as fh:
                fh.write("# kappa(300 K) of the tutorial reproduction at 2/8 (SCPH), NONANALYTIC 3, RTA on the stated BTE mesh; T&T: SCPH q1 8^3, BTE 12^3,\n"
                         "# calculated kappa(300 K) approximately 8.7 W/mK (Fig. 7); measured 11.0 (Martelli 2018).\n")
                fh.write(",".join(krows[0].keys()) + "\n")
                for r in krows:
                    fh.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
            tmp.rename(kout)
    print("11.0 runs done", flush=True)


if __name__ == "__main__":
    main()

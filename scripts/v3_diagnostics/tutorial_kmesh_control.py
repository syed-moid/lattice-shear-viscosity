#!/usr/bin/env python3
"""tutorial_kmesh_control.py - KMESH_SCPH dependence on the tutorial surface (ALAMODE example harmonic
STO222.xml + example anharmonic set, cell 7.363 bohr), unstrained only, SELF_OFFDIAG = 1, 300 K, NONANALYTIC = 3,
TOL_SCPH 1e-10, MIXALPHA 0.2: KMESH_INTERPOLATE/KMESH_SCPH = 2/2, 4/4, 4/8, 4/12, 4/16. Per mesh: iterations,
residual, wall time; dfc2 renormalised XML; anphon frequencies at Gamma, Gamma+x, X, M, R; imaginary counts on an
8^3 list. Writes data/processed/v3_diagnostics/tutorial_kmesh_control_9p4.csv (temp-then-rename). Runs live in the scratchpad."""

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
# (label, KMESH_INTERPOLATE, KMESH_SCPH, mpi ranks, OMP threads). anphon replicates the quartic vertex array
# (nk_irred_interpolate * nk_scph * ns^4 complex, twice during its evaluation) on every rank: 4/8 8.3 GB, 4/12 28 GB,
# 4/16 66 GB peak per rank -> 4/16 does not fit the 36 GB machine and is skipped with a reason row.
MESHES = [("2/2", (2, 2, 2), (2, 2, 2), 4, 1), ("4/4", (4, 4, 4), (4, 4, 4), 4, 1), ("4/8", (4, 4, 4), (8, 8, 8), 2, 4),
          ("4/12", (4, 4, 4), (12, 12, 12), 1, 10)]
Q = {"Gamma": (0, 0, 0), "Gamma+x": (0.001, 0, 0), "X": (0.5, 0, 0), "M": (0.5, 0.5, 0), "R": (0.5, 0.5, 0.5)}


def run(cmd, cwd, log, env):
    t0 = time.time()
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=True)
    return time.time() - t0


def main():
    # optional argv: a list of meshes "ki/ks" and an output suffix, e.g. "2/4 2/8 2/12 --suffix interp2" (T&T protocol)
    global MESHES
    argv = sys.argv[1:]
    suffix = ""
    if "--suffix" in argv:
        i = argv.index("--suffix"); suffix = "_" + argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if argv:
        MESHES = [(m, tuple([int(m.split("/")[0])] * 3), tuple([int(m.split("/")[1])] * 3),
                   1 if int(m.split("/")[1]) >= 12 else (2 if int(m.split("/")[1]) >= 8 else 4),
                   10 if int(m.split("/")[1]) >= 12 else (5 if int(m.split("/")[1]) >= 8 else 2)) for m in argv]
    work = SCRATCH / "run" / "tutorial_kmesh"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("STO_anharm.xml", "bare_fc2.xml", "BORN"):
        if not (work / name).exists():
            (work / name).write_bytes((TUT / name).read_bytes())
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env4 = dict(os.environ, OMP_NUM_THREADS="4")
    anphon = str(BUILD / "anphon" / "anphon")
    rows = []
    out = DIAG / f"tutorial_kmesh_control_9p4{suffix}.csv"
    for label, ki, ks, nproc, nomp in MESHES:
        tag = f"tut_i{ki[0]}s{ks[0]}_od1"
        deck = work / f"scph_{tag}.in"
        if not (work / f"scph_{tag}.scph_dfc2").exists():
            write_anphon_input(deck, f"scph_{tag}", "SCPH", "STO_anharm.xml", ALAT, ID, fc2xml="bare_fc2.xml", born="BORN",
                               nonanalytic=3, temps=(300, 300, 50), kmesh=(8, 8, 8),
                               scph=["SELF_OFFDIAG = 1", "MAXITER = 1500", "MIXALPHA = 0.2",
                                     f"KMESH_INTERPOLATE = {ki[0]} {ki[1]} {ki[2]}", f"KMESH_SCPH = {ks[0]} {ks[1]} {ks[2]}"])
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
                           fc2xml=xml.name, born="BORN", nonanalytic=3)
        run([anphon, f"{p2}.in"], work, work / f"{p2}.log", env4)
        fr = parse_log_frequencies(work / f"{p2}.log")
        f = {lab: np.sort(fr[i][1]) for i, lab in enumerate(Q)}
        mesh8 = np.array([np.sort(fr[i][1]) for i in range(len(Q), len(Q) + 512)])
        n_imag8 = int((mesh8[1:] < -0.5).sum() + (mesh8[0][3:] < -0.5).sum())
        g = np.delete(f["Gamma"], np.argsort(np.abs(f["Gamma"]))[:3])
        gx = np.delete(f["Gamma+x"], np.argsort(np.abs(f["Gamma+x"]))[:3])
        lo = [v for v in gx if np.abs(g - v).min() > 0.2]
        rec = {"mesh": label, "iterations": int(it[0]) if it else -1, "final_DIFF": diffs[-1] if diffs else "",
               "wall_s": wall, "TO1_cm1": g[0], "TO2_cm1": g[3], "LO1_cm1": lo[0] if lo else np.nan, "F2u_cm1": g[6],
               "X_lowest_cm1": f["X"][0], "M_lowest_cm1": f["M"][0], "R_AFD_cm1": f["R"][0], "R_second_cm1": f["R"][1],
               "R5+_cm1": f["R"][3], "n_imaginary_8x8x8": n_imag8}
        rows.append(rec)
        print(rec, flush=True)
        if label == "4/12" and not suffix:
            rows.append({k: ("4/16" if k == "mesh" else ("not run: 66 GB peak per rank exceeds 36 GB RAM" if k == "final_DIFF" else np.nan)) for k in rec})
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            fh.write("# (tutorial_kmesh_control.py): tutorial surface (example STO222.xml harmonic set + example anharmonic set,\n"
                     "# cell 7.363 bohr), unstrained, SELF_OFFDIAG = 1, 300 K, TOL 1e-10; frequencies from anphon (cm-1); imaginary count\n"
                     "# on the interpolated 8^3 list excluding the three Gamma translations (threshold -0.5 cm-1).\n")
            fh.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                fh.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
        tmp.rename(out)


if __name__ == "__main__":
    main()

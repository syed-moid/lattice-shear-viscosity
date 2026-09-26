#!/usr/bin/env python3
"""run_scph_surface.py - SCPH driver: anphon SCPH (SELF_OFFDIAG = 1) for the
unstrained and shear-strained converted harmonic sets, dfc2 renormalised FC2 XML per T, PRINTEVEC on the
11^3 list, and the 8^3 RTA of the unstrained set per T.

Mesh tags: i2s2 = KMESH_INTERPOLATE 2x2x2 / KMESH_SCPH 2x2x2; i4s4 = 4/4; i4s8 = 4/8. IFC supercell of
the anharmonic set: 2x2x2 (fixed). Long runs live in the work directory DYNMAT_SCRATCH (an ALAMODE build
under DYNMAT_SCRATCH/alamode/_build and the input sets under DYNMAT_SCRATCH/run); results are copied to OUT when
complete (temp name + rename). Skips what exists. Appends timing rows to OUT/runs.csv.

Usage: DYNMAT_SCRATCH=<alamode_work> uv run python run_scph_surface.py --mesh i4s8 --temps 300 \
         --sets reference shear_xy_p005 shear_xy_m005 --out data/raw/alamode_sto/own_od1/i4s8 [--xml-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from anphon_io import gamma_centered_mesh, parse_evec, parse_log_frequencies, write_anphon_input  # noqa: E402

SCRATCH = Path(os.environ["DYNMAT_SCRATCH"])
BUILD = SCRATCH / "alamode" / "_build"
ALAT = 7.356754
MESH = {"i2s2": ((2, 2, 2), (2, 2, 2)), "i4s4": ((4, 4, 4), (4, 4, 4)), "i4s8": ((4, 4, 4), (8, 8, 8)),
        "i4s12": ((4, 4, 4), (12, 12, 12)), "i4s16": ((4, 4, 4), (16, 16, 16)), "i2s16": ((2, 2, 2), (16, 16, 16)),
        "i8s16": ((8, 8, 8), (16, 16, 16)), "i2s8": ((2, 2, 2), (8, 8, 8)),
        "i2s4": ((2, 2, 2), (4, 4, 4)), "i2s12": ((2, 2, 2), (12, 12, 12)), "i3s12": ((3, 3, 3), (12, 12, 12))}
SHEAR = {"reference": 0.0, "shear_xy_p005": 0.005, "shear_xy_m005": -0.005, "shear_xy_p010": 0.010, "shear_xy_m010": -0.010}


def shear_of(name):
    """engineering-shear cell of a set name: reference -> 0; shear_xy_p005[_suffix] -> +0.005, m -> -."""
    m = re.search(r"shear_xy_([pm])(\d{3})", name)
    if not m:
        return 0.0
    return (1 if m.group(1) == "p" else -1) * int(m.group(2)) / 1000.0


def cell_rows(s):
    return [[1.0, s, 0.0], [s, 1.0, 0.0], [0.0, 0.0, 1.0]]


def run(cmd, cwd, log, env):
    t0 = time.time()
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, choices=list(MESH))
    ap.add_argument("--temps", type=int, nargs="+", default=[300])
    ap.add_argument("--sets", nargs="+", default=["reference", "shear_xy_p005", "shear_xy_m005"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--xml-dir", type=Path, default=SCRATCH / "run" / "convert",
                    help="directory with <set>_full_fc2.xml and BORN_<set>")
    ap.add_argument("--xml-suffix", default="_full_fc2.xml")
    ap.add_argument("--label", default=None, help="label used in output file names (default: mesh tag)")
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--omp", type=int, default=1, help="OMP_NUM_THREADS for the SCPH/RTA ranks (memory: the quartic vertex array is replicated per rank)")
    ap.add_argument("--skip-rta", action="store_true")
    ap.add_argument("--self-offdiag", type=int, default=1)
    ap.add_argument("--mixalpha", type=float, default=0.2, help="SCPH MIXALPHA (0.2 for unstrained sets; 0.1 for strained sets, where 0.2 can produce periodic limit cycles)")
    ap.add_argument("--maxiter", type=int, default=1500)
    ap.add_argument("--alat", type=float, default=ALAT, help="cell parameter in bohr (own surface 7.356754; tutorial/Z surface 7.363)")
    args = ap.parse_args()
    label = args.label or args.mesh
    alat = args.alat
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = SCRATCH / "run" / f"prod_{label}"
    work.mkdir(parents=True, exist_ok=True)
    anharm = SCRATCH / "run" / "own_surface_asr" / "STO_anharm.xml"
    for d in (work, out):
        if not (d / "STO_anharm.xml").exists():
            (d / "STO_anharm.xml").symlink_to(anharm.resolve())
    env = dict(os.environ, OMP_NUM_THREADS=str(args.omp))
    env4 = dict(os.environ, OMP_NUM_THREADS="4")
    anphon = str(BUILD / "anphon" / "anphon")
    kint, kscph = MESH[args.mesh]
    tmin, tmax = min(args.temps), max(args.temps)
    dt = 50 if len(args.temps) > 1 else 50
    runs_csv = out / "runs.csv"
    if not runs_csv.exists():
        runs_csv.write_text("set,mesh,SELF_OFFDIAG,T_K,stage,wall_s,iterations,final_DIFF\n")

    def log_row(s, T, stage, wall, it="", diff=""):
        with open(runs_csv, "a") as fh:
            fh.write(f"{s},{args.mesh},{args.self_offdiag},{T},{stage},{wall:.1f},{it},{diff}\n")

    for s in args.sets:
        xml = args.xml_dir / f"{s}{args.xml_suffix}"
        born = args.xml_dir / f"BORN_{s}"
        for f in (xml, born):
            for d in (work, out):
                if not (d / f.name).exists():
                    shutil.copy(f, d / f.name)
        prefix = f"scph_{s}_{label}_od{args.self_offdiag}"
        dfc2_file = work / f"{prefix}.scph_dfc2"
        if not dfc2_file.exists():
            deck = work / f"{prefix}.in"
            write_anphon_input(deck, prefix, "SCPH", "STO_anharm.xml", alat, cell_rows(shear_of(s)),
                               fc2xml=xml.name, born=born.name, nonanalytic=3, temps=(tmin, tmax, dt), kmesh=(8, 8, 8),
                               scph=[f"SELF_OFFDIAG = {args.self_offdiag}", f"MAXITER = {args.maxiter}", f"MIXALPHA = {args.mixalpha}",
                                     f"KMESH_INTERPOLATE = {kint[0]} {kint[1]} {kint[2]}",
                                     f"KMESH_SCPH = {kscph[0]} {kscph[1]} {kscph[2]}"])
            wall = run(["mpirun", "-np", str(args.np), anphon, deck.name], work, work / f"{prefix}.log", env)
            txt = (work / f"{prefix}.log").read_text()
            its = re.findall(r"Temp = ([\d.e+]+) : convergence achieved in\s+(\d+) iterations", txt)
            diffs = re.findall(r"DIFF =\s+([\d.eE+-]+)", txt)
            n_temps = len(range(tmin, tmax + 1, dt))
            log_row(s, f"{tmin}-{tmax}", "scph", wall, ";".join(f"{float(t):.0f}K:{n}" for t, n in its) + f";alpha={args.mixalpha}",
                    diffs[-1] if diffs else "")
            if len(its) < n_temps:
                for f in list(work.glob(f"{prefix}.*")):
                    if f.suffix not in (".in", ".unconverged"):
                        f.rename(f.with_name(f.name + ".unconverged"))
                print(f"[{label}] SCPH {s}: NOT CONVERGED ({len(its)}/{n_temps} temperatures, last DIFF {diffs[-1] if diffs else 'n/a'}); outputs renamed *.unconverged, set skipped", flush=True)
                continue
            for f in work.glob(f"{prefix}.*"):
                shutil.copy(f, out / f.name)
            print(f"[{label}] SCPH {s}: {wall:.0f} s, {its}", flush=True)
        for T in args.temps:
            tag = f"{s}_{label}_od{args.self_offdiag}_{T}K"
            rxml = out / f"renorm_{tag}.xml"
            if not rxml.exists():
                tmp = out / f"renorm_{tag}.xml.tmp"
                run([str(BUILD / "tools" / "dfc2"), xml.name, tmp.name, str(dfc2_file.resolve()), str(T)],
                    out, out / f"dfc2_{tag}.log", env)
                tmp.rename(rxml)
            npz = out / f"mesh11_{tag}.npz"
            if not npz.exists():
                p2 = f"mesh11_{tag}"
                write_anphon_input(out / f"{p2}.in", p2, "phonons", "STO_anharm.xml", alat, cell_rows(shear_of(s)),
                                   kpoints=gamma_centered_mesh(11), fc2xml=rxml.name, born=born.name, nonanalytic=3,
                                   analysis=["PRINTEVEC = 1"])
                wall = run([anphon, f"{p2}.in"], out, out / f"{p2}.log", env4)
                fr = parse_log_frequencies(out / f"{p2}.log")
                _, _, ev = parse_evec(out / f"{p2}.evec")
                omega = np.zeros((1331, 15)); evec = np.zeros((1331, 15, 15), dtype=complex)
                for iq in range(1331):
                    order = np.argsort(fr[iq][1]); omega[iq] = fr[iq][1][order]; evec[iq] = ev[iq][order]
                np.savez_compressed(out / f"{p2}.npz.tmp.npz", omega_cm1=omega, evec=evec, q_frac=np.array(gamma_centered_mesh(11)))
                (out / f"{p2}.npz.tmp.npz").rename(npz)
                (out / f"{p2}.evec").unlink()
                log_row(s, T, "mesh11", wall)
            if s.endswith("reference") and not args.skip_rta:
                p3 = f"STO_RTA_{label}_od{args.self_offdiag}_{T}K"
                if not (out / f"{p3}.result").exists():
                    write_anphon_input(out / f"{p3}.in", p3, "RTA", "STO_anharm.xml", alat, cell_rows(0.0),
                                       fc2xml=rxml.name, born=born.name, nonanalytic=3, temps=(T, T, 50), kmesh=(8, 8, 8))
                    wall = run(["mpirun", "-np", str(args.np), anphon, f"{p3}.in"], out, out / f"{p3}.log", env)
                    log_row(s, T, "rta", wall)
                    print(f"[{label}] RTA {T} K: {wall:.0f} s", flush=True)
    print(f"[{label}] all done", flush=True)


if __name__ == "__main__":
    main()

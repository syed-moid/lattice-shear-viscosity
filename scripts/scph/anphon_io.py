#!/usr/bin/env python3
"""Small I/O helpers for the local ALAMODE (anphon) runs of the EPJ B revision.

  write_anphon_input   : &general/&cell/&kpoint(/&scph/&analysis) deck writer
  parse_log_frequencies: frequencies (cm-1) printed by MODE = phonons, KPMODE = 0
  parse_evec           : PREFIX.evec eigenvectors (mass-weighted, orthonormal)
  parse_result         : .result file of MODE = RTA (frequencies, Gamma per T)
  gamma_centered_mesh  : the production 11^3 fractional mesh, same order as
                         latvisc.matdyn_input.gamma_centered_mesh
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def gamma_centered_mesh(n: int = 11):
    return [(i / n, j / n, k / n) for i in range(n) for j in range(n) for k in range(n)]


def write_anphon_input(path, prefix, mode, fcsxml, cell_alat, cell_rows, kpoints=None,
                       fc2xml=None, born=None, nonanalytic=3, temps=None, extra_general=None,
                       scph=None, analysis=None, kmesh=None):
    """cell_rows: three lattice vectors in units of cell_alat (rows)."""
    g = [f" PREFIX = {prefix}", f" MODE = {mode}", " NKD = 3; KD = Sr Ti O", f" FCSXML = {fcsxml}"]
    if fc2xml:
        g.append(f" FC2XML = {fc2xml}")
    if nonanalytic and born:
        g.append(f" NONANALYTIC = {nonanalytic}; BORNINFO = {born}")
    else:
        g.append(" NONANALYTIC = 0")
    if temps is not None:
        tmin, tmax, dt = temps
        g.append(f" TMIN = {tmin}; TMAX = {tmax}; DT = {dt}")
    if extra_general:
        g += [f" {x}" for x in extra_general]
    lines = ["&general"] + g + ["/", "", "&cell", f"{cell_alat:.9f}"]
    lines += [" ".join(f"{v:.9f}" for v in row) for row in cell_rows]
    lines += ["/", ""]
    if scph:
        lines += ["&scph"] + [f" {x}" for x in scph] + ["/", ""]
    if analysis:
        lines += ["&analysis"] + [f" {x}" for x in analysis] + ["/", ""]
    lines += ["&kpoint"]
    if kmesh is not None:
        lines += ["2", " ".join(str(x) for x in kmesh)]
    else:
        lines += ["0"] + [f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f}" for q in kpoints]
    lines += ["/", ""]
    Path(path).write_text("\n".join(lines))


_BLOCK = re.compile(r"# k point\s+(\d+) : \(([^)]+)\)\s*\n\s*Mode, Frequency\s*\n((?:\s*\d+\s+[-\d.]+ cm\^-1.*\n)+)")


def parse_log_frequencies(log):
    """List of (q, freqs_cm1) in file order for KPMODE = 0 runs."""
    out = []
    for _, q, body in _BLOCK.findall(Path(log).read_text()):
        out.append((np.array([float(x) for x in q.split(",")]),
                    np.array([float(l.split()[1]) for l in body.strip().splitlines()])))
    return out


def parse_evec(path):
    """Returns (kpoints (nk,3), eigenvalues (nk, nmodes), evec (nk, nmodes, nmodes) complex),
    evec[k, mode, component] with components atom-major (x, y, z per atom)."""
    txt = Path(path).read_text().splitlines()
    nmodes = nk = None
    for line in txt:
        if line.startswith("# Number of phonon modes"):
            nmodes = int(line.split(":")[1])
        elif line.startswith("# Number of k points"):
            nk = int(line.split(":")[1])
    kpts = np.zeros((nk, 3))
    evals = np.zeros((nk, nmodes))
    evec = np.zeros((nk, nmodes, nmodes), dtype=complex)
    ik = im = -1
    ic = 0
    for line in txt:
        if line.startswith("## kpoint"):
            ik += 1
            kpts[ik] = [float(x) for x in line.split(":")[1].split()]
            im = -1
        elif line.startswith("### mode"):
            im += 1
            evals[ik, im] = float(line.split(":")[1])
            ic = 0
        elif ik >= 0 and im >= 0 and line.strip() and not line.startswith("#"):
            re_, im_ = (float(x) for x in line.split())
            evec[ik, im, ic] = complex(re_, im_)
            ic += 1
    return kpts, evals, evec


def parse_result(path, target_temp=300):
    """Frequencies and Gamma (cm-1) of an RTA .result file at one temperature,
    keyed by (q index 1-based, branch 1-based); also the irreducible k list
    with weights."""
    text = Path(path).read_text()
    kblock = re.search(r"#KPOINT\n(\d+ \d+ \d+)\n(\d+)\n(.*?)#END KPOINT", text, re.S)
    klist = []
    for line in kblock.group(3).strip().splitlines():
        parts = line.replace(":", " ").split()
        klist.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    freq = {}
    sec = re.search(r"##Phonon Frequency\n#[^\n]*\n(.*?)##END Phonon Frequency", text, re.S).group(1)
    for line in sec.strip().splitlines():
        q, b, w = line.split()
        freq[(int(q), int(b))] = float(w)
    tmin, tmax, tstep = (float(x) for x in re.search(r"#TEMPERATURE\n([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+)", text).groups())
    temps = [tmin] if tstep == 0 or tmax == tmin else list(np.arange(tmin, tmax + tstep / 2, tstep))
    t_idx = min(range(len(temps)), key=lambda i: abs(temps[i] - target_temp))
    gamma = {}
    for q, b, mult, body in re.findall(r"#GAMMA_EACH\n(\d+) (\d+)\n(\d+)\n((?:.*\n)*?)#END GAMMA_EACH", text):
        lines = body.strip().splitlines()
        vals = [float(x) for x in lines[int(mult):]]
        gamma[(int(q), int(b))] = vals[t_idx]
    return klist, freq, gamma

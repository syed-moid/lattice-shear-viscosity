#!/usr/bin/env python3
"""Parse matdyn.x frequency files into dispersion CSVs.

Reads dft/qe/<material>/dispersion/<material>.freq (matdyn.x output along
Gamma-X-M-Gamma-R-X, q in 2*pi/a cartesian units, frequencies in cm^-1;
imaginary modes printed as negative) and writes
data/processed/harmonic_dispersion_<material>.csv with columns:

    path_index, qx, qy, qz, path_coord, segment, branch, omega_cm1, omega_thz

Sign convention is preserved: negative omega means an imaginary
(unstable) harmonic mode.

Usage: uv run python scripts/parse_qe_phonons.py [SrTiO3] [BaTiO3]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CM1_TO_THZ = 0.0299792458  # c in cm/s * 1e-12: ordinary-frequency THz per cm^-1

REPO = Path(__file__).resolve().parent.parent

# High-symmetry points of the simple-cubic path, in 2*pi/a units,
# matching matdyn.in (40 interpolation points per segment).
PATH_LABELS = {0: "Gamma", 40: "X", 80: "M", 120: "Gamma", 160: "R", 200: "X"}
SEGMENTS = ["Gamma-X", "X-M", "M-Gamma", "Gamma-R", "R-X"]


def parse_freq_file(path: Path):
    """Return (nbnd, list of (q, path_coord, freqs_cm1)) from a matdyn .freq file."""
    text = path.read_text()
    header = re.match(r"\s*&plot\s+nbnd=\s*(\d+),\s*nks=\s*(\d+)\s*/", text)
    if not header:
        raise ValueError(f"{path}: not a matdyn frequency file")
    nbnd, nks = int(header.group(1)), int(header.group(2))
    tokens = text[header.end():].split()
    values = [float(t) for t in tokens]

    # Each q-point block is a q line followed by nbnd frequencies. The q line
    # has either 3 (qx qy qz) or 4 (qx qy qz path_coord) numbers depending on
    # the matdyn build; detect from the total token count.
    per_point = len(values) // nks
    q_width = per_point - nbnd
    if per_point * nks != len(values) or q_width not in (3, 4):
        raise ValueError(f"{path}: unexpected token layout ({len(values)} values, nks={nks})")

    points = []
    running_coord = 0.0
    previous_q = None
    for i in range(nks):
        block = values[i * per_point : (i + 1) * per_point]
        q = tuple(block[:3])
        if q_width == 4:
            coord = block[3]
        else:
            if previous_q is not None:
                running_coord += sum((a - b) ** 2 for a, b in zip(q, previous_q)) ** 0.5
            coord = running_coord
            previous_q = q
        points.append((q, coord, block[q_width:]))
    return nbnd, points


def segment_of(path_index: int) -> str:
    return SEGMENTS[min(path_index // 40, len(SEGMENTS) - 1)]


def write_csv(material: str) -> Path:
    freq_file = REPO / "dft" / "qe" / material / "dispersion" / f"{material}.freq"
    out_file = REPO / "data" / "processed" / f"harmonic_dispersion_{material}.csv"
    nbnd, points = parse_freq_file(freq_file)

    lines = [
        f"# harmonic_dispersion_{material}.csv - produced by scripts/parse_qe_phonons.py",
        f"# source: {freq_file.relative_to(REPO)} (matdyn.x along Gamma-X-M-Gamma-R-X, "
        "q in 2*pi/a units)",
        "# omega_cm1 < 0 denotes an imaginary (unstable) harmonic mode",
        "path_index,qx,qy,qz,path_coord,segment,branch,omega_cm1,omega_thz",
    ]
    for index, (q, coord, freqs) in enumerate(points):
        for branch, omega_cm1 in enumerate(freqs, start=1):
            lines.append(
                f"{index},{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{coord:.6f},"
                f"{segment_of(index)},{branch},{omega_cm1:.4f},{omega_cm1 * CM1_TO_THZ:.6f}"
            )
    out_file.write_text("\n".join(lines) + "\n")
    print(f"{material}: {len(points)} q-points x {nbnd} branches -> {out_file.relative_to(REPO)}")
    return out_file


if __name__ == "__main__":
    materials = sys.argv[1:] or ["SrTiO3", "BaTiO3"]
    for material in materials:
        write_csv(material)

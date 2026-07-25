#!/usr/bin/env python3
"""Unstable-manifold fractions on the phono3py target mesh (11x11x11).

Reads the uniform-mesh matdyn.x frequency files (1331 Gamma-centered
q-points in crystal coordinates, 15 branches) for each material and
functional, and tabulates:

    mode_fraction : unstable modes / all modes (15 x 1331)
    q_fraction    : q-points with at least one unstable mode / 1331
                    (the BZ-volume fraction of the unstable pockets)

Writes data/processed/unstable_bz_fraction.csv. The per-mode mask itself
is Stage B material (unstable_mask_<material>.csv, built in B2-H).

Usage: uv run python scripts/unstable_fraction.py
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SOURCES = {
    ("SrTiO3", "pbe"): "dft/qe/mesh_audit/SrTiO3.rebuilt.mesh11.freq",
    ("BaTiO3", "pbe"): "dft/qe/mesh_audit/BaTiO3.rebuilt.mesh11.freq",
    ("SrTiO3", "pbesol"): "dft/qe/mesh_audit/SrTiO3_pbesol.mesh11.freq",
    ("BaTiO3", "pbesol"): "dft/qe/BaTiO3/dispersion_pbesol/BaTiO3_pbesol.mesh11.freq",
}
THRESHOLD_CM1 = -1.0  # tolerance for acoustic zeros / interpolation noise


def mesh_fractions(path: Path):
    import re
    text = path.read_text()
    header = re.match(r"\s*&plot\s+nbnd=\s*(\d+),\s*nks=\s*(\d+)\s*/", text)
    nbnd, nks = int(header.group(1)), int(header.group(2))
    values = [float(t) for t in text[header.end():].split()]
    per_point = len(values) // nks
    q_width = per_point - nbnd
    unstable_modes = 0
    unstable_qpoints = 0
    for i in range(nks):
        freqs = values[i * per_point + q_width : (i + 1) * per_point]
        bad = sum(f < THRESHOLD_CM1 for f in freqs)
        unstable_modes += bad
        unstable_qpoints += bad > 0
    return nbnd, nks, unstable_modes, unstable_qpoints


def main() -> None:
    rows = ["material,functional,mesh,n_modes_total,n_modes_unstable,"
            "mode_fraction,n_q_unstable,q_fraction"]
    for (material, functional), rel in SOURCES.items():
        path = REPO / rel
        if not path.exists():
            print(f"skip {material} [{functional}]: {rel} missing")
            continue
        nbnd, nks, bad_modes, bad_q = mesh_fractions(path)
        rows.append(
            f"{material},{functional},11x11x11,{nbnd * nks},{bad_modes},"
            f"{bad_modes / (nbnd * nks):.4f},{bad_q},{bad_q / nks:.4f}"
        )
        print(f"{material:8s} [{functional:6s}]  modes: {bad_modes}/{nbnd*nks} "
              f"({100*bad_modes/(nbnd*nks):.1f}%)   q-points: {bad_q}/{nks} "
              f"({100*bad_q/nks:.1f}%)")
    out = REPO / "data" / "processed" / "unstable_bz_fraction.csv"
    header = [
        "# unstable_bz_fraction.csv - produced by scripts/unstable_fraction.py",
        "# uniform 11x11x11 Gamma-centered mesh (matdyn.x, crystal coords);",
        "# unstable = omega < -1 cm^-1; q_fraction = BZ-volume fraction of",
        "# q-points hosting at least one unstable mode",
    ]
    out.write_text("\n".join(header) + "\n" + "\n".join(rows) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
